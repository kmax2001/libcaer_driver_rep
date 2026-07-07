#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# BlinkTrackStreamer: reusable streaming core for the RL-gated event tracker.
#
# Wraps the (validated) BlinkTrack ECSubseq_Venv + PPO policy so it can be
# driven frame-by-frame from a live source (ROS /events_rep) instead of reading
# precomputed .h5 files. The only change vs the offline pipeline is that the
# ECSubseq.events_2 generator pulls TimeSurface frames from a queue; everything
# else (crop/stack/concat_ts/reference/RL-gate/tracker/keypoint-update) is the
# original, tested code. Validated to be bit-identical to the offline reference.
#
# Must run in the BlinkTrack env (gostop): torch+SB3+pytorch_lightning, with the
# BlinkTrack repo on PYTHONPATH and PYTHONNOUSERSITE=1.
# -----------------------------------------------------------------------------
import os
import queue
import types

import numpy as np
import torch

from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf, open_dict
import hydra

from src.evaluate_eds import CornerConfig
from policy.policies import get_blueprint
from loader.loader_ec_rl_forinfer import ECSubseq_Venv
from util.data import extract_glimpse, concat_ts
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize


def _streaming_events_2(self, use_tqdm=False):
    """ECSubseq.events_2 with the frame source swapped .h5 -> queue.
    Per-frame body is copied verbatim from the offline generator (non-pose_mode).
    Put a (H, W, C) float32 array to advance; put None to stop."""
    self.dones = np.zeros(self.n_tracks, dtype=bool)
    while True:
        input_1 = self._stream_q.get()
        if input_1 is None:
            return
        self.t_now += self.dt
        input_1 = np.array(input_1)
        input_1 = np.transpose(input_1, (2, 0, 1))
        input_1 = torch.from_numpy(input_1).unsqueeze(0).to(self.u_centers.device)
        n = self.u_centers.size(0)
        x = extract_glimpse(input_1.repeat(n, 1, 1, 1),
                            (self.patch_size, self.patch_size), self.u_centers.detach() + 0.5)
        xx = extract_glimpse(input_1.repeat(n, 1, 1, 1),
                             (self.patch_size * 2, self.patch_size * 2), self.u_centers.detach() + 0.5)
        for i in range(len(x)):
            self.stacked_event_x[i].append(x[i])
            self.stacked_event_xx[i].append(xx[i])
        x = torch.stack([concat_ts(self.stacked_event_x[i], len(self.stacked_event_x[i]))
                         for i in range(len(x))], dim=0)
        xx = torch.stack([concat_ts(self.stacked_event_xx[i], len(self.stacked_event_xx[i]))
                          for i in range(len(xx))], dim=0)
        x = torch.cat([x, self.x_ref], dim=1)
        xx = torch.cat([xx, self.x_ref_2], dim=1)
        self.x = x.clone()
        self.xx = xx.clone()
        yield self.t_now, x, xx, None


class BlinkTrackStreamer:
    """Frame-driven RL-gated tracker.

    Lifecycle:
        s = BlinkTrackStreamer(init_keypoints=[[126, 37]])
        s.push(frame_hwc)          # from each /events_rep msg (H,W,C float32)
        s.reset()                  # blocks until first frame; sets up state
        while ...:
            kp, action = s.step()  # blocks until next frame; returns (N,2), (N,)
    """

    def __init__(self, init_keypoints, config_name="eval_real_defaults_rl_infer",
                 config_dir=None, seq_name="boxes_translation_330_410", dt=0.01,
                 device="cuda"):
        bt = os.environ.get("BLINKTRACK_ROOT", "/mnt/ssd_kyh/youngho_ws/BlinkTrack")
        config_dir = config_dir or os.path.join(bt, "configs")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name=config_name)
        OmegaConf.set_struct(cfg, True)
        with open_dict(cfg):
            cfg.model.representation = cfg.representation
        self.cfg = cfg

        # tracker (obs_encoder)
        model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        sd = torch.load(cfg.event_weights_path, map_location=f"{device}:0")["state_dict"]
        sd = {k.replace("joint_encoder", "lstm_predictor"): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        self.model = model.to(device).eval()

        # env (reused for crop/stack/state + reference-patch init from the sequence)
        policy_dict = get_blueprint(cfg.policy_model)
        corner_config = CornerConfig(30, 0.3, 15, 0.15, False, 11)
        init_kp = np.asarray(init_keypoints, dtype=np.float32).reshape(-1, 2)
        env = ECSubseq_Venv(f"{cfg.eval_dataset_path}/ec_subseq", seq_name, -1,
                            cfg.patch_size, cfg.representation, dt, corner_config,
                            image_folder=cfg.image_folder, event_folder=cfg.event_folder,
                            transform_p=cfg.transform_p, obs_dict=policy_dict,
                            num_envs=init_kp.shape[0])
        env.load_obs_encoder(self.model)
        env = VecMonitor(env)
        env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)
        env = VecNormalize.load(str(cfg.policy_path).replace("model.zip", "rms.pkl"), env)
        env.envs.override_keypoints(init_kp)
        env.envs.prepare_list()
        self.model.reset(env.envs.n_tracks)

        rl = PPO.load(cfg.policy_path, env=env, device=device)
        self.policy = rl.policy
        self.policy.eval()
        self.env = env

        # swap frame source: .h5 -> queue
        self._q = queue.Queue()
        sub = env.envs
        sub._stream_q = self._q
        sub.events_2 = types.MethodType(_streaming_events_2, sub)
        self._obs = None

    def push(self, frame_hwc):
        """Enqueue a (H, W, C) float32 TimeSurface frame (or None to stop)."""
        self._q.put(frame_hwc)

    def reset(self):
        """Consume the first frame and build the initial observation."""
        self._obs = self.env.reset()
        return self.env.envs.u_centers.detach().cpu().numpy()

    def step(self):
        """Advance one frame. Returns (keypoints (N,2), action (N,))."""
        action, _ = self.policy.predict(self._obs, state=None, episode_start=None,
                                        deterministic=True)
        self._obs, _, dones, _ = self.env.step(action)
        kp = self.env.envs.u_centers.detach().cpu().numpy()
        return kp, np.asarray(action).reshape(-1)

    @property
    def track_data(self):
        return np.asarray(self.env.tracks_pred.track_data)


# --------- standalone self-test: class output must match ref_track0.txt ---------
if __name__ == "__main__":
    import glob
    import h5py, hdf5plugin  # noqa
    from util.data import read_input

    SP = "/tmp/claude-1000/-home-kyh-libcaer-driver-rep/fa84d16b-5f2a-4bca-a4c5-b9d77a21efec/scratchpad"
    s = BlinkTrackStreamer(init_keypoints=[[126.0, 37.0]])
    sub = s.env.envs
    dt_us = int(round(sub.dt * 1e6))
    n_events = sub.n_events
    for i in range(1, n_events):
        f = sub.dir_representation / f"{str(int(i * dt_us)).zfill(7)}.h5"
        s.push(np.array(read_input(f, sub.representation)))
    s.push(None)

    s.reset()
    for _ in range(n_events - 2):
        s.step()

    td = s.track_data
    ref = np.loadtxt(os.path.join(SP, "ref_track0.txt"))
    d = np.abs(td[:, 2:] - ref[:, 2:])
    print(f"class rows={td.shape} ref={ref.shape} max|dx,dy|={d.max():.6f} mean={d.mean():.6f}")
    print("IDENTICAL" if d.max() < 1e-3 else "DIFFERS")
