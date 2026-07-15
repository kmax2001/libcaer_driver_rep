#!/usr/bin/env python3
# Regenerate te.ts (the TensorRT target_encoder engine) FOR THE LOCAL GPU.
#
# The TRT engine in te.ts is GPU-architecture + TensorRT-version specific, so it
# must be rebuilt on each new x86_64 target GPU. Run this ON THE TARGET MACHINE —
# torch_tensorrt auto-detects the local GPU; you do NOT pass the SM/arch.
#
# Requirements on the target: torch 2.5.1(+cu121), torch_tensorrt 2.5.0,
# tensorrt 10.3.0, CUDA, and the BlinkTrack repo + event weights.
#
# Usage:
#   export BLINKTRACK_ROOT=/path/to/BlinkTrack          # (default below)
#   PYTHONNOUSERSITE=1 python3 regen_te.py <out_dir>    # writes <out_dir>/te.ts
#     e.g. <out_dir> = .../install/libcaer_driver_eventtrack/share/.../models
import os, sys
import numpy as np, torch, torch_tensorrt
import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/te_regen"
os.makedirs(OUT, exist_ok=True)
BT = os.environ.get("BLINKTRACK_ROOT", "/mnt/ssd_kyh/youngho_ws/BlinkTrack")
sys.path.insert(0, BT); os.chdir(BT)
import h5py, hdf5plugin  # noqa
pr = lambda *a: print(*a, flush=True)

pr(f"GPU: {torch.cuda.get_device_name(0)}  (compute {torch.cuda.get_device_capability(0)})")
pr(f"torch {torch.__version__} | torch_tensorrt {torch_tensorrt.__version__}")

# --- load ONLY the model (no EC env) to get target_encoder ---
with initialize_config_dir(config_dir=os.path.join(BT, "configs"), version_base=None):
    cfg = compose(config_name="eval_real_defaults_rl_infer")
OmegaConf.set_struct(cfg, True)
with open_dict(cfg):
    cfg.model.representation = cfg.representation
model = hydra.utils.instantiate(cfg.model, _recursive_=False)
sd = torch.load(cfg.event_weights_path, map_location="cuda:0")["state_dict"]
sd = {k.replace("joint_encoder", "lstm_predictor"): v for k, v in sd.items()}
model.load_state_dict(sd, strict=False)
te = model.to("cuda").eval().target_encoder.eval()

# --- trace, then torch_tensorrt-compile (fp32; fp16 breaks the chaotic loop) ---
ex = torch.rand(1, 10, 62, 62, device="cuda")          # te input: (1,10,62,62)
traced = torch.jit.trace(te, ex)
trt = torch_tensorrt.compile(
    traced, ir="ts",
    inputs=[torch_tensorrt.Input((1, 10, 62, 62), dtype=torch.float32)],
    enabled_precisions={torch.float32},
    require_full_compilation=False,
)
torch.jit.save(trt, os.path.join(OUT, "te.ts"))

# --- validate vs eager ---
with torch.no_grad():
    o_eager = te(ex)
    o_trt = torch.jit.load(os.path.join(OUT, "te.ts")).cuda()(ex)
d = (o_trt - o_eager).abs().max().item()
pr(f"saved {OUT}/te.ts   out shape {tuple(o_trt.shape)}   max|trt-eager| = {d:.3e}")
pr("OK — te.ts valid" if d < 1e-2 else "WARNING — large TRT deviation, check precision")
