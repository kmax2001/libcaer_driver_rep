#! /usr/bin/env python3
# -----------------------------------------------------------------------------
# Loader + visualizer for the tracker's RL I/O capture (written by the tracker
# node on a 'd'-key trigger). Point it at a capture dir:
#     python3 log_view.py /tmp/blinktrack_rllog/capture_<stamp>
# Run with a python that has matplotlib (e.g. the gostop conda env).
#
# Each frame_*.bin = float32: feat[10*62*62] + stacked_length + action
#                             + logit_accumulate + logit_fire + kp_x + kp_y
# -----------------------------------------------------------------------------
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

FEAT_N = 10 * 62 * 62  # 38720


def load(capdir):
    files = sorted(glob.glob(os.path.join(capdir, 'frame_*.bin')))
    if not files:
        sys.exit(f'no frame_*.bin in {capdir}')
    feats, scal = [], []
    for f in files:
        a = np.fromfile(f, dtype=np.float32)
        feats.append(a[:FEAT_N].reshape(10, 62, 62))
        scal.append(a[FEAT_N:])            # [slen, action, logit_acc, logit_fire, kpx, kpy]
    return np.stack(feats), np.stack(scal)


def main():
    capdir = sys.argv[1] if len(sys.argv) > 1 else '.'
    feats, s = load(capdir)
    n = len(s)
    slen, act, l_acc, l_fire, kpx, kpy = (s[:, i] for i in range(6))
    print(f'{n} frames from {capdir}')
    print(f'{"idx":>4} {"slen":>5} {"act":>3} {"l_acc":>8} {"l_fire":>8} {"kp":>14}')
    for i in range(n):
        print(f'{i:>4} {slen[i]:>5.0f} {int(act[i]):>3} {l_acc[i]:>8.3f} {l_fire[i]:>8.3f}'
              f'   ({kpx[i]:6.1f},{kpy[i]:6.1f})')
    fires = int(act.sum())
    print(f'fires={fires}/{n} ({100*fires/n:.0f}%)   '
          f'logit_fire-logit_acc: mean={np.mean(l_fire-l_acc):+.3f} '
          f'max={np.max(l_fire-l_acc):+.3f}  (>=0 -> would fire)')

    # Fig 1 — time trends (the "why not fire" view)
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(l_acc, label='logit accumulate'); ax[0].plot(l_fire, label='logit fire')
    ax[0].plot(l_fire - l_acc, '--', label='fire - acc (>=0 fires)'); ax[0].axhline(0, color='k', lw=0.5)
    ax[0].legend(); ax[0].set_ylabel('policy logits'); ax[0].set_title(os.path.basename(capdir))
    ax[1].plot(slen); ax[1].set_ylabel('stacked_length')
    ax[2].step(range(n), act, where='mid'); ax[2].set_ylabel('action (1=fire)')
    ax[2].set_xlabel('frame'); ax[2].set_ylim(-0.1, 1.1)
    fig.tight_layout()

    # Fig 2 — feature_map: 10 channels of a chosen frame (last by default)
    fi = n - 1
    fig2, axes = plt.subplots(2, 5, figsize=(12, 5))
    for ch, axx in enumerate(axes.ravel()):
        axx.imshow(feats[fi, ch], cmap='viridis'); axx.set_title(f'ch{ch}'); axx.axis('off')
    fig2.suptitle(f'RL input feature_map @frame {fi}  (slen={slen[fi]:.0f}, action={int(act[fi])})')
    fig2.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
