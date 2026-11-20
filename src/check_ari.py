"""Sanity gate #1: does `info['labels']` populate, and do RAM coords map onto pixels?

Run this before anything else (prompt.md 4.1 / 6).

  1. 5-step rollout, print `info['labels']`  -> AtariARI wiring works.
  2. Calibration: segment the white ball in the native 210x160 frame and regress
     its pixel centroid on the RAM `ball_x` / `ball_y`. A near-identity fit
     confirms the rescale used for the ball bbox in reconstruction_metrics.py.
  3. Same check against the 64x64 observation the agent actually sees.
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ari_env import (AtariARI, make_ari_env, sample_random_action, unwrap,  # noqa: E402
                     NATIVE_H, NATIVE_W)

# Pong palette: the ball and the digits are the only near-white pixels.
BALL_RGB = np.array([236, 236, 236])
PLAY_AREA_ROWS = (34, 194)  # exclude the scoreboard so digits are not picked up


def ball_centroid_native(frame):
    """Centroid of white pixels inside the play area, or None if the ball is absent."""
    sub = frame[PLAY_AREA_ROWS[0]:PLAY_AREA_ROWS[1]]
    mask = np.all(np.abs(sub.astype(np.int16) - BALL_RGB) < 20, axis=-1)
    if mask.sum() == 0:
        return None
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean() + PLAY_AREA_ROWS[0])


def ball_centroid_obs(obs):
    """Brightest-pixel centroid in the 64x64 observation, play area only."""
    lo = int(round(PLAY_AREA_ROWS[0] * 64 / NATIVE_H))
    hi = int(round(PLAY_AREA_ROWS[1] * 64 / NATIVE_H))
    sub = obs[lo:hi].astype(np.float32).mean(-1)
    if sub.max() < 150:
        return None
    mask = sub > (sub.max() - 25)
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean() + lo)


PLAYER_RGB = np.array([92, 186, 92])    # right paddle (agent)
ENEMY_RGB = np.array([213, 130, 74])    # left paddle (built-in opponent)


def paddle_centroid_native(frame, rgb):
    """Centroid of a paddle's colour in the play area, or None if not drawn."""
    sub = frame[PLAY_AREA_ROWS[0]:PLAY_AREA_ROWS[1]]
    mask = np.all(np.abs(sub.astype(np.int16) - rgb) < 20, axis=-1)
    if mask.sum() < 8:
        return None
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean() + PLAY_AREA_ROWS[0])


def linfit(x, y):
    a, b = np.polyfit(x, y, 1)
    resid = y - (a * x + b)
    return a, b, float(np.abs(resid).mean()), float(np.abs(resid).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="results/sanity")
    args = ap.parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    env = make_ari_env(seed=args.seed)
    obs = env.reset()
    print("obs keys:", sorted(obs.keys()), "image", obs["image"].shape, obs["image"].dtype)
    print("action space:", env.action_space, "n =", env.action_space.shape)

    print("\n--- 5-step rollout ---")
    for t in range(5):
        action = sample_random_action(env)
        obs, reward, done, info = env.step(action)
        assert "labels" in info, "info['labels'] missing -- AtariARI wiring is broken"
        print(f"t={t} reward={reward} done={done} labels={info['labels']}")

    # ---- calibration over a longer random rollout -------------------------
    env = make_ari_env(seed=args.seed + 1)
    obs = env.reset()
    rows = []
    paddle_rows = []
    frames = []
    for t in range(args.steps):
        obs, reward, done, info = env.step(sample_random_action(env))
        lab = info["labels"]
        native = unwrap(env, AtariARI)._ale.getScreenRGB()
        if native.shape != (NATIVE_H, NATIVE_W, 3):
            native = native.reshape(NATIVE_H, NATIVE_W, 3)
        cn = ball_centroid_native(native)
        co = ball_centroid_obs(obs["image"])
        cp = paddle_centroid_native(native, PLAYER_RGB)
        ce = paddle_centroid_native(native, ENEMY_RGB)
        if cp is not None and ce is not None:
            paddle_rows.append((lab["player_x"], lab["player_y"], cp[0], cp[1],
                                lab["enemy_x"], lab["enemy_y"], ce[0], ce[1]))
        if cn is not None and co is not None:
            rows.append((lab["ball_x"], lab["ball_y"], cn[0], cn[1], co[0], co[1]))
        if len(frames) < 4 and cn is not None:
            frames.append((native.copy(), obs["image"].copy(), dict(lab)))
        if done:
            obs = env.reset()

    arr = np.array(rows, dtype=np.float64)
    print(f"\n--- ball calibration on {len(arr)} frames with a visible ball ---")
    n_uniq = len(np.unique(arr[:, :2], axis=0))
    print(f"unique (ball_x, ball_y) RAM values: {n_uniq}")
    print(f"ball_x RAM range [{arr[:,0].min():.0f}, {arr[:,0].max():.0f}] "
          f"ball_y RAM range [{arr[:,1].min():.0f}, {arr[:,1].max():.0f}]")

    for name, (ri, pi) in {"ball_x": (0, 2), "ball_y": (1, 3)}.items():
        a, b, mae, mx = linfit(arr[:, ri], arr[:, pi])
        print(f"native px {name}: pixel = {a:.4f}*RAM + {b:.2f}   MAE={mae:.2f}  max={mx:.2f}")
    for name, (ri, pi, scale) in {"ball_x": (0, 4, 64 / NATIVE_W),
                                  "ball_y": (1, 5, 64 / NATIVE_H)}.items():
        a, b, mae, mx = linfit(arr[:, ri], arr[:, pi])
        print(f"  64x64 {name}: pixel = {a:.4f}*RAM + {b:.2f}   MAE={mae:.2f}  max={mx:.2f}"
              f"   (naive rescale slope would be {scale:.4f})")

    parr = np.array(paddle_rows, dtype=np.float64)
    print(f"\n--- paddle calibration on {len(parr)} frames ---")
    for name, (ri, pi) in {"player_y": (1, 3), "enemy_y": (5, 7)}.items():
        a, b, mae, mx = linfit(parr[:, ri], parr[:, pi])
        print(f"native px {name}: pixel = {a:.4f}*RAM + {b:.2f}   MAE={mae:.2f}  max={mx:.2f}")
    for name, (ri, pi) in {"player_x": (0, 2), "enemy_x": (4, 6)}.items():
        print(f"native px {name}: RAM {parr[:, ri].min():.0f}..{parr[:, ri].max():.0f} "
              f"-> pixel {parr[:, pi].mean():.1f} (const, sd {parr[:, pi].std():.2f})")

    np.save(outdir / "paddle_calibration.npy", parr)
    np.save(outdir / "ball_calibration.npy", arr)
    np.savez_compressed(outdir / "sanity_frames.npz",
                        native=np.stack([f[0] for f in frames]),
                        obs=np.stack([f[1] for f in frames]))
    print(f"\nwrote {outdir}/ball_calibration.npy")
    print("SANITY OK: info['labels'] populates and the ball is locatable in pixels.")


if __name__ == "__main__":
    main()
