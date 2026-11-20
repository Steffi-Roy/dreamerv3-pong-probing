"""Record the frozen agent playing Pong: mp4 + GIF + the filmstrip figure.

This is the gameplay deliverable (prompt.md 5.7). It is deliberately separate
from the probe pipeline: the probes need the *mixed* behaviour policy (so the
ball covers the screen), whereas the video needs the *greedy* eval policy, which
is what the published Pong returns are measured with.

Two image streams are kept per policy step:
  * the 64x64 observation the world model actually sees, and
  * the native 210x160 ALE screen (`Atari._screen`), used for the video and the
    filmstrip -- upscaling the 64x64 obs instead looks blocky at video size.

The screen is read *after* `Atari.step` has run its 4 emulator frames, i.e. the
same frame the AtariARI labels are aligned to (see ari_env.py), so the filmstrip
frames and the ball-visibility mask refer to the same instant.

  python src/record_gameplay.py --logdir runs/pong_seed0 \
      --checkpoint runs/pong_seed0/checkpoints/<final>.pt \
      --outdir figures --episodes 5
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ari_env import AtariARI, NATIVE_H, NATIVE_W, ball_visible, make_ari_env, unwrap  # noqa: E402
from wm_utils import build_world_model, load_config  # noqa: E402

RANDOM_RETURN = -21.0      # Pong random-policy return, for the honest caption
PUBLISHED_DREAMERV3 = 18.0  # danijar/dreamerv3 Atari100k Pong (burchim repro: 20)


class GreedyPolicy:
    """The eval policy: `actor(feat).mode()`, mirroring `Dreamer._policy(training=False)`."""

    def __init__(self, wm, actor, device="cuda:0", eval_state_mean=False):
        self.wm, self.actor, self.device = wm, actor, device
        self.eval_state_mean = eval_state_mean
        self.state = None

    def reset(self):
        self.state = None

    @torch.no_grad()
    def act(self, image, is_first):
        obs = {
            "image": torch.as_tensor(image[None], device=self.device),
            "is_first": torch.as_tensor(np.array([is_first], np.float32), device=self.device),
            "is_terminal": torch.zeros(1, device=self.device),
        }
        obs = self.wm.preprocess(obs)
        embed = self.wm.encoder(obs)
        latent, prev_action = self.state if self.state is not None else (None, None)
        latent, _ = self.wm.dynamics.obs_step(latent, prev_action, embed, obs["is_first"])
        if self.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self.wm.dynamics.get_feat(latent)
        action = self.actor(feat).mode()
        self.state = ({k: v.detach() for k, v in latent.items()}, action.detach())
        return action[0].detach().cpu().numpy().astype(np.float32)


def upscale(frames, factor):
    """Nearest-neighbour upscale (keeps the crisp Atari pixel edges, no new deps)."""
    if factor <= 1:
        return frames
    return np.repeat(np.repeat(frames, factor, axis=1), factor, axis=2)


def roll_episode(env, policy, max_steps, screen_buf):
    """One episode. Returns (hi-res frames, 64x64 obs, ball-visible mask, return)."""
    obs = env.reset()
    base = unwrap(env, AtariARI)
    if policy is not None:
        policy.reset()

    frames_hi, frames_obs, visible = [], [], []

    def grab(labels):
        base._screen(screen_buf)
        frames_hi.append(screen_buf.copy())
        frames_obs.append(obs["image"].copy())
        visible.append(bool(ball_visible(labels["ball_x"], labels["ball_y"])))

    grab(base.labels())
    ep_return = 0.0
    for t in range(max_steps):
        if policy is None:
            action = np.zeros(env.action_space.shape[0], np.float32)
            action[np.random.randint(len(action))] = 1.0
        else:
            action = policy.act(obs["image"], is_first=(t == 0))
        obs, reward, done, info = env.step(action)
        grab(info["labels"])
        ep_return += float(reward)
        if done:
            break
    return (np.asarray(frames_hi, np.uint8), np.asarray(frames_obs, np.uint8),
            np.asarray(visible, bool), ep_return)


def longest_visible_run(visible, min_len=6):
    """Longest contiguous stretch with the ball on screen == the longest rally."""
    best = cur_start = best_start = 0
    best_len = 0
    for i, v in enumerate(visible):
        if v:
            if cur_start is None:
                cur_start = i
            run = i - cur_start + 1
            if run > best_len:
                best_len, best_start = run, cur_start
        else:
            cur_start = None
    if best_len < min_len:
        return 0, len(visible)
    return best_start, best_start + best_len


def make_filmstrip(frames_hi, start, stop, n_panels, out_path, pad=2):
    """hstack n evenly spaced frames from one rally -> the paper figure."""
    idx = np.linspace(start, stop - 1, n_panels).round().astype(int)
    panels = [frames_hi[i] for i in idx]
    h, w, _ = panels[0].shape
    strip = np.full((h, n_panels * w + (n_panels - 1) * pad, 3), 255, np.uint8)
    for k, p in enumerate(panels):
        x = k * (w + pad)
        strip[:, x:x + w] = p
    import imageio.v2 as imageio
    imageio.imwrite(out_path, strip)
    return idx, strip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="agent checkpoint; omit for the random-policy control clip")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=1500, help="policy steps per episode")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max_seconds", type=float, default=30.0)
    ap.add_argument("--scale", type=int, default=3, help="nearest-neighbour upscale for the video")
    ap.add_argument("--panels", type=int, default=6)
    ap.add_argument("--prefix", default="pong")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import imageio.v2 as imageio

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    screen_buf = np.zeros((NATIVE_H, NATIVE_W, 3), np.uint8)

    env = make_ari_env(seed=args.seed)
    policy = None
    if args.checkpoint:
        config = load_config(args.logdir, device=args.device)
        wm, behavior = build_world_model(config, env.observation_space, env.action_space,
                                         ckpt_path=args.checkpoint, device=args.device,
                                         with_actor=True)
        policy = GreedyPolicy(wm, behavior.actor, device=args.device,
                              eval_state_mean=bool(getattr(config, "eval_state_mean", False)))
        print(f"greedy eval policy from {args.checkpoint}")
    else:
        print("random policy (control clip)")

    episodes, t0 = [], time.time()
    for ep in range(args.episodes):
        env = make_ari_env(seed=args.seed + ep)
        hi, obs64, vis, ret = roll_episode(env, policy, args.max_steps, screen_buf)
        env.close()
        episodes.append(dict(ep=ep, ret=ret, steps=len(hi), hi=hi, obs=obs64, vis=vis))
        print(f"episode {ep}: return {ret:+.0f}  steps {len(hi)}  "
              f"ball visible {vis.mean()*100:.0f}%  ({time.time()-t0:.0f}s)", flush=True)

    returns = np.array([e["ret"] for e in episodes], np.float32)
    print(f"\nreturn over {len(returns)} episodes: mean {returns.mean():+.2f} "
          f"+/- {returns.std():.2f}  (random {RANDOM_RETURN:+.0f}, "
          f"published DreamerV3 ~{PUBLISHED_DREAMERV3:+.0f})")
    beats_random = returns.mean() > RANDOM_RETURN
    print("beats random policy" if beats_random else
          "does NOT beat random -- caption this honestly as an early-training agent")

    # Video from the best episode; the filmstrip wants the longest rally in it.
    best = int(np.argmax(returns))
    ep = episodes[best]
    max_frames = int(args.fps * args.max_seconds)
    frames = ep["hi"]
    start, stop = longest_visible_run(ep["vis"])
    if len(frames) > max_frames:
        # centre the clip on the longest rally rather than truncating the opening
        mid = (start + stop) // 2
        lo = max(0, min(len(frames) - max_frames, mid - max_frames // 2))
        frames = frames[lo:lo + max_frames]
        start, stop = max(0, start - lo), min(max_frames, stop - lo)
    video = upscale(frames, args.scale)

    mp4 = outdir / f"{args.prefix}_agent.mp4"
    gif = outdir / f"{args.prefix}_agent.gif"
    wrote = []
    try:
        imageio.mimsave(mp4, list(video), fps=args.fps, macro_block_size=1)
        wrote.append(str(mp4))
    except Exception as exc:                      # ffmpeg missing / codec refused
        print(f"mp4 failed ({exc}); GIF only")
    imageio.mimsave(gif, list(video[::2]), duration=2.0 / args.fps)
    wrote.append(str(gif))

    strip_path = outdir / f"{args.prefix}_filmstrip.png"
    idx, _ = make_filmstrip(ep["hi"], *longest_visible_run(ep["vis"]),
                            n_panels=args.panels, out_path=strip_path)
    wrote.append(str(strip_path))

    meta = dict(
        checkpoint=str(args.checkpoint), episodes=int(args.episodes),
        returns=[float(r) for r in returns], mean_return=float(returns.mean()),
        std_return=float(returns.std()), best_episode=best,
        random_return=RANDOM_RETURN, published_dreamerv3=PUBLISHED_DREAMERV3,
        beats_random=bool(beats_random), video_frames=int(len(video)), fps=args.fps,
        clip_seconds=round(len(video) / args.fps, 1),
        filmstrip_frame_idx=[int(i) for i in idx],
        rally_span=[int(longest_visible_run(ep["vis"])[0]), int(longest_visible_run(ep["vis"])[1])],
    )
    meta_path = outdir / f"{args.prefix}_gameplay.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    wrote.append(str(meta_path))
    print("\nwrote:\n  " + "\n  ".join(wrote))
    print(f"clip: {meta['clip_seconds']}s @ {args.fps}fps, "
          f"{video.shape[2]}x{video.shape[1]}")


if __name__ == "__main__":
    main()
