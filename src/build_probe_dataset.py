"""Roll the probe dataset: 64x64 observations + aligned AtariARI ground truth.

Behaviour policy is a mix of the frozen final agent and uniform-random actions
(prompt.md 4.3), so the ball visits the whole screen instead of the narrow band
a good Pong agent induces.

Conventions kept identical to DreamerV3's replay format, so the stored sequences
can be fed straight into `wm.dynamics.observe`:
  * `action[t]` is the action that *led to* `image[t]`; `action[0]` is all-zeros;
  * `is_first[0] = True`;
  * labels are read at the last emulator frame of each action-repeat group.

Splits are by EPISODE (never by frame): adjacent Atari frames are near-duplicates
and would leak train information into the test R^2.

  python src/build_probe_dataset.py --logdir runs/pong_seed0 \
      --checkpoint runs/pong_seed0/checkpoints/checkpoint_0100000.pt \
      --out data/probe_pong_seed0.h5 --episodes 200 --max_ep_steps 300
"""

import argparse
import pathlib
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ari_env import (AtariARI, make_ari_env, sample_random_action, unwrap,  # noqa: E402
                     POSITION_FACTORS)
from wm_utils import build_world_model, load_config  # noqa: E402

ALL_FACTORS = ("player_y", "player_x", "enemy_y", "enemy_x",
               "ball_x", "ball_y", "player_score", "enemy_score")
SPLIT_CYCLE = ["train"] * 14 + ["val"] * 3 + ["test"] * 3  # 70/15/15 by episode


class MixedPolicy:
    """eps-mixed frozen agent; falls back to uniform random when no agent is given."""

    def __init__(self, wm=None, actor=None, eps=0.3, device="cuda:0", rng=None):
        self.wm, self.actor, self.eps = wm, actor, eps
        self.device, self.rng = device, rng or np.random.RandomState(0)
        self.state = None

    def reset(self):
        self.state = None

    @torch.no_grad()
    def act(self, image, is_first, num_actions):
        random_action = np.zeros(num_actions, np.float32)
        random_action[self.rng.randint(num_actions)] = 1.0
        if self.wm is None:
            return random_action, True
        obs = {
            "image": torch.as_tensor(image[None], device=self.device),
            "is_first": torch.as_tensor(np.array([is_first], np.float32), device=self.device),
            "is_terminal": torch.zeros(1, device=self.device),
        }
        obs = self.wm.preprocess(obs)
        embed = self.wm.encoder(obs)
        latent, prev_action = self.state if self.state is not None else (None, None)
        latent, _ = self.wm.dynamics.obs_step(latent, prev_action, embed, obs["is_first"])
        feat = self.wm.dynamics.get_feat(latent)
        action = self.actor(feat).sample()
        explore = self.rng.rand() < self.eps
        if explore:
            action = torch.as_tensor(random_action[None], device=self.device)
        self.state = ({k: v.detach() for k, v in latent.items()}, action.detach())
        return action[0].detach().cpu().numpy().astype(np.float32), explore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default=None, help="run dir with config.json (needed for --checkpoint)")
    ap.add_argument("--checkpoint", default=None, help="agent checkpoint for the behaviour policy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--max_ep_steps", type=int, default=300, help="policy steps per episode")
    ap.add_argument("--eps", type=float, default=0.3, help="random-action rate for agent episodes")
    ap.add_argument("--random_frac", type=float, default=0.25, help="fraction of fully random episodes")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    env = make_ari_env(seed=args.seed)
    num_actions = env.action_space.shape[0]
    print("num_actions", num_actions)

    wm = actor = None
    if args.checkpoint:
        config = load_config(args.logdir, device=args.device)
        wm, behavior = build_world_model(config, env.observation_space, env.action_space,
                                         ckpt_path=args.checkpoint, device=args.device,
                                         with_actor=True)
        actor = behavior.actor
        print(f"behaviour policy: {args.checkpoint} with eps={args.eps}")
    else:
        print("behaviour policy: uniform random (no checkpoint given)")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(out, "w")
    dsets = {
        "image": f.create_dataset("image", shape=(0, 64, 64, 3), maxshape=(None, 64, 64, 3),
                                  dtype=np.uint8, chunks=(64, 64, 64, 3), compression="lzf"),
        "action": f.create_dataset("action", shape=(0, num_actions), maxshape=(None, num_actions),
                                   dtype=np.float32),
        "labels": f.create_dataset("labels", shape=(0, len(ALL_FACTORS)),
                                   maxshape=(None, len(ALL_FACTORS)), dtype=np.int16),
        "reward": f.create_dataset("reward", shape=(0,), maxshape=(None,), dtype=np.float32),
        "is_first": f.create_dataset("is_first", shape=(0,), maxshape=(None,), dtype=bool),
        "is_terminal": f.create_dataset("is_terminal", shape=(0,), maxshape=(None,), dtype=bool),
        "episode_id": f.create_dataset("episode_id", shape=(0,), maxshape=(None,), dtype=np.int32),
    }

    def append(name, arr):
        d = dsets[name]
        n = d.shape[0]
        d.resize(n + len(arr), axis=0)
        d[n:] = arr

    ep_starts, ep_lengths, ep_splits, ep_policies, ep_returns = [], [], [], [], []
    total, t0 = 0, time.time()
    for ep in range(args.episodes):
        # deterministic interleave rather than a coin flip: the split cycle is 20
        # episodes long, so every split gets the same random/agent mix
        period = int(round(1 / args.random_frac)) if args.random_frac > 0 else 0
        fully_random = (wm is None) or (period > 0 and ep % period == 0)
        policy = MixedPolicy(None if fully_random else wm,
                             None if fully_random else actor,
                             eps=args.eps, device=args.device,
                             rng=np.random.RandomState(args.seed + 1000 * ep))
        env = make_ari_env(seed=args.seed + ep)
        obs = env.reset()
        base = unwrap(env, AtariARI)

        images, actions, labels, rewards, is_first, is_term = [], [], [], [], [], []
        # t=0: the reset observation, with a zero action (DreamerV3 replay convention)
        images.append(obs["image"])
        actions.append(np.zeros(num_actions, np.float32))
        labels.append([base.labels()[k] for k in ALL_FACTORS])
        rewards.append(0.0)
        is_first.append(True)
        is_term.append(False)

        ep_return = 0.0
        for t in range(args.max_ep_steps):
            action, _ = policy.act(obs["image"], is_first=(t == 0), num_actions=num_actions)
            obs, reward, done, info = env.step(action)
            images.append(obs["image"])
            actions.append(action)
            labels.append([info["labels"][k] for k in ALL_FACTORS])
            rewards.append(float(reward))
            is_first.append(False)
            is_term.append(bool(obs["is_terminal"]))
            ep_return += float(reward)
            if done:
                break

        n = len(images)
        ep_starts.append(total)
        ep_lengths.append(n)
        ep_splits.append(SPLIT_CYCLE[ep % len(SPLIT_CYCLE)])
        ep_policies.append("random" if fully_random else f"agent_eps{args.eps}")
        ep_returns.append(ep_return)
        append("image", np.asarray(images, np.uint8))
        append("action", np.asarray(actions, np.float32))
        append("labels", np.asarray(labels, np.int16))
        append("reward", np.asarray(rewards, np.float32))
        append("is_first", np.asarray(is_first, bool))
        append("is_terminal", np.asarray(is_term, bool))
        append("episode_id", np.full(n, ep, np.int32))
        total += n
        env.close()          # 200 live ALE instances would otherwise pile up
        if (ep + 1) % 10 == 0:
            print(f"episode {ep+1}/{args.episodes} frames={total} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    f.create_dataset("ep_start", data=np.asarray(ep_starts, np.int64))
    f.create_dataset("ep_length", data=np.asarray(ep_lengths, np.int64))
    f.create_dataset("ep_split", data=np.asarray(ep_splits, dtype="S8"))
    f.create_dataset("ep_policy", data=np.asarray(ep_policies, dtype="S24"))
    f.create_dataset("ep_return", data=np.asarray(ep_returns, np.float32))
    f.attrs["factors"] = np.asarray(ALL_FACTORS, dtype="S16")
    f.attrs["position_factors"] = np.asarray(POSITION_FACTORS, dtype="S16")
    f.attrs["checkpoint"] = str(args.checkpoint)
    f.attrs["eps"] = args.eps
    f.attrs["seed"] = args.seed

    lab = f["labels"][:]
    print("\nlabel coverage (min/max/std):")
    for i, name in enumerate(ALL_FACTORS):
        print(f"  {name:14s} {lab[:,i].min():4d} {lab[:,i].max():4d} {lab[:,i].std():7.2f}")
    for split in ("train", "val", "test"):
        idx = [i for i, s in enumerate(ep_splits) if s == split]
        print(f"{split}: {len(idx)} episodes, {sum(ep_lengths[i] for i in idx)} frames")
    print(f"mean episode return {np.mean(ep_returns):.2f}")
    print(f"wrote {out} ({total} frames, {time.time()-t0:.0f}s)")
    f.close()


if __name__ == "__main__":
    main()
