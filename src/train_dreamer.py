"""Train DreamerV3 (NM512/dreamerv3-torch) on Atari Pong with periodic checkpoints.

This is `dreamer.py:main` with three additions the probing study needs:

  * a snapshot every `--ckpt_every` policy steps (`checkpoint_XXXXXXX.pt`), so we
    can probe an early -> mid -> final trajectory rather than only the final model;
  * a step-0 snapshot of the *initialised but untrained* model, which doubles as
    the random-init floor of controls.py;
  * `config.json` dumped next to the checkpoints so the probe stage can rebuild
    the exact same WorldModel without re-deriving the config.

Everything else (config parsing, env construction, the train/eval loop) is the
upstream code, imported rather than copied where possible.

Usage:
  python src/train_dreamer.py --configs atari100k --task atari_pong \
      --logdir runs/pong_seed0 --seed 0 --steps 4e5 --ckpt_every 20000
"""

import argparse
import functools
import json
import pathlib
import sys

import numpy as np
import ruamel.yaml as yaml

REPO = pathlib.Path(__file__).resolve().parents[1] / "repos" / "dreamerv3-torch"
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import dreamer as drm  # noqa: E402
import tools  # noqa: E402
from parallel import Damy, Parallel  # noqa: E402

to_np = drm.to_np


def save_checkpoint(agent, path):
    torch.save(
        {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        },
        path,
    )
    print(f"[ckpt] wrote {path}", flush=True)


def dump_config(config, path):
    def enc(o):
        if isinstance(o, pathlib.Path):
            return str(o)
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        return str(o)

    path.write_text(json.dumps(vars(config), indent=2, default=enc))


def main(config, ckpt_every):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    ckptdir = logdir / "checkpoints"
    ckptdir.mkdir(parents=True, exist_ok=True)
    step = drm.count_steps(config.traindir)
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.")
    directory = config.offline_traindir.format(**vars(config)) if config.offline_traindir else config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    directory = config.offline_evaldir.format(**vars(config)) if config.offline_evaldir else config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode, id: drm.make_env(config, mode, id)
    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]
    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space
    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    dump_config(config, logdir / "config.json")

    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - drm.count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        random_actor = tools.OneHotDist(
            torch.zeros(config.num_actions).repeat(config.envs, 1)
        )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent, train_envs, train_eps, config.traindir, logger,
            limit=config.dataset_size, steps=prefill,
        )
        logger.step += prefill * config.action_repeat

    print("Simulate agent.")
    train_dataset = drm.make_dataset(train_eps, config)
    eval_dataset = drm.make_dataset(eval_eps, config)
    agent = drm.Dreamer(
        train_envs[0].observation_space, train_envs[0].action_space, config, logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False
    else:
        # step-0 snapshot: initialised weights, zero gradient steps.
        save_checkpoint(agent, ckptdir / "checkpoint_0000000.pt")

    next_ckpt = (agent._step // ckpt_every + 1) * ckpt_every
    while agent._step < config.steps + config.eval_every:
        logger.write()
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy, eval_envs, eval_eps, config.evaldir, logger,
                is_eval=True, episodes=config.eval_episode_num,
            )
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))
        print("Start training.")
        state = tools.simulate(
            agent, train_envs, train_eps, config.traindir, logger,
            limit=config.dataset_size, steps=config.eval_every, state=state,
        )
        save_checkpoint(agent, logdir / "latest.pt")
        while agent._step >= next_ckpt:
            save_checkpoint(agent, ckptdir / f"checkpoint_{next_ckpt:07d}.pt")
            next_ckpt += ckpt_every
    save_checkpoint(agent, ckptdir / f"checkpoint_{agent._step:07d}_final.pt")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--ckpt_every", type=int, default=20000,
                        help="policy steps between checkpoint snapshots")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load((REPO / "configs.yaml").read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining), args.ckpt_every)
