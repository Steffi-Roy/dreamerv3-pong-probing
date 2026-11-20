"""Atari env for DreamerV3 that also reports AtariARI ground-truth RAM labels.

The observation pipeline is exactly the one the agent trains on
(`envs.atari.Atari` from NM512/dreamerv3-torch: 4-frame action repeat,
max-pool of the last two frames, opencv resize to 64x64 colour), so probes see
the same pixels the world model does.

Label alignment: `Atari.step` runs `action_repeat` emulator frames and returns
the max-pooled image; we read the RAM *after* that loop, i.e. at the last frame
of the action-repeat group -- the frame the agent's next decision is
conditioned on (see prompt.md 4.3 / 6 "label alignment").
"""

import os
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1] / "repos" / "dreamerv3-torch"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from envs.atari import Atari  # noqa: E402
import envs.wrappers as wrappers  # noqa: E402
from atariari.benchmark.ram_annotations import atari_dict  # noqa: E402

# The four continuous position factors we probe for. player/enemy are the two
# paddles (x is constant in Pong, so only y is informative); the ball moves in
# both axes.
POSITION_FACTORS = ("player_y", "enemy_y", "ball_x", "ball_y")
PADDLE_FACTORS = ("player_y", "enemy_y")
BALL_FACTORS = ("ball_x", "ball_y")

# Native Atari screen the RAM coordinates refer to, before the 64x64 resize.
NATIVE_H, NATIVE_W = 210, 160


class AtariARI(Atari):
    """`Atari` + `info['labels']` from the AtariARI RAM annotation table."""

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        key = name.lower().replace("_", "")
        assert key in atari_dict, f"{name} has no AtariARI annotations"
        self._ram_index = atari_dict[key]

    def labels(self):
        ram = self._ale.getRAM()
        return {k: int(ram[i]) for k, i in self._ram_index.items()}

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info["labels"] = self.labels()
        return obs, reward, done, info

    def reset(self):
        obs = super().reset()
        # `Atari.reset` returns obs only; callers read `env.labels()` directly.
        return obs


def make_ari_env(task="pong", action_repeat=4, size=(64, 64), gray=False, noops=30,
                 lives="unused", sticky=False, actions="needed", resize="opencv",
                 seed=0, time_limit=108000):
    """Build the AtariARI env under the same wrapper stack as `dreamer.make_env`."""
    env = AtariARI(task, action_repeat, size, gray=gray, noops=noops, lives=lives,
                   sticky=sticky, actions=actions, resize=resize, seed=seed)
    env = wrappers.OneHotAction(env)
    env = wrappers.TimeLimit(env, time_limit // action_repeat)
    return env


def unwrap(env, cls):
    """Walk the wrapper chain to the first instance of `cls`.

    gym 0.22's `Wrapper.__getattr__` refuses to forward names starting with '_',
    and `Atari` is not a gym.Wrapper, so `.unwrapped` is unusable here.
    """
    while not isinstance(env, cls):
        env = env.env
    return env


def sample_random_action(env):
    return unwrap(env, wrappers.OneHotAction)._sample_action()


def labels_to_vector(labels, factors=POSITION_FACTORS):
    return np.array([labels[f] for f in factors], dtype=np.float32)


# RAM -> native-screen-pixel offsets, measured by src/check_ari.py over 384 frames
# with a visible ball: pixel_x = ram_x - 48.50 (MAE 0.00 px),
#                      pixel_y = ram_y - 12.38 (MAE 0.11 px).
# The RAM values are NOT raw screen coordinates -- ignoring the offset shifts the
# ball bbox by ~19 px in x on the 64x64 frame, i.e. off the ball entirely.
BALL_RAM_OFFSET_X = 48.50
BALL_RAM_OFFSET_Y = 12.38


def rescale_to_obs(ball_x_ram, ball_y_ram, obs_size=64):
    """AtariARI Pong ball RAM coords -> 64x64 observation pixel coords."""
    x = (np.asarray(ball_x_ram, np.float32) - BALL_RAM_OFFSET_X) * (obs_size / NATIVE_W)
    y = (np.asarray(ball_y_ram, np.float32) - BALL_RAM_OFFSET_Y) * (obs_size / NATIVE_H)
    return x, y


# Paddle geometry, also measured by src/check_ari.py (487 frames):
#   player paddle centre x = 141.5 native px (RAM player_x is constant at 188)
#   enemy  paddle centre x =  17.5 native px (RAM enemy_x  is constant at 64)
#   centre y = 0.962 * ram_y - 1.94 (player), 0.983 * ram_y - 5.18 (enemy);
# the sub-unit slope is the 16-px-tall paddle being clipped at the screen edges.
PADDLE_NATIVE = {
    "player": dict(cx=141.5, a=0.9621, b=-1.94),
    "enemy": dict(cx=17.5, a=0.9825, b=-5.18),
}


def paddle_center_to_obs(which, ram_y, obs_size=64):
    """AtariARI paddle RAM y -> centre of the paddle on the 64x64 observation."""
    p = PADDLE_NATIVE[which]
    x = np.full(np.shape(ram_y), p["cx"], np.float32) * (obs_size / NATIVE_W)
    y = (p["a"] * np.asarray(ram_y, np.float32) + p["b"]) * (obs_size / NATIVE_H)
    return x, y


def ball_visible(ball_x_ram, ball_y_ram):
    """Pong writes (0, 0) while the ball is off screen between points.

    Those frames carry no ball position, so they are masked out of the ball
    probes and the ball-region reconstruction metric.
    """
    return (np.asarray(ball_x_ram) > 0) & (np.asarray(ball_y_ram) > 0)
