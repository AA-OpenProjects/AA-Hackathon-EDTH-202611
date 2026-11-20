"""Smoke-check the participant API against the public environment."""

import numpy as np

from fpv_sim.envs.approach_aviary import ApproachAviary
from handin.agent import Agent


env = ApproachAviary()
try:
    agent = Agent()
    obs, info = env.reset(seed=1)
    agent.reset()
    for _ in range(10):
        action = np.asarray(agent.act(obs), dtype=float)
        if action.shape != (4,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError("Agent.act must return four finite values in [-1, 1]")
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    print("Hand-in API OK; final info:", info)
finally:
    env.close()
