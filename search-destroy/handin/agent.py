"""Participant entry point. Replace act() with your own policy."""

import numpy as np


class Agent:
    def reset(self):
        pass

    def act(self, observation):
        # Hover in place as a minimal valid example.
        return np.zeros(4, dtype=np.float32)
