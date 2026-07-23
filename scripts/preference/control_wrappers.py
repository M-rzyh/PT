"""Control-interface difficulties for LunarLander: sticky actions and action delay.

These corrupt what the *action* does, leaving the view perfect — the opposite of frame
blanking / region masking, which hide the view and leave control intact.

  sticky_p : repeat the PREVIOUS action with probability p (Machado et al. 2018).
             Feels like an unresponsive controller.
  delay_k  : apply each action k STEPS late, via a FIFO primed with no-ops.
             The human feels k/fps seconds of lag; the LANDER experiences k*20 ms
             of game time (physics is 50 steps/s).

Order per step is sticky THEN delay: an unresponsive controller repeats the last
command, and the wire then delays whatever it sent.

This is a deliberate MIRROR of `make_control_corruptor` in IRL3/imitation/human_demo.py.
The two arms inject the difficulty at different points — GAIL corrupts the human's live
keypress, PT bakes it into the trajectories that get rendered and compared — so a given
(p, k) must mean the same thing on both sides. Keep them in sync.
"""
from __future__ import annotations

import collections

import numpy as np


def make_control_corruptor(sticky_p: float, sticky_seed: int, delay_k: int, noop_fn):
    """Returns `(apply, reset)`. Both are no-ops when sticky_p=0 and delay_k=0."""
    rng = np.random.default_rng(sticky_seed)
    state: dict = {"prev": None, "queue": None}

    def reset():
        state["prev"] = None
        state["queue"] = collections.deque(
            [noop_fn() for _ in range(delay_k)], maxlen=delay_k) if delay_k > 0 else None

    def apply(action):
        if sticky_p > 0.0 and state["prev"] is not None and rng.random() < float(sticky_p):
            action = state["prev"]
        state["prev"] = action
        if delay_k > 0:
            q = state["queue"]
            out = q.popleft()      # the action issued k steps ago
            q.append(action)
            return out
        return action

    reset()
    return apply, reset
