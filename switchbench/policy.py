"""Policy interface every SwitchBench submission implements.

The harness calls `reset` once per episode and `act` whenever the policy's action queue is
empty. A policy may return a chunk of actions; they are executed one per env step before
`act` is called again.

Observation passed to `act`:
    obs["agentview_image"]           (256, 256, 3) uint8, raw LIBERO render (not flipped)
    obs["robot0_eye_in_hand_image"]  (256, 256, 3) uint8, raw wrist render (not flipped)
    obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]  proprioception
    (the full LIBERO observation dict is passed through)
Audio passed to `act`:
    1-D float32 array, 16 kHz: everything the user has said from t=0 up to the current step.
    Future audio is never revealed.

Policies receive no text: not the task name, not the transcript. Only pixels, proprioception
and the user's voice.
"""
import numpy as np

DUMMY_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)  # LIBERO no-op (gripper open)


class Policy:
    def reset(self, info):
        """info: {'max_steps', 'sample_rate', 'step_sec', 'episode_id'} - protocol facts only."""

    def act(self, obs, audio, t):
        """Return {'actions': np.ndarray (k, 7) with k >= 1, 'text': optional str}."""
        raise NotImplementedError


class NoopPolicy(Policy):
    """Never moves. Lower bound, and a harness self-test that needs no GPU."""

    def act(self, obs, audio, t):
        return {"actions": DUMMY_ACTION[None].repeat(10, axis=0), "text": None}


class RandomPolicy(Policy):
    """Small random end-effector motions. Harness self-test."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, audio, t):
        a = self.rng.uniform(-0.5, 0.5, size=(10, 7)).astype(np.float32)
        a[:, 6] = np.where(self.rng.random(10) > 0.5, 1.0, -1.0)
        return {"actions": a, "text": None}
