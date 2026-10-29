"""SwitchBench: mid-execution spoken command switching on LIBERO-Goal.

A robot starts a task from a spoken instruction (task A). While it is executing, the user
speaks again. Depending on the episode type, the robot must switch to a new task B (Switch),
keep doing A despite an irrelevant utterance (Keep), or simply complete A (Plain).

Protocol clock: 1 environment step = 0.1 s of audio (STEP_SEC). Audio is 16 kHz mono.
"""

STEP_SEC = 0.1          # one env step corresponds to 0.1 s of the audio stream
SAMPLE_RATE = 16000     # all benchmark audio is 16 kHz mono float32
NUM_STEPS_WAIT = 10     # no-op steps after reset so objects settle (same as the LIBERO/ELLSA protocol)
TASK_BUDGET = 300       # steps a policy gets to finish a task (LIBERO-Goal official horizon)
SUITE = "libero_goal"
RESOLUTION = 256        # camera resolution (same as the ELLSA/OpenVLA LIBERO protocol)

__version__ = "0.1.0"
