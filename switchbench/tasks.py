"""LIBERO-Goal access: task list, environments, and goal checks for any task in the scene.

All 10 LIBERO-Goal tasks share one kitchen scene with the same objects and fixtures, so an
environment built for task A can evaluate the goal of any task B. This was verified for all
90 ordered pairs: object poses are identical for a given initial state, and no task's goal
is already satisfied in any other task's initial states (see checks/ in the repo).
"""
import os
from collections import OrderedDict

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.bddl_utils import robosuite_parse_problem

from . import RESOLUTION, SUITE

_suite = None


def suite():
    global _suite
    if _suite is None:
        _suite = benchmark.get_benchmark_dict()[SUITE]()
    return _suite


def num_tasks():
    return suite().n_tasks


def task_language(task_id):
    return suite().get_task(task_id).language


def bddl_path(task_id):
    t = suite().get_task(task_id)
    return os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)


def init_states(task_id):
    return suite().get_task_init_states(task_id)


_train_states = None


def init_state_for(episode):
    """Initial sim state of an episode: LIBERO's evaluation states for dev/test,
    SwitchBench's own fresh states (assets/train_init_states.npz) for the train split."""
    global _train_states
    if episode.get("init_source") == "train":
        if _train_states is None:
            import numpy as np
            _train_states = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "assets", "train_init_states.npz"))
        return _train_states[f"task{episode['init_task']}"][episode["init_idx"]]
    return init_states(episode["init_task"])[episode["init_idx"]]


_goals = {}


def goal_state(task_id):
    """Parsed BDDL goal predicates of a task, e.g. [['on', 'akita_black_bowl_1', 'plate_1']]."""
    if task_id not in _goals:
        _goals[task_id] = robosuite_parse_problem(bddl_path(task_id))["goal_state"]
    return _goals[task_id]


def goal_satisfied(env, task_id):
    """True if the goal of `task_id` holds in `env` (works for any LIBERO-Goal env)."""
    inner = env.env
    return all(inner._eval_predicate(p) for p in goal_state(task_id))


class EnvCache:
    """Keeps one environment per task (building one takes seconds), least-recently-used."""

    def __init__(self, max_size=10):
        self.max_size = max_size
        self._envs = OrderedDict()

    def get(self, task_id):
        if task_id in self._envs:
            self._envs.move_to_end(task_id)
            return self._envs[task_id]
        env = OffScreenRenderEnv(bddl_file_name=bddl_path(task_id),
                                 camera_heights=RESOLUTION, camera_widths=RESOLUTION)
        env.seed(0)  # same as the official LIBERO evaluation; affects object poses
        self._envs[task_id] = env
        if len(self._envs) > self.max_size:
            _, old = self._envs.popitem(last=False)
            old.close()
        return env

    def close(self):
        for env in self._envs.values():
            env.close()
        self._envs.clear()
