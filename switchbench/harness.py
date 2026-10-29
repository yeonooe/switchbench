"""Run a Policy on SwitchBench episodes and write one JSON line per episode (resumable)."""
import json
import os
import time
import traceback

import numpy as np

from . import NUM_STEPS_WAIT, SAMPLE_RATE, STEP_SEC
from . import tasks
from .audio import SAMPLES_PER_STEP, build_stream
from .policy import DUMMY_ACTION


def done_ids(path):
    if not os.path.exists(path):
        return set()
    out = set()
    for line in open(path):
        try:
            out.add(json.loads(line)["id"])
        except Exception:
            pass
    return out


def run_episode(ep, policy, env_cache, data_root, video_path=None):
    t0 = time.time()
    stream, tl = build_stream(ep, data_root)
    env = env_cache.get(ep["init_task"])
    env.reset()
    obs = env.set_init_state(tasks.init_state_for(ep))
    for _ in range(NUM_STEPS_WAIT):  # let objects settle, as in the official protocol
        obs, _, _, _ = env.step(DUMMY_ACTION.tolist())

    policy.reset(dict(max_steps=tl["max_steps"], sample_rate=SAMPLE_RATE, step_sec=STEP_SEC,
                      episode_id=ep["id"]))
    a, b, kind = ep["task_a"], ep["task_b"], ep["type"]
    queue, texts, notes, frames = [], [], [], []
    success_step, a_done_step, b_done_step = None, None, None
    for t in range(tl["max_steps"]):
        if not queue:
            out = policy.act(obs, stream[: t * SAMPLES_PER_STEP], t)
            acts = np.asarray(out["actions"], dtype=np.float32).reshape(-1, 7)
            queue = list(acts)
            txt = out.get("text")
            if txt and "<silence>" not in txt:
                texts.append([t, txt])
            if out.get("note"):  # wrapper decisions (e.g. restart or not), kept even when the model is silent
                notes.append([t, out["note"]])
        obs, _, _, _ = env.step(queue.pop(0).tolist())
        if video_path is not None and t % 2 == 0:
            frames.append(obs["agentview_image"][::-1, ::-1][::2, ::2].copy())

        if a_done_step is None and tasks.goal_satisfied(env, a):
            a_done_step = t
        if kind == "switch":
            if t >= tl["utt_start"] and tasks.goal_satisfied(env, b):
                b_done_step = success_step = t
                break
        elif a_done_step is not None:
            success_step = a_done_step
            break

    ref = tl["utt_end"] if kind == "switch" else tl["instr_end"]
    rec = dict(
        id=ep["id"], type=kind, task_a=a, task_b=b, gap_sec=ep["gap_sec"], speaker=ep["speaker"],
        init_idx=ep["init_idx"], success=success_step is not None,
        success_step=success_step,
        steps_after_cue=None if success_step is None else success_step - ref,
        # switch only: did the robot complete the old task A after being told to switch?
        stale_completion=(kind == "switch" and a_done_step is not None and a_done_step >= tl["utt_start"]),
        a_done_before_cue=(kind == "switch" and a_done_step is not None and a_done_step < tl["utt_start"]),
        steps=t + 1, max_steps=tl["max_steps"], timeline=tl, texts=texts, notes=notes,
        seconds=round(time.time() - t0, 1),
    )
    if video_path is not None and frames:
        import imageio
        imageio.mimsave(video_path, frames, duration=0.1, loop=0)
        rec["video"] = os.path.basename(video_path)
    return rec


def run(episodes, policy, data_root, out_jsonl, limit=None, video_dir=None, video_first=10, log=print):
    os.makedirs(os.path.dirname(os.path.abspath(out_jsonl)), exist_ok=True)
    finished = done_ids(out_jsonl)
    todo = [e for e in episodes if e["id"] not in finished]
    if limit is not None:
        todo = todo[: max(0, limit - len(finished))]
    log(f"{len(finished)} episodes already done, {len(todo)} to run -> {out_jsonl}")
    cache = tasks.EnvCache()
    try:
        for i, ep in enumerate(todo):
            vpath = None
            if video_dir and len(finished) + i < video_first:
                os.makedirs(video_dir, exist_ok=True)
                vpath = os.path.join(video_dir, ep["id"] + ".gif")
            try:
                rec = run_episode(ep, policy, cache, data_root, vpath)
            except Exception as ex:
                rec = dict(id=ep["id"], type=ep["type"], task_a=ep["task_a"], task_b=ep["task_b"],
                           success=False, error=f"{type(ex).__name__}: {ex}",
                           traceback=traceback.format_exc()[-2000:])
            with open(out_jsonl, "a") as f:
                f.write(json.dumps(rec) + "\n")
            log(f"[{len(finished)+i+1}/{len(finished)+len(todo)}] {rec['id']:16s} success={rec['success']!s:5s} "
                f"steps={rec.get('steps')} {rec.get('seconds')}s {rec.get('error','')}")
    finally:
        cache.close()
