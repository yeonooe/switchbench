"""Build SwitchBench audio from the recordings shipped with the ELLSA test data ("official" engine).

Why: ELLSA reacts badly to out-of-domain voices (with Google TTS it rejected even valid
instructions: "Action cancelled. I cannot execute this command"), which would confound the
switching measurement. The official clips are in ELLSA's training distribution.

  instruction (all types) : official LIBERO-Goal instruction clip of task A
  switch cue              : official instruction clip of task B, heard mid-execution
                            (the user simply says the new instruction)
  keep cue                : a general spoken question from the ELLSA speech-QA test sets
                            (ELLSA was trained to answer these while continuing to act)
Dev and test use disjoint question clips.
"""
import json
import os
import random
import shutil

import soundfile as sf

from .manifest import sentence
from . import tasks

QA_FILES = ["llama_questions.json", "web_questions.json", "triviaqa.json"]


def _qa_clips(data_root, min_sec=1.2, max_sec=4.5):
    clips = []
    for fn in QA_FILES:
        p = os.path.join(data_root, "json", fn)
        if not os.path.exists(p):
            continue
        for item in json.load(open(p))["annotation"]:
            path = item["path"][0] if isinstance(item.get("path"), list) else item.get("path")
            if not path or not os.path.exists(path):
                continue
            try:
                d = sf.info(path).duration
            except Exception:
                continue
            if min_sec <= d <= max_sec:
                clips.append(dict(path=path, text=str(item.get("Q", "")).strip(), source=fn))
    clips.sort(key=lambda c: c["path"])
    return clips


def build(episodes, out_dir, ellsa_data, split):
    instr = json.load(open(os.path.join(ellsa_data, "json", "libero_eval_speech.json")))["libero_goal"]
    qa = _qa_clips(ellsa_data)
    if len(qa) < 20:
        raise SystemExit(f"found only {len(qa)} usable spoken-question clips in {ellsa_data}/json")
    qa = qa[0::2] if split == "dev" else qa[1::2]  # disjoint between splits
    rng = random.Random(f"official-{split}")
    os.makedirs(os.path.join(out_dir, "audio"), exist_ok=True)
    clips = {}

    def put(src, name, text, kind):
        dst = os.path.join(out_dir, "audio", name)
        if not os.path.exists(dst):
            shutil.copy(src, dst)
        clips[name] = dict(text=text, kind=kind, source=os.path.relpath(src, ellsa_data),
                           duration=round(sf.info(dst).duration, 3), asr_wer=None)
        return "audio/" + name

    for e in episodes:
        a, b = e["task_a"], e["task_b"]
        e["instruction_text"] = sentence(tasks.task_language(a))
        e["instruction_audio"] = put(instr[str(a)], f"instr_task{a}.wav", e["instruction_text"], "instruction")
        e["speaker"] = -1  # official clips: one recording per task
        if e["type"] == "switch":
            e["utterance_text"] = sentence(tasks.task_language(b))
            e["utterance_audio"] = put(instr[str(b)], f"instr_task{b}.wav", e["utterance_text"], "instruction")
        elif e["type"] == "keep":
            q = rng.choice(qa)
            name = "q_" + os.path.basename(q["path"])
            e["utterance_text"] = q["text"]
            e["utterance_audio"] = put(q["path"], name, q["text"], "question:" + q["source"])
        else:
            e["utterance_text"], e["utterance_audio"] = None, None
    json.dump(clips, open(os.path.join(out_dir, "audio", "clips.json"), "w"), indent=1)
    return clips
