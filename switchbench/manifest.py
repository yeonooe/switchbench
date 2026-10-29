"""Episode specification and deterministic manifest construction.

Episode types
  switch : instruction A, then (after a gap) a spoken command to do task B instead.
           Success = B's goal holds at some step after the command starts.
  keep   : instruction A, then an irrelevant utterance (backchannel / remark).
           Success = A's goal holds. The robot must not abandon A.
  plain  : instruction A only. Success = A's goal holds. Sanity check / reproduction.

The dev and test splits use disjoint initial states, phrasings and speakers, so a model
cannot overfit the public dev set phrasing.
"""
import json
import random

from . import tasks

SPLITS = {
    "dev": dict(
        seed=1234,
        init_range=(0, 25),  # LIBERO init states 0-24
        switch_templates=[
            "Actually, {b} instead.",
            "Wait, change of plan. Please {b}.",
            "No, stop that. {b}.",
        ],
        keep_utterances=["Okay.", "Uh-huh, go on.", "Sounds good."],
        speaker_slice=(0, 4),
        gaps_sec=[2.0, 4.0, 6.0],
    ),
    "test": dict(
        seed=5678,
        init_range=(25, 50),  # LIBERO init states 25-49
        switch_templates=[
            "Sorry, I meant {b}.",
            "Forget that. {b} instead.",
            "Let's do something else. {b}.",
        ],
        keep_utterances=["Alright.", "Nice, keep going.", "Mm-hmm."],
        speaker_slice=(4, 8),
        gaps_sec=[2.0, 4.0, 6.0],
    ),
}


def sentence(text):
    """'put the bowl on the stove' -> 'Put the bowl on the stove.'"""
    text = text.strip()
    text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


def fix_case(text):
    """Capitalize the first letter of every sentence: 'No, stop that. put ...' -> '... Put ...'."""
    import re
    text = text[0].upper() + text[1:]
    return re.sub(r"([.!?] )([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


def derangement(n, rng):
    """Random permutation with no fixed points: every task A gets a different task B."""
    while True:
        p = list(range(n))
        rng.shuffle(p)
        if all(i != p[i] for i in range(n)):
            return p


def build_manifest(split="dev", n_speakers=None):
    """Return the list of episode dicts (audio paths are filled in later by tts.py)."""
    cfg = SPLITS[split]
    rng = random.Random(cfg["seed"])
    n = tasks.num_tasks()
    lo, hi = cfg["init_range"]
    spk_lo, spk_hi = cfg["speaker_slice"]
    speakers = list(range(spk_lo, spk_hi)) if n_speakers is None else list(range(n_speakers))
    episodes = []

    def ep(kind, a, b, gap, utter_text, k):
        init_idx = rng.randrange(lo, hi)
        spk = rng.choice(speakers)
        e = dict(
            id=f"{split}_{kind}_{k:03d}",
            split=split, type=kind, task_a=a, task_b=b, init_task=a, init_idx=init_idx,
            instruction_text=sentence(tasks.task_language(a)),
            utterance_text=fix_case(utter_text) if utter_text else None,  # second utterance (None for plain)
            gap_sec=gap,                        # silence between end of instruction and utterance
            speaker=spk,
            target_task=b if kind == "switch" else a,
        )
        episodes.append(e)

    k = 0
    # switch: every task appears once as A and once as B, at every gap
    pairs = derangement(n, rng)
    for a in range(n):
        b = pairs[a]
        for gap in cfg["gaps_sec"]:
            tmpl = rng.choice(cfg["switch_templates"])
            ep("switch", a, b, gap, tmpl.format(b=tasks.task_language(b)), k)
            k += 1
    # keep: every task once, random gap and backchannel
    for a in range(n):
        ep("keep", a, None, rng.choice(cfg["gaps_sec"]), rng.choice(cfg["keep_utterances"]), k)
        k += 1
    # plain: every task once
    for a in range(n):
        ep("plain", a, None, None, None, k)
        k += 1

    # interleave types so partial runs already cover all of them
    by_type = {t: [e for e in episodes if e["type"] == t] for t in ("switch", "keep", "plain")}
    order = []
    while any(by_type.values()):
        for t in ("switch", "switch", "switch", "keep", "plain"):
            if by_type[t]:
                order.append(by_type[t].pop(0))
    return order


def save(episodes, path):
    with open(path, "w") as f:
        json.dump({"version": 1, "episodes": episodes}, f, indent=1)


def load(path):
    with open(path) as f:
        return json.load(f)["episodes"]
