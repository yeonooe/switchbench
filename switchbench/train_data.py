"""SwitchBench-train: the official training split (same episode format as dev/test).

Sources (all disjoint from dev/test, checked automatically):
  commands      ELLSA *train* data, libero/instructions/: one spoken instruction per LIBERO demo,
                many voices (ELLSA's own training distribution). LIBERO-Goal folders become
                instruction / switch-cue clips; other suites become extra command examples.
                Optional paraphrases ("Actually, ... instead") are synthesized with gTTS.
  non-commands  General spoken questions from VoiceAssistant-400K (used to train ELLSA),
                plus gTTS backchannels and scene-related questions (hard negatives that mention
                the scene's objects but are not commands).
  scenes        New LIBERO-Goal initial states from fresh seeds (assets/train_init_states.npz),
                none equal to a dev/test state and none with a goal already satisfied.
  task pairs    All ordered pairs except the 10 test pairs, so test measures unseen combinations.

Every clip is checked against the dev/test audio (MD5) and every question text against the
dev/test texts before the split is written; the report is saved as leak_report.json.
"""
import hashlib
import json
import os
import random
import re

import numpy as np
import soundfile as sf

from . import SAMPLE_RATE
from .audio import load_wav, normalize, trim_silence

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = json.load(open(os.path.join(HERE, "assets", "libero_goal_tasks.json")))
TRAIN_STATES = os.path.join(HERE, "assets", "train_init_states.npz")
ELLSA_TRAIN_REPO = "tsinghua-ee/ELLSA_train_data"
VA_REPO = "gpt-omni/VoiceAssistant-400K"

BACKCHANNELS = ["Okay.", "Uh-huh.", "Mm-hmm.", "Right.", "Sure.", "Nice.", "Great, thanks.", "Cool.",
                "I see.", "Alright.", "Got it.", "Sounds good.", "Interesting.", "Oh, really?", "Yeah.",
                "Hmm.", "That's fine.", "Good job so far.", "Take your time.", "No rush."]
OBJECTS = ["the bowl", "the black bowl", "the plate", "the wine bottle", "the cream cheese", "the stove",
           "the cabinet", "the top drawer", "the middle drawer", "the wine rack"]
SCENE_QUESTIONS = ["Where is {o}?", "Is {o} clean?", "What color is {o}?", "Did you see {o}?",
                   "How far away is {o}?", "Is {o} heavy?", "Who left {o} there?", "Have you used {o} before?",
                   "What is next to {o}?", "Is {o} still there?"]
PARAPHRASES = ["Actually, {t} instead.", "No, {t}.", "Wait, {t}.", "Change of plan, {t}.", "Sorry, {t} instead.",
               "Forget that and {t}.", "Let's {t} instead.", "Hold on, {t} please."]


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def envelope(path, frame=800):
    """Loudness envelope (50 ms frames) of the trimmed recording, zero-mean and unit-norm."""
    w = trim_silence(load_wav(path))
    n = len(w) // frame
    e = np.log(np.sqrt((w[: n * frame].reshape(n, frame) ** 2).mean(1)) + 1e-5)
    e = e - e.mean()
    return e / (np.linalg.norm(e) + 1e-9)


def same_recording(env_a, env_b, tol_frames=2, min_corr=0.98):
    """True for the same recording re-encoded or rescaled; different takes of a sentence do not match."""
    if abs(len(env_a) - len(env_b)) > tol_frames:
        return False
    n = min(len(env_a), len(env_b))
    return n > 0 and float(np.dot(env_a[:n], env_b[:n])) >= min_corr


def norm_text(t):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", (t or "").lower()).split())


def task_of_folder(name):
    """'libero_goal__put_the_bowl_on_the_stove_demo__12' -> 1 (None if not a LIBERO-Goal task)."""
    if not name.startswith("libero_goal"):
        return None
    for t in sorted(TASKS, key=lambda t: -len(t["bddl"])):  # longest match first
        if t["bddl"] in name:
            return t["id"]
    return None


# ----------------------------------------------------------------------------- sources
def fetch_ellsa_instructions(cache_dir, max_other=400, token=None, log=print):
    """Download instruction recordings. Returns {task_id: [wav paths]}, [other-suite command wavs]."""
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi(token=token)
    folders = [e.path.split("/")[-1] for e in api.list_repo_tree(ELLSA_TRAIN_REPO, path_in_repo="libero/instructions",
                                                                   repo_type="dataset")]
    goal = [f for f in folders if task_of_folder(f) is not None]
    other = sorted(f for f in folders if not f.startswith("libero_goal"))
    random.Random(0).shuffle(other)
    other = other[:max_other]
    log(f"ELLSA train instructions: {len(folders)} folders, {len(goal)} LIBERO-Goal, using {len(other)} others")
    pats = [f"libero/instructions/{f}/*" for f in goal + other]
    root = snapshot_download(ELLSA_TRAIN_REPO, repo_type="dataset", allow_patterns=pats, local_dir=cache_dir,
                             token=token, max_workers=16)
    by_task, extra = {t["id"]: [] for t in TASKS}, []
    for f in goal + other:
        d = os.path.join(root, "libero", "instructions", f)
        wavs = sorted(os.path.join(d, x) for x in os.listdir(d) if x.lower().endswith((".wav", ".flac"))) if os.path.isdir(d) else []
        if not wavs:
            continue
        tid = task_of_folder(f)
        (by_task[tid] if tid is not None else extra).append(wavs[0])
    log("recordings per LIBERO-Goal task: " + ", ".join(f"{k}:{len(v)}" for k, v in by_task.items()))
    return by_task, extra


def fetch_va_questions(out_dir, n=800, min_sec=1.0, max_sec=6.0, log=print):
    """Stream spoken questions from VoiceAssistant-400K. Returns [(wav_path, text)]."""
    from datasets import load_dataset
    import librosa
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset(VA_REPO, split="train", streaming=True)
    got, seen = [], 0
    for ex in ds:
        seen += 1
        if seen > n * 20 or len(got) >= n:
            break
        audio = next((v for v in ex.values() if isinstance(v, dict) and "array" in v), None)
        text = next((ex[k] for k in ("question", "instruction", "text", "prompt", "query") if isinstance(ex.get(k), str)), "")
        if audio is None:
            continue
        wav = np.asarray(audio["array"], dtype=np.float32)
        if audio.get("sampling_rate", SAMPLE_RATE) != SAMPLE_RATE:
            wav = librosa.resample(wav, orig_sr=audio["sampling_rate"], target_sr=SAMPLE_RATE)
        if not (min_sec <= len(wav) / SAMPLE_RATE <= max_sec):
            continue
        p = os.path.join(out_dir, f"va_{len(got):05d}.wav")
        sf.write(p, normalize(trim_silence(wav)), SAMPLE_RATE)
        got.append((p, text.strip()))
    log(f"VoiceAssistant-400K: kept {len(got)} questions ({seen} streamed)")
    return got


# ----------------------------------------------------------------------------- build
def build(out_dir, eval_dirs, cache_dir, synth=None, n_switch_per_pair=12, n_keep=600, n_plain=200,
          paraphrase_prob=0.3, use_va=True, ellsa_token=None, seed=0, log=print):
    rng = random.Random(seed)
    audio_dir = os.path.join(out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    clips, src_of = {}, {}

    def add_clip(src_wav, text, kind, source, label):
        wav = normalize(trim_silence(load_wav(src_wav)))
        name = f"{kind}_{hashlib.md5((source + text + src_wav).encode()).hexdigest()[:10]}.wav"
        path = os.path.join(audio_dir, name)
        if not os.path.exists(path):
            sf.write(path, wav, SAMPLE_RATE)
        src_of[name] = src_wav
        clips[name] = dict(text=text, kind=kind, source=source, label=label,
                           duration=round(len(wav) / SAMPLE_RATE, 3), src=os.path.basename(src_wav))
        return "audio/" + name

    # 1) commands from ELLSA train recordings (fallback: gTTS)
    by_task, extra = fetch_ellsa_instructions(os.path.join(cache_dir, "ellsa_train"), token=ellsa_token, log=log)
    sentence = lambda t: t[0].upper() + t[1:] + "."
    instr = {}
    for t in TASKS:
        paths = [add_clip(p, sentence(t["language"]), "cmd", "ellsa_train", 1) for p in by_task[t["id"]]]
        if len(paths) < 5 and synth is not None:
            log(f"task {t['id']}: only {len(paths)} recordings, adding gTTS voices")
            for spk in range(8):
                p = os.path.join(cache_dir, f"gtts_instr_{t['id']}_{spk}.wav")
                if not os.path.exists(p):
                    synth.synth(sentence(t["language"]), str(spk), p)
                paths.append(add_clip(p, sentence(t["language"]), "cmd", "gtts", 1))
        instr[t["id"]] = paths
    extra_cmds = [add_clip(p, "", "cmd_other", "ellsa_train_other_suite", 1) for p in extra]

    # 2) paraphrased switch cues (gTTS)
    para = {t["id"]: [] for t in TASKS}
    if synth is not None and paraphrase_prob > 0:
        for t in TASKS:
            for j, tmpl in enumerate(PARAPHRASES):
                text = tmpl.format(t=t["language"])
                text = text[0].upper() + text[1:]
                p = os.path.join(cache_dir, f"gtts_para_{t['id']}_{j}.wav")
                if not os.path.exists(p):
                    synth.synth(text, str(rng.randrange(8)), p)
                para[t["id"]].append(add_clip(p, text, "cmd_para", "gtts", 1))

    # 3) non-commands: VoiceAssistant questions + gTTS backchannels and scene questions
    nonc = []
    if use_va:
        try:
            for p, text in fetch_va_questions(os.path.join(cache_dir, "va"), n=800, log=log):
                nonc.append(add_clip(p, text, "q", "voiceassistant400k", 0))
        except Exception as e:
            log(f"VoiceAssistant-400K unavailable ({type(e).__name__}: {str(e)[:120]}); using synthesized questions only")
    if synth is not None:
        texts = BACKCHANNELS + [q.format(o=o) for q in SCENE_QUESTIONS for o in OBJECTS]
        for j, text in enumerate(texts):
            text = text[0].upper() + text[1:]
            p = os.path.join(cache_dir, f"gtts_nc_{j}.wav")
            if not os.path.exists(p):
                synth.synth(text, str(j % 8), p)
            nonc.append(add_clip(p, text, "nc", "gtts", 0))
    if not nonc:
        raise SystemExit("no non-command clips: enable VoiceAssistant or gTTS")

    # 4) leak checks against dev/test
    eval_md5, eval_env, eval_texts, test_pairs = set(), [], set(), set()
    for d in eval_dirs:
        man = json.load(open(os.path.join(d, "manifest.json")))["episodes"]
        for root, _, files in os.walk(os.path.join(d, "audio")):
            wavs = [os.path.join(root, f) for f in files if f.endswith(".wav")]
            eval_md5 |= {md5(w) for w in wavs}
            eval_env += [envelope(w) for w in wavs]
        eval_texts |= {norm_text(e["utterance_text"]) for e in man if e["type"] == "keep"}
        if os.path.basename(os.path.normpath(d)) == "test":
            test_pairs |= {(e["task_a"], e["task_b"]) for e in man if e["type"] == "switch"}
    def leaked(n):
        if md5(src_of[n]) in eval_md5:
            return True
        e = envelope(src_of[n])
        return any(same_recording(e, x) for x in eval_env)
    leak_audio = [n for n in clips if leaked(n)]
    leak_text = [n for n, c in clips.items() if c["label"] == 0 and norm_text(c["text"]) in eval_texts]
    for n in leak_audio + leak_text:  # drop anything that overlaps
        clips.pop(n, None)
        os.remove(os.path.join(audio_dir, n))
    ok = lambda p: p.split("/")[-1] in clips
    instr = {k: [p for p in v if ok(p)] for k, v in instr.items()}
    para = {k: [p for p in v if ok(p)] for k, v in para.items()}
    nonc = [p for p in nonc if ok(p)]

    # 5) episodes
    states = np.load(TRAIN_STATES)
    n_states = {t["id"]: states[f"task{t['id']}"].shape[0] for t in TASKS}
    pairs = [(a["id"], b["id"]) for a in TASKS for b in TASKS if a["id"] != b["id"] and (a["id"], b["id"]) not in test_pairs]
    eps, k = [], 0

    def episode(kind, a, b, cue_path, cue_text, gap):
        nonlocal k
        e = dict(id=f"train_{kind}_{k:05d}", split="train", type=kind, task_a=a, task_b=b, init_task=a,
                 init_source="train", init_idx=rng.randrange(n_states[a]),
                 instruction_text=sentence(TASKS[a]["language"]), instruction_audio=rng.choice(instr[a]),
                 utterance_text=cue_text, utterance_audio=cue_path, gap_sec=gap, speaker=-1,
                 target_task=b if kind == "switch" else a)
        k += 1
        eps.append(e)

    for a, b in pairs:
        for _ in range(n_switch_per_pair):
            if para[b] and rng.random() < paraphrase_prob:
                cue = rng.choice(para[b]); text = clips[cue.split("/")[-1]]["text"]
            else:
                cue = rng.choice(instr[b]); text = sentence(TASKS[b]["language"])
            episode("switch", a, b, cue, text, round(rng.uniform(1.5, 7.0), 1))
    for _ in range(n_keep):
        a = rng.randrange(len(TASKS)); cue = rng.choice(nonc)
        episode("keep", a, None, cue, clips[cue.split("/")[-1]]["text"], round(rng.uniform(1.5, 7.0), 1))
    for _ in range(n_plain):
        episode("plain", rng.randrange(len(TASKS)), None, None, None, None)
    rng.shuffle(eps)

    json.dump({"version": 1, "episodes": eps, "extra_command_clips": extra_cmds},
              open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    json.dump(clips, open(os.path.join(audio_dir, "clips.json"), "w"), indent=1)
    report = dict(eval_dirs=eval_dirs, eval_clips_checked=len(eval_md5), removed_same_audio=len(leak_audio),
                  removed_same_question=len(leak_text), test_pairs_excluded=sorted(test_pairs),
                  n_pairs=len(pairs), clips=len(clips),
                  counts={t: sum(e["type"] == t for e in eps) for t in ("switch", "keep", "plain")},
                  command_clips=sum(c["label"] == 1 for c in clips.values()),
                  noncommand_clips=sum(c["label"] == 0 for c in clips.values()),
                  sources={s: sum(c["source"] == s for c in clips.values()) for s in {c["source"] for c in clips.values()}})
    json.dump(report, open(os.path.join(out_dir, "leak_report.json"), "w"), indent=1)
    return report
