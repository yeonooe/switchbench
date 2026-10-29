"""Audio helpers and the episode timeline.

Timeline of one episode (t = env step after the NUM_STEPS_WAIT settling steps, 1 step = 0.1 s):

    t=0                instr_end          utt_start      utt_end                   max_steps
    |--- instruction ---|---- gap_sec ------|-- utterance --|------ TASK_BUDGET -------|
                                            (switch/keep only)

    plain:        max_steps = instr_end + TASK_BUDGET
    switch/keep:  max_steps = utt_end   + TASK_BUDGET

The policy only ever receives the audio up to the current step (no look-ahead).
"""
import math
import os

import numpy as np
import soundfile as sf

from . import SAMPLE_RATE, STEP_SEC, TASK_BUDGET

SAMPLES_PER_STEP = int(round(SAMPLE_RATE * STEP_SEC))


def load_wav(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
    return wav.astype(np.float32)


def steps_of(seconds):
    return int(math.ceil(seconds / STEP_SEC - 1e-9))


def build_stream(episode, data_root):
    """Return (stream, timeline) for an episode whose audio files live under data_root."""
    instr = load_wav(os.path.join(data_root, episode["instruction_audio"]))
    instr_end = steps_of(len(instr) / SAMPLE_RATE)
    parts = [instr, np.zeros(instr_end * SAMPLES_PER_STEP - len(instr), np.float32)]
    tl = dict(instr_end=instr_end, utt_start=None, utt_end=None)
    if episode["type"] == "plain":
        tl["max_steps"] = instr_end + TASK_BUDGET
    else:
        utt = load_wav(os.path.join(data_root, episode["utterance_audio"]))
        gap = steps_of(episode["gap_sec"])
        tl["utt_start"] = instr_end + gap
        tl["utt_end"] = tl["utt_start"] + steps_of(len(utt) / SAMPLE_RATE)
        tl["max_steps"] = tl["utt_end"] + TASK_BUDGET
        parts += [np.zeros(gap * SAMPLES_PER_STEP, np.float32), utt,
                  np.zeros((tl["utt_end"] - tl["utt_start"]) * SAMPLES_PER_STEP - len(utt), np.float32)]
    stream = np.concatenate(parts)
    total = tl["max_steps"] * SAMPLES_PER_STEP
    stream = np.concatenate([stream, np.zeros(max(0, total - len(stream)), np.float32)])
    return stream, tl


def trim_silence(wav, threshold_db=-40.0, pad_sec=0.1):
    """Cut leading/trailing silence so utterance timing in the manifest is accurate."""
    if len(wav) == 0:
        return wav
    frame = int(0.02 * SAMPLE_RATE)
    n = len(wav) // frame
    if n == 0:
        return wav
    energy = 20 * np.log10(np.sqrt((wav[: n * frame].reshape(n, frame) ** 2).mean(axis=1)) + 1e-9)
    peak = energy.max()
    voiced = np.where(energy > peak + threshold_db)[0]
    if len(voiced) == 0:
        return wav
    pad = int(pad_sec * SAMPLE_RATE)
    start = max(0, voiced[0] * frame - pad)
    end = min(len(wav), (voiced[-1] + 1) * frame + pad)
    return wav[start:end]


def normalize(wav, peak=0.9):
    m = np.abs(wav).max()
    return wav if m == 0 else (wav / m * peak).astype(np.float32)
