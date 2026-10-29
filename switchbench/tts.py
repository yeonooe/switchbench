"""Synthesize the benchmark audio with CosyVoice2 (zero-shot voice from a prompt recording).

Every (text, speaker) pair is synthesized once and cached. Each clip is checked with Whisper:
if the transcript's word error rate exceeds `max_wer`, it is re-synthesized (up to `retries`).
"""
import hashlib
import json
import os
import re
import sys

import numpy as np
import soundfile as sf

from . import SAMPLE_RATE
from .audio import load_wav, normalize, trim_silence


def norm_words(text):
    return re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()


def wer(ref, hyp):
    r, h = norm_words(ref), norm_words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / max(1, len(r))


def find_prompts(prompt_dir, n, min_sec=2.0, max_sec=15.0):
    """Pick n speaker prompt recordings (deterministic) from a folder of wav files."""
    paths = []
    for root, _, files in os.walk(prompt_dir):
        for fn in files:
            if fn.lower().endswith(".wav"):
                paths.append(os.path.join(root, fn))
    paths.sort()
    good = []
    for p in paths:
        try:
            info = sf.info(p)
            if min_sec <= info.duration <= max_sec:
                good.append(p)
        except Exception:
            pass
    if len(good) < n:
        raise SystemExit(f"need {n} prompt wavs of {min_sec}-{max_sec}s under {prompt_dir}, found {len(good)}")
    step = len(good) / n
    return [good[int(i * step)] for i in range(n)]


class Synth:
    GTTS_ACCENTS = ["com", "co.uk", "com.au", "ca", "co.in", "ie", "co.za", "com.ng"]

    def __init__(self, cosy_dir, ellsa_base, fake=False, asr_model="base.en", device="cuda", engine="cosyvoice"):
        self.fake = fake
        self.engine = engine
        self.asr = None
        if fake:
            return
        if asr_model:
            import whisper
            self.asr = whisper.load_model(asr_model, device=device)
        if engine == "gtts":
            return
        sys.path.append(os.path.join(ellsa_base, "reference/cosyvoice/third_party/Matcha-TTS"))
        sys.path.append(os.path.join(ellsa_base, "reference"))
        from cosyvoice.cli.cosyvoice import CosyVoice2
        self.cosy = CosyVoice2(cosy_dir, load_jit=False, load_trt=False)

    def _tts_once(self, text, prompt_path, seed):
        if self.fake:  # placeholder tone, ~0.3 s per word, for pipeline tests without a GPU
            n = int(0.3 * SAMPLE_RATE * max(1, len(text.split())))
            t = np.arange(n) / SAMPLE_RATE
            return (0.3 * np.sin(2 * np.pi * (220 + 20 * (seed % 5)) * t)).astype(np.float32)
        if self.engine == "gtts":  # fallback engine: speaker = English accent
            import io
            from gtts import gTTS
            buf = io.BytesIO()
            gTTS(text, lang="en", tld=self.GTTS_ACCENTS[int(prompt_path) % len(self.GTTS_ACCENTS)]).write_to_fp(buf)
            buf.seek(0)
            wav, sr = sf.read(buf, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            import librosa
            return librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE).astype(np.float32)
        import torch
        import torchaudio
        torch.manual_seed(seed)
        prompt = torch.from_numpy(load_wav(prompt_path)).unsqueeze(0)
        chunks = [o["tts_speech"] for o in self.cosy.inference_cross_lingual(text, prompt, stream=False)]
        wav = torch.cat(chunks, dim=1)
        wav = torchaudio.functional.resample(wav, self.cosy.sample_rate, SAMPLE_RATE)
        return wav.squeeze(0).cpu().numpy().astype(np.float32)

    def synth(self, text, prompt_path, out_path, retries=3, max_wer=0.25):
        best = None
        for attempt in range(retries):
            wav = normalize(trim_silence(self._tts_once(text, prompt_path, seed=1000 + attempt)))
            if self.asr is None:
                best = (wav, None, None)
                break
            hyp = self.asr.transcribe(wav, language="en", fp16=False)["text"]
            score = wer(text, hyp)
            if best is None or score < best[1]:
                best = (wav, score, hyp)
            if score <= max_wer:
                break
        wav, score, hyp = best
        sf.write(out_path, wav, SAMPLE_RATE)
        return dict(duration=round(len(wav) / SAMPLE_RATE, 3), asr_wer=score, asr_text=hyp)


def clip_name(text, speaker):
    return hashlib.md5(f"{speaker}|{text}".encode()).hexdigest()[:12] + ".wav"


def synthesize_episodes(episodes, out_dir, prompts, synth):
    """Fill instruction_audio / utterance_audio (paths relative to out_dir) for every episode."""
    os.makedirs(os.path.join(out_dir, "audio"), exist_ok=True)
    cache_path = os.path.join(out_dir, "audio", "clips.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    todo = set()
    for e in episodes:
        todo.add((e["instruction_text"], e["speaker"]))
        if e["utterance_text"]:
            todo.add((e["utterance_text"], e["speaker"]))
    for i, (text, spk) in enumerate(sorted(todo)):
        name = clip_name(text, spk)
        path = os.path.join(out_dir, "audio", name)
        if name not in cache or not os.path.exists(path):
            cache[name] = dict(text=text, speaker=spk, prompt=os.path.basename(str(prompts[spk])),
                               **synth.synth(text, prompts[spk], path))
            json.dump(cache, open(cache_path, "w"), indent=1)
        c = cache[name]
        flag = "" if c["asr_wer"] is None or c["asr_wer"] <= 0.25 else "  <-- check"
        print(f"[{i+1}/{len(todo)}] spk{spk} {c['duration']:4.1f}s wer={c['asr_wer']} '{text}'{flag}", flush=True)
    for e in episodes:
        e["instruction_audio"] = "audio/" + clip_name(e["instruction_text"], e["speaker"])
        e["utterance_audio"] = "audio/" + clip_name(e["utterance_text"], e["speaker"]) if e["utterance_text"] else None
    return cache
