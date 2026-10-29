"""Spoken intent gate: is a mid-task utterance a new manipulation command (1) or not (0)?

Model = frozen speech encoder + small trained MLP.
  encoder 'whisper-<size>' : Whisper encoder states over the clip, mean- and max-pooled
  encoder 'mfcc'           : 40 MFCC + deltas, mean and std (cheap ablation, no download)
Trained on SwitchBench-train clips (audio/clips.json labels), validated on held-out sentences,
evaluated on the dev cue recordings (official ELLSA test-data voices, never seen in training).
"""
import json
import math
import os
import random

import numpy as np

from . import SAMPLE_RATE
from .audio import load_wav


class Encoder:
    def __init__(self, name="whisper-base", device=None):
        self.name = name
        if name.startswith("whisper"):
            import torch
            import whisper
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = whisper.load_model(name.split("-", 1)[1], device=self.device)
            self.dim = 2 * self.model.dims.n_audio_state
        elif name == "mfcc":
            self.dim = 160
        else:
            raise ValueError(name)

    def __call__(self, wav):
        if self.name == "mfcc":
            import librosa
            m = librosa.feature.mfcc(y=wav, sr=SAMPLE_RATE, n_mfcc=40)
            f = np.concatenate([m, librosa.feature.delta(m)])
            return np.concatenate([f.mean(1), f.std(1)]).astype(np.float32)
        import torch
        import whisper
        with torch.no_grad():
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(torch.from_numpy(wav))).to(self.device)
            h = self.model.encoder(mel[None])[0]                      # (1500, d), 20 ms per frame
            n = max(1, min(h.shape[0], math.ceil(len(wav) / SAMPLE_RATE / 0.02)))
            h = h[:n]
            return torch.cat([h.mean(0), h.max(0).values]).float().cpu().numpy()


def augment(wav, rng):
    """Gain, background noise, speed and edge padding - recording conditions differ at test time."""
    import librosa
    w = wav * rng.uniform(0.4, 1.4)
    rate = rng.uniform(0.9, 1.1)
    w = librosa.resample(w, orig_sr=SAMPLE_RATE, target_sr=int(SAMPLE_RATE / rate))
    snr = rng.uniform(10, 35)
    p = float(np.mean(w ** 2)) + 1e-9
    w = w + np.random.default_rng(rng.randrange(1 << 30)).normal(0, math.sqrt(p / 10 ** (snr / 10)), len(w))
    pad = lambda: np.zeros(int(rng.uniform(0, 0.3) * SAMPLE_RATE), np.float32)
    return np.concatenate([pad(), w.astype(np.float32), pad()])


def train_clips(train_dir):
    clips = json.load(open(os.path.join(train_dir, "audio", "clips.json")))
    return [dict(path=os.path.join(train_dir, "audio", n), label=c["label"],
                 group=c["text"] or c["src"], source=c["source"]) for n, c in clips.items()]


def dev_clips(dev_dir):
    eps = json.load(open(os.path.join(dev_dir, "manifest.json")))["episodes"]
    out, seen = [], set()
    for e in eps:
        if e["type"] in ("switch", "keep") and e["utterance_audio"] not in seen:
            seen.add(e["utterance_audio"])
            out.append(dict(path=os.path.join(dev_dir, e["utterance_audio"]), label=int(e["type"] == "switch"),
                            text=e["utterance_text"]))
    return out


class Gate:
    """Trained MLP on top of an Encoder."""

    def __init__(self, ckpt_path, device=None):
        import torch
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.encoder = Encoder(ck["encoder"], device=device)
        self.mean, self.std, self.threshold = ck["mean"], ck["std"], ck["threshold"]
        self.net = make_mlp(ck["dim"])
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()

    def prob(self, wav):
        import torch
        x = (self.encoder(wav) - self.mean) / self.std
        with torch.no_grad():
            return float(torch.sigmoid(self.net(torch.from_numpy(x[None]).float()))[0, 0])

    def is_command(self, wav):
        p = self.prob(wav)
        return p >= self.threshold, p


def make_mlp(dim):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))


def metrics(y, p, thr=0.5):
    y, pred = np.asarray(y), np.asarray(p) >= thr
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    rec_cmd = tp / max(1, tp + fn)
    rec_non = tn / max(1, tn + fp)
    return dict(acc=(tp + tn) / max(1, len(y)), command_recall=rec_cmd, noncommand_recall=rec_non,
                balanced_acc=(rec_cmd + rec_non) / 2, tp=tp, tn=tn, fp=fp, fn=fn, n=len(y))


def train(train_dir, dev_dir, encoder="whisper-base", n_aug=2, epochs=300, val_frac=0.15, seed=0, log=print):
    import torch
    rng = random.Random(seed)
    torch.manual_seed(seed)
    enc = Encoder(encoder)
    items = train_clips(train_dir)
    groups = sorted({it["group"] for it in items})
    rng.shuffle(groups)
    val_groups = set(groups[: int(len(groups) * val_frac)])  # unseen sentences / recordings
    X, Y, Xv, Yv = [], [], [], []
    for i, it in enumerate(items):
        wav = load_wav(it["path"])
        is_val = it["group"] in val_groups
        (Xv if is_val else X).append(enc(wav)); (Yv if is_val else Y).append(it["label"])
        if not is_val:
            for _ in range(n_aug):
                X.append(enc(augment(wav, rng))); Y.append(it["label"])
        if (i + 1) % 200 == 0:
            log(f"  embedded {i+1}/{len(items)} clips")
    X, Y, Xv, Yv = np.stack(X), np.array(Y), np.stack(Xv), np.array(Yv)
    mean, std = X.mean(0), X.std(0) + 1e-6
    net = make_mlp(X.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    pos_w = torch.tensor([(Y == 0).sum() / max(1, (Y == 1).sum())], dtype=torch.float32)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    xt = torch.from_numpy((X - mean) / std).float()
    yt = torch.from_numpy(Y).float()[:, None]
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xt))
        for b in range(0, len(xt), 256):
            idx = perm[b: b + 256]
            opt.zero_grad()
            loss = lossf(net(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pv = torch.sigmoid(net(torch.from_numpy((Xv - mean) / std).float()))[:, 0].numpy() if len(Xv) else np.array([])
    val = metrics(Yv, pv) if len(Xv) else {}
    log(f"train clips {len(Y)} (with aug) | val clips {len(Yv)} (unseen sentences): {val}")

    ck = dict(encoder=encoder, dim=X.shape[1], mean=mean, std=std, threshold=0.5, state_dict=net.state_dict(), val=val)
    dev_report = None
    if dev_dir:
        dv = dev_clips(dev_dir)
        pd = []
        for d in dv:
            x = (enc(load_wav(d["path"])) - mean) / std
            with torch.no_grad():
                pd.append(float(torch.sigmoid(net(torch.from_numpy(x[None]).float()))[0, 0]))
        dev_report = dict(metrics=metrics([d["label"] for d in dv], pd),
                          items=[dict(text=d["text"], label=d["label"], prob=round(p, 3)) for d, p in zip(dv, pd)])
        ck["dev"] = dev_report["metrics"]
        log(f"dev cue recordings (official voices, never trained on): {dev_report['metrics']}")
    return ck, dev_report
