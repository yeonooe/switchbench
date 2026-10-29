"""Training-free policies built on ELLSA that restart it on a new spoken command.

B1  ResetOnSpeechPolicy   : restart ELLSA on EVERY utterance after the first one.
P1  IntentGatedResetPolicy: transcribe the utterance (Whisper) and restart only if it is a
                            manipulation command; questions and chit-chat are left to ELLSA.

Why: on SwitchBench B0 (ELLSA) heard mid-task speech correctly (it answered every keep-question,
and named the new object in two switch episodes) but never treated a new instruction as a
command. P1 adds exactly that missing decision; B1 shows what happens without it.

Restart = ELLSA's internal history is cleared and it is fed the audio stream starting at the new
utterance, as if that utterance were the episode's first instruction. ELLSA consumes one 1 s audio
block per model call, so after a restart it re-listens to the utterance (a few seconds of lag) and
then acts. It only ever receives audio that has already been heard: no look-ahead.

Utterances are found with a streaming energy VAD: an utterance ends after `end_sil` seconds of
silence. The first utterance of an episode is the original instruction and never triggers a restart.
"""
import re

import numpy as np

from switchbench.policy import Policy

COMMAND_VERBS = {"put", "place", "open", "close", "push", "pull", "turn", "pick", "move", "take",
                 "grab", "stack", "set", "bring", "lift", "drop", "switch", "start", "stop"}
QUESTION_START = {"who", "what", "when", "where", "why", "how", "which", "whose", "is", "are", "was",
                  "were", "do", "does", "did", "can", "could", "would", "will", "should", "have", "has",
                  "tell", "name"}


def is_command(text):
    """Rule-based intent: an imperative manipulation sentence, not a question."""
    t = text.strip().lower()
    if not t or t.endswith("?"):
        return False
    words = re.findall(r"[a-z']+", t)
    if not words or words[0] in QUESTION_START:
        return False
    # imperative: a manipulation verb within the first few words ("actually, put ...", "please open ...")
    return any(w in COMMAND_VERBS for w in words[:4])


class StreamingVAD:
    """Energy VAD over 20 ms frames; returns utterances (start, end) in samples once they end."""

    def __init__(self, sr=16000, frame_sec=0.02, threshold=0.01, end_sil=0.8, min_len=0.3):
        self.sr, self.frame = sr, int(sr * frame_sec)
        self.threshold, self.end_sil, self.min_len = threshold, end_sil, min_len
        self.reset()

    def reset(self):
        self.pos, self.start, self.last_voiced, self.done = 0, None, None, []

    def update(self, audio):
        new = []
        while self.pos + self.frame <= len(audio):
            f = audio[self.pos: self.pos + self.frame]
            voiced = float(np.sqrt(np.mean(f * f))) > self.threshold
            if voiced:
                if self.start is None:
                    self.start = self.pos
                self.last_voiced = self.pos + self.frame
            elif self.start is not None and self.pos - self.last_voiced >= self.end_sil * self.sr:
                if self.last_voiced - self.start >= self.min_len * self.sr:
                    new.append((self.start, self.last_voiced))
                self.start = None
            self.pos += self.frame
        self.done += new
        return new


class ResetOnSpeechPolicy(Policy):
    """B1: restart ELLSA on every utterance after the instruction."""

    name = "b1_reset_on_speech"

    def __init__(self, inner=None):
        if inner is None:
            from baselines.ellsa_policy import ELLSAPolicy
            inner = ELLSAPolicy()
        self.inner = inner
        self.vad = StreamingVAD()

    def reset(self, info):
        self.info = dict(info)
        self.vad.reset()
        self.offset = 0          # sample where the inner policy's audio stream starts
        self.n_utts = 0
        self.inner.reset(self.info)

    def decide(self, segment_audio):
        return True, "reset"

    def _restart(self, start, t):
        self.offset = start
        step = int(start / (self.info["sample_rate"] * self.info["step_sec"]))
        self.inner.reset(dict(self.info, max_steps=max(1, self.info["max_steps"] - step)))

    def act(self, obs, audio, t):
        note = None
        for s, e in self.vad.update(audio):
            self.n_utts += 1
            if self.n_utts == 1:
                continue  # the original instruction
            restart, why = self.decide(audio[s:e])
            note = f"[{self.name}] utterance {s/16000:.1f}-{e/16000:.1f}s: {why}"
            if restart:
                self._restart(s, t)
        out = self.inner.act(obs, audio[self.offset:], t)
        if note:
            out = dict(out, note=note)
        return out


class IntentGatedResetPolicy(ResetOnSpeechPolicy):
    """P1: restart ELLSA only when the new utterance is a manipulation command."""

    name = "p1_intent_gated_reset"

    def __init__(self, inner=None, transcribe=None, asr_model="base.en"):
        super().__init__(inner)
        if transcribe is None:
            import whisper
            asr = whisper.load_model(asr_model)
            transcribe = lambda wav: asr.transcribe(wav.astype(np.float32), language="en", fp16=False)["text"]
        self.transcribe = transcribe

    def decide(self, segment_audio):
        text = self.transcribe(segment_audio).strip()
        cmd = is_command(text)
        return cmd, f"heard '{text}' -> {'COMMAND, reset' if cmd else 'not a command, keep going'}"


class LearnedGateResetPolicy(ResetOnSpeechPolicy):
    """P2: restart ELLSA only when a gate trained on SwitchBench-train says 'command'.

    Unlike P1 there is no transcript and no word list: the decision is made from the audio.
    The gate checkpoint path comes from the argument or $SWITCHBENCH_GATE.
    """

    name = "p2_learned_gate_reset"

    def __init__(self, inner=None, gate=None, gate_path=None):
        super().__init__(inner)
        if gate is None:
            import os
            from switchbench.intent import Gate
            gate = Gate(gate_path or os.environ["SWITCHBENCH_GATE"])
        self.gate = gate

    def decide(self, segment_audio):
        cmd, p = self.gate.is_command(segment_audio.astype(np.float32))
        return cmd, f"p(command)={p:.2f} -> {'COMMAND, reset' if cmd else 'not a command, keep going'}"
