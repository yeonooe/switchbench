"""ELLSA (End-to-end Listen, Look, Speak and Act, ICLR 2026) as a SwitchBench policy.

Uses the public checkpoint unchanged (zero-shot on SwitchBench). The model call mirrors the
official LIBERO evaluation (evaluate_libero_emu_contemporary.py, `--speech --moe --silence`):
same constructor flags, 128-bin lhotse fbank, one model call per 10-action chunk (1 s block),
images rotated 180 degrees, generate_text=True.

Difference from the official script: the official script computes the fbank of the WHOLE
episode audio up front. Here the policy only receives audio up to the current step, and the
unknown future is filled with silence before computing the fbank. ELLSA's speech encoder is
causal, so past blocks are unaffected; this simply guarantees no look-ahead.

Requires the environment variables used by the ELLSA code: ELLSA_BASE_PATH, LLAMA_CKPT_PATH,
UNIVLA_CKPT_PATH, COSY_CKPT_PATH, VISION_VQ_PATH, and ELLSA_CKPT (checkpoint folder).
"""
import os
import sys

import numpy as np
import torch

from switchbench.policy import Policy

_BASE = os.environ["ELLSA_BASE_PATH"]
sys.path.insert(0, os.path.join(_BASE, "reference/RoboVLMs"))
sys.path.insert(0, os.path.join(_BASE, "reference/RoboVLMs/eval/libero"))


class ELLSAPolicy(Policy):
    def __init__(self, ckpt=None, vision_hub=None):
        from lhotse import Fbank, FbankConfig
        from model_wrapper_emu import EmuVLAModel
        from libero_utils import get_libero_image, get_libero_wrist_image, quat2axisangle
        self._img, self._wrist, self._q2a = get_libero_image, get_libero_wrist_image, quat2axisangle
        self.fbank = Fbank(FbankConfig(num_mel_bins=128))
        self.model = EmuVLAModel(
            emu_hub=ckpt or os.environ["ELLSA_CKPT"], vq_hub="",
            vision_hub=vision_hub or os.environ["VISION_VQ_PATH"],
            device=torch.device("cuda"), speech=True, moe=True, generate=False,
            merge_speech_lora=True, lora_modules="qkv", encoder_type="zipformer2",
            time_block=1.0, predict_action_frames=10)
        self.horizon = None
        self.sr = 16000

    def reset(self, info):
        self.model.reset()
        self.sr = info["sample_rate"]
        # same length as the full episode stream in the official script
        self.horizon = int(round(info["max_steps"] * info["step_sec"] * self.sr))

    def act(self, obs, audio, t):
        observation = {
            "full_image": self._img(obs),
            "wrist_image": self._wrist(obs),
            "state": np.concatenate((obs["robot0_eef_pos"], self._q2a(obs["robot0_eef_quat"]),
                                     obs["robot0_gripper_qpos"])),
        }
        stream = np.zeros(max(self.horizon, len(audio)), dtype=np.float32)
        stream[: len(audio)] = audio  # future = silence (no look-ahead)
        feat = self.fbank.extract(torch.from_numpy(stream), sampling_rate=self.sr)
        if not torch.is_tensor(feat):
            feat = torch.from_numpy(np.asarray(feat))
        text, action = self.model.step_withspeech(observation, feat, feat.size(0), t, generate_text=True)
        return {"actions": np.asarray(action, dtype=np.float32).reshape(-1, 7), "text": text}
