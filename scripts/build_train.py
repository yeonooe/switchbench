"""Build SwitchBench-train (the official training split).

    python scripts/build_train.py --out data --eval data/dev data/test --cache /content/sb_cache

Needs internet (ELLSA train recordings, VoiceAssistant-400K, gTTS). No GPU, no simulator.
Writes data/train/{manifest.json, audio/, leak_report.json}.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from switchbench import train_data, tts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval", nargs="+", required=True, help="dev and test folders (for leak checks)")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--no_va", action="store_true", help="skip VoiceAssistant-400K questions")
    ap.add_argument("--no_gtts", action="store_true", help="no synthesized clips")
    ap.add_argument("--switch_per_pair", type=int, default=12)
    ap.add_argument("--keep", type=int, default=600)
    ap.add_argument("--plain", type=int, default=200)
    args = ap.parse_args()
    out = os.path.join(args.out, "train")
    os.makedirs(out, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)
    synth = None if args.no_gtts else tts.Synth(None, None, asr_model=None, engine="gtts")
    rep = train_data.build(out, args.eval, args.cache, synth=synth, n_switch_per_pair=args.switch_per_pair,
                           n_keep=args.keep, n_plain=args.plain, use_va=not args.no_va,
                           ellsa_token=os.environ.get("HF_TOKEN"), log=lambda m: print(m, flush=True))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
