"""Train the P2 spoken intent gate on SwitchBench-train and evaluate it on the dev cues.

    python scripts/train_intent.py --train data/train --dev data/dev --encoder whisper-base --out results/gate
Writes <out>/gate.pt, <out>/report.json (validation + dev metrics, per-clip dev probabilities).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from switchbench import intent  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--encoder", default="whisper-base", help="whisper-tiny|whisper-base|whisper-small|mfcc")
    ap.add_argument("--aug", type=int, default=2, help="augmented copies per training clip")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch
    os.makedirs(args.out, exist_ok=True)
    ck, dev = intent.train(args.train, args.dev, encoder=args.encoder, n_aug=args.aug, epochs=args.epochs,
                           log=lambda m: print(m, flush=True))
    torch.save(ck, os.path.join(args.out, "gate.pt"))
    json.dump(dict(encoder=args.encoder, val=ck["val"], dev=dev), open(os.path.join(args.out, "report.json"), "w"), indent=1)
    if dev:
        wrong = [d for d in dev["items"] if (d["prob"] >= 0.5) != bool(d["label"])]
        print(f"\nmisclassified dev cues: {len(wrong)}")
        for d in wrong:
            print(f"  label={d['label']} p={d['prob']:.2f}  '{d['text']}'")
    print("saved", os.path.join(args.out, "gate.pt"))


if __name__ == "__main__":
    main()
