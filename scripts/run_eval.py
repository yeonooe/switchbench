"""Evaluate a policy on a SwitchBench split. Resumable: re-running skips finished episodes.

    python scripts/run_eval.py --data data/switchbench/dev --policy ellsa --out results/ellsa_dev
    python scripts/run_eval.py --data data/switchbench/dev --policy noop  --out results/noop_dev

--policy accepts 'ellsa', 'noop', 'random', or 'module.path:ClassName' for your own policy.
"""
import argparse
import importlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from switchbench import harness, manifest, metrics  # noqa: E402


def make_policy(name):
    if name == "noop":
        from switchbench.policy import NoopPolicy
        return NoopPolicy()
    if name == "random":
        from switchbench.policy import RandomPolicy
        return RandomPolicy()
    if name == "ellsa":
        from baselines.ellsa_policy import ELLSAPolicy
        return ELLSAPolicy()
    if name == "b1":
        from baselines.reset_policies import ResetOnSpeechPolicy
        return ResetOnSpeechPolicy()
    if name == "p1":
        from baselines.reset_policies import IntentGatedResetPolicy
        return IntentGatedResetPolicy()
    if name == "p2":
        from baselines.reset_policies import LearnedGateResetPolicy
        return LearnedGateResetPolicy()
    mod, cls = name.split(":")
    return getattr(importlib.import_module(mod), cls)()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="split folder with manifest.json")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="stop after this many episodes in total")
    ap.add_argument("--types", default="switch,keep,plain")
    ap.add_argument("--videos", type=int, default=10, help="save GIFs for the first N episodes")
    args = ap.parse_args()

    episodes = [e for e in manifest.load(os.path.join(args.data, "manifest.json"))
                if e["type"] in args.types.split(",")]
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    policy = make_policy(args.policy)
    print(f"policy '{args.policy}' ready in {time.time()-t0:.0f}s", flush=True)
    out_jsonl = os.path.join(args.out, "episodes.jsonl")
    harness.run(episodes, policy, args.data, out_jsonl, limit=args.limit,
                video_dir=os.path.join(args.out, "videos"), video_first=args.videos,
                log=lambda m: print(m, flush=True))

    s = metrics.summarize(metrics.load_records(out_jsonl))
    json.dump(s, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    md = metrics.to_markdown(s, f"SwitchBench {os.path.basename(args.data)} - {args.policy}")
    open(os.path.join(args.out, "summary.md"), "w").write(md + "\n")
    print("\n" + md)


if __name__ == "__main__":
    main()
