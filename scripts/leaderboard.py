"""Build the SwitchBench leaderboard from several result folders.

Each result folder holds episodes.jsonl (written by run_eval.py) and an optional meta.json:
    {"name": "ELLSA (zero-shot)", "kind": "baseline", "training": "none", "note": "..."}

    python scripts/leaderboard.py results/b0_ellsa_dev results/my_model_dev --out leaderboard

Writes <out>.md (table) and <out>.json (rows, for the web page), ranked by Score.
Runs missing any switch or keep episode (the two the Score uses) are marked partial and not ranked.
Ties on Score are broken by PSR. PSR is not in the Score, but a run whose PSR falls below
PSR_FLOOR (the lower 95% bound of the zero-shot ELLSA baseline on dev, 9/10 -> 60%) is flagged:
it may have traded basic manipulation for switching.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from switchbench import metrics  # noqa: E402

N_SWITCH, N_KEEP = 30, 10
PSR_FLOOR = 0.60


def pct(r):
    return "-" if not r["n"] else f"{100 * r['rate']:.1f}% ({r['k']}/{r['n']})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="result folders containing episodes.jsonl")
    ap.add_argument("--out", default="leaderboard")
    args = ap.parse_args()

    rows = []
    for run in args.runs:
        meta_path = os.path.join(run, "meta.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        s = metrics.summarize(metrics.load_records(os.path.join(run, "episodes.jsonl")))
        rows.append(dict(
            name=meta.get("name", os.path.basename(os.path.normpath(run))),
            kind=meta.get("kind", "submission"), training=meta.get("training", "?"),
            note=meta.get("note", ""), episodes=s["n_episodes"], partial=s["SSR"]["n"] < N_SWITCH or s["KSR"]["n"] < N_KEEP,
            score=None if s["score"] != s["score"] else round(100 * s["score"], 1),
            SSR=s["SSR"], KSR=s["KSR"], PSR=s["PSR"], stale=s["stale_completion"]))
        r = rows[-1]
        r["psr_flag"] = bool(r["PSR"]["n"]) and r["PSR"]["rate"] < PSR_FLOOR
    ranked = sorted([r for r in rows if not r["partial"]],
                    key=lambda r: (-(r["score"] or 0), -(r["PSR"]["rate"] if r["PSR"]["n"] else 0)))
    others = [r for r in rows if r["partial"]]

    lines = ["| rank | model | type | training | Score | SSR | KSR | PSR | stale completion |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(ranked + others, 1):
        rank = str(i) if not r["partial"] else f"- (partial, {r['episodes']} ep)"
        lines.append(f"| {rank} | {r['name']} | {r['kind']} | {r['training']} | {r['score']} | "
                     f"{pct(r['SSR'])} | {pct(r['KSR'])} | {pct(r['PSR'])}{' ⚠' if r['psr_flag'] else ''} | {pct(r['stale'])} |")
    if any(r["psr_flag"] for r in rows):
        lines.append(f"\n⚠ PSR below {PSR_FLOOR:.0%}: basic manipulation may be degraded (not in the Score).")
    open(args.out + ".md", "w").write("\n".join(lines) + "\n")
    json.dump(ranked + others, open(args.out + ".json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
