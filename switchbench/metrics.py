"""SwitchBench scoring.

Primary score  = (SSR + KSR) / 2
  SSR  Switch Success Rate : switch episodes where the NEW task B was completed
  KSR  Keep Success Rate   : keep episodes where the ORIGINAL task A was completed despite
                             an irrelevant utterance
  A policy that always switches on any speech, or never reacts to speech, cannot score well.
Diagnostics (not part of the score)
  PSR  Plain Success Rate  : no second utterance (basic speech-conditioned manipulation)
  Stale completion rate    : switch episodes where the robot finished the OLD task A after the cue
  A-done-before-cue rate   : switch episodes where A was already finished before the cue arrived
  Steps after cue          : median steps from the end of the cue to success (lower = faster)
All rates come with Wilson 95% confidence intervals; dev sets are small, read them with care.
"""
import json
import math
from collections import defaultdict


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rate(recs, key=lambda r: r["success"]):
    k, n = sum(1 for r in recs if key(r)), len(recs)
    lo, hi = wilson(k, n)
    return dict(rate=(k / n if n else float("nan")), k=k, n=n, ci95=[lo, hi])


def median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def summarize(records):
    by = defaultdict(list)
    for r in records:
        by[r["type"]].append(r)
    sw, kp, pl = by["switch"], by["keep"], by["plain"]
    s = dict(
        n_episodes=len(records),
        n_errors=sum(1 for r in records if r.get("error")),
        SSR=rate(sw), KSR=rate(kp), PSR=rate(pl),
        stale_completion=rate(sw, lambda r: r.get("stale_completion")),
        a_done_before_cue=rate(sw, lambda r: r.get("a_done_before_cue")),
        median_steps_after_cue_switch=median([r["steps_after_cue"] for r in sw if r["success"]]),
        SSR_by_gap={str(g): rate([r for r in sw if r.get("gap_sec") == g])
                    for g in sorted({r.get("gap_sec") for r in sw if r.get("gap_sec") is not None})},
    )
    s["score"] = (s["SSR"]["rate"] + s["KSR"]["rate"]) / 2 if sw and kp else float("nan")
    return s


def fmt(r):
    if not r["n"]:
        return "n/a"
    return f"{100*r['rate']:5.1f}%  ({r['k']}/{r['n']}, 95% CI {100*r['ci95'][0]:.0f}-{100*r['ci95'][1]:.0f}%)"


def to_markdown(s, title="SwitchBench results"):
    lines = [f"### {title}", "", "| metric | value |", "|---|---|",
             f"| **Score = (SSR + KSR) / 2** | **{100*s['score']:.1f}** |",
             f"| SSR (switch success) | {fmt(s['SSR'])} |",
             f"| KSR (keep success) | {fmt(s['KSR'])} |",
             f"| PSR (plain success) | {fmt(s['PSR'])} |",
             f"| stale completion (switch) | {fmt(s['stale_completion'])} |",
             f"| A done before cue (switch) | {fmt(s['a_done_before_cue'])} |",
             f"| median steps after cue (successful switches) | {s['median_steps_after_cue_switch']} |"]
    for g, r in s["SSR_by_gap"].items():
        lines.append(f"| SSR, gap {g}s | {fmt(r)} |")
    lines.append(f"| episodes / errors | {s['n_episodes']} / {s['n_errors']} |")
    return "\n".join(lines)


def load_records(path):
    return [json.loads(l) for l in open(path) if l.strip()]
