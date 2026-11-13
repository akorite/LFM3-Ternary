#!/usr/bin/env python3
"""Single owner of the retention-gate arithmetic.

Tier-0 (avg-5 lm-eval rung gate):
  pass      avg retention >= 0.88 AND every task retention >= 0.80
  escalate  otherwise, while a later rung can still try
  kill      avg retention < 0.80, or a miss with no rungs left

Tier-1 (BFCLv3):
  pass  retention >= 0.88; kill below that.

Every verdict prints the arithmetic and is emitted as JSON
{verdict, avg_retention, per_task_retention, reason} to stdout
and to --out when given. A missing or unreadable score file is a
hard error (nonzero exit), never a silent zero baseline.
"""
import argparse
import json
import math
import sys
from pathlib import Path

TASKS = ["piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande"]
RUNGS = (1, 2, 3)

AVG_PASS = 0.88
TASK_PASS = 0.80
AVG_KILL = 0.80


def load_scores(path, label):
    try:
        data = json.loads(Path(path).read_text())
    except OSError as exc:
        sys.exit(f"error: cannot read {label} file {path}: {exc}")
    except json.JSONDecodeError as exc:
        sys.exit(f"error: corrupt {label} file {path}: {exc}")
    if not isinstance(data, dict):
        sys.exit(f"error: {label} file {path} is not a JSON object of task scores")
    scores = {}
    for task in TASKS:
        value = data.get(task)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            sys.exit(
                f"error: {label} file {path} has no usable score for {task} "
                f"(got {value!r}); refusing to treat it as 0"
            )
        scores[task] = float(value)
    return scores


def emit(verdict, out_path):
    print(json.dumps(verdict, indent=2, sort_keys=True))
    if out_path:
        with open(out_path, "w") as handle:
            json.dump(verdict, handle, indent=2, sort_keys=True)
            handle.write("\n")


def tier0(args):
    if args.remaining != len(RUNGS) - args.rung:
        sys.exit(
            f"error: --remaining {args.remaining} contradicts --rung {args.rung} "
            f"({len(RUNGS)} rungs, so {len(RUNGS) - args.rung} remain)"
        )
    baseline = load_scores(args.baseline, "baseline")
    scores = load_scores(args.scores, "scores")

    per_task = {}
    print(f"rung {args.rung} tier-0 retention vs baseline:")
    for task in TASKS:
        per_task[task] = scores[task] / baseline[task]
        print(f"  {task}: {scores[task]:.4f} / {baseline[task]:.4f} = {per_task[task]:.4f}")
    avg_base = sum(baseline.values()) / len(TASKS)
    avg_score = sum(scores.values()) / len(TASKS)
    avg_retention = avg_score / avg_base
    print(f"  avg-5: {avg_score:.4f} / {avg_base:.4f} = {avg_retention:.4f}")

    weakest = min(per_task.items(), key=lambda kv: kv[1])
    if avg_retention >= AVG_PASS and all(r >= TASK_PASS for r in per_task.values()):
        verdict = "pass"
        reason = (
            f"avg retention {avg_retention:.4f} >= {AVG_PASS} and every task "
            f">= {TASK_PASS} (weakest {weakest[0]} = {weakest[1]:.4f})"
        )
    elif avg_retention < AVG_KILL:
        verdict = "kill"
        reason = (
            f"avg retention {avg_retention:.4f} < {AVG_KILL}: no remaining rung "
            "can be judged above the floor from this artifact"
        )
    elif args.remaining <= 0:
        verdict = "kill"
        reason = (
            f"rung {args.rung} misses the gate (avg {avg_retention:.4f}, weakest "
            f"{weakest[0]} {weakest[1]:.4f}) and no rungs remain"
        )
    else:
        verdict = "escalate"
        reason = (
            f"avg retention {avg_retention:.4f} or task floor missed (weakest "
            f"{weakest[0]} {weakest[1]:.4f}); {args.remaining} rung(s) remain"
        )

    print(f"verdict: {verdict} — {reason}")
    emit(
        {
            "verdict": verdict,
            "avg_score": round(avg_score, 4),
            "avg_retention": round(avg_retention, 4),
            "per_task_retention": {t: round(r, 4) for t, r in per_task.items()},
            "reason": reason,
        },
        args.out,
    )


def tier1(args):
    if args.baseline_score <= 0:
        sys.exit(f"error: baseline score {args.baseline_score} is not positive")
    retention = args.score / args.baseline_score
    print(f"tier-1 BFCLv3 retention: {args.score:.4f} / {args.baseline_score:.4f} = {retention:.4f}")
    if retention >= AVG_PASS:
        verdict = "pass"
        reason = f"retention {retention:.4f} >= {AVG_PASS}"
    else:
        verdict = "kill"
        reason = f"retention {retention:.4f} < {AVG_PASS}"
    print(f"verdict: {verdict} — {reason}")
    emit(
        {
            "verdict": verdict,
            "avg_retention": round(retention, 4),
            "per_task_retention": {"bfclv3": round(retention, 4)},
            "reason": reason,
        },
        args.out,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="gate", required=True)

    t0 = sub.add_parser("tier0", help="avg-5 lm-eval rung gate")
    t0.add_argument("--baseline", required=True, help="JSON of baseline task scores")
    t0.add_argument("--scores", required=True, help="JSON of rung task scores")
    t0.add_argument("--rung", type=int, choices=[1, 2, 3], required=True)
    t0.add_argument("--remaining", type=int, required=True, help="rungs left after this one")
    t0.add_argument("--out", help="optional path for the verdict JSON")
    t0.set_defaults(fn=tier0)

    t1 = sub.add_parser("tier1", help="BFCLv3 retention gate")
    t1.add_argument("--baseline-score", type=float, required=True, help="BF16 BFCLv3 score")
    t1.add_argument("--score", type=float, required=True, help="ternary BFCLv3 score")
    t1.add_argument("--out", help="optional path for the verdict JSON")
    t1.set_defaults(fn=tier1)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
