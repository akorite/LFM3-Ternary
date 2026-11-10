#!/usr/bin/env python3
"""Render REPORT.md strictly from committed result files.

Every number in the report comes out of `results/*.json`; a missing file says
so instead of being filled with an estimate. That is the mechanism behind the
honest-claims constraint: a report cannot contain a number nobody measured.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TASKS = ["piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande"]
TASK_LABEL = {"arc_easy": "ARC-E", "arc_challenge": "ARC-C", "hellaswag": "HellaSwag",
              "winogrande": "Winogrande", "piqa": "PIQA"}


def _load(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _pct(x) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _num(x, digits=1) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def render(results: Path, plan: str) -> str:
    base = _load(results / "baseline.json")
    rungs = {n: _load(results / f"rung{n}.json") for n in (1, 2, 3)}
    verdicts = {n: _load(results / f"rung{n}_verdict.json") for n in (1, 2, 3)}
    bfcl = _load(results / "bfcl.json")
    bench = _load(results / "benchmarks.json")

    out = ["# LFM3-1.7B ternary compression — results", ""]
    out += [f"Plan: `{plan}`. Every number below is read from `results/*.json`; "
            "a dash means that measurement has not been run.", ""]

    out += ["## Tier-0 quality (lm-eval zero-shot)", ""]
    if base is None:
        out.append("No BF16 baseline recorded yet (`results/baseline.json` missing).")
    else:
        out += ["| Task | BF16 | " + " | ".join(f"rung {n}" for n in rungs) + " |",
                "|---|---|" + "---|" * len(rungs)]
        for t in TASKS:
            cells = []
            for n, scores in rungs.items():
                raw = (scores or {}).get(t)
                ret = ((verdicts[n] or {}).get("per_task_retention", {}) or {}).get(t)
                cells.append("—" if raw is None else f"{raw:.4f} ({_pct(ret)})")
            out.append(f"| {TASK_LABEL[t]} | {base.get(t, float('nan')):.4f} | " + " | ".join(cells) + " |")
        out += ["", "| Rung | avg-5 | avg retention | verdict |", "|---|---|---|---|"]
        for n, v in verdicts.items():
            scores = rungs[n] or {}
            avg = sum(scores[t] for t in TASKS) / len(TASKS) if all(t in scores for t in TASKS) else None
            out.append(f"| {n} | {_num(avg, 4)} | {_pct((v or {}).get('avg_retention'))} | "
                       f"{(v or {}).get('verdict', '—')} |")
    out.append("")

    out += ["## Tier-1 tool calling (BFCLv3)", ""]
    if bfcl is None:
        out.append("Not measured yet (`results/bfcl.json` missing).")
    else:
        b16, art = bfcl.get("bf16_score"), bfcl.get("artifact_score")
        ret = (art / b16) if (b16 and art is not None) else None
        t1v = _load(results / "bfcl_verdict.json") or {}
        out += [f"- BF16: {_num(b16, 2)}",
                f"- ternary: {_num(art, 2)}",
                f"- retention: {_pct(ret)}",
                f"- verdict: {t1v.get('verdict', '—')}"]
    out.append("")

    out += ["## Decode speed (llama-bench, same GPU and settings)", ""]
    if bench is None:
        out.append("Not measured yet (`results/benchmarks.json` missing).")
    else:
        out += ["| Variant | runtime | tg128 tok/s | pp512 tok/s |", "|---|---|---|---|"]
        for name, row in bench.items():
            out.append(f"| {name} | {row.get('runtime', '—')} | "
                       f"{_num(row.get('tg128_t_s'))} | {_num(row.get('pp512_t_s'))} |")
        t, b = (bench.get("ternary") or {}), (bench.get("bf16") or {})
        if t.get("tg128_t_s") and b.get("tg128_t_s"):
            out.append(f"\nTernary/BF16 decode ratio: {t['tg128_t_s'] / b['tg128_t_s']:.2f}× "
                       f"(cross-runtime: {t.get('runtime')} vs {b.get('runtime')})")
    out.append("")

    out += ["## Artifact", ""]
    size = _load(results / "artifact.json")
    if size is None:
        out.append("Artifact size not recorded yet (`results/artifact.json` missing).")
    else:
        out += [f"- file: `{size.get('path')}`",
                f"- size: {size.get('bytes', 0) / 1e9:.2f} GB",
                f"- runtime: {size.get('runtime')}",
                f"- tensor census: {size.get('census')}"]
    out.append("")
    out += ["## Reading caution", "",
            "Retention, not absolute score, is the metric that matters at this size. "
            "A ternary 1.7B model can hold a respectable average while collapsing on "
            "tool calling — that is why the BFCLv3 gate exists and why no speed or "
            "size number is quoted without the quality number beside it.", ""]
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--plan",
                   default="docs/plans/2026-11-10-1024-feat-lfm3-ternary-compression-plan.md")
    p.add_argument("--out", type=Path, default=Path("REPORT.md"))
    args = p.parse_args()
    args.out.write_text(render(args.results, args.plan))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
