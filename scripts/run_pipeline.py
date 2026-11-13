#!/usr/bin/env python3
"""Drive the plan's escalation ladder on one GPU host, unattended.

Runs the baseline, then each rung in order, evaluating and gating after every
rung, and stops where the plan says to stop: a rung that passes its gate goes
to tier-1, a rung that fails escalates, and a kill verdict ends the run with the
best artifact named. Export, benchmarks and the report follow a surviving rung.

Every step is skipped when its output already exists, so a session that dies
mid-run resumes by re-invoking the same command.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=REPO).returncode


def exists(*parts: str) -> bool:
    return (REPO.joinpath(*parts)).exists()


def verdict(path: str) -> str:
    return json.loads((REPO / path).read_text())["verdict"]


def tier0(baseline: str, scores: str, rung: int, remaining: int, out: str) -> str:
    rc = run([sys.executable, "scripts/gates.py", "tier0", "--baseline", baseline,
              "--scores", scores, "--rung", str(rung), "--remaining", str(remaining),
              "--out", out])
    if rc != 0:
        raise SystemExit(f"gate arithmetic failed for rung {rung}")
    return verdict(out)


def tier1(bf16: str, artifact: str, out: str) -> str:
    rc = run([sys.executable, "scripts/gates.py", "tier1",
              "--baseline-score", bf16, "--score", artifact, "--out", out])
    if rc != 0:
        raise SystemExit("tier-1 gate arithmetic failed")
    return verdict(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="LiquidAI/LFM3-1.7B")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--start-rung", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--stop-after", default="export",
                   choices=["gate", "tier1", "export", "bench", "report"],
                   help="how far past a passing gate to continue")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = p.parse_args()

    steps: list[str] = []
    if not args.skip_baseline and not exists("results", "baseline.json"):
        steps.append("baseline")
    if not exists("data", "calib", "c4_512x2048.pt"):
        steps.append("c4")
    for rung in range(args.start_rung, 4):
        steps.append(f"rung{rung}")

    if args.dry_run:
        print(json.dumps({"steps": steps, "stop_after": args.stop_after,
                          "start_rung": args.start_rung}, indent=1))
        return

    if "baseline" in steps:
        if run(["bash", "scripts/eval_lm_eval.sh", args.model,
                "results/baseline.json"]) != 0:
            raise SystemExit("baseline eval failed")
    if "c4" in steps:
        if run([sys.executable, "scripts/build_calib_c4.py", "--model", args.model,
                "--out-dir", "data/calib"]) != 0:
            raise SystemExit("calibration build failed")

    passed_rung = None
    for rung in range(args.start_rung, 4):
        scores = f"results/rung{rung}.json"
        if rung == 2 and not exists("data", "calib", "ayot", "ayot-000.jsonl"):
            if run([sys.executable, "scripts/build_calib_ayot.py", "--model", args.model,
                    "--out-dir", "data/calib/ayot", "--tokens", "4000000"]) != 0:
                raise SystemExit("AYOT calibration build failed")
        if not exists(scores):
            calib = ("data/calib/c4_512x2048.pt" if rung == 1 else "data/calib/ayot")
            if rung == 3:
                # QAT starts from the best PTQ checkpoint and its own corpus
                if not exists("data", "calib", "qat", "corpus.pt"):
                    run([sys.executable, "scripts/build_qat_data.py",
                         "--out-dir", "data/calib/qat", "--tokens", "102000000"])
                rc = run([sys.executable, "scripts/train_qat.py",
                          "--model", "checkpoints/rung2/hf",
                          "--data", "data/calib/qat/corpus.pt", "--out", "checkpoints/rung3"])
            else:
                rc = run([sys.executable, "scripts/run_catq.py", "--model", args.model,
                          "--calib", calib, "--out", f"checkpoints/rung{rung}"])
            if rc != 0:
                raise SystemExit(f"rung {rung} quantization failed")
            if rung < 3 and run(["bash", "scripts/eval_lm_eval.sh",
                                 f"checkpoints/rung{rung}/hf", scores]) != 0:
                raise SystemExit(f"rung {rung} evaluation failed")
            if rung == 3 and run(["bash", "scripts/eval_lm_eval.sh",
                                  "checkpoints/rung3/hf", scores]) != 0:
                raise SystemExit("rung 3 evaluation failed")
        remaining = 3 - rung
        decision = tier0("results/baseline.json", scores, rung, remaining,
                         f"results/rung{rung}_verdict.json")
        print(f"rung {rung}: {decision}")
        if decision == "pass":
            passed_rung = rung
            break
        if decision == "kill":
            run([sys.executable, "scripts/report.py"])
            raise SystemExit(f"killed at rung {rung}; REPORT.md records the evidence")
        if remaining == 0:
            raise SystemExit("no rung remains")

    if passed_rung is None:
        raise SystemExit("no rung passed")
    if args.stop_after == "gate":
        return

    checkpoint = f"checkpoints/rung{passed_rung}/hf"
    if not exists("results", "bfcl.json"):
        run(["bash", "scripts/eval_bfcl.sh", args.model, "baseline"])
        if run(["bash", "scripts/eval_bfcl.sh", checkpoint, "artifact"]) != 0:
            raise SystemExit("tier-1 evaluation failed")
    if args.stop_after == "tier1":
        return

    bfcl = json.loads((REPO / "results/bfcl.json").read_text())
    b16 = (bfcl.get("bf16_score") or {}).get("score")
    art = (bfcl.get("artifact_score") or {}).get("score")
    if b16 is None or art is None:
        raise SystemExit("tier-1 scores incomplete")
    if tier1(str(b16), str(art), "results/bfcl_verdict.json") != "pass":
        run([sys.executable, "scripts/report.py"])
        raise SystemExit("killed at the tier-1 gate")

    if not exists("gguf", "LFM3-1.7B-ternary-pq2_0.gguf"):
        manifest = f"checkpoints/rung{passed_rung}/ternary"
        if run([sys.executable, "scripts/export_gguf.py", "--model", checkpoint,
                "--manifest", manifest, "--out", "gguf/LFM3-1.7B-ternary-pq2_0.gguf"]) != 0:
            raise SystemExit("ternary export failed")
    if args.stop_after == "export":
        run([sys.executable, "scripts/report.py"])
        return

    if run(["bash", "scripts/build_baseline_ggufs.sh"]) != 0:
        raise SystemExit("comparator build failed")
    if run(["bash", "scripts/bench.sh"]) != 0:
        raise SystemExit("benchmark failed")
    run([sys.executable, "scripts/report.py"])


if __name__ == "__main__":
    main()
