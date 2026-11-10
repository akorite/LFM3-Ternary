"""Report renderer: every number comes from a result file, none is invented."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from report import TASKS, render


def _write(path, obj):
    path.write_text(json.dumps(obj))


def test_empty_results_report_claims_nothing(tmp_path):
    md = render(tmp_path, "plan.md")
    assert "missing" in md.lower()
    assert "%" not in md  # no retention quoted without a measurement
    assert "×" not in md  # no speed ratio either


def test_measured_results_are_rendered(tmp_path):
    base = {t: 0.6 for t in TASKS}
    rung = {t: 0.55 for t in TASKS}
    rung["piqa"] = 0.5
    _write(tmp_path / "baseline.json", base)
    _write(tmp_path / "rung1.json", rung)
    _write(tmp_path / "rung1_verdict.json",
           {"verdict": "pass", "avg_retention": 0.9167,
            "per_task_retention": {t: 0.9167 for t in TASKS} | {"piqa": 0.8333}})
    md = render(tmp_path, "plan.md")
    assert "| PIQA | 0.6000 | 0.5000 (83.3%)" in md
    assert "| 1 | 0.5400 | 91.7% | pass |" in md


def test_bfcl_and_bench_sections(tmp_path):
    _write(tmp_path / "bfcl.json", {"bf16_score": 71.8, "artifact_score": 64.6})
    _write(tmp_path / "bfcl_verdict.json", {"verdict": "pass"})
    _write(tmp_path / "benchmarks.json", {
        "bf16": {"tg128_t_s": 93.5, "pp512_t_s": 500.0, "runtime": "upstream"},
        "ternary": {"tg128_t_s": 285.1, "pp512_t_s": 1200.0, "runtime": "prism"},
    })
    md = render(tmp_path, "plan.md")
    assert "retention: 90.0%" in md  # 64.6 / 71.8
    assert "| ternary | prism | 285.1 | 1200.0 |" in md
    assert "Ternary/BF16 decode ratio: 3.05×" in md
    assert "cross-runtime: prism vs upstream" in md
