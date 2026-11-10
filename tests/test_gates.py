import json
import subprocess
import sys
from pathlib import Path

import pytest

GATES = Path(__file__).resolve().parents[1] / "scripts" / "gates.py"
TASKS = ["piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande"]


def write_scores(path, values):
    path.write_text(json.dumps(dict(zip(TASKS, values))))
    return str(path)


def run_tier0(tmp_path, baseline, scores, rung=1, remaining=2):
    baseline_file = write_scores(tmp_path / "baseline.json", baseline)
    scores_file = write_scores(tmp_path / "scores.json", scores)
    out_file = tmp_path / "verdict.json"
    result = subprocess.run(
        [
            sys.executable, str(GATES), "tier0",
            "--baseline", baseline_file,
            "--scores", scores_file,
            "--rung", str(rung),
            "--remaining", str(remaining),
            "--out", str(out_file),
        ],
        capture_output=True,
        text=True,
    )
    verdict = json.loads(out_file.read_text()) if out_file.exists() else None
    return result, verdict


def run_tier1(baseline_score, score):
    out = subprocess.run(
        [
            sys.executable, str(GATES), "tier1",
            "--baseline-score", str(baseline_score),
            "--score", str(score),
        ],
        capture_output=True,
        text=True,
    )
    return out


def test_tier0_pass_avg89_min81(tmp_path):
    result, verdict = run_tier0(tmp_path, [1.0] * 5, [0.81, 0.91, 0.91, 0.91, 0.91])
    assert result.returncode == 0, result.stderr
    assert verdict["verdict"] == "pass"
    assert verdict["avg_retention"] == pytest.approx(0.89)
    assert verdict["per_task_retention"]["piqa"] == pytest.approx(0.81)


def test_tier0_escalate_avg87(tmp_path):
    result, verdict = run_tier0(tmp_path, [1.0] * 5, [0.87] * 5, rung=1, remaining=2)
    assert result.returncode == 0, result.stderr
    assert verdict["verdict"] == "escalate"


def test_tier0_kill_avg79(tmp_path):
    result, verdict = run_tier0(tmp_path, [1.0] * 5, [0.79] * 5, rung=1, remaining=2)
    assert result.returncode == 0, result.stderr
    assert verdict["verdict"] == "kill"


def test_tier0_escalate_task_below_80_with_ok_avg(tmp_path):
    result, verdict = run_tier0(tmp_path, [1.0] * 5, [0.79, 0.905, 0.905, 0.905, 0.905])
    assert result.returncode == 0, result.stderr
    assert verdict["verdict"] == "escalate"
    assert verdict["per_task_retention"]["piqa"] == pytest.approx(0.79)


def test_tier0_fail_with_no_rungs_left_is_kill(tmp_path):
    result, verdict = run_tier0(tmp_path, [1.0] * 5, [0.87] * 5, rung=3, remaining=0)
    assert result.returncode == 0, result.stderr
    assert verdict["verdict"] == "kill"


def test_missing_baseline_fails_loudly(tmp_path):
    scores_file = write_scores(tmp_path / "scores.json", [0.9] * 5)
    result = subprocess.run(
        [
            sys.executable, str(GATES), "tier0",
            "--baseline", str(tmp_path / "nonexistent.json"),
            "--scores", scores_file,
            "--rung", "1", "--remaining", "2",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert result.stderr.strip()


def test_corrupt_baseline_fails_loudly(tmp_path):
    baseline_file = tmp_path / "baseline.json"
    baseline_file.write_text("{not json")
    scores_file = write_scores(tmp_path / "scores.json", [0.9] * 5)
    result = subprocess.run(
        [
            sys.executable, str(GATES), "tier0",
            "--baseline", str(baseline_file),
            "--scores", scores_file,
            "--rung", "1", "--remaining", "2",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert result.stderr.strip()


def test_tier1_pass_89():
    result = run_tier1(1.0, 0.89)
    assert result.returncode == 0, result.stderr
    assert '"pass"' in result.stdout


def test_tier1_kill_87():
    result = run_tier1(1.0, 0.87)
    assert result.returncode == 0, result.stderr
    assert '"kill"' in result.stdout
