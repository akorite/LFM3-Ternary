"""Stage routing and the escalation ladder's decision points."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import remote_stage


def test_every_stage_has_an_artifact_list():
    for stage, artifacts in remote_stage.ARTIFACTS.items():
        assert remote_stage.stage_command(stage, []), stage
        assert isinstance(artifacts, list), stage
    assert "baseline.json" in " ".join(remote_stage.ARTIFACTS["baseline"])


def test_rung1_command_uses_the_c4_corpus_and_writes_one_layout():
    cmd = remote_stage.stage_command("rung1", [])
    joined = " ".join(cmd)
    assert "data/calib/c4_512x2048.pt" in joined
    assert "--out checkpoints/rung1" in joined


def test_extra_args_pass_through_to_the_stage():
    cmd = remote_stage.stage_command("rung2", ["--epochs", "2"])
    assert cmd[-2:] == ["--epochs", "2"]


def test_eval_stage_takes_model_and_output():
    cmd = remote_stage.stage_command("eval", ["checkpoints/rung1/hf", "results/x.json"])
    assert cmd[-2:] == ["checkpoints/rung1/hf", "results/x.json"]


def test_unknown_stage_is_rejected():
    try:
        remote_stage.stage_command("nope", [])
    except SystemExit as exc:
        assert "unknown stage" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_pipeline_dry_run_lists_the_ladder(tmp_path, monkeypatch, capsys):
    import run_pipeline

    monkeypatch.setattr(run_pipeline, "REPO", tmp_path)
    monkeypatch.setattr(sys, "argv",
                        ["run_pipeline.py", "--dry-run", "--skip-baseline"])
    run_pipeline.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["steps"] == ["c4", "rung1", "rung2", "rung3"]


def test_pipeline_skips_existing_outputs(tmp_path, monkeypatch, capsys):
    import run_pipeline

    (tmp_path / "results").mkdir()
    (tmp_path / "data" / "calib").mkdir(parents=True)
    (tmp_path / "data" / "calib" / "c4_512x2048.pt").write_text("x")
    (tmp_path / "results" / "baseline.json").write_text("{}")
    monkeypatch.setattr(run_pipeline, "REPO", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--dry-run"])
    run_pipeline.main()
    assert json.loads(capsys.readouterr().out)["steps"] == ["rung1", "rung2", "rung3"]


def test_pipeline_stops_at_kill_without_running_later_rungs(tmp_path, monkeypatch):
    """A kill verdict must end the ladder and write the report, not try rung 2."""
    import run_pipeline

    (tmp_path / "results").mkdir()
    (tmp_path / "data" / "calib").mkdir(parents=True)
    (tmp_path / "data" / "calib" / "c4_512x2048.pt").write_text("x")
    (tmp_path / "results" / "baseline.json").write_text("{}")
    (tmp_path / "results" / "rung1.json").write_text("{}")
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(run_pipeline, "REPO", tmp_path)
    monkeypatch.setattr(run_pipeline, "run", fake_run)
    monkeypatch.setattr(run_pipeline, "tier0", lambda *a, **k: "kill")
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--skip-baseline"])
    try:
        run_pipeline.main()
    except SystemExit as exc:
        assert "killed at rung 1" in str(exc)
    else:
        raise AssertionError("expected SystemExit")
    assert ["scripts/report.py"] in [c[1:] for c in calls]
    # rung 2 must never be attempted
    assert not any("checkpoints/rung2" in " ".join(c) for c in calls)


def test_seed_banked_copies_only_missing_tensors(tmp_path, monkeypatch):
    src = tmp_path / "banked" / "rung1" / "tensors"
    src.mkdir(parents=True)
    (src / "a.pt").write_bytes(b"one")
    (src / "b.pt").write_bytes(b"two")
    repo = tmp_path / "repo"
    existing = repo / "checkpoints" / "rung1" / "ternary" / "tensors"
    existing.mkdir(parents=True)
    (existing / "b.pt").write_bytes(b"newer")   # host copy must win
    monkeypatch.setenv("BANKED_DIR", str(src))
    import remote_stage
    assert remote_stage.seed_banked(repo, "rung1") == 1
    assert (existing / "a.pt").read_bytes() == b"one"
    assert (existing / "b.pt").read_bytes() == b"newer"


def test_seed_banked_noop_without_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("BANKED_DIR", raising=False)
    import remote_stage
    assert remote_stage.seed_banked(tmp_path / "repo", "rung1") == 0


def test_eval_stage_names_the_rung_not_the_stage():
    cmd = remote_stage.stage_command("eval", ["checkpoints/rung1/hf", "rung1"])
    assert cmd[-1] == "results/rung1.json"
    cmd = remote_stage.stage_command("eval", ["checkpoints/rung2/hf"])
    assert cmd[-1] == "results/rung_eval.json"
