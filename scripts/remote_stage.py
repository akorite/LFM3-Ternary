#!/usr/bin/env python3
"""Run one pipeline stage on a remote GPU host (Colab or Kaggle).

Executed on the remote VM by `scripts/colab_stage.sh` or
`scripts/kaggle_stage.sh`; never needs the caller's shell state. It syncs this
repository, installs the pinned environment once, runs the stage's command, and
prints a JSON record of the artifacts the caller should download.

Stages are named so a driver can drive the whole plan without a notebook:
`c4`, `baseline`, `ayot`, `rung1`, `rung2`, `qat`, `eval`, `bfcl`, `export`,
`census`, `report`, `pipeline`.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

MODEL = "LiquidAI/LFM3-1.7B"
C4_CORPUS = "data/calib/c4_512x2048.pt"
AYOT_CORPUS = "data/calib/ayot"
QAT_CORPUS = "data/calib/qat/corpus.pt"
TERNARY_GGUF = "gguf/LFM3-1.7B-ternary-pq2_0.gguf"

# artifact paths each stage produces, relative to the repo root
ARTIFACTS = {
    "c4": [C4_CORPUS, "data/calib/c4_512x2048.manifest.json"],
    "baseline": ["results/baseline.json"],
    "ayot": ["data/calib/ayot/manifest.json"],
    "rung1": ["checkpoints/rung1/rung.json", "checkpoints/rung1/ternary/ternary_codes.pt"],
    "rung2": ["checkpoints/rung2/rung.json", "checkpoints/rung2/ternary/ternary_codes.pt"],
    "qat": ["checkpoints/rung3/qat.json", "checkpoints/rung3/ternary/ternary_codes.pt"],
    "eval": ["results/rung_eval.json"],
    "bfcl": ["results/bfcl.json"],
    "export": [TERNARY_GGUF, "results/artifact.json"],
    "census": [],
    "report": ["REPORT.md"],
    "pipeline": ["results", "REPORT.md"],
    "status": [],
}


def missing_input(stage: str) -> list[str] | None:
    """The build command a stage needs first when its input is absent, if any."""
    py = sys.executable
    if stage == "rung1" and not Path(C4_CORPUS).exists():
        return [py, "scripts/build_calib_c4.py", "--model", MODEL, "--out-dir", "data/calib"]
    if stage == "rung2" and not Path(AYOT_CORPUS, "ayot-000.jsonl").exists():
        return [py, "scripts/build_calib_ayot.py", "--model", MODEL,
                "--out-dir", AYOT_CORPUS, "--tokens", "4000000"]
    return None


def log_path(repo: Path, stage: str) -> Path:
    return repo / "logs" / f"{stage}.log"


def pid_path(repo: Path, stage: str) -> Path:
    return repo / "logs" / f"{stage}.pid"


def launch(repo: Path, stage: str, stage_args: list[str]) -> None:
    """Start the stage detached so a dropped CLI connection does not kill it."""
    (repo / "logs").mkdir(exist_ok=True)
    cmd = stage_command(stage, stage_args)
    env = {**os.environ, "DETACHED": "0"}
    with open(log_path(repo, stage), "w") as log:
        proc = subprocess.Popen(cmd, cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)
    pid_path(repo, stage).write_text(str(proc.pid))
    print("STAGE_LAUNCHED " + json.dumps({"stage": stage, "pid": proc.pid,
                                          "log": str(log_path(repo, stage).relative_to(repo))}),
          flush=True)


def watch_state(repo: Path, stage: str) -> dict:
    """Whether a detached stage is still running, and what it has produced."""
    log = log_path(repo, stage)
    pid_file = pid_path(repo, stage)
    pid = int(pid_file.read_text()) if pid_file.exists() else None
    alive = False
    if pid:
        try:
            os.kill(pid, 0)
            alive = True
        except OSError:
            alive = False
    tail = ""
    if log.exists():
        with open(log, errors="replace") as fh:
            tail = "".join(fh.readlines()[-15:])
    found = [a for a in ARTIFACTS.get(stage, []) if (repo / a).exists()]
    return {"stage": stage, "running": alive, "log_tail": tail, "artifacts": found}


def _arg(args: list[str], index: int, default: str) -> str:
    return args[index] if len(args) > index else default


def stage_command(stage: str, args: list[str]) -> list[str]:
    """The argv that carries out `stage` inside the repository root."""
    py = sys.executable
    extra = args
    table = {
        "c4": lambda: [py, "scripts/build_calib_c4.py", "--model", MODEL,
                       "--out-dir", "data/calib", *extra],
        "baseline": lambda: ["bash", "scripts/eval_lm_eval.sh", MODEL,
                             "results/baseline.json", *extra],
        "ayot": lambda: [py, "scripts/build_calib_ayot.py", "--model", MODEL,
                         "--out-dir", AYOT_CORPUS, "--tokens", "4000000", *extra],
        "rung1": lambda: [py, "scripts/run_catq.py", "--model", MODEL,
                          "--calib", C4_CORPUS, "--out", "checkpoints/rung1", *extra],
        "rung2": lambda: [py, "scripts/run_catq.py", "--model", MODEL,
                          "--calib", AYOT_CORPUS, "--out", "checkpoints/rung2", *extra],
        "qat": lambda: [py, "scripts/train_qat.py",
                        "--model", extra[0] if extra else "checkpoints/rung2/hf",
                        "--data", QAT_CORPUS, "--out", "checkpoints/rung3"],
        "eval": lambda: ["bash", "scripts/eval_lm_eval.sh", _arg(extra, 0, MODEL),
                         _arg(extra, 1, "results/rung_eval.json")],
        "bfcl": lambda: ["bash", "scripts/eval_bfcl.sh", _arg(extra, 0, MODEL),
                         _arg(extra, 1, "artifact")],
        "export": lambda: [py, "scripts/export_gguf.py", "--model", _arg(extra, 0, MODEL),
                           "--manifest", _arg(extra, 1, "absmean"), "--out", TERNARY_GGUF],
        "census": lambda: [py, "scripts/export_gguf.py", "--model", _arg(extra, 0, MODEL),
                           "--census"],
        "report": lambda: [py, "scripts/report.py", "--out", "REPORT.md"],
        "pipeline": lambda: [py, "scripts/run_pipeline.py", *extra],
        "status": lambda: [py, "-c", "pass"],
    }
    if stage not in table:
        raise SystemExit(f"unknown stage {stage!r}; known: {sorted(table)}")
    return table[stage]()


def sync_repo(repo: Path, url: str, branch: str) -> str:
    if not (repo / ".git").is_dir():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--branch", branch, "--depth", "1", url, str(repo)],
                       check=True)
    else:
        subprocess.run(["git", "-C", str(repo), "fetch", "--depth", "1", "origin", branch],
                       check=True)
        subprocess.run(["git", "-C", str(repo), "reset", "--hard", "FETCH_HEAD"], check=True)
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def install(repo: Path) -> None:
    # a marker keeps repeated stages from re-resolving the environment
    marker = repo / ".deps-installed"
    if marker.exists():
        return
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                    str(repo / "requirements.txt")], check=True)
    marker.write_text("ok\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("stage", nargs="?", default=os.environ.get("STAGE"))
    p.add_argument("args", nargs="*", help="passed through to the stage command")
    p.add_argument("--repo", default=os.environ.get("REPO_DIR", "/content/LFM3-Ternary"))
    p.add_argument("--url", default=os.environ.get("REPO_URL",
                                                   "https://github.com/akorite/LFM3-Ternary"))
    p.add_argument("--branch", default=os.environ.get("BRANCH", "feat/lfm3-ternary"))
    p.add_argument("--skip-sync", action="store_true")
    p.add_argument("--skip-install", action="store_true")
    p.add_argument("--print-command", action="store_true",
                   help="print the stage argv and exit (no sync, no run)")
    # Remote kernels execute this file through IPython, so argv belongs to the
    # kernel launcher; Colab and Kaggle hand the stage over in the environment.
    if (os.environ.get("STAGE") and os.environ.get("DETACHED") == "1"
            and os.environ["STAGE"] != "status"):
        repo = Path(os.environ.get("REPO_DIR", "/content/LFM3-Ternary"))
        stage = os.environ["STAGE"]
        runner = os.environ.get("RUNNER", str(repo / "runner.py"))
        (repo / "logs").mkdir(parents=True, exist_ok=True)
        log = open(log_path(repo, stage), "w")
        env = {**os.environ, "DETACHED": "0"}
        proc = subprocess.Popen([sys.executable, runner], stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)
        (repo / "logs").mkdir(exist_ok=True)
        pid_path(repo, stage).write_text(str(proc.pid))
        print("STAGE_LAUNCHED " + json.dumps({"stage": stage, "pid": proc.pid}), flush=True)
        raise SystemExit(0)

    if os.environ.get("STAGE"):
        stage = os.environ["STAGE"]
        stage_args = json.loads(os.environ.get("STAGE_ARGS") or "[]")
        repo = Path(os.environ.get("REPO_DIR", "/content/LFM3-Ternary"))
        head = sync_repo(repo, os.environ.get("REPO_URL",
                                              "https://github.com/akorite/LFM3-Ternary"),
                         os.environ.get("BRANCH", "feat/lfm3-ternary")) \
            if os.environ.get("SKIP_SYNC") != "1" else None
        if os.environ.get("SKIP_INSTALL") != "1":
            install(repo)
        if stage == "status":
            print(json.dumps(watch_state(repo, stage_args[0] if stage_args else "")), flush=True)
            raise SystemExit(0)

        cmd = stage_command(stage, stage_args)
        print(json.dumps({"stage": stage, "head": head, "command": cmd}), flush=True)
        if os.environ.get("DETACHED") == "1":
            launch(repo, stage, stage_args)
            raise SystemExit(0)
        pre = missing_input(stage)
        if pre:
            print("building a missing stage input: " + " ".join(pre), flush=True)
            subprocess.run(pre, cwd=repo, check=True)
        log_dir = repo / "logs"
        log_dir.mkdir(exist_ok=True)
        stage_log = log_dir / f"{stage}.log"
        rc = 0
        lines: list[str] = []
        with open(stage_log, "w") as log:
            proc = subprocess.Popen(cmd, cwd=repo, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                lines.append(line)
            rc = proc.wait()
        if rc != 0:
            print("--- last lines of " + str(stage_log) + " ---", flush=True)
            print("".join(lines[-40:]), flush=True)
        found = [a for a in ARTIFACTS[stage] if (repo / a).exists()]
        print("STAGE_RESULT " + json.dumps({"stage": stage, "returncode": rc,
                                            "log": str(stage_log.relative_to(repo)),
                                            "artifacts": found + [str(stage_log.relative_to(repo))]}),
              flush=True)
        raise SystemExit(rc)

    argv = sys.argv[1:]
    passthrough: list[str] = []
    if "--" in argv:          # everything after `--` belongs to the stage
        cut = argv.index("--")
        argv, passthrough = argv[:cut], argv[cut + 1:]
    args = p.parse_args(argv)
    if passthrough:
        args.args = passthrough + list(args.args or [])
    if not args.stage:
        p.error("a stage is required (argument or STAGE env)")
    if not args.args and os.environ.get("STAGE_ARGS"):
        args.args = json.loads(os.environ["STAGE_ARGS"])

    cmd = stage_command(args.stage, args.args)
    if args.print_command:
        print(" ".join(shlex.quote(c) for c in cmd))
        return

    repo = Path(args.repo)
    head = None if args.skip_sync else sync_repo(repo, args.url, args.branch)
    if not args.skip_install:
        install(repo)
    print(json.dumps({"stage": args.stage, "head": head, "command": cmd}), flush=True)

    done = subprocess.run(cmd, cwd=repo)
    artifacts = [a for a in ARTIFACTS[args.stage] if (repo / a).exists()]
    print("STAGE_RESULT " + json.dumps({"stage": args.stage, "returncode": done.returncode,
                                        "artifacts": artifacts}), flush=True)
    raise SystemExit(done.returncode)


if __name__ == "__main__":
    main()
