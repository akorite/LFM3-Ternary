#!/usr/bin/env bash
# Run one pipeline stage on a Kaggle GPU kernel, then pull its outputs back.
#
#   scripts/kaggle_stage.sh <stage> [stage args...]
#
# Env: KAGGLE_USER (default: kaggle config username), TIMEOUT (minutes, default
# 720), ACCELERATOR (NvidiaTeslaT4 | NvidiaL4 | NvidiaA100, default NvidiaA100),
# REPO_URL, BRANCH. Kaggle kernels run headless and return their working
# directory, which is where the stage's artifacts land.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
STAGE="${1:?usage: kaggle_stage.sh <stage> [stage args...]}"
shift || true
[[ "${1:-}" == "--" ]] && shift   # runbook form: <stage> -- args...

USER_NAME="${KAGGLE_USER:-$(kaggle config view 2>/dev/null | awk '/- username:/ {print $3}')}"
[[ -n "$USER_NAME" ]] || { echo "set KAGGLE_USER or run 'kaggle config set -n username'" >&2; exit 1; }
SLUG="lfm3-${STAGE//_/-}"
ACCELERATOR="${ACCELERATOR:-NvidiaA100}"
TIMEOUT="${TIMEOUT:-720}"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/$SLUG"

python3 - "$WORK/$SLUG/kernel-metadata.json" "$USER_NAME" "$SLUG" "$ACCELERATOR" "$TIMEOUT" <<'PY'
import json, sys
path, user, slug, accelerator, timeout = sys.argv[1:6]
import os
datasets = [d for d in os.environ.get("KAGGLE_DATASETS", "").split(",") if d]
json.dump({
    "id": f"{user}/{slug}",
    "title": slug,
    "code_file": "stage.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "accelerator": accelerator,
    "timeout": int(timeout),
    "dataset_sources": datasets,
    "competition_sources": [],
    "kernel_sources": [],
}, open(path, "w"), indent=1)
PY

# Kaggle runs the script with cwd /kaggle/working and no arguments, so stage and
# its arguments arrive through the environment.
python3 - "$HERE/remote_stage.py" "$WORK/$SLUG/stage.py" "$STAGE" "$@" <<'PY'
import json, sys
src, dst, stage, *stage_args = sys.argv[1:]
body = open(src).read()
body = body.replace('\nif __name__ == "__main__":\n    main()\n', '\n')
body += f'''

# Kaggle entry point: cwd is /kaggle/working, which is also the output root.
import os as _os
_os.chdir("/kaggle/working")
_os.environ["STAGE"] = {stage!r}
_os.environ["STAGE_ARGS"] = {json.dumps(stage_args)!r}
_os.environ["REPO_DIR"] = "/kaggle/working/LFM3-Ternary"
_banked = {json.dumps([d for d in __import__("os").environ.get("KAGGLE_DATASETS", "").split(",") if d])!r}
if _banked:
    _os.environ.setdefault("BANKED_DIR", f"/kaggle/input/{{_banked[0].split('/')[-1]}}")
_os.environ.setdefault("BRANCH", "feat/lfm3-ternary")
if __name__ == "__main__":
    main()
'''
open(dst, "w").write(body)
PY

(cd "$WORK/$SLUG" && kaggle kernels push)

echo "waiting for $USER_NAME/$SLUG to finish (timeout ${TIMEOUT}m)"
for _ in $(seq 1 $((TIMEOUT * 2))); do
  STATE="$(kaggle kernels status "$USER_NAME/$SLUG" 2>&1 | tr -d '\r')"
  case "$STATE" in
    *complete*|*COMPLETE*) break ;;
    *error*|*ERROR*|*cancel*|*CANCEL*)
      kaggle kernels output "$USER_NAME/$SLUG" -p "$WORK/out" || true
      echo "kernel failed: $STATE" >&2
      cp -R "$WORK/out" "$ROOT/logs/kaggle-$STAGE-$SLUG" 2>/dev/null || true
      tail -n 80 "$WORK/out"/* 2>/dev/null || true
      exit 1 ;;
  esac
  sleep 30
done

mkdir -p "$WORK/out"
kaggle kernels output "$USER_NAME/$SLUG" -p "$WORK/out"
# results/ and REPORT.md land in the repo subtree; copy what the stage produced.
# out_tensors/ is returned even when the kernel was killed mid-rung
if [[ -d "$WORK/out/out_tensors" ]]; then
  mkdir -p "$ROOT/checkpoints/$STAGE/ternary/tensors"
  cp -n "$WORK/out/out_tensors"/*.pt "$ROOT/checkpoints/$STAGE/ternary/tensors/" 2>/dev/null || true
fi
for rel in results REPORT.md checkpoints; do
  if [[ -e "$WORK/out/LFM3-Ternary/$rel" ]]; then
    mkdir -p "$ROOT/$(dirname "$rel")"
    cp -R "$WORK/out/LFM3-Ternary/$rel" "$ROOT/$rel"
  fi
done
echo "downloaded into $ROOT"
