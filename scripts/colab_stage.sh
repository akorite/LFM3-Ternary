#!/usr/bin/env bash
# Run one pipeline stage on a Colab GPU VM, then pull its artifacts back.
#
#   scripts/colab_stage.sh [--session NAME] [--gpu A100] [--new] <stage> [stage args...]
#
# Env: TIMEOUT (seconds, default 21600), SYNC_DOWN (1 = download artifacts,
# default 1), REPO_URL, BRANCH. The session must hold the GPU type you want;
# `--new` (re)creates it, which is how a different accelerator is chosen.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
SESSION="${SESSION:-lfm3-ternary}"
GPU="${GPU:-A100}"
TIMEOUT="${TIMEOUT:-21600}"
SYNC_DOWN="${SYNC_DOWN:-1}"
CREATE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session) SESSION="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --new) CREATE=1; shift ;;
    --) shift; break ;;
    *) break ;;
  esac
done

STAGE="${1:?usage: colab_stage.sh [--session NAME] [--gpu A100] [--new] <stage> [stage args...]}"
shift || true

if [[ "$CREATE" == "1" ]]; then
  colab stop -s "$SESSION" >/dev/null 2>&1 || true
  colab new -s "$SESSION" --gpu "$GPU" --high-mem
fi

# The stage script lives in the repo, so it is uploaded from this checkout and
# syncs the remote clone itself.
CTRL="$(mktemp -d)"
trap 'rm -rf "$CTRL"' EXIT
cp "$HERE/remote_stage.py" "$CTRL/remote_stage.py"

ARGS_JSON="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@")"

colab exec -s "$SESSION" -f "$CTRL/remote_stage.py" --timeout "$TIMEOUT" \
  --env "STAGE=$STAGE" \
  --env "STAGE_ARGS=$ARGS_JSON" \
  --env "REPO_URL=${REPO_URL:-https://github.com/akorite/LFM3-Ternary}" \
  --env "BRANCH=${BRANCH:-feat/lfm3-ternary}" \
  2>&1 | tee "$CTRL/stage.log"

if [[ "$SYNC_DOWN" != "1" ]]; then
  echo "artifacts left on the VM (SYNC_DOWN=0); stage log above"
  exit 0
fi

# Artifact list is echoed by the stage runner; download whatever exists.
python3 - "$CTRL/stage.log" > "$CTRL/artifacts.txt" <<'PY'
import json, sys
for line in open(sys.argv[1], errors="replace"):
    if line.startswith("STAGE_RESULT "):
        for a in json.loads(line[len("STAGE_RESULT "):]).get("artifacts", []):
            print(a)
PY
mkdir -p "$ROOT/results" "$ROOT/gguf"
while read -r artifact; do
  [[ -z "$artifact" ]] && continue
  if [[ "$artifact" == "results" ]]; then
    for f in baseline.json rung1.json rung1_verdict.json rung2.json rung2_verdict.json \
             rung3.json rung3_verdict.json rung_eval.json bfcl.json bfcl_verdict.json \
             benchmarks.json artifact.json; do
      colab download -s "$SESSION" "/content/LFM3-Ternary/results/$f" "$ROOT/results/$f" \
        >/dev/null 2>&1 || true
    done
  else
    mkdir -p "$(dirname "$ROOT/$artifact")"
    colab download -s "$SESSION" "/content/LFM3-Ternary/$artifact" "$ROOT/$artifact" \
      || echo "could not download $artifact" >&2
  fi
done < "$CTRL/artifacts.txt"
echo "downloaded into $ROOT"
