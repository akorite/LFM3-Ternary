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
[[ "${1:-}" == "--" ]] && shift   # runbook form: <stage> -- args...

if [[ "$CREATE" == "1" ]]; then
  colab stop -s "$SESSION" >/dev/null 2>&1 || true
  colab new -s "$SESSION" --gpu "$GPU" --high-mem
fi

# The stage script lives in the repo, so it is uploaded from this checkout and
# syncs the remote clone itself.
CTRL="$(mktemp -d)"
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/colab-$STAGE.log"
trap 'rm -rf "$CTRL"' EXIT
cp "$HERE/remote_stage.py" "$CTRL/remote_stage.py"

ARGS_JSON="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@")"

# Long stages must outlive the CLI connection: the runner is placed on disk and
# starts the stage in its own session, so the exec returns in seconds and the
# poll below watches the host-side log.
colab upload -s "$SESSION" "$CTRL/remote_stage.py" /content/runner.py >/dev/null
colab exec -s "$SESSION" -f "$CTRL/remote_stage.py" --timeout 120 \
  --env "STAGE=$STAGE" \
  --env "STAGE_ARGS=$ARGS_JSON" \
  --env "DETACHED=1" \
  --env "RUNNER=/content/runner.py" \
  --env "REPO_URL=${REPO_URL:-https://github.com/akorite/LFM3-Ternary}" \
  --env "BRANCH=${BRANCH:-feat/lfm3-ternary}" \
  2>&1 | tee "$LOG" | tail -5

python3 - "$LOG" <<'PY'
import json, sys
launched = False
for line in open(sys.argv[1], errors="replace"):
    if line.startswith("STAGE_LAUNCHED "):
        print(json.loads(line[len("STAGE_LAUNCHED "):])["stage"], "launched")
        launched = True
        break
sys.exit(0 if launched else 1)
PY
[[ $? -eq 0 ]] || { echo "stage $STAGE did not start; see $LOG" >&2; exit 1; }

ELAPSED=0
while :; do
  OUT="$(colab exec -s "$SESSION" -f "$CTRL/remote_stage.py" --timeout 180 \
        --env "STAGE=status" --env "STAGE_ARGS=[\"$STAGE\"]" --env "DETACHED=0" \
        --env "REPO_DIR=/content/LFM3-Ternary" --env "SKIP_SYNC=1" --env "SKIP_INSTALL=1" \
        2>&1)"
  STATE="$(printf '%s\n' "$OUT" | grep '^{' | tail -1)"
  printf '%s\n' "$OUT" >> "$LOG"
  printf '%s\n' "$STATE" | python3 -c '
import json, sys
raw = sys.stdin.read().strip()
if not raw:
    print("no status"); raise SystemExit(0)
d = json.loads(raw)
print(("running" if d["running"] else "stopped"), "| artifacts:", d["artifacts"])
' 
  printf '%s\n' "$STATE" | grep -q '"running": false' && break
  sleep 60
  ELAPSED=$((ELAPSED + 60))
  if [[ "$ELAPSED" -ge "$TIMEOUT" ]]; then
    echo "stage $STAGE still running after ${TIMEOUT}s; log is on the host at logs/$STAGE.log" >&2
    exit 1
  fi
done

# the detached run writes its own log on the host; fetch it for the local record
colab download -s "$SESSION" "/content/LFM3-Ternary/logs/$STAGE.log" "$LOG" >/dev/null 2>&1 \
  || echo "could not fetch the remote stage log" >&2

if [[ "$SYNC_DOWN" != "1" ]]; then
  echo "artifacts left on the VM (SYNC_DOWN=0); stage log above"
  exit 0
fi

# Artifact list is echoed by the stage runner; download whatever exists.
python3 - "$LOG" > "$CTRL/artifacts.txt" <<'PY'
import json, sys
for line in open(sys.argv[1], errors="replace"):
    if line.startswith("STAGE_RESULT "):
        for a in json.loads(line[len("STAGE_RESULT "):]).get("artifacts", []):
            print(a)
PY
mkdir -p "$ROOT/results" "$ROOT/gguf"
FAILED=0
while read -r artifact; do
  [[ -z "$artifact" ]] && continue
  if [[ "$artifact" == "results" ]]; then
    for f in baseline.json rung1.json rung1_verdict.json rung2.json rung2_verdict.json \
             rung3.json rung3_verdict.json rung_eval.json bfcl.json bfcl_verdict.json \
             benchmarks.json artifact.json; do
      colab download -s "$SESSION" "/content/LFM3-Ternary/results/$f" "$ROOT/results/$f" \
        >/dev/null 2>&1 || true      # optional per-file: not every stage writes all of them
    done
    continue
  fi
  mkdir -p "$(dirname "$ROOT/$artifact")"
  if ! colab download -s "$SESSION" "/content/LFM3-Ternary/$artifact" "$ROOT/$artifact"; then
    echo "could not download $artifact" >&2
    FAILED=$((FAILED + 1))
  fi
done < "$CTRL/artifacts.txt"
if [[ "$FAILED" != "0" ]]; then
  echo "$FAILED artifact(s) failed to download; the stage log is $LOG" >&2
  exit 1
fi
echo "downloaded into $ROOT"
