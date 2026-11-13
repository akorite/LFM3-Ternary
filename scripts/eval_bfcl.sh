#!/usr/bin/env bash
# Tier-1 BFCLv3 evaluation against an HF model dir.
#
#   eval_bfcl.sh <model-dir> <baseline|artifact> [-- ...extra run_bfcl.py args]
#
# Harness fallback chain (first usable wins):
#   1. Official gorilla-llm bfcl_eval harness — used when $BFCL_PATH is set to
#      a checked-out gorilla repo AND bfcl_eval imports. It only supports model
#      configs it already registers; Lfm3ForCausalLM (custom modeling,
#      trust_remote_code) is not registered upstream, so this branch prints the
#      official commands and exits for you to wire the model config in, unless
#      BFCL_MODEL_CONFIG makes the model name resolvable. When it does run,
#      merge its score with: run_bfcl.py --record-score S --role R.
#   2. EvalScope + vLLM (the Bonsai card's path) — used when the evalscope and
#      vllm packages are installed AND VLLM_MODEL is servable by vLLM. LFM3's
#      hybrid conv/attention arch needs vLLM's Lfm3 support; if vllm can't load
#      the dir this branch is skipped. Score lands in evalscope outputs; merge
#      it the same --record-score way.
#   3. Direct HF run via scripts/run_bfcl.py (default on CPU/GPU alike): loads
#      the dir with transformers + trust_remote_code, runs the AST-scored
#      BFCLv3 categories, and merges into $OUT (results/bfcl.json).
#
# Env:
#   BFCL_DATA   dir of BFCL_v3_*.json + possible_answer/ + multi_turn_func_doc/
#               (default: data/bfcl; auto-downloaded from the HF dataset if empty)
#   BFCL_PATH   optional path to the gorilla repo checkout (branch 1)
#   BFCL_MODEL_CONFIG  model name registered inside bfcl_eval (enables branch 1)
#   EVALSCOPE_DATASET  evalscope dataset id (default: bfcl_v3)
#   CATEGORIES  space-separated categories for branch 3 (default: run_bfcl.py's)
#   OUT         results JSON (default: results/bfcl.json)
set -euo pipefail

MODEL="${1:?usage: eval_bfcl.sh <model-dir> <baseline|artifact>}"
ROLE="${2:?role must be baseline or artifact}"
shift 2

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
OUT="${OUT:-$ROOT/results/bfcl.json}"
BFCL_DATA="${BFCL_DATA:-$ROOT/data/bfcl}"
DATASET_BASE="$(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import run_bfcl; print(run_bfcl.DATASET_BASE)' "$HERE")"

fetch_bfcl_data() {
  local files
  read -r -a files <<<"$(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import run_bfcl; print(" ".join(run_bfcl.ALL_CATEGORIES))' "$HERE")"
  mkdir -p "$BFCL_DATA/possible_answer" "$BFCL_DATA/multi_turn_func_doc"
  for f in "${files[@]}"; do
    [ -s "$BFCL_DATA/BFCL_v3_$f.json" ] || \
      curl -fsSL "$DATASET_BASE/BFCL_v3_$f.json" -o "$BFCL_DATA/BFCL_v3_$f.json"
    case "$f" in irrelevance|live_irrelevance|live_relevance) continue ;; esac
    [ -s "$BFCL_DATA/possible_answer/BFCL_v3_$f.json" ] || \
      curl -fsSL "$DATASET_BASE/possible_answer/BFCL_v3_$f.json" \
        -o "$BFCL_DATA/possible_answer/BFCL_v3_$f.json"
  done
  for d in gorilla_file_system math_api message_api posting_api ticket_api \
           trading_bot travel_booking vehicle_control; do
    [ -s "$BFCL_DATA/multi_turn_func_doc/$d.json" ] || \
      curl -fsSL "$DATASET_BASE/multi_turn_func_doc/$d.json" \
        -o "$BFCL_DATA/multi_turn_func_doc/$d.json"
  done
}

record_score() {
  python3 "$HERE/run_bfcl.py" --record-score "$1" --role "$ROLE" \
    --n-samples "${2:-0}" --out "$OUT"
}

# ---- branch 1: official bfcl_eval harness ---------------------------------
if [ -n "${BFCL_PATH:-}" ] && \
   PYTHONPATH="$BFCL_PATH/berkeley-function-call-leaderboard${PYTHONPATH:+:$PYTHONPATH}" \
     python3 -c "import bfcl_eval" 2>/dev/null; then
  if [ -n "${BFCL_MODEL_CONFIG:-}" ]; then
    echo "== bfcl_eval harness (model config: $BFCL_MODEL_CONFIG)"
    PYTHONPATH="$BFCL_PATH/berkeley-function-call-leaderboard" \
      bfcl generate --model "$BFCL_MODEL_CONFIG" --test-category all \
      --run-ids || echo "bfcl generate failed — check harness logs"
    echo "Run 'bfcl evaluate' and merge with:"
    echo "  $HERE/run_bfcl.py --record-score <score> --role $ROLE --out $OUT"
    exit 0
  fi
  echo "bfcl_eval present but no registered model config for this model."
  echo "Register one in bfcl_eval/constants/model_config.py, then re-run with"
  echo "BFCL_MODEL_CONFIG=<name>. Falling through to next option."
fi

# ---- branch 2: EvalScope + vLLM (Bonsai card path) -------------------------
if python3 -c "import evalscope, vllm" 2>/dev/null && \
   python3 - "$MODEL" <<'PY' 2>/dev/null
import sys
from vllm import LLM
# probe only config resolution, not a full load
from vllm.transformers_utils.config import get_config
get_config(sys.argv[1], trust_remote_code=True)
PY
then
  echo "== EvalScope + vLLM path"
  python3 - "$MODEL" <<'PY'
import sys
from evalscope import TaskConfig, run_task
run_task(task_cfg=TaskConfig(
    model=sys.argv[1],
    eval_type="service",
    datasets=["bfcl_v3"],
    generation_config={"max_new_tokens": 512, "temperature": 0.0},
))
PY
  echo "EvalScope finished. Merge its reported score with:"
  echo "  $HERE/run_bfcl.py --record-score <score> --role $ROLE --out $OUT"
  exit 0
fi

# ---- branch 3: direct HF run ----------------------------------------------
echo "== direct HF run (run_bfcl.py)"
[ -s "$BFCL_DATA/BFCL_v3_simple.json" ] || fetch_bfcl_data
python3 "$HERE/run_bfcl.py" --model "$MODEL" --role "$ROLE" \
  --bfcl-data "$BFCL_DATA" --out "$OUT" \
  ${CATEGORIES:+--categories $CATEGORIES} "$@"

echo "results written to $OUT"
echo "verdict (once both scores exist): python3 $HERE/gates.py tier1 \\"
echo "    --baseline-score <bf16_score> --score <artifact_score> --out $ROOT/results/bfcl_verdict.json"
