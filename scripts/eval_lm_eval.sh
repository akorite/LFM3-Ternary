#!/usr/bin/env bash
# Tier-0 five-task evaluation for LFM3-1.7B — the only score source gates.py trusts.
# Mirrors BitTern cat-q's auto_test_one.sh task selection: PIQA, ARC-E, ARC-C,
# HellaSwag, Winogrande, zero-shot, acc_norm preferred with acc fallback.
#
# Pinned for like-for-like retention across rungs:
#   lm-eval==0.4.7   (BitTern cat-q pin)
#   --seed 2         (BitTern default)
#   num_fewshot 0 / full test sets (no --limit)
#
# Usage:
#   ./scripts/eval_lm_eval.sh <model_path_or_hf_id> <output_scores.json> [hf_token_env]
# Output is a flat JSON {task: score} that scripts/gates.py consumes.
set -euo pipefail

MODEL="${1:?usage: eval_lm_eval.sh <model_path_or_hf_id> <output_scores.json>}"
OUT="${2:?usage: eval_lm_eval.sh <model_path_or_hf_id> <output_scores.json>}"

TASKS="piqa,arc_easy,arc_challenge,hellaswag,winogrande"
RAW_DIR="$(mktemp -d)"
trap 'rm -rf "${RAW_DIR}"' EXIT

lm_eval \
  --model hf \
  --model_args "pretrained=${MODEL},dtype=bfloat16,trust_remote_code=True" \
  --tasks "${TASKS}" \
  --num_fewshot 0 \
  --batch_size auto:4 \
  --seed 2 \
  --output_path "${RAW_DIR}"

# lm-eval writes <output_path>/<model-slug>/results_<ts>.json; glob it.
python3 - "${RAW_DIR}" "${OUT}" "${TASKS}" <<'PYEOF'
import glob
import json
import sys

raw_dir, out_path, tasks = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
hits = glob.glob(f"{raw_dir}/**/results_*.json", recursive=True)
if not hits:
    sys.exit(f"error: no lm-eval results file under {raw_dir}")
raw = json.loads(open(sorted(hits)[-1]).read())["results"]

scores = {}
missing = []
for task in tasks:
    entry = raw.get(task)
    if entry is None:
        missing.append(task)
        continue
    value = entry.get("acc_norm,none", entry.get("acc,none"))
    if value is None:
        missing.append(f"{task} (no acc_norm/acc metric)")
        continue
    scores[task] = round(float(value), 4)
if missing:
    sys.exit(f"error: lm-eval results missing tasks: {', '.join(missing)}")

with open(out_path, "w") as handle:
    json.dump(scores, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(f"wrote {out_path}: {scores}")
PYEOF
