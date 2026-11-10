#!/usr/bin/env bash
# Load the exported ternary GGUF, generate from a fixed prompt, and record what
# was served and on which runtime (the artifact record REPORT.md reads).
set -euo pipefail

MODEL="${1:-gguf/LFM3-1.7B-ternary-pq2_0.gguf}"
CLI="${LLAMA_CLI:?set LLAMA_CLI to the llama-cli of the runtime that has the packed ternary kernels}"
PROMPT="${PROMPT:-The capital of France is}"
N="${N:-32}"
RUNTIME="${RUNTIME:-PrismML fork (prism branch)}"
OUT="${OUT:-results}"
mkdir -p "$OUT"

[[ -f "$MODEL" ]] || { echo "no such model: $MODEL" >&2; exit 1; }
out="$("$CLI" -m "$MODEL" -p "$PROMPT" -n "$N" --temp 0 -no-cnv 2>&1)" || {
  echo "llama-cli failed:" >&2; echo "$out" >&2; exit 1; }
echo "$out"
continuation="${out#"$PROMPT"}"
[[ -n "${continuation//[[:space:]]/}" ]] || { echo "no generated text" >&2; exit 1; }

python3 - "$MODEL" "$RUNTIME" "$OUT/artifact.json" <<'PY'
import json, sys
from pathlib import Path

import gguf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_gguf import GGML_TYPE_Q2_0

model, runtime, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
reader = gguf.GGUFReader(str(model))
q2_0 = sum(1 for t in reader.tensors if t.tensor_type == GGML_TYPE_Q2_0)
floats = len(reader.tensors) - q2_0
out.write_text(json.dumps({
    "path": str(model),
    "bytes": model.stat().st_size,
    "runtime": runtime,
    "census": {"ternary": q2_0, "float": floats},
}, indent=1))
print(f"recorded {out}: {q2_0} ternary / {floats} float tensors")
PY
echo "serve check OK"
