#!/usr/bin/env bash
# Load the exported ternary GGUF and generate from a fixed prompt.
# Fails loudly when the runtime cannot produce text — the one serving gate.
set -euo pipefail

MODEL="${1:-gguf/LFM3-1.7B-ternary-pq2_0.gguf}"
CLI="${LLAMA_CLI:?set LLAMA_CLI to the llama-cli of the runtime that has the packed ternary kernels}"
PROMPT="${PROMPT:-The capital of France is}"
N="${N:-32}"

[[ -f "$MODEL" ]] || { echo "no such model: $MODEL" >&2; exit 1; }
out="$("$CLI" -m "$MODEL" -p "$PROMPT" -n "$N" --temp 0 -no-cnv 2>&1)" || {
  echo "llama-cli failed:" >&2; echo "$out" >&2; exit 1; }
echo "$out"
# A runtime that emits no continuation beyond the prompt is not serving.
continuation="${out#"$PROMPT"}"
[[ -n "${continuation//[[:space:]]/}" ]] || { echo "no generated text" >&2; exit 1; }
echo "serve check OK"
