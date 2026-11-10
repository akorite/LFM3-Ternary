#!/usr/bin/env bash
# Build the two comparator GGUFs for the speed benchmark, then (optionally)
# build the ternary file's upstream-runnable variant.
#
# Comparators: BF16 and NVFP4 of LiquidAI/LFM3-1.7B via upstream llama.cpp
# (LFM3 support merged in PR #31252; NVFP4 enablement in PR #31608). If this
# llama.cpp cannot produce NVFP4 for the hybrid LFM3 graph, the script records
# the substitution to Q4_K_M — same INT4-class bandwidth — and the two-way
# comparison still runs. Never silently drops a comparator.
set -euo pipefail

LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
LLAMA_CPP_PRISM="${LLAMA_CPP_PRISM:-$HOME/llama.cpp-prism}"
MODEL="${MODEL:-LiquidAI/LFM3-1.7B}"
OUT="${OUT:-gguf}"
mkdir -p "$OUT"

COMBINED="$OUT/LFM3-1.7B-BF16.gguf"
if [[ ! -f "$COMBINED" ]]; then
  python3 "$LLAMA_CPP/convert_hf_to_gguf.py" "$MODEL" \
    --outfile "$COMBINED" --outtype bf16
fi

quantize() {  # $1 = type, $2 = output
  "$LLAMA_CPP/build/bin/llama-quantize" "$COMBINED" "$2" "$1"
}

if quantize NVFP4 "$OUT/LFM3-1.7B-NVFP4.gguf" 2>"$OUT/nvfp4.err"; then
  echo "NVFP4 comparator built"
else
  echo "NVFP4 unavailable for this LFM3 architecture ($(tail -1 "$OUT/nvfp4.err")); using Q4_K_M" \
    | tee "$OUT/comparator-substitution.txt"
  quantize Q4_K_M "$OUT/LFM3-1.7B-Q4_K_M.gguf"
fi

# The packed g128 ternary layout is the PrismML fork's; stock llama.cpp has no
# such type (its TQ2_0 carries one scale per 256 values), so a ternary
# comparator is only buildable there. Record that rather than fabricate one.
echo "ternary comparator: PrismML fork binary ($LLAMA_CPP_PRISM), not stock" \
  > "$OUT/ternary-runtime.txt"
ls -l "$OUT"
