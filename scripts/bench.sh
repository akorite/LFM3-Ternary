#!/usr/bin/env bash
# llama-bench all three variants on the same GPU with identical settings.
# Ternary runs on its own serving runtime (PrismML fork); cross-runtime numbers
# are labeled, never blended.
set -euo pipefail

OUT="${OUT:-results}"
mkdir -p "$OUT"
GPU_ARGS="${GPU_ARGS:--ngl 99}"           # no GPU flag => CPU, still comparable
CTX="${CTX:-2048}"
GEN="${GEN:-128}"
PROMPT="${PROMPT:-512}"

LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
LLAMA_CPP_PRISM="${LLAMA_CPP_PRISM:-$HOME/llama.cpp-prism}"
# the comparator builder records which INT4 file it actually produced
INT4_DEFAULT="gguf/LFM3-1.7B-NVFP4.gguf"
[[ -f "$INT4_DEFAULT" ]] || INT4_DEFAULT="gguf/LFM3-1.7B-Q4_K_M.gguf"

run_bench() {  # $1 = label, $2 = llama-bench binary, $3 = model file
  [[ -f "$3" ]] || { echo "missing $3 — skipping $1" >&2; return 0; }
  "$2" -m "$3" $GPU_ARGS -p "$PROMPT" -n "$GEN" -r 3 -o json > "$OUT/bench-$1.json"
  echo "benchmarked $1 -> $OUT/bench-$1.json"
}

BENCH="$LLAMA_CPP/build/bin/llama-bench"
BENCH_PRISM="$LLAMA_CPP_PRISM/build/bin/llama-bench"

run_bench bf16   "$BENCH"       "${BF16_GGUF:-gguf/LFM3-1.7B-BF16.gguf}"
run_bench int4   "$BENCH"       "${INT4_GGUF:-$INT4_DEFAULT}"
run_bench ternary "$BENCH_PRISM" "${TERNARY_GGUF:-gguf/LFM3-1.7B-ternary-pq2_0.gguf}"

python3 - "$OUT" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
res, runtimes = {}, {"bf16": "upstream llama.cpp", "int4": "upstream llama.cpp",
                     "ternary": "PrismML fork (prism branch)"}
for name in ("bf16", "int4", "ternary"):
    f = out / f"bench-{name}.json"
    if not f.exists():
        continue
    rows = json.loads(f.read_text())
    res[name] = {"tg128_t_s": next((r["avg_ts"] for r in rows if r.get("n_gen") and not r.get("n_prompt")), None),
                 "pp512_t_s": next((r["avg_ts"] for r in rows if r.get("n_prompt") and not r.get("n_gen")), None),
                 "runtime": runtimes[name], "model": rows[0]["model_filename"] if rows else None}
(out / "benchmarks.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
PY
