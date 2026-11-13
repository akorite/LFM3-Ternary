# Ternary export and evaluation pitfalls (LFM3 pipeline)

Three failures cost real debugging time here because each one is silent, passes
tests, and only shows up when a 1.7B model is actually converted.

## 1. Read bf16 safetensors through torch, never numpy

`safetensors.safe_open(..., framework="numpy").get_tensor(name)` raises
`TypeError: data type 'bfloat16' not understood`. Every checkpoint this pipeline
produces is bf16, so a numpy read makes the exporter fail on the first real
model while float32 fixtures keep the suite green. Read with
`framework="pt"`, convert to `float32` (the widening is exact for bf16), and
keep the float32 values until the GGUF writer narrows them.

## 2. A packed GGUF tensor takes one byte row per output channel

`block_q2_0` is `fp16 scale + 32 packed bytes` per 128 weights. A weight of
shape `(out, in)` becomes a byte array of shape `(out, in/128 * 34)`. Passing a
3-D array (`(out, groups, 34)`) is accepted by `GGUFWriter` without complaint,
but the reader derives `ne` by expanding only the last dimension, so the file
declares `ne = [128, groups, out]` — a K dimension of 128 on every matmul. The
symptom is a model that loads and computes nonsense, not a load error. Flatten
grouped codes to `(out, in)` before packing, and assert the written `ne` dims in
a test (the fixture's `_read_gguf_header` parses them).

## 3. Bake ternary weights before saving, or the gate measures a different model

QAT applies absmean ternary inside `forward()` only. Saving the student with the
wrappers stripped writes the raw STE master weights, so lm-eval/BFCL score a
float model while the exported GGUF is quantized — the gate certifies quality
the artifact does not have. Bake
`w <- round(w / mean|w|_group) * mean|w|_group` into every wrapped weight at the
final step, write the matching codes/scales manifest from the same tensors, and
export from that manifest. Evaluating the baked checkpoint and exporting the
baked codes then describe one model by construction.
