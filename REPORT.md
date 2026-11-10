# LFM3-1.7B ternary compression — results

Plan: `docs/plans/2026-11-10-1024-feat-lfm3-ternary-compression-plan.md`. Every number below is read from `results/*.json`; a dash means that measurement has not been run.

## Tier-0 quality (lm-eval zero-shot)

No BF16 baseline recorded yet (`results/baseline.json` missing).

## Tier-1 tool calling (BFCLv3)

Not measured yet (`results/bfcl.json` missing).

## Decode speed (llama-bench, same GPU and settings)

Not measured yet (`results/benchmarks.json` missing).

## Artifact

Artifact size not recorded yet (`results/artifact.json` missing).

## Reading caution

Retention, not absolute score, is the metric that matters at this size. A ternary 1.7B model can hold a respectable average while collapsing on tool calling — that is why the BFCLv3 gate exists and why no speed or size number is quoted without the quality number beside it.
