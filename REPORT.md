# LFM3-1.7B ternary compression — results

Plan: `docs/plans/2026-11-10-1024-feat-lfm3-ternary-compression-plan.md`. Every number below is read from `results/*.json`; a dash means that measurement has not been run.

## Tier-0 quality (lm-eval zero-shot)

| Task | BF16 | rung 1 | rung 2 | rung 3 |
|---|---|---|---|---|
| PIQA | 0.5996 | — | — | — |
| ARC-E | 0.3851 | — | — | — |
| ARC-C | 0.2645 | — | — | — |
| HellaSwag | 0.5371 | — | — | — |
| Winogrande | 0.5454 | — | — | — |

| Rung | avg-5 | avg retention | verdict |
|---|---|---|---|
| 1 | — | — | — |
| 2 | — | — | — |
| 3 | — | — | — |

## Tier-1 tool calling (BFCLv3)

Not measured yet (`results/bfcl.json` missing).

## Decode speed (llama-bench, same GPU and settings)

Not measured yet (`results/benchmarks.json` missing).

## Calibration

Calibration corpora not recorded yet (`results/calibration.json` missing).

## Artifact

Artifact size not recorded yet (`results/artifact.json` missing).

## Reading caution

Retention, not absolute score, is the metric that matters at this size. A ternary 1.7B model can hold a respectable average while collapsing on tool calling — that is why the BFCLv3 gate exists and why no speed or size number is quoted without the quality number beside it.
