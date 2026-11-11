# Runbook: LFM3 ternary compression on remote GPUs

All training and inference runs through the Colab CLI or the Kaggle CLI. There
is no notebook path: a stage is a command, its inputs are files, and its output
comes back to `results/` and `gguf/`.

## Inputs

- `colab` (authenticated) or `kaggle` (token in `~/.kaggle`), plus a GPU quota.
- Nothing else: the model (`LiquidAI/LFM3-1.7B`), C4, and the BFCLv3 dataset are
  public, and the VM clones this branch itself.

**Pick the cheaper host per stage.** Kaggle's GPU hours are a separate, larger
quota than Colab compute units, so prefer it for anything that fits a 12-hour
kernel: calibration builds, a PTQ rung, evaluations, export. Keep Colab for a
run that will exceed Kaggle's kernel limit or when Kaggle's queue is long — its
session keeps a warm environment between stages. Waiting in a Kaggle queue costs
no Colab compute units; an idle Colab session does.

## One stage at a time (Colab)

```bash
scripts/colab_stage.sh --new --gpu A100 baseline    # -> results/baseline.json
scripts/colab_stage.sh c4                           # -> data/calib/c4_512x2048.pt
scripts/colab_stage.sh rung1                        # -> checkpoints/rung1/{hf,ternary}
scripts/colab_stage.sh eval -- checkpoints/rung1/hf results/rung1.json
scripts/colab_stage.sh export -- checkpoints/rung1/hf checkpoints/rung1/ternary
scripts/colab_stage.sh report                       # -> REPORT.md
```

`--new` recreates the session at the requested accelerator; without it the named
session is reused (`SESSION=...`). `TIMEOUT` bounds each call, `SYNC_DOWN=0`
leaves artifacts on the VM.

## The whole ladder, unattended

```bash
scripts/colab_stage.sh --new --gpu A100 pipeline -- --stop-after export
```

`scripts/run_pipeline.py` runs the baseline, then each rung, evaluating and gating
after every rung. It stops at `kill`, passes a surviving rung to tier-1 (BFCLv3),
then exports and writes the report. Every step is skipped when its output
already exists, so a session that dies resumes by re-running the same command.
`--stop-after gate|tier1|export|bench|report` decides how far past a passing gate
to continue.

## Kaggle (preferred for most stages)

```bash
scripts/kaggle_stage.sh pipeline -- --stop-after export
ACCELERATOR=NvidiaL4 scripts/kaggle_stage.sh baseline
ACCELERATOR=NvidiaA100 TIMEOUT=720 scripts/kaggle_stage.sh rung1
```

Each call pushes a private kernel that runs `scripts/remote_stage.py`, waits for
it, and copies `results/` and `REPORT.md` back. Kaggle caps kernel runtime at 12
hours, so the QAT rung's resumable checkpoints matter there.

## Stage reference

| Stage | Runs | Writes |
|---|---|---|
| `baseline` | `scripts/eval_lm_eval.sh` on BF16 | `results/baseline.json` |
| `c4` | `scripts/build_calib_c4.py` | `data/calib/c4_512x2048.pt` |
| `ayot` | `scripts/build_calib_ayot.py` | `data/calib/ayot/` |
| `rung1` / `rung2` | `scripts/run_catq.py` | `checkpoints/rungN/{hf,ternary}` |
| `qat` | `scripts/train_qat.py` from rung 2 | `checkpoints/rung3/{hf,ternary}` |
| `eval` | lm-eval on a checkpoint | `results/<name>.json` |
| `bfcl` | BFCLv3 baseline or artifact | `results/bfcl.json` |
| `export` | `scripts/export_gguf.py` | `gguf/LFM3-1.7B-ternary-pq2_0.gguf` |
| `census` | tensor dtype audit | stdout |
| `report` | `scripts/report.py` | `REPORT.md` |
| `pipeline` | the ladder above | all of the above |

`scripts/serve_check.sh` and `scripts/bench.sh` run wherever a llama.cpp build
exists (the ternary kernels live in the PrismML fork); they take the GGUF and a
`LLAMA_CLI` / llama-bench path, so run them on the VM after `export`.

## Harness pin

`lm-eval==0.4.13`, not the `0.4.7` the published CAT-Q numbers used: 0.4.7 reads
`transformers.AutoModelForVision2Seq` at import, which no longer exists in the
`transformers>=5.12` release LFM3 needs. Task set, `--num_fewshot 0` and
`--seed 2` are unchanged, and baseline and rungs share the pin, so retentions
stay internally comparable while absolute scores are not directly comparable to
BitTern's table.

## Kaggle needs internet

A Kaggle kernel only reaches GitHub and the Hugging Face Hub when the account is
phone-verified and `enable_internet` is honoured; unverified accounts get
`Could not resolve host: github.com` and cannot run any stage. Colab always has
network. Check with one trivial kernel before queueing a multi-hour one.

## Measured cost (first run, A100)

- baseline lm-eval (5 tasks, zero-shot): ~20 min on A100
- rung 1 CAT-Q fit (512 × 2048 C4, 60 epochs, ~100 tensors): ~5.5–6 h on A100,
  ~15 min per decoder layer, the 2048×7680 MLP linears dominating
- session is billed ~6.7–10.6 CU/hr depending on sharing; budget a rung at ≥7 h
- `checkpoints/<rung>/ternary/tensors/` lets a killed rung resume at the first
  unfinished tensor — the resume path re-splices saved codes, weights match a
  single-pass run bit-for-bit
- `py-spy dump --locals --pid $(pgrep -f run_catq)` names the exact layer,
  tensor, and epoch in flight when the log is silent between layer lines
- known hazard: bf16 `tanh` saturates at exactly -1 for `d_alpha < ~-4`, which
  zeroed `alpha_eff` and NaN'd a 6h rung at epoch 48/60 — fixed in `cb51066`,
  kept here because the same pattern in any future modulation code will kill
  hours just as silently
- measured per-tensor rate on A100 (rung1 log): conv 109s, attn ~110s,
  feed_forward 150-300s => ~5.5-6h per 153-tensor rung, ~35-40 CU
- T4 is ~16x slower than A100 on this workload (measured 53.9ms vs ~3.3ms
  per 4096x2048x7680 matmul): the same rung is ~50-75h on T4, i.e. 5+
  12h-capped Kaggle kernels and ~2-3 weekly quota cycles — T4 is a
  bank-while-free path, not a finish path; a ~34 CU Colab top-up ends it
- resume point after the first interrupted attempt is layer 7

## When a host is unavailable

Both providers can refuse work without failing loudly: a Colab `new` can time
out on capacity, and a Kaggle kernel can sit `QUEUED` while the weekly GPU quota
is exhausted. Neither is an error in the pipeline — the stages are idempotent,
so retry the same command later. Do not switch to another path in the meantime:
every stage resumes from the artifacts already on disk.
