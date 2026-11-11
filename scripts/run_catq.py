#!/usr/bin/env python3
"""Run the CAT-Q PTQ rung on LFM3 and save a quantized HF checkpoint.

Loads the BF16 model, streams calibration batches from a token corpus, runs
the sliding-layer reconstruction over every decoder layer, writes a spliced
checkpoint plus the ternary manifest, and (optionally) checks reload identity.

Both PTQ rungs use this script; `--calib` picks the corpus (C4 for rung 1,
AYOT traces for rung 2) and nothing else changes.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logger = logging.getLogger("run_catq")
import torch


def load_token_tensor(path: Path):
    """Load a saved int token tensor, tolerating a plain list."""
    import torch
    data = torch.load(path, map_location="cpu", weights_only=True)
    return data if torch.is_tensor(data) else torch.tensor(data, dtype=torch.int32)


def load_corpus(path: Path, seq_len: int, batch_size: int, encode=None):
    """Yield [batch, seq] token batches.

    Accepts a token-id tensor (.pt) or AYOT triplet JSONL (a file or a dir of
    shards); triplets are concatenated per question and encoded with `encode`.
    """
    import torch
    path = Path(path)
    if path.suffix == ".jsonl" or path.is_dir():
        if encode is None:
            raise ValueError("triplet corpora need a tokenizer")
        files = sorted(path.glob("ayot-*.jsonl")) if path.is_dir() else [path]
        ids = []
        for f in files:
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                t = json.loads(line)
                ids.extend(encode(f"{t['question']}\n{t['trace']}\n{t['answer']}"))
        data = torch.tensor(ids, dtype=torch.int32)
    else:
        data = load_token_tensor(path)
    if data.ndim == 1:
        data = data[: data.numel() // seq_len * seq_len].reshape(-1, seq_len)
    for start in range(0, data.shape[0], batch_size):
        yield data[start:start + batch_size].to(torch.long)


def _state_dicts_equal(a: dict, b: dict) -> bool:
    """Key-ordered comparison: safetensors can drop or rename buffers, so
    positional pairing would both mask a missing tensor and false-match."""
    import torch
    if a.keys() != b.keys():
        return False
    return all(torch.equal(a[k], b[k]) for k in a)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="LiquidAI/LFM3-1.7B")
    p.add_argument("--calib", required=True, type=Path, help="token corpus .pt")
    p.add_argument("--out", required=True, type=Path, help="output checkpoint dir")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--s0", type=float, default=30.0)
    p.add_argument("--delta0", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--fit-dtype", default="",
                   help="dtype for the two fit GEMMs (e.g. fp16 on Turing); "
                        "params, anchors, and the loss stay fp32/bf16")
    p.add_argument("--max-batches", type=int, default=0,
                   help="cap replay batches per layer pass; 0 replays the whole corpus")
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ternarq.reconstruction import CatqReconstructor

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    def forward(m, batch):
        return m(input_ids=batch.to(next(m.parameters()).device))

    batches = list(load_corpus(args.calib, 2048, args.batch_size,
                               lambda t: tok(t, add_special_tokens=False)["input_ids"]))
    if args.max_batches:
        batches = batches[: args.max_batches]
    if not batches:
        raise SystemExit(f"calibration corpus {args.calib} yielded no batches")
    logger.info("replaying %d calibration batches", len(batches))
    fit_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}.get(args.fit_dtype) if args.fit_dtype else None
    recon = CatqReconstructor(model, args.group_size, args.gamma, args.s0,
                              args.delta0, args.lr, args.epochs,
                              resume_dir=args.out / "ternary",
                              fit_dtype=fit_dtype)
    recon.run(batches, forward)

    hf_dir = args.out / "hf"
    hf_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(hf_dir, safe_serialization=True)
    tok.save_pretrained(hf_dir)
    manifest = recon.save_manifest(args.out / "ternary")
    logger.info("wrote checkpoint %s and manifest %s", args.out, manifest)

    # CPU keeps the check off the card: two resident model copies OOM a 16GB
    # GPU, and equality is dtype/device-agnostic
    reloaded = AutoModelForCausalLM.from_pretrained(
        hf_dir, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="cpu")
    same = _state_dicts_equal(model.state_dict(), reloaded.state_dict())
    meta = {"calib": str(args.calib), "n_batches": len(batches),
            "group_size": args.group_size, "epochs": args.epochs,
            "reload_identical": same,
            "tensors": len(recon.manifest["tensors"])}
    (args.out / "rung.json").write_text(json.dumps(meta, indent=1))
    record_path = Path("results/calibration.json")
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    record[f"rung:{args.out.name}"] = {
        "corpus": str(args.calib), "batches": len(batches),
        "sequences": len(batches) * args.batch_size,
        "epochs": args.epochs, "group_size": args.group_size,
    }
    record_path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    if not same:
        raise SystemExit("reloaded checkpoint differs from the quantized model")
    logger.info("reload identity verified")


if __name__ == "__main__":
    main()
