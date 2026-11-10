#!/usr/bin/env python3
"""Build the QAT training corpus: ~102M domain-matched tokens.

Domain-matched means C4 prose plus reasoning-shaped text, matching what the
QAT rung must recover (the calibration corpora are 1M and 4M tokens; QAT is the
escalation that pays for volume). Output is a flat int32 token tensor that
`train_qat.py` slices into sequences.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

# Rough mix; the exact documents depend on the stream, the volume does not.
SOURCES = [
    ("allenai/c4", "en", "en/c4-train.00000-of-01024.json.gz", "text", 0.7),
    ("meta-math/MetaMathQA", None, None, "query", 0.15),
    ("ise-uiuc/Magicoder-OSS-Instruct-75K", None, None, "problem", 0.15),
]


def build_corpus(streams, encode, target_tokens: int, seed: int = 2):
    """Interleave `streams` ([{weight, docs}]) until `target_tokens` is reached."""
    import random
    rng = random.Random(seed)
    docs = []
    for weight, docs_iter in streams:
        docs.extend([(weight, d) for d in docs_iter])
    ids: list[int] = []
    while len(ids) < target_tokens:
        if not docs:
            raise RuntimeError(f"sources exhausted at {len(ids)}/{target_tokens} tokens")
        _, text = docs[rng.randrange(len(docs))]
        ids.extend(encode(text))
    return ids[:target_tokens]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="LiquidAI/LFM3-1.7B")
    p.add_argument("--out-dir", default="data/calib/qat")
    p.add_argument("--tokens", type=int, default=102_000_000)
    p.add_argument("--doc-pool", type=int, default=200_000,
                   help="documents materialized per source before interleaving")
    p.add_argument("--seed", type=int, default=2)
    args = p.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    streams = []
    for name, config, data_files, field, weight in SOURCES:
        kw = {"split": "train", "streaming": True}
        if config:
            kw["name"] = config
        if data_files:
            kw["data_files"] = data_files
        ds = load_dataset(name, **kw)
        streams.append((weight, (d[field] for d in itertools.islice(ds, args.doc_pool))))

    ids = build_corpus(streams, lambda t: tok(t, add_special_tokens=False)["input_ids"],
                       args.tokens, args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensor = torch.tensor(ids, dtype=torch.int32)
    torch.save(tensor, out / "corpus.pt")
    (out / "manifest.json").write_text(json.dumps({
        "tokens": len(ids),
        "sources": [s[0] for s in SOURCES],
        "seed": args.seed,
        "tokenizer": args.model,
    }, indent=2))
    print(f"wrote {out / 'corpus.pt'}: {len(ids)} tokens")


if __name__ == "__main__":
    main()
