"""Build the C4 calibration corpus: 512 sequences x 2048 tokens.

BitTern's released repo ships no calibration loader (eval/export only); its
configs pin `calib_dataset: c4`, `nsamples: 512`, `seed: 2`. This mirrors the
standard recipe: documents are taken in deterministic order from the first
C4/en train shard, and each sample is a seeded random crop of `seq_len`
tokens, so the corpus is a pure function of (tokenizer, seed, dataset order).
Train is used rather than validation to match the published recipe; the eval
tasks are disjoint from C4 so the split choice cannot contaminate the gates.
"""

import argparse
import itertools
import json
import random
from pathlib import Path


def build_samples(texts, encode, n_seq=512, seq_len=2048, seed=2):
    """Crop `n_seq` samples of `seq_len` tokens from the document pool.

    `texts` is any iterable of document strings; `encode` maps a string to a
    list of token ids. Documents at or below `seq_len` tokens are skipped.
    """
    rng = random.Random(seed)
    pool = list(texts)
    samples = []
    attempts = 0
    while len(samples) < n_seq:
        attempts += 1
        if attempts > n_seq * 200:
            raise RuntimeError(
                f"document pool too small: {len(samples)}/{n_seq} samples after {attempts} draws"
            )
        ids = encode(pool[rng.randrange(len(pool))])
        if len(ids) <= seq_len:
            continue
        start = rng.randrange(len(ids) - seq_len)
        samples.append(ids[start : start + seq_len])
    return samples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="LiquidAI/LFM3-1.7B", help="HF id for the tokenizer")
    p.add_argument("--out-dir", default="data/calib")
    p.add_argument("--n-seq", type=int, default=512)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--doc-pool", type=int, default=20000,
                   help="documents to materialize from the first C4/en train shard")
    args = p.parse_args()

    from datasets import load_dataset  # lazy: heavy dep
    from transformers import AutoTokenizer  # lazy: heavy dep
    import torch  # lazy: heavy dep

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    stream = load_dataset(
        "allenai/c4", "en",
        data_files="en/c4-train.00000-of-01024.json.gz",
        split="train", streaming=True,
    )
    texts = (doc["text"] for doc in itertools.islice(stream, args.doc_pool))
    samples = build_samples(
        texts,
        lambda t: tokenizer(t, add_special_tokens=False)["input_ids"],
        n_seq=args.n_seq, seq_len=args.seq_len, seed=args.seed,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"c4_{args.n_seq}x{args.seq_len}"
    tensor = torch.tensor(samples, dtype=torch.int32)
    torch.save(tensor, out_dir / f"{stem}.pt")
    (out_dir / f"{stem}.manifest.json").write_text(json.dumps({
        "n_seq": args.n_seq,
        "seq_len": args.seq_len,
        "source": "allenai/c4 en train shard-00000, deterministic order",
        "seed": args.seed,
        "tokenizer": args.model,
    }, indent=2))
    print(f"wrote {stem}.pt: {tuple(tensor.shape)}")


if __name__ == "__main__":
    main()
