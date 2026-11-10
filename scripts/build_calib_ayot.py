"""Build the AYOT self-trace calibration corpus.

Samples domain-matched questions (math reasoning, code reasoning, and
function-calling prompts, since tool calling is the tier-1 eval), generates
reasoning traces with the BF16 model itself, and writes [question, trace,
answer] JSONL shards under data/calib/ayot/. Target ~4M generated tokens
(accepted band 3.5M-4.5M). `--dry-run` emits the question mix without a model.
"""

import argparse
import itertools
import json
import random
import re
from pathlib import Path

TARGET_TOKENS = 4_000_000
TOKEN_BAND = (3_500_000, 4_500_000)
QUESTIONS_PER_DOMAIN = 2000
QUESTION_SOURCES = {
    "math": ("meta-math/MetaMathQA", "query"),
    "code": ("ise-uiuc/Magicoder-OSS-Instruct-75K", "problem"),
    "function_calling": ("glaiveai/glaive-function-calling-v2", "chat"),
}
EVAL_DATASETS = {
    "piqa": ["goal"],
    "arc_easy": ["question"],
    "arc_challenge": ["question"],
    "hellaswag": ["ctx"],
    "winogrande": ["sentence"],
}

WORD_RE = re.compile(r"[a-z0-9]+")


def _norm(text):
    return " ".join(WORD_RE.findall(text.lower()))


def assert_no_contamination(questions, eval_prompts):
    """Assert no question text collides with any eval prompt."""
    eval_set = {_norm(p) for p in eval_prompts}
    collisions = [q for q in questions if _norm(q) in eval_set]
    assert not collisions, f"{len(collisions)} questions collide with eval prompts: {collisions[:5]}"


def assert_token_band(total, band=TOKEN_BAND):
    lo, hi = band
    if not lo <= total <= hi:
        raise ValueError(f"generated {total} tokens, outside the {lo}-{hi} band")


def split_trace(text):
    """Split a generated trace into (reasoning, final-answer line)."""
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if len(lines) == 1:
        return lines[0], lines[0]
    return "\n".join(lines[:-1]), lines[-1]


def sample_questions(sources, per_domain=QUESTIONS_PER_DOMAIN, seed=0):
    """`sources`: {domain: iterable of question strings}. Returns shuffled
    [{domain, question}] with up to `per_domain` per domain."""
    rng = random.Random(seed)
    questions = []
    for domain, texts in sources.items():
        picked = list(itertools.islice(texts, per_domain))
        questions.extend({"domain": domain, "question": q} for q in picked)
    rng.shuffle(questions)
    return questions


def build_triplets(questions, generate, count_tokens, out_dir,
                   target_tokens=TARGET_TOKENS, shard_size=500):
    """Generate triplets until `target_tokens` is reached or questions run out.

    `generate`: question -> generated text. `count_tokens`: text -> int.
    Returns {"triplets": [...], "total_tokens": int}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    triplets, total = [], 0
    shard, written = 0, 0
    fh = None
    for q in questions:
        if total >= target_tokens:
            break
        text = generate(q["question"])
        trace, answer = split_trace(text)
        triplet = {"question": q["question"], "trace": trace, "answer": answer}
        if fh is None or written % shard_size == 0:
            if fh is not None:
                fh.close()
            fh = open(out_dir / f"ayot-{shard:03d}.jsonl", "w")
            shard += 1
        fh.write(json.dumps(triplet) + "\n")
        written += 1
        total += count_tokens(text)
        triplets.append(triplet)
    if fh is not None:
        fh.close()
    return {"triplets": triplets, "total_tokens": total}


def iter_eval_prompts(bfcl_file=None):
    """Yield every prompt text from the five tier-0 eval sets and BFCLv3."""
    from datasets import load_dataset  # lazy

    for name, fields in EVAL_DATASETS.items():
        ds = load_dataset(name)
        for split in ds:
            for entry in ds[split]:
                for f in fields:
                    if entry.get(f):
                        yield entry[f]
    if bfcl_file:
        for line in Path(bfcl_file).read_text().splitlines():
            entry = json.loads(line)
            qs = entry.get("question") or []
            for turn in qs:
                for msg in (turn if isinstance(turn, list) else [turn]):
                    if isinstance(msg, dict) and msg.get("content"):
                        yield msg["content"]


def _load_questions(seed):
    from datasets import load_dataset  # lazy

    sources = {}
    for domain, (hf_id, field) in QUESTION_SOURCES.items():
        ds = load_dataset(hf_id, split="train", streaming=True)
        sources[domain] = (entry[field] for entry in ds)
    return sample_questions(sources, seed=seed)


def _load_model(model_id):
    import torch  # lazy
    from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    return model, tok


def _make_generate(model, tok, max_new_tokens):
    import torch  # lazy

    def generate(question):
        ids = tok(question, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=0.7, top_p=0.9)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="LiquidAI/LFM3-1.7B")
    p.add_argument("--out-dir", default="data/calib/ayot")
    p.add_argument("--tokens", type=int, default=TARGET_TOKENS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--bfcl-file", default=None,
                   help="optional BFCLv3 questions JSONL for the contamination check")
    p.add_argument("--dry-run", action="store_true",
                   help="emit the question mix only; no model, no generation")
    p.add_argument("--sample-review", type=int, default=0, metavar="N",
                   help="also write the first N triplets to ayot_review.jsonl")
    args = p.parse_args()

    questions = _load_questions(args.seed)
    assert_no_contamination([q["question"] for q in questions],
                            iter_eval_prompts(args.bfcl_file))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        path = out_dir / "questions.jsonl"
        with open(path, "w") as f:
            for q in questions:
                f.write(json.dumps(q) + "\n")
        mix = {}
        for q in questions:
            mix[q["domain"]] = mix.get(q["domain"], 0) + 1
        print(f"dry run: {len(questions)} questions {mix} -> {path}")
        return

    model, tok = _load_model(args.model)
    meta = build_triplets(
        questions,
        _make_generate(model, tok, args.max_new_tokens),
        lambda t: len(tok(t, add_special_tokens=False)["input_ids"]),
        out_dir, target_tokens=args.tokens,
    )
    assert_token_band(meta["total_tokens"])
    if args.sample_review:
        (out_dir / "ayot_review.jsonl").write_text(
            "".join(json.dumps(t) + "\n" for t in meta["triplets"][: args.sample_review]))
    (out_dir / "manifest.json").write_text(json.dumps({
        "n_triplets": len(meta["triplets"]),
        "total_tokens": meta["total_tokens"],
        "source": "self-generated by BF16 " + args.model,
        "seed": args.seed,
    }, indent=2))
    print(f"wrote {len(meta['triplets'])} triplets, {meta['total_tokens']} tokens")


if __name__ == "__main__":
    main()
