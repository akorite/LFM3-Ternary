#!/usr/bin/env python3
"""Run the QAT rung: absmean g=128 ternary forward, STE, frozen-teacher KD.

Starts from a PTQ checkpoint (not BF16), keeps a frozen BF16 teacher for the
KD term, and checkpoints every `--save-every` steps so a Colab session that
dies mid-run resumes instead of restarting.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logger = logging.getLogger("train_qat")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, help="PTQ checkpoint to start from")
    p.add_argument("--teacher", default="LiquidAI/LFM3-1.7B")
    p.add_argument("--data", required=True, type=Path, help="QAT token corpus .pt")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--tokens", type=int, default=102_000_000)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--keep-last", type=int, default=2,
                   help="rolling checkpoints to keep; each is model + optimizer state")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--max-steps", type=int, default=0, help="0 = derive from --tokens")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import torch
    from transformers import AutoModelForCausalLM

    from ternarq.qat import (FakeQuantLinear, bake_ternary, install_qat, load_checkpoint,
                            qat_loss, save_checkpoint, save_ternary_manifest,
                            uninstall_qat, zero_ratio)
    from ternarq.lfm3_adapter import decoder_layers
    from run_catq import load_token_tensor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    student = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=device)
    install_qat(student, args.group_size)
    student.train()
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=device)
    teacher.eval()
    for prm in teacher.parameters():
        prm.requires_grad_(False)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)
    tokens_per_step = args.seq_len * args.batch_size
    total_steps = args.max_steps or max(args.tokens // tokens_per_step, 1)
    start = load_checkpoint(args.resume, student, opt) if args.resume else 0
    logger.info("resuming at step %d of %d", start, total_steps)

    ids = load_token_tensor(args.data)
    n_chunks = ids.numel() // args.seq_len
    if n_chunks == 0:
        raise SystemExit(f"corpus holds {ids.numel()} tokens, fewer than one sequence")

    # the collapse sentinel is a global signature, so sampling four layers is enough
    probes = [m for layer in list(decoder_layers(student))[:4]
              for m in layer.modules() if isinstance(m, FakeQuantLinear)]
    if not probes:
        raise SystemExit("no quantized linears to monitor")

    args.out.mkdir(parents=True, exist_ok=True)
    for step in range(start, total_steps):
        i = step % n_chunks
        span = args.seq_len * args.batch_size
        chunk = ids[i * span:(i + 1) * span].to(device, torch.long).reshape(
            args.batch_size, args.seq_len)
        with torch.no_grad():
            t_logits = teacher(input_ids=chunk).logits
        s_logits = student(input_ids=chunk).logits
        loss = qat_loss(s_logits, chunk, t_logits, args.temperature, args.lam)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0:
            zr = max(zero_ratio(m.linear.weight, m.group_size) for m in probes)
            logger.info("step %d loss %.4f zero_ratio_max %.3f", step, loss.item(), zr)
        final = step + 1 == total_steps
        if (step + 1) % args.save_every == 0 or final:
            path = save_checkpoint(args.out / f"step{step + 1}.pt", student, opt, step + 1,
                                   keep_last=args.keep_last)
            logger.info("saved %s", path)
        if final:
            # Bake the ternary values into the weights so the evaluated checkpoint,
            # the exported GGUF and the reported numbers are the same model.
            manifest = bake_ternary(student, args.group_size)
            uninstall_qat(student)
            student.save_pretrained(args.out / "hf", safe_serialization=True)
            save_ternary_manifest(args.out / "ternary", manifest, args.group_size)
            (args.out / "qat.json").write_text(json.dumps(
                {"step": step + 1, "total_steps": total_steps, "loss": loss.item(),
                 "checkpoint": str(path), "baked_tensors": len(manifest)}, indent=1))
            logger.info("baked %d ternary tensors into %s/hf", len(manifest), args.out)


if __name__ == "__main__":
    main()
