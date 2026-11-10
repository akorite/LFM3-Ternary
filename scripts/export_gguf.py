#!/usr/bin/env python3
"""Export a CAT-Q ternary LFM3 checkpoint as a packed-ternary GGUF.

Adapted from BitTern `projects/cat-q/quantize/gguf_export.py` (Apache-2.0) with
the LFM3 tensor map: quantized projections are written as `Q2_0`, the
group-128 packed-ternary type of the PrismML llama.cpp fork (branch `prism`) —
1 fp16 scale + 128 packed 2-bit codes = 34 bytes per group (2.125 bits/weight).
BF16-scoped tensors (depthwise conv kernel, all norms, tied embeddings,
rope_freqs) are written as floats exactly as a stock conversion would.

Upstream llama.cpp cannot hold this layout: its TQ2_0 superblock carries one
scale per 256 elements and would drop every second group scale, so no
stock-llama.cpp ternary variant of this file exists. The BF16 and INT4-class
comparators come from `scripts/build_baseline_ggufs.sh`.

Inputs: an HF model dir (config.json + safetensors + tokenizer files) and the
ternary manifest + codes produced by the CAT-Q reconstruction run.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("export_gguf")

QK2_0 = 128
BLOCK_BYTES = 2 + QK2_0 // 4  # fp16 scale + 32 packed bytes, per 128 weights
GGML_TYPE_Q2_0 = 42           # fork's ternary block type id
LLAMA_FTYPE_MOSTLY_Q2_0 = 41  # fork's file-type id

FLOAT_DTYPES = {"f16": "F16", "bf16": "BF16", "f32": "F32"}
DEFAULT_GGUF = Path("gguf/LFM3-1.7B-ternary-pq2_0.gguf")

LFM3_TENSOR_MAP = {
    "model.embed_tokens": "token_embd",
    "model.embedding_norm": "output_norm",
}
ATTN_MAP = {
    "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
    "self_attn.gate_proj": "attn_gate",
    "self_attn.q_norm": "attn_q_norm", "self_attn.k_norm": "attn_k_norm",
}
CONV_MAP = {
    "conv.conv": "shortconv.conv",
    "conv.in_proj": "shortconv.in_proj", "conv.out_proj": "shortconv.out_proj",
}
BLOCK_MAP = {
    "operator_norm": "attn_norm", "post_operator_norm": "post_attention_norm",
    "ffn_norm": "ffn_norm", "post_ffn_norm": "post_ffw_norm",
    "feed_forward.w1": "ffn_gate", "feed_forward.w3": "ffn_up",
    "feed_forward.w2": "ffn_down",
    **ATTN_MAP, **CONV_MAP,
}


def register_q2_0() -> int:
    from gguf.constants import GGML_QUANT_SIZES
    GGML_QUANT_SIZES.setdefault(GGML_TYPE_Q2_0, (QK2_0, BLOCK_BYTES))
    return GGML_TYPE_Q2_0


def pack_q2_0(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """int8 {-1,0,+1} codes + per-group scales -> 34-byte Q2_0 blocks.

    `codes` is one row per output channel, `(out, in)` with `in % 128 == 0`;
    the result is `(out, in/128 * 34)`, the byte shape a GGUF quantized tensor
    expects.
    """
    codes = np.asarray(codes)
    if codes.ndim != 2:
        raise ValueError(f"expected a 2-D (out, in) weight, got {codes.shape}")
    out_f, in_f = codes.shape
    if in_f % QK2_0:
        raise ValueError(f"in_features={in_f} not a multiple of {QK2_0}")
    row_bytes = in_f // QK2_0 * BLOCK_BYTES
    n_blocks = codes.size // QK2_0
    scales = np.asarray(scales, np.float32).reshape(-1)
    if scales.size != n_blocks:
        raise ValueError(f"expected {n_blocks} scales, got {scales.size}")
    q = codes.astype(np.int8).reshape(n_blocks, QK2_0)
    if q.min() < -1 or q.max() > 1:
        raise ValueError("codes must lie in {-1, 0, +1}")
    levels = (q + np.int8(1)).astype(np.uint8).reshape(n_blocks, QK2_0 // 4, 4)
    shifted = levels << np.array([0, 2, 4, 6], np.uint8).reshape(1, 1, 4)
    qs = shifted[..., 0] | shifted[..., 1] | shifted[..., 2] | shifted[..., 3]
    d = scales.astype(np.float16).reshape(n_blocks, 1).view(np.uint8)
    return np.concatenate([d, qs], -1).reshape(out_f, row_bytes)


def gguf_name(hf_name: str) -> str:
    """HF tensor name (with .weight) -> GGUF tensor name, or '' to skip."""
    name = hf_name.removesuffix(".weight")
    if name == "lm_head":  # tied to embed_tokens; no separate tensor
        return ""
    if name in LFM3_TENSOR_MAP:
        return LFM3_TENSOR_MAP[name] + ".weight"
    parts = name.split(".")
    if parts[:2] == ["model", "layers"] and parts[2].isdigit():
        bid = int(parts[2])
        sub = ".".join(parts[3:])
        gg = BLOCK_MAP.get(sub)
        if gg is None:
            raise ValueError(f"unmapped LFM3 block tensor: {hf_name}")
        return f"blk.{bid}.{gg}.weight"
    raise ValueError(f"unmapped tensor: {hf_name}")


def _is_ternary_target(hf_name: str, shape) -> bool:
    """True for the matmul tensors the recipe ternarizes (mirrors the adapter)."""
    if len(shape) != 2 or hf_name.startswith("lm_head"):
        return False
    stem = hf_name.removesuffix(".weight")
    parts = stem.split(".")
    if parts[:2] != ["model", "layers"] or not parts[2].isdigit():
        return False
    sub = ".".join(parts[3:])
    return sub in BLOCK_MAP and BLOCK_MAP[sub] not in {
        "attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm",
        "attn_q_norm", "attn_k_norm", "shortconv.conv"}


def rope_freqs(head_dim: int, partial_rotary_factor: float) -> np.ndarray:
    n_pairs = head_dim // 2
    n_rot = int(partial_rotary_factor * head_dim) // 2
    return np.array([1.0] * n_rot + [1e30] * (n_pairs - n_rot), np.float32)


def load_hf_tensors(model_dir: Path) -> dict[str, np.ndarray]:
    """Read every safetensors shard as float32 numpy.

    bf16 is the checkpoint dtype and numpy has no bf16, so the read goes through
    torch and the values widen exactly.
    """
    import torch
    from safetensors.torch import load_file
    out = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        for k, v in load_file(str(f)).items():
            out[k] = v.to(torch.float32).numpy()
    return out


def count_gguf_types(path: Path) -> dict:
    """Tensor census of a written GGUF (counts the packed type without gguf's enum)."""
    import struct
    raw = np.fromfile(path, dtype=np.uint8)
    _, _, n_tensors, n_kv = struct.unpack("<4sIQQ", bytes(raw[:24]))
    off = 24

    def rd_str(o):
        n = struct.unpack("<Q", bytes(raw[o:o + 8]))[0]
        return o + 8 + n

    def rd_val(t, o):
        if t == 8:
            return rd_str(o)
        if t == 9:
            it, n = struct.unpack("<IQ", bytes(raw[o:o + 12]))
            o += 12
            for _ in range(n):
                o = rd_val(it, o)
            return o
        size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[t]
        return o + size

    for _ in range(n_kv):
        off = rd_str(off)
        t = struct.unpack("<I", bytes(raw[off:off + 4]))[0]
        off = rd_val(t, off + 4)
    ternary = floats = 0
    for _ in range(n_tensors):
        off = rd_str(off)
        nd = struct.unpack("<I", bytes(raw[off:off + 4]))[0]
        off += 4 + 8 * nd
        t = struct.unpack("<I", bytes(raw[off:off + 4]))[0]
        off += 4 + 8
        if t == GGML_TYPE_Q2_0:
            ternary += 1
        else:
            floats += 1
    return {"ternary": ternary, "float": floats}


def _is_f32(name: str, shape) -> bool:
    """Norms, the depthwise conv kernel and every 1-D tensor stay float32."""
    return "norm" in name or len(shape) <= 1 or "shortconv.conv" in name


def _float_tensor(tensor: np.ndarray, dtype: str):
    import gguf
    if dtype == "F32":
        return tensor.astype(np.float32), gguf.GGMLQuantizationType.F32
    if dtype == "BF16":
        t = tensor.astype(np.float32).view(np.uint32)
        return ((t + 0x8000) >> 16).astype(np.uint16).view(np.uint16), gguf.GGMLQuantizationType.BF16
    return tensor.astype(np.float16), gguf.GGMLQuantizationType.F16


def set_vocab(writer, model_dir: Path, vocab_size: int):
    """GPT-2 BPE vocab with the lfm2 pre-tokenizer, from tokenizer.json."""
    import gguf
    tok = json.loads((model_dir / "tokenizer.json").read_text())
    model = tok["model"]
    assert model["type"] == "BPE", "expected BPE tokenizer"
    vocab, merges = model["vocab"], model["merges"]
    added = {a["id"]: a for a in tok.get("added_tokens", [])}
    added_by_content = {a["content"]: i for i, a in added.items()}
    cfg = json.loads((model_dir / "tokenizer_config.json").read_text()) \
        if (model_dir / "tokenizer_config.json").exists() else {}
    tokens = [f"[PAD{i}]" for i in range(vocab_size)]
    toktypes = [gguf.TokenType.UNUSED] * vocab_size
    for piece, idx in vocab.items():
        if idx >= vocab_size:
            continue
        tokens[idx] = piece
        toktypes[idx] = gguf.TokenType.NORMAL
    for idx, a in added.items():
        if idx >= vocab_size:
            continue
        tokens[idx] = a["content"]
        toktypes[idx] = gguf.TokenType.CONTROL if a.get("special") else gguf.TokenType.USER_DEFINED

    def _tok_id(key, default=-1):
        t = cfg.get(key)
        if isinstance(t, dict):
            return t.get("id", default)
        if isinstance(t, str):
            return added_by_content.get(t, default)
        return t if isinstance(t, int) else default

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("lfm2")
    writer.add_token_list(tokens)
    writer.add_token_types(toktypes)
    writer.add_token_merges(merges)
    writer.add_bos_token_id(_tok_id("bos_token", 124894))
    writer.add_eos_token_id(_tok_id("eos_token", 124900))
    writer.add_pad_token_id(_tok_id("pad_token", 124893))
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)
    tpl = cfg.get("chat_template")
    if tpl:
        writer.add_chat_template(tpl)


def absmean_codes(weight: np.ndarray, group_size: int = QK2_0):
    """Derive {-1,0,+1} codes and per-group scales from raw weights."""
    n_groups = weight.shape[-1] // group_size
    g = weight.astype(np.float32).reshape(*weight.shape[:-1], n_groups, group_size)
    alpha = np.clip(np.abs(g).mean(axis=-1, keepdims=True), 1e-8, None)
    return np.round(g / alpha).clip(-1, 1).astype(np.int8), alpha


def export(model_dir: Path, manifest_dir: Path | None, outfile: Path,
           float_type: str = "BF16") -> Path:
    import gguf
    import torch
    config = json.loads((model_dir / "config.json").read_text())
    layer_types = config["layer_types"]

    packed = {}
    if manifest_dir == "absmean":
        pass  # codes are derived per tensor in the loop below
    elif manifest_dir:
        codes = torch.load(manifest_dir / "ternary_codes.pt",
                           map_location="cpu", weights_only=True)
        for name, e in codes.items():
            flat = e["codes"].numpy()
            packed[name + ".weight"] = pack_q2_0(
                flat.reshape(flat.shape[0], -1), e["alpha"].numpy())

    register_q2_0()
    writer = gguf.GGUFWriter(path=None, arch="lfm3", use_temp_file=True)

    # parameters — mirrored from the official LFM3 GGUF conversion
    writer.add_block_count(config["num_hidden_layers"])
    writer.add_context_length(config["max_position_embeddings"])
    writer.add_embedding_length(config["hidden_size"])
    writer.add_feed_forward_length(config["intermediate_size"])
    writer.add_head_count(config["num_attention_heads"])
    n_kv = config["num_key_value_heads"]
    writer.add_head_count_kv([n_kv if lt != "conv" else 0 for lt in layer_types])
    writer.add_key_length(config["head_dim"])
    writer.add_value_length(config["head_dim"])
    writer.add_rope_dimension_count(config["head_dim"])
    writer.add_sliding_window_pattern([lt == "sliding_attention" for lt in layer_types])
    writer.add_sliding_window(config["sliding_window"])
    writer.add_shortconv_l_cache(config["conv_L_cache"])
    rope = config["rope_parameters"]
    writer.add_rope_freq_base(rope["full_attention"]["rope_theta"])
    writer.add_rope_freq_base_swa(rope["sliding_attention"]["rope_theta"])
    writer.add_layer_norm_rms_eps(config["norm_eps"])
    writer.add_type(gguf.GGUFType.MODEL)

    set_vocab(writer, model_dir, config["vocab_size"])

    tensors = load_hf_tensors(model_dir)
    n_ternary = 0
    for hf_name, tensor in sorted(tensors.items()):
        name = gguf_name(hf_name)
        if name == "":
            continue
        if hf_name in packed:
            writer.add_tensor(name, packed.pop(hf_name), raw_dtype=GGML_TYPE_Q2_0)
            n_ternary += 1
            continue
        if manifest_dir == "absmean" and _is_ternary_target(hf_name, tensor.shape):
            codes, alpha = absmean_codes(tensor)
            writer.add_tensor(name, pack_q2_0(codes.reshape(codes.shape[0], -1), alpha),
                              raw_dtype=GGML_TYPE_Q2_0)
            n_ternary += 1
            continue
        if "shortconv.conv" in name and tensor.ndim == 3:
            tensor = tensor.squeeze(1)  # depthwise conv stored 2-D in GGUF
        data, dtype = _float_tensor(tensor, "F32" if _is_f32(name, tensor.shape) else float_type)
        writer.add_tensor(name, data, raw_dtype=dtype)

    writer.add_tensor("rope_freqs.weight",
                      rope_freqs(config["head_dim"], rope["full_attention"].get("partial_rotary_factor", 1.0)),
                      raw_dtype=gguf.GGMLQuantizationType.F32)

    if packed:
        raise RuntimeError(f"{len(packed)} manifest tensors absent from checkpoint: {sorted(packed)[:4]}")
    if not n_ternary:
        raise RuntimeError(
            "no tensor was packed as ternary; pass --manifest <dir> or --manifest absmean"
        )
    writer.add_file_type(LLAMA_FTYPE_MOSTLY_Q2_0)

    writer.write_header_to_file(path=outfile)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    logger.info("wrote %s (%d ternary tensors)", outfile, n_ternary)
    return outfile


def census(model_dir: Path, manifest_dir: Path | None = None) -> dict:
    """Count the tensors each export dtype will receive, without loading data."""
    from safetensors import safe_open

    ternary = set()
    if manifest_dir:
        ternary = {k + ".weight" for k in
                   json.loads((manifest_dir / "ternary_manifest.json").read_text())["tensors"]}
    counts: dict[str, int] = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(f), framework="numpy") as h:
            for hf in h.keys():
                name = gguf_name(hf)
                if name == "":
                    continue
                if hf in ternary:
                    kind = "ternary"
                elif _is_f32(name, h.get_slice(hf).get_shape()):
                    kind = "f32"
                else:
                    kind = "float"
                counts[kind] = counts.get(kind, 0) + 1
    counts["f32"] = counts.get("f32", 0) + 1  # the generated rope_freqs mask
    return counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, type=Path, help="HF checkpoint dir")
    p.add_argument("--manifest", default=None,
                   help="dir with ternary_manifest.json + ternary_codes.pt, "
                        "or 'absmean' to derive codes from the checkpoint weights")
    p.add_argument("--out", type=Path, default=None, help="output .gguf path")
    p.add_argument("--float-type", choices=sorted(FLOAT_DTYPES), default="bf16")
    p.add_argument("--census", action="store_true",
                   help="print the tensor census only, do not write a file")
    args = p.parse_args()
    manifest = args.manifest if isinstance(args.manifest, str) and args.manifest == "absmean" \
        else (Path(args.manifest) if args.manifest else None)
    if args.census:
        print(json.dumps(census(args.model, manifest), indent=1))
        return
    out = args.out or DEFAULT_GGUF
    out.parent.mkdir(parents=True, exist_ok=True)
    export(args.model, manifest, out, FLOAT_DTYPES[args.float_type])


if __name__ == "__main__":
    main()
