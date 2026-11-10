"""Export-path tests: block packing, name mapping, and a real tiny GGUF write."""
import json
import struct

import numpy as np
import pytest
import torch

from export_gguf import (
    BLOCK_BYTES,
    census,
    export,
    gguf_name,
    pack_q2_0,
    rope_freqs,
)

LAYER_TYPES = ["conv", "conv", "full_attention", "conv", "sliding_attention", "conv"]
H, FF, KVH, HD = 128, 256, 2, 8  # hidden, mlp, kv heads, head_dim; in_features % 128 == 0


def _fake_checkpoint(tmp_path):
    """Minimal LFM3-shaped HF dir: config + safetensors + tokenizer files."""
    d = tmp_path / "ckpt"
    d.mkdir()
    cfg = {
        "layer_types": LAYER_TYPES, "num_hidden_layers": 6,
        "num_attention_heads": 4, "num_key_value_heads": 2,
        "hidden_size": H, "intermediate_size": FF, "head_dim": HD,
        "sliding_window": 16, "conv_L_cache": 7,
        "tie_word_embeddings": True, "max_position_embeddings": 512,
        "vocab_size": 128, "norm_eps": 1e-5,
        "rope_parameters": {
            "full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 1e6, "rope_type": "default"},
            "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"},
        },
    }
    (d / "config.json").write_text(json.dumps(cfg))
    sd = {"model.embed_tokens.weight": np.random.randn(128, H).astype(np.float32),
          "model.embedding_norm.weight": np.ones(H, np.float32),
          "lm_head.weight": np.random.randn(128, H).astype(np.float32)}  # tied
    for i, lt in enumerate(LAYER_TYPES):
        p = f"model.layers.{i}"
        sd[f"{p}.operator_norm.weight"] = np.ones(H, np.float32)
        sd[f"{p}.post_operator_norm.weight"] = np.ones(H, np.float32)
        sd[f"{p}.ffn_norm.weight"] = np.ones(H, np.float32)
        sd[f"{p}.post_ffn_norm.weight"] = np.ones(H, np.float32)
        sd[f"{p}.feed_forward.w1.weight"] = np.random.randn(FF, H).astype(np.float32)
        sd[f"{p}.feed_forward.w3.weight"] = np.random.randn(FF, H).astype(np.float32)
        sd[f"{p}.feed_forward.w2.weight"] = np.random.randn(H, FF).astype(np.float32)
        if lt == "conv":
            sd[f"{p}.conv.conv.weight"] = np.random.randn(H, 1, 7).astype(np.float32)
            sd[f"{p}.conv.in_proj.weight"] = np.random.randn(3 * H, H).astype(np.float32)
            sd[f"{p}.conv.out_proj.weight"] = np.random.randn(H, H).astype(np.float32)
        else:
            sd[f"{p}.self_attn.q_proj.weight"] = np.random.randn(H, H).astype(np.float32)
            sd[f"{p}.self_attn.k_proj.weight"] = np.random.randn(KVH * HD, H).astype(np.float32)
            sd[f"{p}.self_attn.v_proj.weight"] = np.random.randn(KVH * HD, H).astype(np.float32)
            sd[f"{p}.self_attn.o_proj.weight"] = np.random.randn(H, H).astype(np.float32)
            sd[f"{p}.self_attn.gate_proj.weight"] = np.random.randn(4, H).astype(np.float32)
            sd[f"{p}.self_attn.q_norm.weight"] = np.ones(HD, np.float32)
            sd[f"{p}.self_attn.k_norm.weight"] = np.ones(HD, np.float32)
    from safetensors.numpy import save_file
    save_file(sd, str(d / "model.safetensors"))
    # minimal BPE tokenizer files
    vocab = {chr(33 + i): i for i in range(118)}
    added = [{"id": 120 + i, "content": f"<|t{i}|>", "special": i == 0, "normalized": False}
             for i in range(8)]
    tok = {"version": "1.0", "model": {"type": "BPE", "vocab": vocab,
                                       "merges": ["Ġ t", "Ġa b"]},
           "added_tokens": added}
    (d / "tokenizer.json").write_text(json.dumps(tok))
    (d / "tokenizer_config.json").write_text(json.dumps({
        "bos_token": "<|t0|>", "eos_token": "<|t1|>", "pad_token": "<|t2|>",
        "chat_template": "{{ bos_token }}"}))
    return d


def _manifest(tmp_path):
    m = tmp_path / "manifest"
    m.mkdir()
    tensors = {}
    for i, lt in enumerate(LAYER_TYPES):
        p = f"model.layers.{i}"
        for n in ("feed_forward.w1", "feed_forward.w3", "feed_forward.w2"):
            tensors[f"{p}.{n}"] = {"shape": {"feed_forward.w1": [FF, H], "feed_forward.w3": [FF, H],
                                             "feed_forward.w2": [H, FF]}[n]}
        if lt == "conv":
            for n in ("conv.in_proj", "conv.out_proj"):
                tensors[f"{p}.{n}"] = {"shape": [3 * H, H] if n == "conv.in_proj" else [H, H]}
        else:
            shapes = {"self_attn.q_proj": [H, H], "self_attn.k_proj": [KVH * HD, H],
                      "self_attn.v_proj": [KVH * HD, H], "self_attn.o_proj": [H, H],
                      "self_attn.gate_proj": [4, H]}
            for n, s in shapes.items():
                tensors[f"{p}.{n}"] = {"shape": s}
    (m / "ternary_manifest.json").write_text(json.dumps({"group_size": 128, "tensors": tensors}))
    codes = {}
    g = 128  # packed block size; pad group count to full blocks
    for name, e in tensors.items():
        o, i = e["shape"]
        ng = -(-i // g)
        codes[name] = {
            "codes": torch.randint(-1, 2, (o, ng, g), dtype=torch.int8),
            "alpha": torch.rand(o, ng, 1) + 0.1,
        }
    torch.save(codes, m / "ternary_codes.pt")
    return m


def test_pack_q2_0_pins_the_external_block_layout():
    """Bytes must match the fork's dequantize: y[j] = (q-1)*d, 2 bits per weight."""
    codes = np.zeros((1, 128), np.int8)
    codes[0, :4] = [-1, 0, 1, -1]
    codes[0, 4:] = 1
    alpha = np.array([[0.5]], np.float32)
    block = pack_q2_0(codes, alpha)
    assert block.shape == (1, BLOCK_BYTES)
    assert bytes(block[0, :2]) == struct.pack("<e", 0.5)   # fp16 scale, little endian
    # levels (code+1) = 0,1,2,0 packed 2 bits each, low bits first -> 0b00_10_01_00
    assert block[0, 2] == 0b00100100
    assert (block[0, 3:] == 0b10101010).all()               # levels 2 (+1) repeated


def test_pack_shape_and_scale_count():
    rng = np.random.RandomState(0)
    codes = rng.randint(-1, 2, (4, 256)).astype(np.int8)
    alpha = (rng.rand(4, 2, 1) + 0.5).astype(np.float32)
    packed = pack_q2_0(codes, alpha)
    assert packed.shape == (4, 2 * BLOCK_BYTES)


def test_pack_rejects_bad_inputs():
    with pytest.raises(ValueError):
        pack_q2_0(np.zeros((4, 100), np.int8), np.zeros(4))
    with pytest.raises(ValueError):
        pack_q2_0(np.full((1, 128), 2, np.int8), np.ones(1))
    with pytest.raises(ValueError):
        pack_q2_0(np.zeros((1, 128), np.int8), np.ones(3))

def test_gguf_name_map():
    assert gguf_name("model.layers.0.conv.in_proj.weight") == "blk.0.shortconv.in_proj.weight"
    assert gguf_name("model.layers.2.self_attn.gate_proj.weight") == "blk.2.attn_gate.weight"
    assert gguf_name("model.layers.2.self_attn.q_norm.weight") == "blk.2.attn_q_norm.weight"
    assert gguf_name("model.layers.4.post_ffn_norm.weight") == "blk.4.post_ffw_norm.weight"
    assert gguf_name("model.layers.0.operator_norm.weight") == "blk.0.attn_norm.weight"
    assert gguf_name("model.embed_tokens.weight") == "token_embd.weight"
    assert gguf_name("model.embedding_norm.weight") == "output_norm.weight"
    assert gguf_name("lm_head.weight") == ""  # tied -> skipped
    with pytest.raises(ValueError):
        gguf_name("model.layers.0.unknown.weight")


def test_rope_freqs_mask():
    f = rope_freqs(128, 0.5)
    assert f.shape == (64,)
    assert (f[:32] == 1.0).all() and (f[32:] == 1e30).all()


def test_census(tmp_path):
    ck = _fake_checkpoint(tmp_path)
    m = _manifest(tmp_path)
    c = census(ck, m)
    assert c["ternary"] == 36
    assert c.get("f32", 0) > 0  # norms + conv kernels + 1-d
    n_float = c.get("float", 0) + c.get("f32", 0)
    assert n_float + c["ternary"] == 6 * 7 + 2 * 7 + 4 * 3 + 2 + 1


def _read_gguf_header(path):
    data = np.fromfile(path, dtype=np.uint8)
    magic, ver, nt, nkv = struct.unpack('<4sIQQ', bytes(data[:24]))
    assert magic == b"GGUF"
    off = 24

    def rd_str(o):
        n = struct.unpack('<Q', bytes(data[o:o + 8]))[0]
        return bytes(data[o + 8:o + 8 + n]).decode('utf8', 'replace'), o + 8 + n

    def rd_val(t, o):
        if t == 8:
            return rd_str(o)
        if t == 9:
            it, n = struct.unpack('<IQ', bytes(data[o:o + 12]))
            o += 12
            vals = []
            for _ in range(n):
                v, o = rd_val(it, o)
                vals.append(v)
            return vals, o
        fmt = {0: 'B', 1: 'b', 2: '<H', 3: '<h', 4: '<I', 5: '<i',
               6: '<f', 7: '?', 10: '<Q', 11: '<q', 12: '<d'}[t]
        sz = struct.calcsize(fmt)
        return struct.unpack(fmt, bytes(data[o:o + sz]))[0], o + sz

    kvs = {}
    for _ in range(nkv):
        k, off = rd_str(off)
        t = struct.unpack('<I', bytes(data[off:off + 4]))[0]
        v, off = rd_val(t, off + 4)
        kvs[k] = v
    names, dims, types = [], {}, {}
    for _ in range(nt):
        k, off = rd_str(off)
        names.append(k)
        nd = struct.unpack('<I', bytes(data[off:off + 4]))[0]
        off += 4
        dims[k] = struct.unpack('<' + 'Q' * nd, bytes(data[off:off + 8 * nd]))
        off += 8 * nd
        types[k] = struct.unpack('<I', bytes(data[off:off + 4]))[0]
        off += 4 + 8
    return kvs, names, dims, types


def test_export_writes_valid_gguf(tmp_path):
    ck = _fake_checkpoint(tmp_path)
    m = _manifest(tmp_path)
    out = tmp_path / "tiny-ternary.gguf"
    export(ck, m, out)
    kvs, names, dims, types = _read_gguf_header(out)
    assert kvs["general.architecture"] == "lfm3"
    assert kvs["general.file_type"] == 41
    assert kvs["lfm3.attention.sliding_window_pattern"] == [False, False, False, False, True, False]
    assert kvs["lfm3.attention.head_count_kv"] == [0, 0, 2, 0, 2, 0]
    assert kvs["lfm3.rope.freq_base"] == 1e6
    assert kvs["lfm3.rope.freq_base_swa"] == 1e4
    assert kvs["lfm3.shortconv.l_cache"] == 7
    assert kvs["tokenizer.ggml.pre"] == "lfm2"
    assert kvs["tokenizer.ggml.bos_token_id"] == 120
    assert "rope_freqs.weight" in names
    assert "token_embd.weight" in names and "output.weight" not in names
    assert "output_norm.weight" in names
    expected = {"token_embd.weight", "output_norm.weight", "rope_freqs.weight"}
    for i, lt in enumerate(LAYER_TYPES):
        b = f"blk.{i}"
        expected |= {f"{b}.attn_norm.weight", f"{b}.post_attention_norm.weight",
                     f"{b}.ffn_norm.weight", f"{b}.post_ffw_norm.weight",
                     f"{b}.ffn_gate.weight", f"{b}.ffn_up.weight", f"{b}.ffn_down.weight"}
        if lt == "conv":
            expected |= {f"{b}.shortconv.conv.weight", f"{b}.shortconv.in_proj.weight",
                         f"{b}.shortconv.out_proj.weight"}
        else:
            expected |= {f"{b}.attn_q.weight", f"{b}.attn_k.weight", f"{b}.attn_v.weight",
                         f"{b}.attn_output.weight", f"{b}.attn_gate.weight",
                         f"{b}.attn_q_norm.weight", f"{b}.attn_k_norm.weight"}
    assert set(names) == expected
    # packed rows keep the logical [in, out] shape; the block count must not leak
    assert dims["blk.0.ffn_up.weight"] == (H, FF)
    assert dims["blk.2.attn_q.weight"] == (H, H)
    assert dims["blk.0.shortconv.in_proj.weight"] == (H, 3 * H)
    # R2 scope by dtype: matmuls packed (42), embedding BF16 (30), norms/kernel F32 (0)
    assert types["blk.0.ffn_up.weight"] == 42
    assert types["token_embd.weight"] == 30
    assert types["blk.0.attn_norm.weight"] == 0
    assert types["blk.0.shortconv.conv.weight"] == 0
    assert types["blk.2.attn_q_norm.weight"] == 0


def test_absmean_mode_packs_every_matmul(tmp_path):
    ck = _fake_checkpoint(tmp_path)
    out = tmp_path / "absmean-ternary.gguf"
    export(ck, "absmean", out)
    kvs, names, dims, types = _read_gguf_header(out)
    # every matmul is packed; norms, conv kernel, embed and lm_head are not
    assert "blk.0.shortconv.in_proj.weight" in names
    assert "blk.2.attn_q.weight" in names
    assert "blk.0.ffn_up.weight" in names
    assert "token_embd.weight" in names and "output.weight" not in names
    assert kvs["general.file_type"] == 41


def test_manifest_from_bake_round_trips_into_export(tmp_path):
    """The QAT path: bake codes, save the manifest, export reads it back."""
    from test_lfm3_adapter import _CausalLM, _Inner, _Layer
    from ternarq.qat import bake_ternary, install_qat, save_ternary_manifest

    torch.manual_seed(7)

    class _Model(_CausalLM):          # same layer types as the export fixture
        def __init__(self):
            super().__init__()
            self.model = _Inner()
            self.model.layers = torch.nn.ModuleList(
                [_Layer("conv" if lt == "conv" else "attention", d=H) for lt in LAYER_TYPES])
            self.lm_head = torch.nn.Linear(H, 128, bias=False)
            self.lm_head.weight = self.model.embed_tokens.weight

    model = _Model()
    install_qat(model, 128)
    manifest = bake_ternary(model, 128)
    manifest_dir = tmp_path / "ternary"
    save_ternary_manifest(manifest_dir, manifest, 128)

    ck = _fake_checkpoint(tmp_path)
    out = tmp_path / "qat-ternary.gguf"
    export(ck, manifest_dir, out)          # must not raise "absent from checkpoint"
    _, names, dims, types = _read_gguf_header(out)
    assert types["blk.2.attn_q.weight"] == 42
    assert dims["blk.2.attn_q.weight"] == (H, H)
