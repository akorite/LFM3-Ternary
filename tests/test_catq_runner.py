"""Corpus loading for the PTQ rung: token tensors and AYOT triplet shards."""
import json
from pathlib import Path

import torch


from run_catq import load_corpus


def test_tensor_corpus_chunks_into_sequences(tmp_path):
    p = tmp_path / "c4.pt"
    torch.save(torch.arange(40, dtype=torch.int32), p)
    batches = list(load_corpus(p, seq_len=8, batch_size=2))
    assert [b.shape for b in batches] == [(2, 8), (2, 8), (1, 8)]


def test_2d_tensor_corpus_passes_through(tmp_path):
    p = tmp_path / "c4.pt"
    torch.save(torch.zeros((3, 8), dtype=torch.int32), p)
    batches = list(load_corpus(p, seq_len=8, batch_size=2))
    assert [b.shape for b in batches] == [(2, 8), (1, 8)]


def test_triplet_shards_are_tokenized(tmp_path):
    d = tmp_path / "ayot"
    d.mkdir()
    trip = {"question": "q", "trace": "t", "answer": "a"}
    (d / "ayot-000.jsonl").write_text(json.dumps(trip) + "\n" + json.dumps(trip) + "\n")
    seen = []

    def encode(text):
        seen.append(text)
        return [1, 2, 3, 4, 5, 6, 7, 8]

    batches = list(load_corpus(d, seq_len=8, batch_size=1, encode=encode))
    assert len(seen) == 2 and seen[0] == "q\nt\na"   # question, trace, answer joined
    assert len(batches) == 2 and batches[0][0].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]


def test_triplet_corpus_requires_tokenizer(tmp_path):
    f = tmp_path / "x.jsonl"
    f.write_text(json.dumps({"question": "q", "trace": "t", "answer": "a"}) + "\n")
    try:
        list(load_corpus(f, seq_len=4, batch_size=1))
    except ValueError as exc:
        assert "tokenizer" in str(exc)
    else:
        raise AssertionError("expected a ValueError without an encoder")


def test_state_dicts_equal_compares_by_key():
    from scripts.run_catq import _state_dicts_equal
    a = {"w": torch.ones(2), "v": torch.zeros(2)}
    # same keys, same tensors, different ordering: still equal
    b = {"v": torch.zeros(2), "w": torch.ones(2)}
    assert _state_dicts_equal(a, b)
    # a dropped key: not equal, even if remaining values match positionally
    assert not _state_dicts_equal(a, {"w": torch.ones(2)})
    # a differing value: not equal
    assert not _state_dicts_equal(a, {"v": torch.zeros(2), "w": torch.zeros(2)})


def test_state_dicts_equal_spots_a_positional_false_match():
    """The bug the smoke check existed to catch: two tensors with different
    names that would line up positionally must not count as equal."""
    from scripts.run_catq import _state_dicts_equal
    a = {"x": torch.ones(2), "y": torch.zeros(2)}
    b = {"y": torch.zeros(2), "z": torch.ones(2)}  # reordered + renamed
    assert not _state_dicts_equal(a, b)
