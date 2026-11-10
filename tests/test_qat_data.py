"""Corpus mix: the QAT build honours the source weights it declares."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_qat_data import build_corpus


def _encode(text):
    return [len(text)]


def test_mix_follows_the_declared_weights():
    # source B has 20x the weight of A and each doc is one token
    streams = [(0.05, iter(["a"] * 5000)), (0.95, iter(["bb"] * 5000))]
    ids = build_corpus(streams, _encode, target_tokens=2000)
    heavy = sum(1 for v in ids if v == 2)
    assert len(ids) == 2000
    assert heavy > 1700


def test_exhausted_source_drops_out():
    streams = [(0.5, iter(["a"])), (0.5, iter(["bb"] * 50))]
    ids = build_corpus(streams, _encode, target_tokens=40)
    assert len(ids) == 40
    assert set(ids) <= {1, 2}


def test_all_sources_exhausted_raises():
    try:
        build_corpus([(1.0, iter(["a"]))], _encode, target_tokens=10)
    except RuntimeError as exc:
        assert "exhausted" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
