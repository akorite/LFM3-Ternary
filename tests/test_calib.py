import json
import sys
from pathlib import Path

import pytest


import build_calib_ayot as ayot
import build_calib_c4 as c4


def _fake_encode(text):
    # deterministic stand-in for a tokenizer: one pseudo-token per char
    return [ord(c) % 65536 for c in text]


def test_c4_builds_exactly_512x2048():
    texts = ["x" * 6000 + str(i) for i in range(64)]  # every doc > 2048 tokens
    samples = c4.build_samples(texts, _fake_encode, n_seq=512, seq_len=2048, seed=2)
    assert len(samples) == 512
    assert all(len(s) == 2048 for s in samples)


def test_c4_is_deterministic_for_same_seed():
    texts = ["x" * 6000 + str(i) for i in range(64)]
    a = c4.build_samples(texts, _fake_encode, n_seq=8, seq_len=128, seed=7)
    b = c4.build_samples(list(texts), _fake_encode, n_seq=8, seq_len=128, seed=7)
    assert a == b


def test_ayot_triplets_are_well_formed(tmp_path):
    questions = [{"domain": "math", "question": f"what is {i}+{i}?"} for i in range(20)]

    def stub_generate(question):
        return f"reasoning about {question}\nmore steps\nThe answer is 42"

    meta = ayot.build_triplets(
        questions,
        stub_generate,
        count_tokens=lambda s: len(s.split()),
        out_dir=tmp_path / "ayot",
        target_tokens=50,
    )
    shards = sorted((tmp_path / "ayot").glob("*.jsonl"))
    assert shards, "no shards written"
    triplets = [json.loads(l) for sh in shards for l in sh.read_text().splitlines()]
    assert triplets
    for t in triplets:
        assert set(t) == {"question", "trace", "answer"}
        assert all(isinstance(t[k], str) and t[k].strip() for k in t)
    assert meta["total_tokens"] >= 50


def test_contamination_check_flags_collision():
    questions = [
        "What is the capital of France?",
        "completely unrelated question",
    ]
    eval_prompts = iter(["what  is the capital   of france?", "some other prompt"])
    with pytest.raises(AssertionError):
        ayot.assert_no_contamination(questions, eval_prompts)


def test_contamination_check_passes_clean_set():
    questions = ["question a", "question b"]
    eval_prompts = iter(["prompt x", "prompt y"])
    ayot.assert_no_contamination(questions, eval_prompts)


def test_token_band_rejects_under_target():
    with pytest.raises(ValueError):
        ayot.assert_token_band(3_499_999)
    with pytest.raises(ValueError):
        ayot.assert_token_band(4_500_001)
    ayot.assert_token_band(4_000_000)


def test_imports_stay_light():
    """Heavy deps load lazily, so `pytest` and `--help` run on a CPU-only box."""
    import subprocess
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, sys.argv[1]);"
         "import build_calib_c4, build_calib_ayot;"
         "heavy = {'datasets', 'transformers', 'torch'} & set(sys.modules);"
         "assert not heavy, heavy",
         str(Path(__file__).resolve().parents[1] / "scripts")],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
