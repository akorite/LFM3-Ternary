import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_bfcl

RUNNER = ROOT / "scripts" / "run_bfcl.py"

TRIANGLE_FUNC = {
    "name": "calculate_triangle_area",
    "description": "Calculate the area of a triangle given its base and height.",
    "parameters": {
        "type": "dict",
        "properties": {
            "base": {"type": "integer", "description": "base"},
            "height": {"type": "integer", "description": "height"},
            "unit": {"type": "string", "description": "unit"},
        },
        "required": ["base", "height"],
    },
}
TRIANGLE_GT = {"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}

SPOTIFY_FUNC = {
    "name": "spotify.play",
    "description": "Play tracks from an artist for a duration.",
    "parameters": {
        "type": "dict",
        "properties": {
            "artist": {"type": "string", "description": "artist"},
            "duration": {"type": "integer", "description": "minutes"},
        },
        "required": ["artist", "duration"],
    },
}
SPOTIFY_GT = [
    {"spotify.play": {"artist": ["Taylor Swift"], "duration": [20]}},
    {"spotify.play": {"artist": ["Maroon 5"], "duration": [15]}},
]

CURVE_FUNC = {
    "name": "calculate_area_under_curve",
    "description": "integrate",
    "parameters": {
        "type": "dict",
        "properties": {
            "function": {"type": "string", "description": "f"},
            "interval": {"type": "tuple", "items": {"type": "float"}, "description": "i"},
            "method": {"type": "string", "description": "m"},
        },
        "required": ["function", "interval"],
    },
}
CURVE_GT = {
    "calculate_area_under_curve": {
        "function": ["x**2", "lambda x: x**2", "y=x**2"],
        "interval": [[1.0, 3.0]],
        "method": ["", "trapezoidal"],
    }
}


# ---------- simple AST checker ----------


def test_simple_exact_match():
    model = {"calculate_triangle_area": {"base": 10, "height": 5}}
    assert run_bfcl.simple_checker(TRIANGLE_FUNC, model, TRIANGLE_GT)


def test_optional_param_omitted_is_fine():
    model = {"calculate_triangle_area": {"base": 10, "height": 5}}
    assert run_bfcl.simple_checker(TRIANGLE_FUNC, model, TRIANGLE_GT)
    model_full = {"calculate_triangle_area": {"base": 10, "height": 5, "unit": "units"}}
    assert run_bfcl.simple_checker(TRIANGLE_FUNC, model_full, TRIANGLE_GT)


def test_wrong_function_fails():
    model = {"determine_body_mass_index": {"weight": 70, "height": 1.8}}
    assert not run_bfcl.simple_checker(TRIANGLE_FUNC, model, TRIANGLE_GT)


def test_wrong_value_fails():
    model = {"calculate_triangle_area": {"base": 11, "height": 5}}
    assert not run_bfcl.simple_checker(TRIANGLE_FUNC, model, TRIANGLE_GT)


def test_missing_required_param_fails():
    model = {"calculate_triangle_area": {"base": 10}}
    assert not run_bfcl.simple_checker(TRIANGLE_FUNC, model, TRIANGLE_GT)


def test_unexpected_param_fails():
    model = {"calculate_triangle_area": {"base": 10, "height": 5, "flavor": "x"}}
    assert not run_bfcl.simple_checker(TRIANGLE_FUNC, model, TRIANGLE_GT)


def test_string_value_case_and_punctuation_insensitive():
    func = dict(SPOTIFY_FUNC)
    model = {"spotify.play": {"artist": "taylor swift", "duration": 20}}
    assert run_bfcl.simple_checker(func, model, SPOTIFY_GT[0])


def test_nested_tuple_param():
    # BFCL nested-type check: inner items must match the declared item type
    model = {
        "calculate_area_under_curve": {"function": "x**2", "interval": [1.0, 3.0]}
    }
    assert run_bfcl.simple_checker(CURVE_FUNC, model, CURVE_GT)
    bad = {
        "calculate_area_under_curve": {"function": "x**2", "interval": [1.0, 4.0]}
    }
    assert not run_bfcl.simple_checker(CURVE_FUNC, bad, CURVE_GT)


def test_int_accepted_for_float_param():
    func = {
        "name": "calc_binomial_probability",
        "description": "d",
        "parameters": {
            "type": "dict",
            "properties": {
                "n": {"type": "integer"},
                "k": {"type": "integer"},
                "p": {"type": "float"},
            },
            "required": ["n", "k", "p"],
        },
    }
    gt = {"calc_binomial_probability": {"n": [20], "k": [5], "p": [0.6]}}
    assert run_bfcl.simple_checker(func, {"calc_binomial_probability": {"n": 20, "k": 5, "p": 0.6}}, gt)


# ---------- category dispatch ----------


def test_parallel_unordered_match():
    swapped = [
        {"spotify.play": {"artist": "Maroon 5", "duration": 15}},
        {"spotify.play": {"artist": "Taylor Swift", "duration": 20}},
    ]
    assert run_bfcl.ast_score("parallel", [SPOTIFY_FUNC], swapped, SPOTIFY_GT)


def test_parallel_wrong_count_fails():
    one = [{"spotify.play": {"artist": "Taylor Swift", "duration": 20}}]
    assert not run_bfcl.ast_score("parallel", [SPOTIFY_FUNC], one, SPOTIFY_GT)


def test_parallel_wrong_param_fails():
    bad = [
        {"spotify.play": {"artist": "Taylor Swift", "duration": 20}},
        {"spotify.play": {"artist": "Maroon 5", "duration": 99}},
    ]
    assert not run_bfcl.ast_score("parallel", [SPOTIFY_FUNC], bad, SPOTIFY_GT)


def test_simple_category_rejects_multiple_calls():
    two = [
        {"calculate_triangle_area": {"base": 10, "height": 5}},
        {"calculate_triangle_area": {"base": 10, "height": 5}},
    ]
    assert not run_bfcl.ast_score("simple", [TRIANGLE_FUNC], two, [TRIANGLE_GT])


def test_multi_turn_ordered_check():
    docs = [
        {"name": "cd", "parameters": {"properties": {"folder": {"type": "string"}}, "required": []}},
        {"name": "mkdir", "parameters": {"properties": {"dir_name": {"type": "string"}}, "required": []}},
        {"name": "mv", "parameters": {"properties": {"source": {"type": "string"}, "destination": {"type": "string"}}, "required": ["source", "destination"]}},
    ]
    gt = [
        {"cd": {"folder": ["document"]}},
        {"mkdir": {"dir_name": ["temp"]}},
        {"mv": {"source": ["final_report.pdf"], "destination": ["temp"]}},
    ]
    ordered = [
        {"cd": {"folder": "document"}},
        {"mkdir": {"dir_name": "temp"}},
        {"mv": {"source": "final_report.pdf", "destination": "temp"}},
    ]
    assert run_bfcl.parallel_enforce_order(docs, ordered, gt)
    reordered = [ordered[0], ordered[2], ordered[1]]
    assert not run_bfcl.parallel_enforce_order(docs, reordered, gt)


# ---------- decoding ----------


def test_decode_python_call_with_kwargs():
    calls = run_bfcl.decode_model_output(
        "cd(folder='document')",
        [{"name": "cd", "parameters": {"properties": {"folder": {}}}}],
    )
    assert calls == [{"cd": {"folder": "document"}}]


def test_decode_positional_args_mapped_by_param_order():
    docs = [{"name": "sort", "parameters": {"properties": {"file_name": {}, "ascending": {}}}}]
    calls = run_bfcl.decode_model_output("sort('final_report.pdf')", docs)
    assert calls == [{"sort": {"file_name": "final_report.pdf"}}]


def test_decode_dotted_method_name():
    calls = run_bfcl.decode_model_output(
        "math.factorial(number=5)", [{"name": "math.factorial", "parameters": {"properties": {"number": {}}}}]
    )
    assert calls == [{"math.factorial": {"number": 5}}]


def test_decode_multiple_calls_one_per_line():
    text = "cd(folder='document')\nmkdir(dir_name='temp')"
    docs = [
        {"name": "cd", "parameters": {"properties": {"folder": {}}}},
        {"name": "mkdir", "parameters": {"properties": {"dir_name": {}}}},
    ]
    calls = run_bfcl.decode_model_output(text, docs)
    assert calls == [{"cd": {"folder": "document"}}, {"mkdir": {"dir_name": "temp"}}]


def test_decode_json_tool_calls():
    text = '[{"name": "spotify.play", "arguments": {"artist": "Taylor Swift", "duration": 20}}]'
    calls = run_bfcl.decode_model_output(text, [SPOTIFY_FUNC])
    assert calls == [{"spotify.play": {"artist": "Taylor Swift", "duration": 20}}]


def test_decode_bfcl_dict_style():
    text = '[{"calculate_triangle_area": {"base": 10, "height": 5}}]'
    calls = run_bfcl.decode_model_output(text, [TRIANGLE_FUNC])
    assert calls == [{"calculate_triangle_area": {"base": 10, "height": 5}}]


def test_decode_malformed_output_is_empty_not_crash():
    assert run_bfcl.decode_model_output("I'm sorry, I can't help with that.", [TRIANGLE_FUNC]) == []
    assert run_bfcl.decode_model_output("", [TRIANGLE_FUNC]) == []
    assert run_bfcl.decode_model_output("{not json at all", [TRIANGLE_FUNC]) == []


# ---------- irrelevance / relevance ----------


def test_irrelevance_scoring():
    assert run_bfcl.irrelevance_correct([])
    assert not run_bfcl.irrelevance_correct([{"f": {}}])


# ---------- results merge ----------


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(RUNNER), *args], capture_output=True, text=True
    )


def test_help_works_without_transformers():
    r = run_cli("--help")
    assert r.returncode == 0
    assert "bfcl-data" in r.stdout


def test_merge_baseline_then_artifact(tmp_path):
    out = tmp_path / "bfcl.json"
    r1 = run_cli("--record-score", "0.71", "--role", "baseline",
                 "--n-samples", "400", "--out", str(out))
    assert r1.returncode == 0, r1.stderr
    r2 = run_cli("--record-score", "0.63", "--role", "artifact",
                 "--n-samples", "400", "--out", str(out))
    assert r2.returncode == 0, r2.stderr
    data = json.loads(out.read_text())
    assert data["bf16_score"] == 0.71
    assert data["artifact_score"] == 0.63
    assert data["n_samples"] == 400


def test_artifact_without_baseline_writes(tmp_path):
    out = tmp_path / "bfcl.json"
    r = run_cli("--record-score", "0.5", "--role", "artifact", "--out", str(out))
    assert r.returncode == 0, r.stderr
    data = json.loads(out.read_text())
    assert data["artifact_score"] == 0.5
    assert "bf16_score" not in data
