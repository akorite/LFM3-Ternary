#!/usr/bin/env python3
"""Model-agnostic BFCLv3 runner: loads an HF causal LM, runs the AST-scored
categories, and records scores into results/bfcl.json for gates.py tier1.

Scoring is a port of gorilla-llm's BFCL AST checker (bfcl_eval/
eval_checker/ast_eval/ast_checker.py): per-call simple_function_checker
with required/unexpected/optional param rules, type conversion, and
standardized string/list/dict value matching; no-order matching for
parallel categories; enforced order for multi-turn ground-truth steps.
Multi-turn turns are scored by AST match against ground_truth call lists,
which approximates the official stateful-backend exec check.

Execution-backed categories (exec_*, sql, rest) and chatable are skipped
unless the official bfcl_eval harness is present (see eval_bfcl.sh).

Usage:
  run_bfcl.py --model <hf-dir> --role baseline --bfcl-data <dir> \
      --categories simple parallel ... [--out results/bfcl.json]
  run_bfcl.py --record-score 0.71 --role artifact --out results/bfcl.json
"""
import argparse
import ast
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------- data ----

AST_CATEGORIES = [
    "simple", "multiple", "parallel", "parallel_multiple",
    "java", "javascript",
    "irrelevance",
    "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple",
    "live_irrelevance", "live_relevance",
    "multi_turn_base", "multi_turn_composite", "multi_turn_long_context",
    "multi_turn_miss_func", "multi_turn_miss_param",
]
EXEC_CATEGORIES = [
    "exec_simple", "exec_multiple", "exec_parallel", "exec_parallel_multiple",
    "rest", "sql", "chatable",
]

DATASET_BASE = (
    "https://huggingface.co/datasets/"
    "gorilla-llm/Berkeley-Function-Calling-Leaderboard/raw/main"
)


def _snake(name):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def load_jsonl(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_dataset(bfcl_data, categories):
    """{category: {id: entry, 'answers': {id: ground_truth}}}"""
    data = {}
    data_dir = Path(bfcl_data)
    for cat in categories:
        entry_file = data_dir / f"BFCL_v3_{cat}.json"
        if not entry_file.exists():
            print(f"warning: {entry_file} missing, skipping {cat}", file=sys.stderr)
            continue
        entries = {row["id"]: row for row in load_jsonl(entry_file)}
        answers = {}
        ans_file = data_dir / "possible_answer" / f"BFCL_v3_{cat}.json"
        if ans_file.exists():
            answers = {row["id"]: row["ground_truth"] for row in load_jsonl(ans_file)}
        func_docs = {}
        for entry in entries.values():
            for cls in entry.get("involved_classes") or []:
                doc_file = data_dir / "multi_turn_func_doc" / f"{_snake(cls)}.json"
                if cls not in func_docs and doc_file.exists():
                    func_docs[cls] = load_jsonl(doc_file)
        data[cat] = {"entries": entries, "answers": answers, "func_docs": func_docs}
    return data


def multi_turn_docs(entry, func_docs):
    docs = []
    for cls in entry.get("involved_classes") or []:
        docs.extend(func_docs.get(cls, []))
    return docs


# ------------------------------------------------------------ decoding ----

def _call_to_dict(call, func_docs):
    if isinstance(call.func, ast.Attribute):
        parts = []
        node = call.func
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        name = ".".join(reversed(parts))
    elif isinstance(call.func, ast.Name):
        name = call.func.id
    else:
        return None
    doc = next((d for d in func_docs if d.get("name") in (name, name.replace(".", "_"))), None)
    props = list(((doc or {}).get("parameters") or {}).get("properties") or {})

    def value(node):
        try:
            return ast.literal_eval(node)
        except (ValueError, SyntaxError):
            return ast.unparse(node)

    params = {kw.arg: value(kw.value) for kw in call.keywords if kw.arg}
    if call.args:
        # positional args map onto properties in declared order (BFCL decoder convention)
        positional = [value(a) for a in call.args]
        for i, v in enumerate(positional):
            params[props[i] if i < len(props) else f"arg{i}"] = v
    return {name: params}


def _calls_from_code(text, func_docs):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    calls = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            parsed = _call_to_dict(node.value, func_docs)
            if parsed is None:
                return None
            calls.append(parsed)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        else:
            return None
    return calls if calls else None


def _normalize_json_calls(obj):
    items = obj if isinstance(obj, list) else [obj]
    calls = []
    for item in items:
        if not isinstance(item, dict):
            return None
        if "name" in item and ("arguments" in item or "parameters" in item or "args" in item):
            calls.append({item["name"]: item.get("arguments") or item.get("parameters") or item.get("args") or {}})
        elif len(item) == 1:
            name, params = next(iter(item.items()))
            calls.append({name: params if isinstance(params, dict) else {}})
        else:
            return None
    return calls


def decode_model_output(text, func_docs):
    """Best-effort decode of model text into [{func: {param: value}}].

    Accepts Python call syntax (BFCL convention, incl. positional args and
    dotted method names), JSON tool-call objects, and the BFCL dict style.
    Anything undecodable yields [] so the sample scores 0, not a crash.
    """
    text = text.strip()
    text = re.sub(r"^```(?:python|json)?\s*|\s*```$", "", text).strip()
    if not text:
        return []

    for candidate in (text, text[text.find("["):text.rfind("]") + 1] if "[" in text else None):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        calls = _normalize_json_calls(parsed)
        if calls is not None:
            return calls

    calls = _calls_from_code(text, func_docs)
    if calls is not None:
        return calls

    # last resort: one call per line
    calls = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parsed = _calls_from_code(line, func_docs)
        if not parsed:
            return []
        calls.extend(parsed)
    return calls


# --------------------------------------------------- AST checker (port) ---

TYPE_MAPPING = {
    "string": str, "integer": int, "float": float, "boolean": bool,
    "array": list, "tuple": list, "dict": dict, "any": str,
    # Java/JS type spellings used by the java/javascript categories; values
    # for those types are strings in model output, so they type-check as str.
    "String": str, "int": int, "long": int, "float64": float,
    "double": float, "boolean2": bool, "Boolean": bool,
    "Array": list, "ArrayList": list, "HashMap": dict, "Hashtable": dict,
    "char": str, "byte": int, "short": int, "BigInteger": int,
    "number": float, "object": dict, "BigDecimal": float,
}
NESTED_TYPES = {"array", "tuple", "Array", "ArrayList"}


def standardize_string(value):
    return re.sub(r"[ \,\.\/\-\_\*\^]", "", value).lower().replace("'", '"')


def _possible_type(possible):
    for answer in possible:
        if answer != "":
            return type(answer)
    return None


def type_checker(param, value, possible, expected_desc, expected_conv, nested_conv):
    """Port of bfcl_eval type_checker (Python branch, incl. is_variable)."""
    result = {"valid": True, "is_variable": False}
    is_variable = False
    pa_type = _possible_type(possible)
    if pa_type is not None and pa_type != expected_conv:
        is_variable = True

    if type(value) == expected_conv:
        if nested_conv is None:
            result["is_variable"] = is_variable
            return result
        for pa_item in possible:
            flag = True
            if type(pa_item) == list:
                for item in value:
                    check = type_checker(param, item, pa_item, str(nested_conv), nested_conv, None)
                    if not check["valid"]:
                        flag = False
                        break
            if flag:
                return {"valid": True, "is_variable": is_variable}
        result["valid"] = False
        result["error_type"] = "type_error:nested"

    pa_type = _possible_type(possible)
    if pa_type is not None and type(value) == pa_type:
        result["is_variable"] = True
        return result

    result["valid"] = False
    result["error_type"] = "type_error:simple"
    return result


def string_checker(model_value, possible):
    target = standardize_string(model_value)
    standard = [standardize_string(p) for p in possible if type(p) == str]
    return target in standard


def list_checker(model_value, possible):
    model_list = list(model_value)
    model_list = [standardize_string(v) if type(v) == str else v for v in model_list]
    standard = []
    for pa in possible:
        standard.append([standardize_string(v) if type(v) == str else v for v in pa])
    return model_list in standard


def dict_checker(model_dict, possible_answers):
    for possible in possible_answers:
        if possible == "":
            continue
        flag = True
        for key, value in model_dict.items():
            if key not in possible:
                flag = False
                break
            std_value = standardize_string(value) if type(value) == str else value
            std_possible = [
                standardize_string(v) if type(v) == str else v
                for v in possible[key]
            ]
            if std_value not in std_possible:
                flag = False
                break
        for key, value in possible.items():
            if key not in model_dict and "" not in value:
                flag = False
                break
        if flag:
            return True
    return False


def _match_key(model_output, func_name):
    """OSS models may emit '_' where the doc has '.'; accept either."""
    if func_name in model_output:
        return func_name
    alt = func_name.replace(".", "_")
    return alt if alt in model_output else None


def simple_checker(func_description, model_call, possible_answer_call):
    """Port of bfcl_eval simple_function_checker. Returns bool."""
    possible = list(possible_answer_call.values())[0]
    func_name = func_description["name"]
    param_details = func_description["parameters"]["properties"]
    required = func_description["parameters"].get("required") or []

    key = _match_key(model_call, func_name)
    if key is None:
        return False
    model_params = model_call[key]
    if not isinstance(model_params, dict):
        return False

    for param in required:
        if param not in model_params:
            return False

    for param, value in model_params.items():
        if param not in param_details or param not in possible:
            return False
        expected_desc = param_details[param].get("type", "any")
        expected_conv = TYPE_MAPPING.get(expected_desc, str)
        nested_conv = None
        if expected_desc in NESTED_TYPES:
            item_desc = (param_details[param].get("items") or {}).get("type", "any")
            nested_conv = TYPE_MAPPING.get(item_desc, str)
        if expected_desc == "tuple" and type(value) == tuple:
            value = list(value)
        if expected_desc == "float" and type(value) == int:
            value = float(value)

        check = type_checker(param, value, possible[param], expected_desc, expected_conv, nested_conv)
        if not check["valid"]:
            return False
        if not check["is_variable"]:
            if expected_conv == dict:
                if not dict_checker(value, possible[param]):
                    return False
                continue
            if expected_conv == list and nested_conv == dict:
                if not any(
                    len(value) == len(pa) and
                    all(dict_checker(m, [p]) for m, p in zip(value, pa))
                    for pa in possible[param] if pa != ""
                ):
                    return False
                continue
            if expected_conv == str:
                if not string_checker(value, possible[param]):
                    return False
                continue
            if expected_conv == list:
                if not list_checker(value, possible[param]):
                    return False
                continue
        if value not in possible[param]:
            return False

    # ground-truth params absent from the model call must be optional ("")
    for param, allowed in possible.items():
        if param not in model_params and "" not in allowed:
            return False
    return True


def _find_doc(func_descriptions, name):
    if isinstance(func_descriptions, dict):
        return func_descriptions
    for doc in func_descriptions:
        if doc.get("name") == name:
            return doc
    return None


def parallel_enforce_order(func_descriptions, model_calls, possible_answers):
    """Port of parallel_function_checker_enforce_order (multi-turn turns)."""
    if len(model_calls) != len(possible_answers):
        return False
    for model_call, possible in zip(model_calls, possible_answers):
        doc = _find_doc(func_descriptions, next(iter(possible)))
        if doc is None or not simple_checker(doc, model_call, possible):
            return False
    return True


def parallel_no_order(func_descriptions, model_calls, possible_answers):
    """Port of parallel_function_checker_no_order."""
    if len(model_calls) != len(possible_answers):
        return False
    matched = set()
    for possible in possible_answers:
        doc = _find_doc(func_descriptions, next(iter(possible)))
        found = False
        for idx, model_call in enumerate(model_calls):
            if idx in matched:
                continue
            if doc is not None and simple_checker(doc, model_call, possible):
                matched.add(idx)
                found = True
                break
        if not found:
            return False
    return True


def ast_score(category, func_descriptions, model_calls, possible_answers):
    """Dispatch on test_category exactly as bfcl_eval ast_checker does."""
    if "parallel" in category:
        return parallel_no_order(func_descriptions, model_calls, possible_answers)
    if "multiple" in category:
        if len(model_calls) != len(possible_answers):
            return False
        doc = _find_doc(func_descriptions, next(iter(possible_answers[0])))
        return doc is not None and simple_checker(doc, model_calls[0], possible_answers[0])
    if len(model_calls) != 1:
        return False
    return simple_checker(func_descriptions[0], model_calls[0], possible_answers[0])


def irrelevance_correct(model_calls):
    """Model should not emit a function call when none applies."""
    return len(model_calls) == 0


# ------------------------------------------------------------ inference ---

def _build_prompt(tokenizer, messages, tools):
    if hasattr(tokenizer, "apply_chat_template"):
        for kwargs in ({"tools": tools, "add_generation_prompt": True},
                       {"tools": tools}, {"add_generation_prompt": True}, {}):
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, **kwargs
                )
            except Exception:
                continue
    rendered = []
    if tools:
        rendered.append(
            "You have access to the following functions:\n"
            + json.dumps(tools, indent=1)
            + "\nReply with function calls only, one per line, e.g. "
            "func_name(arg=1). Do not call a function if none applies."
        )
    for msg in messages:
        rendered.append(f"{msg['role']}: {msg['content']}")
    rendered.append("assistant:")
    return "\n\n".join(rendered)


def load_model(model_path):
    import torch  # noqa: F401  (lazy: keeps --help and tests CPU-free)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="bfloat16" if torch.cuda.is_available() else "float32",
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return tokenizer, model


def generate_reply(tokenizer, model, messages, tools, max_new_tokens, temperature):
    import torch

    prompt = _build_prompt(tokenizer, messages, tools)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    kwargs = {"max_new_tokens": max_new_tokens}
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature)
    else:
        kwargs.update(do_sample=False)
    with torch.no_grad():
        out = model.generate(**inputs, **kwargs)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# --------------------------------------------------------------- scoring ---

def evaluate_entry(cat, entry, answers, func_docs, tokenizer, model, max_new, temp):
    """Run one entry; returns True/False for correct."""
    eid = entry["id"]
    tools = entry.get("function") or multi_turn_docs(entry, func_docs)

    if cat == "chatable":
        return None  # not scored

    gt = answers.get(eid)

    if cat.startswith("multi_turn"):
        history = []
        all_ok = True
        for turn_idx, turn in enumerate(entry["question"]):
            history.extend(turn)
            reply = generate_reply(tokenizer, model, history, tools, max_new, temp)
            calls = decode_model_output(reply, tools)
            history.append({"role": "assistant", "content": reply})
            if gt is None:
                all_ok = False
                continue
            expected = gt[turn_idx] if turn_idx < len(gt) else []
            parsed = []
            for call_str in expected:
                for call in decode_model_output(call_str, tools):
                    parsed.append({
                        name: {p: v if isinstance(v, list) else [v]
                               for p, v in params.items()}
                        for name, params in call.items()
                    })
            if not parallel_enforce_order(tools, calls, parsed):
                all_ok = False
        return all_ok

    # single-turn
    messages = entry["question"][0]
    reply = generate_reply(tokenizer, model, messages, tools, max_new, temp)
    calls = decode_model_output(reply, tools)

    if cat.endswith("irrelevance"):
        return irrelevance_correct(calls)
    if cat == "live_relevance":
        return len(calls) > 0
    if gt is None:
        print(f"warning: no ground truth for {eid}, scoring 0", file=sys.stderr)
        return False
    return ast_score(cat, tools, calls, gt)


def run_eval(model_path, bfcl_data, categories, out_path, role, max_new, temp, limit):
    tokenizer, model = load_model(model_path)
    dataset = load_dataset(bfcl_data, categories)
    per_cat = {}
    total_n = 0
    total_correct = 0
    for cat, pack in dataset.items():
        entries = list(pack["entries"].values())
        if limit:
            entries = entries[:limit]
        correct = 0
        n = 0
        for entry in entries:
            result = evaluate_entry(
                cat, entry, pack["answers"], pack["func_docs"],
                tokenizer, model, max_new, temp,
            )
            if result is None:
                continue
            n += 1
            correct += int(result)
        if n:
            per_cat[cat] = {"n": n, "accuracy": round(correct / n, 4)}
            total_n += n
            total_correct += correct
        print(f"{cat}: {correct}/{n}", file=sys.stderr)
    score = round(total_correct / total_n, 4) if total_n else 0.0
    record(out_path, role, score, total_n, per_cat)
    print(json.dumps({"score": score, "n_samples": total_n, "categories": per_cat}, indent=2))


# --------------------------------------------------------------- results ---

def record(out_path, role, score, n_samples, categories):
    out_path = Path(out_path)
    data = {}
    if out_path.exists():
        data = json.loads(out_path.read_text())
    key = "bf16_score" if role == "baseline" else "artifact_score"
    data[key] = score
    if n_samples:
        data["n_samples"] = n_samples
    if categories:
        data["categories"] = categories
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="HF model dir or hub id to evaluate")
    parser.add_argument("--role", choices=["baseline", "artifact"], required=True,
                        help="which score slot this run writes in results/bfcl.json")
    parser.add_argument("--bfcl-data", help="dir with BFCL_v3_*.json files "
                        "(dataset: huggingface.co/datasets/gorilla-llm/"
                        "Berkeley-Function-Calling-Leaderboard)")
    parser.add_argument("--categories", nargs="+", default=AST_CATEGORIES,
                        help="BFCLv3 categories to run (default: all AST-scored)")
    parser.add_argument("--out", default="results/bfcl.json")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="cap entries per category (smoke runs)")
    parser.add_argument("--record-score", type=float,
                        help="merge an externally-measured score instead of running a model")
    parser.add_argument("--n-samples", type=int, default=0, help="with --record-score")
    args = parser.parse_args()

    if args.record_score is not None:
        record(args.out, args.role, args.record_score, args.n_samples, None)
        print(f"recorded {args.role} score {args.record_score} in {args.out}")
        return
    if not args.model:
        parser.error("--model is required unless --record-score is given")
    if not args.bfcl_data:
        parser.error("--bfcl-data is required (point at the cloned BFCLv3 dataset dir)")
    skip = [c for c in args.categories if c in EXEC_CATEGORIES]
    if skip:
        print(f"note: {skip} need the official bfcl_eval execution backend; "
              "use eval_bfcl.sh's harness paths for those", file=sys.stderr)
    run_eval(args.model, args.bfcl_data, args.categories, args.out,
             args.role, args.max_new_tokens, args.temperature, args.limit)


if __name__ == "__main__":
    main()
