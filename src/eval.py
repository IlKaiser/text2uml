#!/usr/bin/env python3
"""
eval.py – Evaluation script for text2uml.

Reads configuration from eval_config.yaml (or a path passed via --config).
Runs evaluation over dataset result files, writes a CSV, and saves all plots
to the graph/ directory (configurable).
"""

import argparse
import csv
import os
import re
import time
import traceback

# Disable LangSmith tracing before any LangChain code is imported.
# Without this, LangChain's background thread spams stderr with
# "tenant exceeded usage limits" errors whenever the free quota is hit.
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from itertools import chain, islice
from pathlib import Path

import yaml
from dotenv import load_dotenv
from lark import Lark, Tree as _LarkTree
from tqdm.auto import tqdm

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend – no display needed
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "dataset_path": "dataset",
    "grammar_path": "grammar.ebnf",
    "output_csv": "crash_evaluation_results_llm.csv",
    "graph_dir": "graph",
    "max_workers": 12,
    "stall_timeout": 300,
    "poll_interval": 1.0,
    "ignore_list": ["deepseek-reasoner"],
    "selected_models": ["o3-mini", "claude-3-7-sonnet-20250219", "Qwen2.5-14B-Instruct-4bit"],
    "subfolder_model": "claude-sonnet-4-6",
    "subfolder_technique": "CoT",
    "run_evaluation": True,
    "run_plots": True,
    "show_plots": False,
    "dpi": 300,
    "prefixes": {
        "one": "one-shot",
        "few": "two-shots",
        "cot": "CoT",
        "cot2": "CoT-multi",
    },
    "strip_prompt": {
        "enabled": False,
        "patterns": [
            r"\[INST\].*?\[/INST\]",
            r"<\|user\|>.*?<\|assistant\|>",
            r"<\|im_start\|>.*?<\|im_end\|>",
            r"<\|system\|>.*?<\|end\|>",
            r"<start_of_turn>user.*?<start_of_turn>model",
            r"<｜User｜>.*?<｜Assistant｜>",
            r"(?:Human|User):\s.*?(?=Assistant:|$)",
            r"Assistant:\s",
            r"###\s*Instruction:.*?###\s*Response:",
        ],
        "keep_after_last_match": False,
    },
}


def load_config(path: str) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path) as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg.update(user_cfg)
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _resolve(base: str, path: str) -> str:
    """Resolve path relative to a base directory."""
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(base)), path)


def save_fig(fig_or_ax, filename: str, graph_dir: str, show: bool = False, dpi: int = 300):
    os.makedirs(graph_dir, exist_ok=True)
    out = os.path.join(graph_dir, filename)
    plt.savefig(out, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Prefix / technique extraction
# ──────────────────────────────────────────────────────────────────────────────

def make_extractor(prefixes: dict):
    """Return extract_base_and_type_(model_name) -> (base, technique) closure."""
    def extract(model_name: str):
        if not isinstance(model_name, str):
            return str(model_name), "zero-shot"
        for prefix, label in prefixes.items():
            if model_name.startswith(f"{prefix}-"):
                return model_name[len(prefix) + 1:], label
        return model_name, "zero-shot"
    return extract


# ──────────────────────────────────────────────────────────────────────────────
# Prompt / assistant stripping
# ──────────────────────────────────────────────────────────────────────────────

def get_strip_cfg_for_file(path: str, strip_cfg: dict) -> dict | None:
    """
    Return a single-pattern strip config matched to the result filename,
    or None if no model_pattern matches (file is left untouched).
    """
    if not strip_cfg or not strip_cfg.get("enabled"):
        return None
    filename = os.path.basename(path)
    for model_regex, rule in strip_cfg.get("model_patterns", {}).items():
        #print(f"Checking strip rule: regex={model_regex} against filename={filename}")
        if re.search(model_regex, filename, re.IGNORECASE):
            return {
                "enabled": True,
                "patterns": [rule["pattern"]],
                "keep_after_last_match": strip_cfg.get("keep_after_last_match", True),
                "save_stripped": strip_cfg.get("save_stripped", False),
                "stripped_suffix": strip_cfg.get("stripped_suffix", "_stripped"),
            }
    return None  # no rule matched → no stripping for this file


def strip_prompt_noise(text: str, strip_cfg: dict) -> str:
    """
    Remove prompt scaffolding (system prompt, user turn, assistant tag) from a
    result file so that only the model's PlantUML output remains.

    strip_cfg keys:
        enabled              – bool, master switch
        patterns             – list[str], regex patterns applied with re.DOTALL
        keep_after_last_match – bool, if True keep only the text after the last
                               matched region's end (useful when the full prompt
                               precedes a single assistant answer)
    """
    if not strip_cfg.get("enabled", False):
        return text

    patterns = strip_cfg.get("patterns", [])
    keep_after = strip_cfg.get("keep_after_last_match", False)

    if keep_after:
        last_end = 0
        for pat in patterns:
            for m in re.finditer(pat, text, flags=re.DOTALL):
                if m.end() > last_end:
                    last_end = m.end()
        if last_end:
            text = text[last_end:]
    else:
        for pat in patterns:
            text = re.sub(pat, "", text, flags=re.DOTALL)

    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Grammar / UML parsing
# ──────────────────────────────────────────────────────────────────────────────

def init_parser(grammar_path: str) -> Lark:
    with open(grammar_path, encoding="utf-8") as f:
        return Lark(f.read())


def parse_text(parser: Lark, ref: str, strip_cfg: dict | None = None):
    # Strip prompt / assistant scaffolding first (before any other cleanup)
    if strip_cfg:
        ref = strip_prompt_noise(ref, strip_cfg)
    ref = ref.replace("`", "").replace("plantuml", "").replace("plaintext", "")
    # If </think> is present (full block or orphaned closing tag after prompt-stripping),
    # keep only the content that follows it — that is the actual model output.
    if "</think>" in ref:
        ref = ref[ref.rfind("</think>") + len("</think>"):]
    else:
        ref = re.sub(r"<think>", "", ref)
    # Normalise inline class bodies:  class Foo { A x, B y }  →  class Foo {\n  A x\n  B y\n}
    # Some models emit comma-separated attributes on one line; the grammar requires one per line.
    def _expand_inline_body(m):
        body = m.group(1)
        attrs = "\n  ".join(a.strip() for a in body.split(","))
        return "{\n  " + attrs + "\n}"
    # Only expand if the body contains a comma (comma-separated attributes on one line).
    # This avoids falsely rewriting single-attribute bodies or method signatures.
    ref = re.sub(r"\{\s*([^}\n]*,[^}\n]*)\s*\}", _expand_inline_body, ref)
    # If multiple @startuml blocks exist (e.g. prompt contained a few-shot example),
    # keep only the last one — that is always the model's actual output.
    if ref.count("@startuml") > 1:
        ref = "@startuml" + ref.rsplit("@startuml", 1)[1]
    if "@startuml" not in ref:
        ref = "@startuml\n" + ref
    if "@enduml" not in ref:
        ref = ref + "\n@enduml"
    return parser.parse(ref)


def _node_data(node) -> str:
    """Return the rule name of a Lark Tree node.

    In parsed trees Lark sets Tree.data to a Token object (which has a .value
    attribute equal to the rule name).  However, grammar *aliases* (rules
    written as  rule: ... -> alias_name) produce Tree nodes whose .data is
    already a plain str.  Both cases must be handled uniformly.
    """
    d = node.data
    return d.value if hasattr(d, "value") else d


def get_from_parsed(parsed_text, to_get: str = "class"):
    found = parsed_text.find_pred(lambda v: _node_data(v) == to_get)
    to_ret = []
    for f in found:
        if to_get == "relationship":
            if _node_data(f.children[0]) == "relationship_state":
                continue
            one_rel = f.children[0].children
            if _node_data(one_rel[0]) == "arrow_end_1":
                first_rel = one_rel[0].children[0].value
            else:
                first_rel = (
                    one_rel[0].children[0].value + ", " + one_rel[0].children[1].value
                )
            try:
                first_card = one_rel[1].children[0].value
            except Exception:
                first_card = '"0..*"'
            try:
                second_card = one_rel[5].children[0].value
            except Exception:
                second_card = '"0..*"'
            second_rel = one_rel[-2].children[0].value
            to_ret.append({first_rel: first_card, second_rel: second_card})
        elif to_get == "class":
            to_ret.append(f.children[0].children[0].value)
    return to_ret


def _extract_var_string(variable_node):
    var_name = type_name = None
    for child in variable_node.children:
        if isinstance(child, _LarkTree):
            child_type = child.data.value if hasattr(child.data, "value") else child.data
            if child_type == "var":
                var_name = child.children[0].value
            elif child_type == "type":
                type_name = child.children[0].value
    if var_name and type_name:
        return f"{var_name}:{type_name}"
    return var_name or type_name or None


def get_attributes_from_parsed(parsed_text):
    result = []
    for class_node in parsed_text.find_pred(lambda v: _node_data(v) == "class"):
        class_name = class_node.children[0].children[0].value
        for child in class_node.children:
            if not isinstance(child, _LarkTree):
                continue
            child_type = child.data.value if hasattr(child.data, "value") else child.data
            if child_type in ("stereotype", "class_name", "method", "class_group_title"):
                continue
            if child_type == "variable":
                attr_str = _extract_var_string(child)
                if attr_str:
                    result.append((class_name, attr_str))
            elif child_type == "attribute":
                for sub in child.children:
                    if isinstance(sub, _LarkTree):
                        sub_type = sub.data.value if hasattr(sub.data, "value") else sub.data
                        if sub_type == "variable":
                            attr_str = _extract_var_string(sub)
                            if attr_str:
                                result.append((class_name, attr_str))
                            break
    return result


def _is_inheritance_rel(one_rel):
    for child in one_rel:
        if isinstance(child, _LarkTree):
            child_type = child.data.value if hasattr(child.data, "value") else child.data
            if child_type in ("arrow_head_1", "arrow_head_2"):
                for token in child.children:
                    if hasattr(token, "value") and token.value.strip() in ("<|", "|>"):
                        return True
    return False


def get_inheritance_from_parsed(parsed_text):
    result = []
    seen = set()
    for rel_node in parsed_text.find_pred(lambda v: _node_data(v) == "relationship"):
        if _node_data(rel_node.children[0]) == "relationship_state":
            continue
        one_rel = rel_node.children[0].children
        if not _is_inheritance_rel(one_rel):
            continue
        end1 = one_rel[0].children[0].value
        end2 = one_rel[-2].children[0].value
        parent, child = end1, end2
        for c in one_rel:
            if isinstance(c, _LarkTree):
                ctype = c.data.value if hasattr(c.data, "value") else c.data
                if ctype == "arrow_head_2":
                    for token in c.children:
                        if hasattr(token, "value") and token.value.strip() == "|>":
                            parent, child = end2, end1
        key = (parent, child)
        if key not in seen:
            seen.add(key)
            result.append(key)
    # Also extract inheritance declared via "extends" in class definitions
    for class_node in parsed_text.find_pred(lambda v: _node_data(v) == "class"):
        class_name_nodes = [
            c for c in class_node.children
            if isinstance(c, _LarkTree)
            and (c.data.value if hasattr(c.data, "value") else c.data) == "class_name"
        ]
        if len(class_name_nodes) >= 2:
            child_class = class_name_nodes[0].children[0].value
            parent_class = class_name_nodes[1].children[0].value
            key = (parent_class, child_class)
            if key not in seen:
                seen.add(key)
                result.append(key)
    return result


def parse_path(path_to_uml: str, parser: Lark, strip_cfg: dict | None = None):
    if not os.path.isfile(path_to_uml):
        raise FileNotFoundError(path_to_uml)
    with open(path_to_uml) as f:
        ref = f.read()

    # Resolve the effective strip rule for this specific file
    effective_strip = get_strip_cfg_for_file(path_to_uml, strip_cfg)

    if effective_strip and effective_strip.get("save_stripped"):
        stripped = strip_prompt_noise(ref, effective_strip)
        suffix = effective_strip.get("stripped_suffix", "_stripped")
        base, ext = os.path.splitext(path_to_uml)
        stripped_path = base + suffix + ext
        with open(stripped_path, "w", encoding="utf-8") as f_out:
            f_out.write(stripped)

    parsed = parse_text(parser, ref, strip_cfg=effective_strip)
    return (
        get_from_parsed(parsed, "class"),
        get_from_parsed(parsed, "relationship"),
        get_attributes_from_parsed(parsed),
        get_inheritance_from_parsed(parsed),
    )


def get_class_and_assoc_size(path: str, parser: Lark, strip_cfg: dict | None = None):
    class_uml, rel_uml, attrs_uml, inh_uml = parse_path(path, parser, strip_cfg=strip_cfg)
    return len(class_uml), len(rel_uml), len(attrs_uml), len(inh_uml)


# ──────────────────────────────────────────────────────────────────────────────
# Metric computation
# ──────────────────────────────────────────────────────────────────────────────

def check_relations(predicted, correct, path_to_check=None):
    try:
        total_score = 0
        tp_fn = len(correct)
        tp_fp = len(predicted)
        # Each predicted relation can match at most one correct relation (and
        # vice-versa). Without this guard, a single predicted relation whose
        # endpoints happen to equal several correct ones would increment
        # true_pos multiple times, inflating recall artificially.
        def _norm(v: str) -> str:
            """Normalise a cardinality token to lo..hi form (no surrounding quotes).

            Canonical equivalences applied before comparison:
              *          →  0..*   (bare star means zero-or-more)
              1          →  1..1   (bare integer means exact)
              <|, |>, '' →  0..*   (arrow-head tokens from inheritance arrows)
            """
            v = v.replace('"', "").strip()
            # Arrow-head tokens and bare star all mean 0..*
            if v in ("<|", "|>", "", "*"):
                v = "0..*"
            if ".." not in v:
                # Bare integer n  →  n..n
                v = f"{v}..{v}"
            return v

        def _rel_score(rel: dict, rel_: dict) -> int:
            """Score a candidate match: 1 (endpoint match) + up to 2 per endpoint.

            Both the natural key alignment and the swapped alignment are tried;
            the higher score is returned.  This handles associations written in
            the opposite direction in the reference vs. the prediction
            (e.g. reference: A "1"--"0..*" B  vs.  prediction: B "1"--"0..*" A).
            """
            keys = list(rel.keys())

            def _score_alignment(pairs):
                s = 1
                for k_ref, k_pred in pairs:
                    lo,  hi  = _norm(rel[k_ref]).split("..")
                    lo_, hi_ = _norm(rel_[k_pred]).split("..")
                    if lo == lo_: s += 1
                    if hi == hi_: s += 1
                return s

            score = _score_alignment([(k, k) for k in keys])
            if len(keys) == 2:
                # Also try swapped: cardinality of A in ref matched against B in pred
                swapped = [(keys[0], keys[1]), (keys[1], keys[0])]
                score = max(score, _score_alignment(swapped))
            return score

        used_predicted: set[int] = set()
        matched_correct = 0
        for rel in correct:
            # Find the best-scoring available predicted relation with the same endpoints.
            best_score = 0
            best_idx = -1
            for i, rel_ in enumerate(predicted):
                if i in used_predicted:
                    continue
                if set(rel.keys()) == set(rel_.keys()):
                    s = _rel_score(rel, rel_)
                    if s > best_score:
                        best_score = s
                        best_idx = i
            if best_idx >= 0:
                matched_correct += 1
                used_predicted.add(best_idx)
            total_score += best_score
        if total_score > len(correct) * 5:
            total_score = len(correct) * 5
        if matched_correct > 0:
            precision = matched_correct / tp_fp if tp_fp > 0 else 0.0
            recall = matched_correct / tp_fn if tp_fn > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            precision = recall = f1 = 0.0
        return {
            "total_score": total_score,
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1": round(f1, 2),
            "len": len(correct) * 5,
            "len_": len(correct),
            "n_pred": tp_fp,
        }
    except Exception:
        return {"total_score": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "len": len(correct) * 5, "len_": len(correct), "n_pred": 0}


def check_class(predicted, correct, path_to_check=""):
    pred_set = set(predicted) if not isinstance(predicted, set) else predicted
    corr_set = set(correct) if not isinstance(correct, set) else correct
    if not corr_set and not pred_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "len_": 0, "n_pred": 0}
    if not pred_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "len_": len(corr_set), "n_pred": 0}
    precision = len(pred_set & corr_set) / len(pred_set)
    recall = len(pred_set & corr_set) / len(corr_set) if corr_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 2), "recall": round(recall, 2),
            "f1": round(f1, 2), "len_": len(corr_set), "n_pred": len(pred_set)}


def check_class_syn(predicted, correct, path_to_check=""):
    try:
        from nltk.corpus import wordnet
    except Exception:
        return check_class(predicted, correct)

    def get_syn_list(word):
        try:
            synonyms = wordnet.synsets(word)
            return list(set(chain.from_iterable([w.lemma_names() for w in synonyms])))
        except Exception:
            return []

    intr = []
    for cl in predicted:
        for cl_ in correct:
            if cl == cl_ or cl in get_syn_list(cl_):
                intr.append(cl_)
    try:
        precision = len(intr) / len(predicted)
        recall = len(intr) / len(correct)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"precision": round(precision, 2), "recall": round(recall, 2),
                "f1": round(f1, 2), "len_": len(correct)}
    except Exception:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "len_": len(correct)}


def _parse_class_alias_pairs(res: str, predicted: list, correct: list) -> dict:
    """Parse ``Correct -> Predicted`` lines into ``{predicted_name: correct_name}``.

    Keys are normalised to the *exact* strings from ``predicted`` (not the LLM's
    possibly-reworded echo of them) so the map can be used to rewrite relation
    endpoint names, which come from the same parsed predicted-class strings.
    """
    pred_lookup = {p.lower(): p for p in predicted}
    corr_lookup = {c.lower(): c for c in correct}
    alias_map: dict = {}
    for line in res.splitlines():
        if "->" not in line:
            continue
        left, right = line.split("->", 1)
        left = left.strip().strip("\"'-*• ")
        right = right.strip().strip("\"'-*• ")
        corr_name = corr_lookup.get(left.lower())
        pred_name = pred_lookup.get(right.lower())
        if corr_name and pred_name:
            alias_map[pred_name] = corr_name
    return alias_map


def check_class_llm(predicted, correct, path_to_check):
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_openai import ChatOpenAI
        from langchain_core.output_parsers import StrOutputParser
    except ImportError:
        return check_class(predicted, correct)

    prompt = PromptTemplate.from_template(
        """You will be asked by the user to compare two lists of class names extracted from a text specification.
You will receive the specification text in input and two lists of class names: the predicted classes and the correct classes.
For every class in the correct list that refers to the same real-world concept as a class in the predicted list
(even if the name differs, e.g. synonyms or paraphrases), output one line in the form:

CorrectClassName -> PredictedClassName

Use the context from the specification to judge whether two differently-named classes refer to the same concept.
Match each correct class to at most one predicted class (its best match). Omit correct classes with no matching
predicted class. Do not invent class names that are not in either list.

For example, if the correct class list is ["Person"] and the predicted class list is ["Human"], output:
Person -> Human

Output only the pairs, one per line, without further explanation. If there are no matches, output nothing.

##############

The specification text is:

{text}

The predicted list is:

{predicted}

The correct list is:

{correct}

##############

The pairs are:
"""
    )
    try:
        model = ChatOpenAI(model="gpt-4o-mini")
        chain = prompt | model | StrOutputParser()
        spec_path = "/".join(path_to_check.split("/")[:-1]) + "/description.md"
        res = chain.invoke({"text": open(spec_path).read(), "predicted": predicted, "correct": correct})
        alias_map = _parse_class_alias_pairs(res, predicted, correct)
        intr = set(alias_map.values())
        tp = min(len(intr), len(predicted))  # LLM can hallucinate extras; cap TP at |predicted|
        precision = tp / len(predicted) if predicted else 0.0
        recall = len(intr) / len(correct) if correct else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"precision": round(precision, 2), "recall": round(recall, 2),
                "f1": round(f1, 2), "len_": len(correct), "n_pred": len(predicted),
                "alias_map": alias_map}
    except Exception:
        return check_class(predicted, correct)


def check_attributes(predicted_attrs, correct_attrs, path_to_check=""):
    pred_set = set((c.lower(), a.lower()) for c, a in predicted_attrs)
    corr_set = set((c.lower(), a.lower()) for c, a in correct_attrs)
    if not corr_set and not pred_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "len_": 0, "n_pred": 0}
    if not pred_set or not corr_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "len_": len(corr_set), "n_pred": len(pred_set)}
    tp = len(pred_set & corr_set)
    precision = tp / len(pred_set)
    recall = tp / len(corr_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 2), "recall": round(recall, 2),
            "f1": round(f1, 2), "len_": len(corr_set), "n_pred": len(pred_set)}


def check_attributes_syn(predicted_attrs, correct_attrs, path_to_check=""):
    try:
        from nltk.corpus import wordnet
    except Exception:
        return check_attributes(predicted_attrs, correct_attrs)

    def get_syn_list(word):
        try:
            synonyms = wordnet.synsets(word)
            return list(set(chain.from_iterable([w.lemma_names() for w in synonyms])))
        except Exception:
            return []

    def attr_name(a):
        return a.split(":")[0].lower() if ":" in a else a.lower()

    if not correct_attrs and not predicted_attrs:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "len_": 0}
    if not predicted_attrs or not correct_attrs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "len_": len(correct_attrs)}

    try:
        intr = []
        used_predicted: set[int] = set()
        for cls_c, attr_c in correct_attrs:
            for i_p, (cls_p, attr_p) in enumerate(predicted_attrs):
                if i_p in used_predicted:
                    continue
                cls_c_l, cls_p_l = cls_c.lower(), cls_p.lower()
                class_match = cls_c_l == cls_p_l or cls_p_l in get_syn_list(cls_c_l)
                if not class_match:
                    continue
                an_c, an_p = attr_name(attr_c), attr_name(attr_p)
                if an_c == an_p or an_p in get_syn_list(an_c):
                    intr.append((cls_c, attr_c))
                    used_predicted.add(i_p)
                    break

        precision = len(intr) / len(predicted_attrs) if predicted_attrs else 0.0
        recall = len(intr) / len(correct_attrs) if correct_attrs else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"precision": round(precision, 2), "recall": round(recall, 2),
                "f1": round(f1, 2), "len_": len(correct_attrs)}
    except Exception:
        return check_attributes(predicted_attrs, correct_attrs)


def check_attributes_llm(predicted_attrs, correct_attrs, path_to_check=""):
    try:
        from openai import OpenAI
    except ImportError:
        return check_attributes(predicted_attrs, correct_attrs)

    if not correct_attrs and not predicted_attrs:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "len_": 0}
    if not predicted_attrs or not correct_attrs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "len_": len(correct_attrs)}

    def fmt(attrs):
        return [f"{c}.{a}" for c, a in attrs]

    spec_path = "/".join(path_to_check.split("/")[:-1]) + "/description.md"
    spec_text = open(spec_path).read() if os.path.exists(spec_path) else ""

    prompt = (
        'You will compare two lists of UML class attributes. Each attribute is formatted as "ClassName.attributeName:Type".\n'
        "Find which attributes from the correct list are also present in the predicted list, "
        "considering that similar names and types are acceptable.\n"
        "Include only attributes from the correct list in the intersection.\n"
        "Use the specification context to judge semantic equivalence.\n\n"
        "Output only the intersection as a Python list of strings, "
        'e.g. ["ClassName.attr:Type", ...], without explanation.\n\n'
        "##############\n\n"
        f"Specification:\n{spec_text}\n\n"
        f"Predicted attributes:\n{fmt(predicted_attrs)}\n\n"
        f"Correct attributes:\n{fmt(correct_attrs)}\n\n"
        "##############\n\n"
        "Intersection:"
    )
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        res = response.choices[0].message.content or ""
        intr = {
            x.strip()
            for x in res.replace("[", "").replace("]", "").replace('"', "").replace("'", "").split(",")
            if x.strip()
        }
        tp = min(len(intr), len(predicted_attrs))  # LLM can hallucinate extras; cap TP at |predicted|
        precision = tp / len(predicted_attrs) if predicted_attrs else 0.0
        recall = len(intr) / len(correct_attrs) if correct_attrs else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"precision": round(precision, 2), "recall": round(recall, 2),
                "f1": round(f1, 2), "len_": len(correct_attrs)}
    except Exception:
        return check_attributes(predicted_attrs, correct_attrs)


def check_inheritance(predicted_inh, correct_inh, path_to_check=""):
    pred_set = set((p.lower(), c.lower()) for p, c in predicted_inh)
    corr_set = set((p.lower(), c.lower()) for p, c in correct_inh)
    if not corr_set and not pred_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "len_": 0, "n_pred": 0}
    if not pred_set or not corr_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "len_": len(corr_set), "n_pred": len(pred_set)}
    tp = len(pred_set & corr_set)
    precision = tp / len(pred_set)
    recall = tp / len(corr_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 2), "recall": round(recall, 2),
            "f1": round(f1, 2), "len_": len(corr_set), "n_pred": len(pred_set)}


def _alias_relation_endpoints(rel_list: list, alias_map: dict) -> list:
    """Rewrite relation endpoint class names through a predicted->gold alias map.

    ``check_relations`` matches endpoints by exact string equality, so a
    predicted relation between differently-but-equivalently-named classes
    (e.g. "Line" vs. gold's "BusLine") would otherwise never match even
    though ``check_class_llm`` already established the two names refer to
    the same concept. A combined key ("ClassA, ClassB") can appear for
    n-ary association endpoints, so each comma-separated part is aliased
    individually.
    """
    if not alias_map:
        return rel_list
    aliased = []
    for rel in rel_list:
        new_rel = {}
        for key, card in rel.items():
            parts = [alias_map.get(p, p) for p in key.split(", ")]
            new_rel[", ".join(parts)] = card
        aliased.append(new_rel)
    return aliased


def evaluate(path_uml: str, path_to_check: str, parser: Lark, strip_cfg: dict | None = None):
    # Gold standard: never strip (it's a clean plantuml.txt file)
    class_uml, rel_uml, attrs_uml, inh_uml = parse_path(path_uml, parser, strip_cfg=None)
    # Candidate: apply stripping to remove prompt scaffolding if enabled
    class_check, rel_check, attrs_check, inh_check = parse_path(path_to_check, parser, strip_cfg=strip_cfg)
    res_cl = check_class_llm(class_check, class_uml, path_to_check=path_to_check)
    rel_check = _alias_relation_endpoints(rel_check, res_cl.get("alias_map", {}))
    res_re = check_relations(rel_check, rel_uml, path_to_check=path_to_check)
    res_at_naive = check_attributes(attrs_check, attrs_uml)
    res_at_syn = check_attributes_syn(attrs_check, attrs_uml)
    res_at_llm = check_attributes_llm(attrs_check, attrs_uml, path_to_check=path_to_check)
    res_in = check_inheritance(inh_check, inh_uml)
    return res_cl, res_re, res_at_naive, res_at_syn, res_at_llm, res_in


# ──────────────────────────────────────────────────────────────────────────────
# Worker process (must be top-level for pickling)
# ──────────────────────────────────────────────────────────────────────────────

_GRAMMAR_PATH: str = ""  # kept for reference / debugging
_IGNORE_LIST: set = set()
_STRIP_CFG: dict = {}
_PARSER: "Lark | None" = None  # initialised once per worker in _init_worker


def _init_worker(grammar_path: str, ignore_list: set, strip_cfg: dict):
    global _GRAMMAR_PATH, _IGNORE_LIST, _STRIP_CFG, _PARSER
    _GRAMMAR_PATH = grammar_path
    _IGNORE_LIST = ignore_list
    _STRIP_CFG = strip_cfg
    _PARSER = init_parser(grammar_path)  # parse grammar once per worker, not per task
    load_dotenv()  # make API keys available in worker processes


def _contains_uml(path: str, strip_cfg: dict) -> bool:
    """Return False if the file contains no PlantUML after prompt-stripping — i.e. plain text output."""
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        effective = get_strip_cfg_for_file(path, strip_cfg)
        if effective:
            content = strip_prompt_noise(content, effective)
        if "</think>" in content:
            content = content[content.rfind("</think>") + len("</think>"):]
        else:
            content = re.sub(r"<think>", "", content)
        content = content.replace("`", "").replace("plantuml", "").replace("plaintext", "")
        if content.count("@startuml") > 1:
            content = "@startuml" + content.rsplit("@startuml", 1)[1]
        return "@startuml" in content or "@enduml" in content
    except Exception:
        return True  # let the normal error path handle it


def process_result_file(args):
    sub_folder, sub_folder_path, filename = args
    model_name = "-".join(filename.replace(".txt", "").split("_")[1:]) if filename else "unknown"
    class_n = assoc_n = attr_n = inh_n = 0
    exception_occurred = False
    try:
        parser = _PARSER  # reuse the per-worker parser; avoids re-parsing grammar per task
        uml_file = os.path.join(sub_folder_path, "plantuml.txt")
        result_file = os.path.join(sub_folder_path, filename)
        if not _contains_uml(result_file, _STRIP_CFG):
            return ("__SKIPPED__", sub_folder, model_name)  # plain-text output
        class_n, assoc_n, attr_n, inh_n = get_class_and_assoc_size(uml_file, parser)
        results = evaluate(uml_file, result_file, parser, strip_cfg=_STRIP_CFG or None)
        return [
            sub_folder, model_name,
            results[0]["precision"], results[0]["recall"], results[0]["f1"], class_n, results[0].get("n_pred", 0),
            results[1]["precision"], results[1]["recall"], results[1]["f1"],
            results[1]["total_score"], results[1]["len"], assoc_n, results[1].get("n_pred", 0),
            results[2]["precision"], results[2]["recall"], results[2]["f1"], attr_n, results[2].get("n_pred", 0),
            results[3]["precision"], results[3]["recall"], results[3]["f1"],
            results[4]["precision"], results[4]["recall"], results[4]["f1"],
            results[5]["precision"], results[5]["recall"], results[5]["f1"], inh_n, results[5].get("n_pred", 0),
            exception_occurred,
        ]
    except Exception as e:
        exception_occurred = True
        try:
            with open("trace.txt", "a", encoding="utf-8") as t:
                t.write(f"[{sub_folder}/{filename}] {type(e).__name__}: {e}\n")
        except Exception:
            pass
        return [
            sub_folder, model_name,
            0.0, 0.0, 0.0, class_n, 0,
            0.0, 0.0, 0.0, 0.0, 0, assoc_n, 0,
            0.0, 0.0, 0.0, attr_n, 0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, inh_n, 0,
            exception_occurred,
        ]


_CSV_HEADER = [
    "sub_folder_name", "model_name",
    "precision_class", "recall_class", "f1_class", "n_classes", "n_pred_class",
    "precision_rel", "recall_rel", "f1_rel", "score_rel", "max_score", "n_rel", "n_pred_rel",
    "precision_attr_naive", "recall_attr_naive", "f1_attr_naive", "n_attrs", "n_pred_attrs",
    "precision_attr_syn", "recall_attr_syn", "f1_attr_syn",
    "precision_attr_llm", "recall_attr_llm", "f1_attr_llm",
    "precision_inh", "recall_inh", "f1_inh", "n_inh", "n_pred_inh",
    "exception_occurred",
]


def _make_timeout_row(args):
    sub_folder, sub_folder_path, filename = args
    model_name = "-".join(filename.replace(".txt", "").split("_")[1:]) if filename else "unknown"
    return [sub_folder, model_name,
            0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0,
            0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, True]


def analyze_crashes(results):
    total_counts = defaultdict(lambda: defaultdict(int))
    exception_counts = defaultdict(lambda: defaultdict(int))
    for entry in results:
        sub_folder, model_name, *metrics, exception_occurred = entry
        technique = next(
            (t for t in ["cot-", "cot2-", "one-", "few-"] if model_name.startswith(t)), "zero"
        )
        total_counts[technique][model_name] += 1
        if exception_occurred:
            exception_counts[technique][model_name] += 1
    crash_percentages = {
        tech: {m: (exception_counts[tech][m] / total_counts[tech][m]) * 100
               for m in total_counts[tech]}
        for tech in total_counts
    }
    exception_counts_summary = {tech: dict(exception_counts[tech]) for tech in exception_counts}
    return crash_percentages, exception_counts_summary


def process_folders(
    root_folder: str,
    csv_file: str,
    grammar_path: str,
    ignore_list: set,
    strip_cfg: dict | None = None,
    max_workers: int = 32,
    stall_timeout_s: float = 300,
    poll_interval_s: float = 1.0,
    prefixes: dict | None = None,
):
    # Suffix appended to stripped copies (e.g. "_stripped") – must be excluded
    # from evaluation because they are intermediate artefacts, not model outputs.
    stripped_suffix = (strip_cfg or {}).get("stripped_suffix", "_stripped")

    tasks = []
    for sub_folder in os.listdir(root_folder):
        sub_folder_path = os.path.join(root_folder, sub_folder)
        if not os.path.isdir(sub_folder_path):
            continue
        uml_file = os.path.join(sub_folder_path, "plantuml.txt")
        if not os.path.exists(uml_file):
            continue
        for filename in os.listdir(sub_folder_path):
            if not (filename.startswith("result_") and filename.endswith(".txt")):
                continue
            # Skip stripped copies written by the strip-prompt step
            if filename.endswith(f"{stripped_suffix}.txt"):
                continue
            model_name = "-".join(filename.replace(".txt", "").split("_")[1:])
            # Strip any technique prefix to get the base model name for the
            # ignore-list check (e.g. "few-deepseek-reasoner" → "deepseek-reasoner")
            base_model = model_name
            for pfx in (prefixes or {}).keys():
                if model_name.startswith(f"{pfx}-"):
                    base_model = model_name[len(pfx) + 1:]
                    break
            if model_name in ignore_list or base_model in ignore_list:
                continue
            tasks.append((sub_folder, sub_folder_path, filename))

    # Write CSV header
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(_CSV_HEADER)

    results = []
    skipped = []   # ("__SKIPPED__", sub_folder, model_name) entries
    if not tasks:
        return analyze_crashes(results), skipped, tasks

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(grammar_path, ignore_list, strip_cfg or {}),
    ) as ex, tqdm(total=len(tasks), dynamic_ncols=True) as pbar:

        fut_to_args = {}
        start_times = {}
        warned = set()
        task_iter = iter(tasks)

        # Submit only the first batch so that start_times reflects when each
        # task actually begins executing rather than when it was enqueued.
        # New tasks are submitted one-for-one as workers complete, keeping the
        # pool full without pre-loading the entire task list.
        for args in islice(task_iter, max_workers):
            fut = ex.submit(process_result_file, args)
            fut_to_args[fut] = args
            start_times[fut] = time.time()

        pending = set(fut_to_args)
        while pending:
            done, not_done = wait(pending, timeout=poll_interval_s, return_when=FIRST_COMPLETED)
            for fut in done:
                args = fut_to_args[fut]
                pending.remove(fut)
                try:
                    res = fut.result()
                except Exception as e:
                    sf, _, fn = args
                    with open("trace.txt", "a", encoding="utf-8") as t:
                        t.write(f"[EXC] {sf}/{fn}: {type(e).__name__}: {e}\n")
                        t.write("".join(traceback.format_exception(type(e), e, e.__traceback__)) + "\n")
                    res = _make_timeout_row(args)
                if isinstance(res, tuple) and res[0] == "__SKIPPED__":
                    skipped.append(res)
                elif res is not None:
                    results.append(res)
                    if len(results) % 100 == 0:
                        with open(csv_file, "a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerows(results[-100:])
                pbar.update(1)

                # Keep the pool full: submit next task as soon as one finishes
                next_args = next(task_iter, None)
                if next_args is not None:
                    new_fut = ex.submit(process_result_file, next_args)
                    fut_to_args[new_fut] = next_args
                    start_times[new_fut] = time.time()
                    pending.add(new_fut)

            now = time.time()
            for fut in list(not_done):
                elapsed = now - start_times.get(fut, now)
                if elapsed > stall_timeout_s and fut not in warned:
                    warned.add(fut)
                    sf, _, fn = fut_to_args[fut]
                    with open("trace.txt", "a", encoding="utf-8") as t:
                        t.write(f"[TIMEOUT] >{stall_timeout_s}s: {sf}/{fn}\n")
                    fut.cancel()
                    if fut in pending:
                        pending.remove(fut)
                        row = _make_timeout_row(fut_to_args[fut])
                        results.append(row)
                        with open(csv_file, "a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerows([row])
                        pbar.update(1)
                        # Cancelled future freed a worker slot — fill it
                        next_args = next(task_iter, None)
                        if next_args is not None:
                            new_fut = ex.submit(process_result_file, next_args)
                            fut_to_args[new_fut] = next_args
                            start_times[new_fut] = time.time()
                            pending.add(new_fut)

    # Final write
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_CSV_HEADER)
        w.writerows(results)

    return analyze_crashes(results), skipped, tasks


# ──────────────────────────────────────────────────────────────────────────────
# Plot functions  (all save to graph_dir)
# ──────────────────────────────────────────────────────────────────────────────

def _apply_extract(df: pd.DataFrame, extract, col_names=("base_model_name", "prefix")) -> pd.DataFrame:
    """Apply the extract function to model_name and append the result columns.

    Using pd.concat instead of direct multi-column assignment avoids a
    ValueError when df is empty (apply returns shape (0,0) on an empty Series,
    which cannot be unpacked into named columns via __setitem__).
    """
    extracted = df["model_name"].apply(lambda x: pd.Series(extract(x)))
    extracted.columns = list(col_names)
    return pd.concat([df, extracted], axis=1)


_ZERO_METRIC_COLS = ["f1_class", "f1_rel", "score_rel", "f1_attr_llm", "f1_inh"]


def _drop_all_zero_models(df: pd.DataFrame) -> pd.DataFrame:
    """Drop base_model_name groups where every metric is 0 across all their rows."""
    if "base_model_name" not in df.columns:
        return df
    metric_cols = [c for c in _ZERO_METRIC_COLS if c in df.columns]
    if not metric_cols:
        return df
    # Keep only models that have at least one non-zero value in any metric
    nonzero_mask = df.groupby("base_model_name")[metric_cols].max().gt(0).any(axis=1)
    keep = nonzero_mask[nonzero_mask].index
    dropped = set(df["base_model_name"].unique()) - set(keep)
    if dropped:
        print(f"  [filter] dropping all-zero models: {sorted(dropped)}")
    return df[df["base_model_name"].isin(keep)]


def _clean_df(df: pd.DataFrame, extract) -> pd.DataFrame:
    df = df.copy()
    df["model_name"] = df["model_name"].str.replace("mlx-community-", "", regex=False)
    df = _apply_extract(df, extract, col_names=("base_model_name", "result_type"))
    return _drop_all_zero_models(df)


def _barplot_a4(data: pd.DataFrame, x: str, y: str, hue: str,
                hue_order: list, palette: dict,
                title: str, ylabel: str,
                filename: str, graph_dir: str, show: bool, dpi: int,
                ylim: tuple = (0.0, 1.05)) -> None:
    """Shared bar-chart renderer for all plot_data_* functions.

    A4-landscape width (9.5 in), vertical x-labels, single-column legend
    outside the right edge — mirrors the style of plot_crash_data.
    """
    fig, ax = plt.subplots(figsize=(9.5, 3.5))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=data, x=x, y=y, hue=hue, hue_order=hue_order,
                palette=palette, ax=ax)
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(*ylim)
    ax.tick_params(axis="y", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.legend(title="Technique", title_fontsize=8, fontsize=8,
              loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0,
              frameon=True, framealpha=0.9)
    fig.subplots_adjust(left=0.07, right=0.78, top=0.90, bottom=0.32)
    save_fig(None, filename, graph_dir, show, dpi)


_ORDERED_TECHNIQUES = ["zero-shot", "CoT", "CoT-multi", "one-shot", "two-shots"]

# Palette order matches the Eval.ipynb notebook (Paired, same colour assignment)
_PALETTE_COLOR_ORDER = ["one-shot", "two-shots", "CoT", "CoT-multi", "zero-shot"]
_TECHNIQUE_PALETTE: dict = {}   # populated lazily on first call


def _build_full_palette():
    """Build a Paired palette in the same order as the Eval.ipynb notebook."""
    colors = sns.color_palette("Paired", len(_PALETTE_COLOR_ORDER))
    return {t: colors[i] for i, t in enumerate(_PALETTE_COLOR_ORDER)}


def _technique_palette(techniques) -> tuple[dict, list]:
    """
    Return (palette_dict, hue_order) guaranteed to contain every value that
    will appear in the hue column.

    - palette_dict  covers ALL known techniques plus any extras in `techniques`
    - hue_order     lists only the techniques actually present, in canonical order
    """
    global _TECHNIQUE_PALETTE
    if not _TECHNIQUE_PALETTE:
        _TECHNIQUE_PALETTE = _build_full_palette()

    # Normalise to plain strings, drop NaN / 'nan'
    present = sorted(
        {str(t) for t in techniques if t is not None and str(t) not in ("nan", "None", "")},
        key=lambda t: (_ORDERED_TECHNIQUES.index(t) if t in _ORDERED_TECHNIQUES else 999, t),
    )

    # Extend global palette with any unexpected technique names
    extra_colors = sns.color_palette("Set2", max(len(present), 1))
    palette = dict(_TECHNIQUE_PALETTE)
    for i, t in enumerate(present):
        if t not in palette:
            palette[t] = extra_colors[i % len(extra_colors)]

    return palette, present


def plot_crash_data(csv_file: str, extract, graph_dir: str, show: bool, dpi: int):
    df = pd.read_csv(csv_file)
    df["model_name"] = df["model_name"].str.replace("mlx-community-", "", regex=False)
    df[["base_model", "technique"]] = df["model_name"].apply(lambda x: pd.Series(extract(x)))

    total_counts = df.groupby(["technique", "base_model"]).size().reset_index(name="total")
    exception_counts = df.groupby(["technique", "base_model"])["exception_occurred"].sum().reset_index()
    crash_data = total_counts.merge(exception_counts, on=["technique", "base_model"], how="left")
    crash_data["exception_percentage"] = (
        crash_data["exception_occurred"].fillna(0) / crash_data["total"] * 100
    )

    high_error_models = (
        crash_data.groupby("base_model")["exception_percentage"]
        .max()
        .loc[lambda s: s > 5]
        .index
    )
    crash_data = crash_data[crash_data["base_model"].isin(high_error_models)]

    ordered_techniques = ["zero-shot", "CoT", "CoT-multi", "one-shot", "two-shots"]
    crash_data["technique"] = pd.Categorical(
        crash_data["technique"], categories=ordered_techniques, ordered=True
    )

    # Use average error per model to colour bars (darker = higher average error rate)
    avg_error_by_model = crash_data.groupby("base_model")["exception_percentage"].mean()
    norm = plt.Normalize(avg_error_by_model.min(), avg_error_by_model.max())
    cmap = plt.cm.Reds
    palette = {m: cmap(norm(v)) for m, v in avg_error_by_model.items()}

    # A4 landscape printable width ≈ 9.5 in; reserve ~2.8 in on the right for the legend
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    sns.set_theme(style="whitegrid")
    barplot = sns.barplot(
        data=crash_data, x="technique", y="exception_percentage", hue="base_model",
        dodge=True, width=0.85, palette=palette, ax=ax
    )
    ax.set_xlabel("Technique", fontsize=9)
    ax.set_ylabel("Exception %", fontsize=9)
    ax.set_title("Exception Rate per Model & Technique (>5%) — darker = higher avg error",
                 fontsize=10, pad=6)
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=8)

    handles, labels = barplot.get_legend_handles_labels()
    seen = []
    seen_h = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.append(l)
            seen_h.append(h)
    sorted_models = sorted(seen, key=lambda m: avg_error_by_model.get(m, 0), reverse=True)
    sorted_handles = [seen_h[seen.index(m)] for m in sorted_models]
    sorted_labels = [f"{m}  ({avg_error_by_model.get(m, 0):.0f}%)" for m in sorted_models]

    # Single-column legend outside the right edge — no bottom clutter
    ax.legend(sorted_handles, sorted_labels,
              title="Model (avg error)", title_fontsize=7,
              fontsize=7, frameon=True, framealpha=0.9,
              loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)

    # left: room for y-label; right: room for legend; bottom/top: tight
    fig.subplots_adjust(left=0.07, right=0.66, top=0.90, bottom=0.10)
    save_fig(None, "error_dist_gradient.svg", graph_dir, show, dpi)


def plot_data_class(csv_file: str, extract, graph_dir: str, show: bool, dpi: int):
    df = _clean_df(pd.read_csv(csv_file), extract)
    avg = df.groupby(["base_model_name", "result_type"]).apply(
        lambda g: (g["f1_class"] * g["n_classes"]).sum() / g["n_classes"].sum()
        if g["n_classes"].sum() > 0 else np.nan, include_groups=False
    ).reset_index(name="f1_class_avg")
    color_map, hue_order = _technique_palette(avg["result_type"].unique())
    _barplot_a4(avg, "base_model_name", "f1_class_avg", "result_type",
                hue_order, color_map,
                "Weighted F1 Class per Model", "Weighted F1 Class",
                "data_class.svg", graph_dir, show, dpi)


def plot_data_rel(csv_file: str, extract, graph_dir: str, show: bool, dpi: int):
    df = _clean_df(pd.read_csv(csv_file), extract)
    avg = df.groupby(["base_model_name", "result_type"]).apply(
        lambda g: (g["f1_rel"] * g["n_rel"]).sum() / g["n_rel"].sum()
        if g["n_rel"].sum() > 0 else np.nan, include_groups=False
    ).reset_index(name="f1_rel_avg")
    color_map, hue_order = _technique_palette(avg["result_type"].unique())
    _barplot_a4(avg, "base_model_name", "f1_rel_avg", "result_type",
                hue_order, color_map,
                "Weighted F1 Association per Model", "Weighted F1 Assoc.",
                "data_rel.svg", graph_dir, show, dpi)


def plot_data_score_perc(csv_file: str, extract, graph_dir: str, show: bool, dpi: int):
    df = _clean_df(pd.read_csv(csv_file), extract)
    avg = df.groupby(["base_model_name", "result_type"]).agg(
        score_sum=("score_rel", "sum"), max_score_sum=("max_score", "sum")
    ).reset_index()
    avg["score_relative"] = avg["score_sum"] / avg["max_score_sum"]
    color_map, hue_order = _technique_palette(avg["result_type"].unique())
    _barplot_a4(avg, "base_model_name", "score_relative", "result_type",
                hue_order, color_map,
                "Relative Cardinality Score per Model", "Score / Max Score",
                "data_score_relative.svg", graph_dir, show, dpi)


def plot_data_attr(csv_file: str, extract, graph_dir: str, show: bool, dpi: int):
    df = _clean_df(pd.read_csv(csv_file), extract)
    avg = df.groupby(["base_model_name", "result_type"]).apply(
        lambda g: (g["f1_attr_llm"] * g["n_attrs"]).sum() / g["n_attrs"].sum()
        if g["n_attrs"].sum() > 0 else np.nan, include_groups=False
    ).reset_index(name="f1_attr_avg")
    color_map, hue_order = _technique_palette(avg["result_type"].unique())
    _barplot_a4(avg, "base_model_name", "f1_attr_avg", "result_type",
                hue_order, color_map,
                "Weighted F1 Attributes per Model", "Weighted F1 Attr.",
                "data_attr.svg", graph_dir, show, dpi)


def plot_aggregated_f1(csv_file: str, extract, graph_dir: str, show: bool, dpi: int):
    df = _clean_df(pd.read_csv(csv_file), extract)
    _grp = ["base_model_name", "model_name", "result_type"]
    avg_class = df.groupby(_grp).apply(
        lambda g: (g["f1_class"] * g["n_classes"]).sum() / g["n_classes"].sum()
        if g["n_classes"].sum() > 0 else np.nan, include_groups=False,
    ).reset_index(name="f1_class_avg")
    avg_rel = df.groupby(_grp).apply(
        lambda g: (g["f1_rel"] * g["n_rel"]).sum() / g["n_rel"].sum()
        if g["n_rel"].sum() > 0 else np.nan, include_groups=False,
    ).reset_index(name="f1_rel_avg")
    avg_score = df.groupby(_grp).agg(
        score_avg=("score_rel", "sum"), max_score=("max_score", "sum"), count=("model_name", "count"),
    ).reset_index()
    avg_score["score_relative"] = avg_score["score_avg"] / avg_score["max_score"]
    avg = avg_class.merge(avg_rel, on=_grp).merge(avg_score, on=_grp)

    avg_attr = df.groupby(_grp).apply(
        lambda g: (g["f1_attr_llm"] * g["n_attrs"]).sum() / g["n_attrs"].sum() if g["n_attrs"].sum() > 0 else 0,
        include_groups=False,
    ).reset_index(name="f1_attr_avg")
    avg = avg.merge(avg_attr, on=_grp, how="left")
    avg["f1_attr_avg"] = avg["f1_attr_avg"].fillna(0)

    avg_inh = df.groupby(_grp).apply(
        lambda g: (g["f1_inh"] * g["n_inh"]).sum() / g["n_inh"].sum() if g["n_inh"].sum() > 0 else 0,
        include_groups=False,
    ).reset_index(name="f1_inh_avg")
    avg = avg.merge(avg_inh, on=_grp, how="left")
    avg["f1_inh_avg"] = avg["f1_inh_avg"].fillna(0)

    avg["f1_aggregated"] = (
        avg["f1_class_avg"] + avg["f1_rel_avg"] + avg["score_relative"]
        + avg["f1_attr_avg"] + avg["f1_inh_avg"]
    ) / 5

    color_map, hue_order = _technique_palette(avg["result_type"].unique())
    _barplot_a4(avg, "base_model_name", "f1_aggregated", "result_type",
                hue_order, color_map,
                "Aggregated F1 Score per Model", "Aggregated F1",
                "data_f1_aggregated.svg", graph_dir, show, dpi)

    latex_table = avg[["base_model_name", "result_type", "f1_class_avg", "f1_rel_avg",
                        "score_relative", "f1_aggregated"]].to_latex(index=False, float_format="%.3f")
    with open(os.path.join(graph_dir, "aggregated_f1_table.tex"), "w") as f:
        f.write(latex_table)


def plot_data_score(csv_file: str, extract, graph_dir: str, show: bool, dpi: int):
    df = _clean_df(pd.read_csv(csv_file), extract)
    avg = df.groupby(["base_model_name", "result_type"]).agg(
        score_avg=("score_rel", "mean")
    ).reset_index()
    color_map, hue_order = _technique_palette(avg["result_type"].unique())
    _barplot_a4(avg, "base_model_name", "score_avg", "result_type",
                hue_order, color_map,
                "Average Raw Cardinality Score per Model", "Avg. Score",
                "data_score.svg", graph_dir, show, dpi,
                ylim=(0.0, avg["score_avg"].max() * 1.15 + 0.01))


def plot_data_class_pc(csv_file: str, extract, graph_dir: str, show: bool, dpi: int):
    df = _clean_df(pd.read_csv(csv_file), extract)
    avg = df.groupby(["sub_folder_name", "base_model_name", "model_name", "result_type"]).agg(
        f1_class_avg=("f1_class", "mean"), count=("model_name", "count"),
        n_classes=("n_classes", "first")
    ).reset_index()
    sub_order = avg.groupby("sub_folder_name")["f1_class_avg"].mean().sort_values().index
    label_map = {r["sub_folder_name"]: f"{r['sub_folder_name']} ({r['n_classes']})"
                 for _, r in avg.iterrows()}
    avg["sub_folder_label"] = avg["sub_folder_name"].map(label_map)

    plt.figure(figsize=(12, 6))
    sns.barplot(x="sub_folder_label", y="f1_class_avg", hue="result_type", data=avg,
                palette="husl", order=[label_map[s] for s in sub_order])
    plt.title("Average F1 Classes per Case")
    plt.xlabel("Model Name (Number of Classes)")
    plt.ylabel("Average F1 Classes")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Result Type")
    plt.tight_layout()
    save_fig(None, "data_class_pc.svg", graph_dir, show, dpi)


def plot_data_rel_pc(csv_file: str, extract, graph_dir: str, show: bool, dpi: int):
    df = _clean_df(pd.read_csv(csv_file), extract)
    avg = df.groupby(["sub_folder_name", "base_model_name", "model_name", "result_type"]).agg(
        f1_rel_avg=("f1_rel", "mean"), count=("model_name", "count"),
        n_classes=("n_rel", "first")
    ).reset_index()
    sub_order = avg.groupby("sub_folder_name")["f1_rel_avg"].mean().sort_values().index
    label_map = {r["sub_folder_name"]: f"{r['sub_folder_name']} ({r['n_classes']})"
                 for _, r in avg.iterrows()}
    avg["sub_folder_label"] = avg["sub_folder_name"].map(label_map)

    plt.figure(figsize=(12, 6))
    sns.barplot(x="sub_folder_label", y="f1_rel_avg", hue="result_type", data=avg,
                palette="husl", order=[label_map[s] for s in sub_order])
    plt.title("Average F1 Association per Case")
    plt.xlabel("Model Name (Number of Associations)")
    plt.ylabel("Average F1 Assoc")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Result Type")
    plt.tight_layout()
    save_fig(None, "data_rel_pc.svg", graph_dir, show, dpi)


def plot_data_score_pc(csv_file: str, extract, graph_dir: str, show: bool, dpi: int):
    df = _clean_df(pd.read_csv(csv_file), extract)
    avg = df.groupby(["sub_folder_name", "base_model_name", "model_name", "result_type"]).agg(
        f1_rel_avg=("score_rel", "mean"), count=("model_name", "count"),
        n_classes=("max_score", "first")
    ).reset_index()
    sub_order = avg.groupby("sub_folder_name")["f1_rel_avg"].mean().sort_values().index
    label_map = {r["sub_folder_name"]: f"{r['sub_folder_name']} ({r['n_classes']})"
                 for _, r in avg.iterrows()}
    avg["sub_folder_label"] = avg["sub_folder_name"].map(label_map)

    plt.figure(figsize=(12, 6))
    sns.barplot(x="sub_folder_label", y="f1_rel_avg", hue="result_type", data=avg,
                palette="husl", order=[label_map[s] for s in sub_order])
    plt.title("Average Score per Case")
    plt.xlabel("Model Name (Total score)")
    plt.ylabel("Average Score")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Result Type")
    plt.tight_layout()
    save_fig(None, "data_score_pc.svg", graph_dir, show, dpi)


def plot_distributions_bubble_sizes(csv_file: str, graph_dir: str, show: bool, dpi: int):
    df = pd.read_csv(csv_file)
    df_filtered = df[(df["n_classes"] > 0) & (df["n_rel"] > 0)]
    df_unique = df_filtered.groupby("sub_folder_name").first().reset_index()
    df_counts = df_unique.groupby(["n_classes", "n_rel"]).size().reset_index(name="count")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 4))
    ax = plt.gca()
    ax.scatter(
        x=df_counts["n_classes"], y=df_counts["n_rel"],
        s=df_counts["count"] * 200, alpha=0.7, edgecolor="k"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    x_ticks = np.sort(df_counts["n_classes"].unique())
    y_ticks = np.sort(df_counts["n_rel"].unique())
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)
    ax.set_ylim(bottom=0.5 * 1.1, top=ax.get_ylim()[1] * 1.1)
    plt.xlabel("Number of Classes (Log Scale)")
    plt.ylabel("Number of Associations (Log Scale)")
    plt.title("Joint Distribution of Classes and Associations (Bubble Chart)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    save_fig(None, "dist2_bubble.svg", graph_dir, show, dpi)


def plot_distributions_no_aggregation_size_by_overlap(
    csv_file: str, graph_dir: str, show: bool, dpi: int,
    base_size: int = 80, bubble_color: str = "#3b82f6", alpha: float = 0.7
):
    from matplotlib.ticker import LogLocator, FuncFormatter

    df = pd.read_csv(csv_file)
    for col in ["n_classes", "n_rel"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df_unique = (
        df[(df["n_classes"] > 0) & (df["n_rel"] > 0)]
        .groupby("sub_folder_name", as_index=False)
        .first()
    )
    cell_counts = df_unique.groupby(["n_classes", "n_rel"]).size().reset_index(name="count")
    count_map = {(int(r["n_classes"]), int(r["n_rel"])): int(r["count"]) for _, r in cell_counts.iterrows()}

    def size_for_count(c):
        return base_size * np.sqrt(c)

    sizes = [size_for_count(count_map.get((int(r["n_classes"]), int(r["n_rel"])), 1))
             for _, r in df_unique.iterrows()]

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.scatter(df_unique["n_classes"], df_unique["n_rel"],
               s=np.asarray(sizes, dtype=float), c=bubble_color, alpha=alpha,
               edgecolors="black", linewidths=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")

    def set_sparse_log_ticks(axis, values):
        vmin, vmax = float(np.min(values)), float(np.max(values))
        if vmax / max(vmin, 1e-12) < 32:
            locator = LogLocator(base=2.0)
        else:
            locator = LogLocator(base=10.0)
        fmt = FuncFormatter(lambda v, _: f"{int(v)}" if v >= 1 and float(v).is_integer() else "")
        axis.set_major_locator(locator)
        axis.set_major_formatter(fmt)

    set_sparse_log_ticks(ax.xaxis, df_unique["n_classes"].to_numpy())
    set_sparse_log_ticks(ax.yaxis, df_unique["n_rel"].to_numpy())
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.grid(False, which="minor")
    ax.set_xlabel("Number of Classes (log scale)")
    ax.set_ylabel("Number of Associations (log scale)")
    ax.set_title("Classes vs Associations — Individual Cases (size indicates overlaps)")

    example_counts = sorted(set([1, 2, 5, int(cell_counts["count"].max())]) & set(range(1, 1000)))
    handles = [plt.scatter([], [], s=size_for_count(c), edgecolors="black",
                           linewidths=0.5, alpha=alpha, c=bubble_color)
               for c in example_counts]
    ax.legend(handles, [f"×{c}" for c in example_counts], title="Cases at cell", loc="best")
    plt.tight_layout()
    save_fig(None, "dist_no_agg_size_by_overlap.svg", graph_dir, show, dpi)


def _sort_by_technique(avg: pd.DataFrame) -> pd.DataFrame:
    """Sort by canonical technique order using a temporary Categorical column."""
    ORDERED = ["zero-shot", "CoT", "CoT-multi", "one-shot", "two-shots"]
    avg = avg.copy()
    avg["_sort_key"] = pd.Categorical(avg["prefix"], categories=ORDERED, ordered=True)
    avg = avg.sort_values(["_sort_key", "base_model_name"]).drop(columns="_sort_key")
    return avg


def plot_bubble_f1_class_(csv_file: str, extract, selected_models: list,
                          graph_dir: str, show: bool, dpi: int):
    df = pd.read_csv(csv_file)
    df["model_name"] = df["model_name"].str.replace("mlx-community-", "", regex=False)
    df = _apply_extract(df, extract)
    df = df[df["base_model_name"].isin(selected_models)]

    avg = df.groupby(["prefix", "base_model_name", "sub_folder_name"]).agg(
        f1_class_avg=("f1_class", "mean"), n_classes=("n_classes", "first")
    ).reset_index()
    avg = _sort_by_technique(avg)
    avg["x_label"] = avg["prefix"] + " - " + avg["base_model_name"]
    color_map, hue_order = _technique_palette(avg["prefix"].unique())

    plt.figure(figsize=(14, 4))
    sns.scatterplot(data=avg, x="x_label", y="f1_class_avg", size="n_classes",
                    hue="prefix", hue_order=hue_order, palette=color_map, sizes=(50, 500),
                    edgecolor="black", alpha=0.7)
    plt.title("Average F1 Class Score per Model and Technique")
    plt.xlabel("Model Name and Technique")
    plt.ylabel("Average F1 Class Score")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Technique", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    save_fig(None, "bubble_chart_f1_class_.svg", graph_dir, show, dpi)


def plot_bubble_f1_rel_(csv_file: str, extract, selected_models: list,
                        graph_dir: str, show: bool, dpi: int):
    df = pd.read_csv(csv_file)
    df["model_name"] = df["model_name"].str.replace("mlx-community-", "", regex=False)
    df = _apply_extract(df, extract)
    df = df[df["base_model_name"].isin(selected_models)]

    avg = df.groupby(["prefix", "base_model_name", "sub_folder_name"]).agg(
        f1_rel_avg=("f1_rel", "mean"), n_rel=("n_rel", "first")
    ).reset_index()
    avg = _sort_by_technique(avg)
    avg["x_label"] = avg["prefix"] + " - " + avg["base_model_name"]
    color_map, hue_order = _technique_palette(avg["prefix"].unique())

    plt.figure(figsize=(14, 4))
    sns.scatterplot(data=avg, x="x_label", y="f1_rel_avg", size="n_rel",
                    hue="prefix", hue_order=hue_order, palette=color_map, sizes=(50, 500),
                    edgecolor="black", alpha=0.7)
    plt.title("Average F1 Association Score per Model and Technique")
    plt.xlabel("Model Name and Technique")
    plt.ylabel("Average F1 Association Score")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Technique", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    save_fig(None, "bubble_chart_f1_rel_.svg", graph_dir, show, dpi)


def plot_bubble_f1_score_(csv_file: str, extract, selected_models: list,
                          graph_dir: str, show: bool, dpi: int):
    df = pd.read_csv(csv_file)
    df["model_name"] = df["model_name"].str.replace("mlx-community-", "", regex=False)
    df = _apply_extract(df, extract)
    df = df[df["base_model_name"].isin(selected_models)]

    avg = df.groupby(["prefix", "base_model_name", "sub_folder_name"]).agg(
        score_avg=("score_rel", "mean"), max_score=("max_score", "first")
    ).reset_index()
    avg["f1_relative"] = avg["score_avg"] / avg["max_score"]
    avg = _sort_by_technique(avg)
    avg["x_label"] = avg["prefix"] + " - " + avg["base_model_name"]
    color_map, hue_order = _technique_palette(avg["prefix"].unique())

    plt.figure(figsize=(14, 4))
    sns.scatterplot(data=avg, x="x_label", y="f1_relative", size="max_score",
                    hue="prefix", hue_order=hue_order, palette=color_map, sizes=(50, 500),
                    edgecolor="black", alpha=0.7)
    plt.title("Relative Average Cardinality Score per Model and Technique")
    plt.xlabel("Model Name and Technique")
    plt.ylabel("Relative Average Card. Score (Percentage of Max)")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Technique", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    save_fig(None, "bubble_chart_f1_score_relative.svg", graph_dir, show, dpi)


def plot_bubble_f1_subfolders_(csv_file: str, extract, subfolder_model: str,
                               subfolder_technique: str, graph_dir: str, show: bool, dpi: int):
    from matplotlib.lines import Line2D

    df = pd.read_csv(csv_file)
    df["model_name"] = df["model_name"].str.replace("mlx-community-", "", regex=False)
    df = _apply_extract(df, extract)
    df = df[(df["base_model_name"] == subfolder_model) & (df["prefix"] == subfolder_technique)]

    if df.empty:
        print(f"  [warn] no data for {subfolder_model} / {subfolder_technique}, skipping subfolder plot")
        return

    # Order cases by total complexity (classes + relations + score) descending
    size_sums = (
        df.groupby("sub_folder_name")[["n_classes", "n_rel", "max_score"]]
        .sum().sum(axis=1).sort_values(ascending=False)
    )
    ordered_cases = list(size_sums.index)
    n = len(ordered_cases)

    df["score_rel_norm"] = df["score_rel"] / df["max_score"].replace(0, float("nan"))

    # A4 landscape printable width ≈ 9.5 in (247 mm with 25 mm margins).
    # Vertical labels + small font let all 45 case names fit without collision.
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    sns.set_theme(style="whitegrid")

    # Numeric x with per-series horizontal jitter so bubbles don't fully overlap.
    x_map = {c: i for i, c in enumerate(ordered_cases)}
    df["_x"] = df["sub_folder_name"].map(x_map)

    series_cfg = [
        ("f1_class",      "n_classes", "#4C72B0", "F1 Class",          -0.20),
        ("f1_rel",        "n_rel",     "#55A868", "F1 Association",     -0.07),
        ("score_rel_norm","max_score", "#C44E52", "Cardinality Score",   0.07),
        ("f1_attr_llm",   "n_attrs",   "#DD8452", "F1 Attributes",       0.20),
    ]

    for y_col, size_col, color, label, dx in series_cfg:
        vals = df[size_col].fillna(0)
        vmin, vmax = vals.min(), vals.max()
        # Bubble area normalised to [15, 180] pt² — smaller so they don't bleed
        s = 15 + (vals - vmin) / (vmax - vmin) * 165 if vmax > vmin else np.full(len(vals), 60)
        ax.scatter(df["_x"] + dx, df[y_col], s=s,
                   color=color, alpha=0.75, edgecolors="black", linewidths=0.4,
                   zorder=3, label=label)

    # Vertical x-axis labels: take no horizontal space, just vertical room
    ax.set_xticks(range(n))
    ax.set_xticklabels(ordered_cases, rotation=90, ha="center", fontsize=7)
    ax.set_xlim(-0.6, n - 0.4)

    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("")                          # label redundant — cases speak for themselves
    ax.set_ylabel("Score", fontsize=9)
    ax.set_title(f"Scores per Case — {subfolder_model}  ·  {subfolder_technique}",
                 fontsize=10, pad=6)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_axisbelow(True)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=7, label=lbl)
        for _, _, c, lbl, _ in series_cfg
    ]
    ax.legend(handles=legend_elements,
              loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0,
              framealpha=0.9, fontsize=8, title="Metric", title_fontsize=8)

    # right: leave room for the legend (~0.10); bottom: room for vertical labels
    fig.subplots_adjust(bottom=0.30, top=0.91, left=0.06, right=0.88)
    save_fig(None, "bubble_chart_f1_subfolders.svg", graph_dir, show, dpi)


def generate_latex_table(csv_file: str, extract, graph_dir: str, selected_models=None):
    df = pd.read_csv(csv_file)
    df["model_name"] = df["model_name"].str.replace("mlx-community-", "", regex=False)
    df = _apply_extract(df, extract)
    if selected_models:
        df = df[df["base_model_name"].isin(selected_models)]

    wf1_class = df.groupby(["prefix", "base_model_name"]).apply(
        lambda g: (g["f1_class"] * g["n_classes"]).sum() / g["n_classes"].sum()
        if g["n_classes"].sum() > 0 else np.nan, include_groups=False
    ).reset_index(name="weighted_f1_class")

    wf1_rel = df.groupby(["prefix", "base_model_name"]).apply(
        lambda g: (g["f1_rel"] * g["n_rel"]).sum() / g["n_rel"].sum()
        if g["n_rel"].sum() > 0 else np.nan, include_groups=False
    ).reset_index(name="weighted_f1_rel")

    score_metrics = df.groupby(["prefix", "base_model_name"]).agg(
        score_sum=("score_rel", "sum"), max_score_sum=("max_score", "sum")
    ).reset_index()
    score_metrics["weighted_score_relative"] = score_metrics["score_sum"] / score_metrics["max_score_sum"]

    merged = wf1_class.merge(wf1_rel, on=["prefix", "base_model_name"])
    merged = merged.merge(score_metrics[["prefix", "base_model_name", "weighted_score_relative"]],
                          on=["prefix", "base_model_name"])

    order = {"zero-shot": 0, "CoT": 1, "CoT-multi": 2, "one-shot": 3, "two-shots": 4}
    merged["technique_order"] = merged["prefix"].map(order)
    merged = merged.sort_values(["technique_order", "base_model_name"])

    latex = ("\\begin{table}[htbp]\n\\centering\n"
             "\\caption{Weighted Performance Metrics by Technique and Model}\n"
             "\\begin{tabular}{llccc}\n\\toprule\n"
             "\\textbf{Technique} & \\textbf{Model} & "
             "\\textbf{Weighted F1 Class} & \\textbf{Weighted F1 Relation} & "
             "\\textbf{Weighted Score} \\\\\n\\midrule\n")
    current_tech = None
    for _, row in merged.iterrows():
        tech = row["prefix"]
        if current_tech and current_tech != tech:
            latex += "\\midrule\n"
        current_tech = tech
        latex += (f"{tech} & {row['base_model_name']} & "
                  f"{row['weighted_f1_class']:.3f} & {row['weighted_f1_rel']:.3f} & "
                  f"{row['weighted_score_relative']:.3f} \\\\\n")
    latex += "\\bottomrule\n\\end{tabular}\n\\label{tab:weighted_performance}\n\\end{table}"

    out_path = os.path.join(graph_dir, "weighted_performance_table.tex")
    os.makedirs(graph_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(latex)
    return latex


# ──────────────────────────────────────────────────────────────────────────────
# Blank / missing output counter
# ──────────────────────────────────────────────────────────────────────────────

def count_blank_outputs(
    root_folder: str,
    strip_cfg: dict | None = None,
    prefixes: dict | None = None,
    graph_dir: str = "graph",
) -> pd.DataFrame:
    """
    Scan every result_*.txt file in the dataset and report how many are empty
    (0 bytes) per model and technique.

    Returns a DataFrame with columns [base_model, technique, total, blank, pct_blank]
    and also saves it to graph_dir/blank_outputs.csv.
    """
    stripped_suffix = (strip_cfg or {}).get("stripped_suffix", "_stripped")

    rows: list[dict] = []
    for sub_folder in sorted(os.listdir(root_folder)):
        sub_folder_path = os.path.join(root_folder, sub_folder)
        if not os.path.isdir(sub_folder_path):
            continue
        if not os.path.exists(os.path.join(sub_folder_path, "plantuml.txt")):
            continue
        for filename in os.listdir(sub_folder_path):
            if not (filename.startswith("result_") and filename.endswith(".txt")):
                continue
            if filename.endswith(f"{stripped_suffix}.txt"):
                continue
            fpath = os.path.join(sub_folder_path, filename)
            model_name = "-".join(filename.replace(".txt", "").split("_")[1:])
            # Strip technique prefix to get the base model name
            base_model = model_name
            technique = "zero-shot"
            for pfx, label in (prefixes or {}).items():
                if model_name.startswith(f"{pfx}-"):
                    base_model = model_name[len(pfx) + 1:]
                    technique = label
                    break
            blank = os.path.getsize(fpath) == 0
            rows.append({"base_model": base_model, "technique": technique,
                         "sub_folder": sub_folder, "blank": blank})

    if not rows:
        print("No result files found.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["base_model", "technique"])
        .agg(total=("blank", "count"), blank=("blank", "sum"))
        .reset_index()
    )
    summary["pct_blank"] = (summary["blank"] / summary["total"] * 100).round(1)
    summary = summary.sort_values(["technique", "base_model"]).reset_index(drop=True)

    # Print table
    print("\n── Blank / empty outputs per model ──────────────────────────────────")
    print(f"{'Technique':<12} {'Model':<45} {'Blank':>6} {'Total':>6} {'%':>6}")
    print("─" * 78)
    for _, r in summary.iterrows():
        if r["blank"] > 0:
            print(f"{r['technique']:<12} {r['base_model']:<45} {int(r['blank']):>6} {int(r['total']):>6} {r['pct_blank']:>5.1f}%")
    no_blank = summary[summary["blank"] == 0]
    if not no_blank.empty:
        print(f"  (no blanks: {', '.join(no_blank['base_model'].unique())})")
    print("─" * 78)

    os.makedirs(graph_dir, exist_ok=True)
    out = os.path.join(graph_dir, "blank_outputs.csv")
    summary.to_csv(out, index=False)
    print(f"Saved → {out}\n")
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Attribute bubble chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_bubble_f1_attr_(csv_file: str, extract, selected_models: list,
                         graph_dir: str, show: bool, dpi: int):
    """Bubble chart: F1 attribute score per model/technique (bubble size = n_attrs)."""
    df = pd.read_csv(csv_file)
    df["model_name"] = df["model_name"].str.replace("mlx-community-", "", regex=False)
    df = _apply_extract(df, extract)
    df = df[df["base_model_name"].isin(selected_models)]

    avg = df.groupby(["prefix", "base_model_name", "sub_folder_name"]).agg(
        f1_attr_avg=("f1_attr_llm", "mean"), n_attrs=("n_attrs", "first")
    ).reset_index()
    avg = _sort_by_technique(avg)
    avg["x_label"] = avg["prefix"] + " - " + avg["base_model_name"]
    color_map, hue_order = _technique_palette(avg["prefix"].unique())

    plt.figure(figsize=(14, 4))
    sns.scatterplot(data=avg, x="x_label", y="f1_attr_avg", size="n_attrs",
                    hue="prefix", hue_order=hue_order, palette=color_map, sizes=(50, 500),
                    edgecolor="black", alpha=0.7)
    plt.title("Average F1 Attribute Score per Model and Technique")
    plt.xlabel("Model Name and Technique")
    plt.ylabel("Average F1 Attribute Score")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Technique", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    save_fig(None, "bubble_chart_f1_attr_.svg", graph_dir, show, dpi)


def report_skipped(skipped: list, all_tasks: list, prefixes: dict, graph_dir: str):
    """
    Print and save a table of plain-text (non-UML) result files per model/technique,
    including the percentage of total tasks that were skipped.
    Saved to graph_dir/skipped_outputs.csv.
    """
    def _decode(model_name):
        base_model = model_name
        technique = "zero-shot"
        for pfx, label in (prefixes or {}).items():
            if model_name.startswith(f"{pfx}-"):
                base_model = model_name[len(pfx) + 1:]
                technique = label
                break
        return base_model, technique

    # Build total counts from all tasks
    total_rows = []
    for sub_folder, _path, filename in all_tasks:
        model_name = "-".join(filename.replace(".txt", "").split("_")[1:]) if filename else "unknown"
        base_model, technique = _decode(model_name)
        total_rows.append({"base_model": base_model, "technique": technique})
    totals = (
        pd.DataFrame(total_rows).groupby(["base_model", "technique"]).size().reset_index(name="total")
        if total_rows else pd.DataFrame(columns=["base_model", "technique", "total"])
    )

    if not skipped:
        print("No plain-text (non-UML) result files detected.")
        return pd.DataFrame()

    skip_rows = []
    for _, sub_folder, model_name in skipped:
        base_model, technique = _decode(model_name)
        skip_rows.append({"base_model": base_model, "technique": technique, "sub_folder": sub_folder})

    df = pd.DataFrame(skip_rows)
    summary = (
        df.groupby(["base_model", "technique"])
        .agg(skipped=("sub_folder", "count"))
        .reset_index()
    )
    summary = summary.merge(totals, on=["base_model", "technique"], how="left")
    summary["total"] = summary["total"].fillna(0).astype(int)
    summary["pct_skipped"] = (summary["skipped"] / summary["total"] * 100).round(1)
    summary = summary.sort_values(["technique", "base_model"]).reset_index(drop=True)

    print("\n── Plain-text (non-UML) outputs per model ───────────────────────────")
    print(f"{'Technique':<12} {'Model':<45} {'Skipped':>8} {'Total':>7} {'%':>6}")
    print("─" * 80)
    for _, r in summary.iterrows():
        print(f"{r['technique']:<12} {r['base_model']:<45} {int(r['skipped']):>8} {int(r['total']):>7} {r['pct_skipped']:>5.1f}%")
    print("─" * 80)

    os.makedirs(graph_dir, exist_ok=True)
    out = os.path.join(graph_dir, "skipped_outputs.csv")
    summary.to_csv(out, index=False)
    print(f"Saved → {out}\n")
    return summary


def generate_performances_csv(csv_file: str, extract, graph_dir: str):
    """
    Compute per-model / per-technique aggregated performance metrics from the
    raw evaluation CSV and write them to graph_dir/performances.csv.

    Columns:
        prefix, base_model_name,
        weighted_f1_class, weighted_f1_rel, weighted_score_relative,
        weighted_f1_attr, weighted_f1_inh,
        f1_aggregated   (mean of the five weighted metrics above)
    """
    _PERF_COLS = [
        "prefix", "base_model_name",
        "weighted_f1_class", "weighted_f1_rel", "weighted_score_relative",
        "weighted_precision_attr_naive", "weighted_recall_attr_naive", "weighted_f1_attr_naive",
        "weighted_precision_attr_syn", "weighted_recall_attr_syn", "weighted_f1_attr_syn",
        "weighted_precision_attr_llm", "weighted_recall_attr_llm", "weighted_f1_attr_llm",
        "weighted_f1_inh", "f1_aggregated",
    ]
    df = pd.read_csv(csv_file)
    if df.empty:
        print("Warning: evaluation CSV has no rows — performances.csv will be empty.")
        os.makedirs(graph_dir, exist_ok=True)
        pd.DataFrame(columns=_PERF_COLS).to_csv(
            os.path.join(graph_dir, "performances.csv"), index=False
        )
        return pd.DataFrame(columns=_PERF_COLS)
    df["model_name"] = df["model_name"].str.replace("mlx-community-", "", regex=False)
    # apply() on an empty Series returns an empty DataFrame with shape (0,0) which
    # cannot be unpacked into two named columns; the concat form is safe either way.
    extracted = df["model_name"].apply(lambda x: pd.Series(extract(x)))
    extracted.columns = ["base_model_name", "prefix"]
    df = pd.concat([df, extracted], axis=1)

    group_cols = ["prefix", "base_model_name"]

    # Weighted F1 class (weight = n_classes across all rows)
    wf1_class = df.groupby(group_cols).apply(
        lambda g: (g["f1_class"] * g["n_classes"]).sum() / g["n_classes"].sum()
        if g["n_classes"].sum() > 0 else np.nan, include_groups=False
    ).reset_index(name="weighted_f1_class")

    # Weighted F1 rel (weight = n_rel across all rows)
    wf1_rel = df.groupby(group_cols).apply(
        lambda g: (g["f1_rel"] * g["n_rel"]).sum() / g["n_rel"].sum()
        if g["n_rel"].sum() > 0 else np.nan, include_groups=False
    ).reset_index(name="weighted_f1_rel")

    # Weighted cardinality score (global sums across all rows)
    score_agg = df.groupby(group_cols).agg(
        score_sum=("score_rel", "sum"), max_score_sum=("max_score", "sum")
    ).reset_index()
    score_agg["weighted_score_relative"] = score_agg["score_sum"] / score_agg["max_score_sum"]

    # Weighted precision/recall/F1 attr — naive (exact), wordnet (syn), LLM (semantic) (weight = n_attrs)
    wprec_attr_naive = df.groupby(group_cols).apply(
        lambda g: (g["precision_attr_naive"] * g["n_attrs"]).sum() / g["n_attrs"].sum()
        if g["n_attrs"].sum() > 0 else np.nan, include_groups=False
    ).reset_index(name="weighted_precision_attr_naive")
    wrec_attr_naive = df.groupby(group_cols).apply(
        lambda g: (g["recall_attr_naive"] * g["n_attrs"]).sum() / g["n_attrs"].sum()
        if g["n_attrs"].sum() > 0 else np.nan, include_groups=False
    ).reset_index(name="weighted_recall_attr_naive")
    wf1_attr_naive = df.groupby(group_cols).apply(
        lambda g: (g["f1_attr_naive"] * g["n_attrs"]).sum() / g["n_attrs"].sum()
        if g["n_attrs"].sum() > 0 else np.nan, include_groups=False
    ).reset_index(name="weighted_f1_attr_naive")

    wprec_attr_syn = df.groupby(group_cols).apply(
        lambda g: (g["precision_attr_syn"] * g["n_attrs"]).sum() / g["n_attrs"].sum()
        if g["n_attrs"].sum() > 0 else np.nan, include_groups=False
    ).reset_index(name="weighted_precision_attr_syn")
    wrec_attr_syn = df.groupby(group_cols).apply(
        lambda g: (g["recall_attr_syn"] * g["n_attrs"]).sum() / g["n_attrs"].sum()
        if g["n_attrs"].sum() > 0 else np.nan, include_groups=False
    ).reset_index(name="weighted_recall_attr_syn")
    wf1_attr_syn = df.groupby(group_cols).apply(
        lambda g: (g["f1_attr_syn"] * g["n_attrs"]).sum() / g["n_attrs"].sum()
        if g["n_attrs"].sum() > 0 else np.nan, include_groups=False
    ).reset_index(name="weighted_f1_attr_syn")

    wprec_attr = df.groupby(group_cols).apply(
        lambda g: (g["precision_attr_llm"] * g["n_attrs"]).sum() / g["n_attrs"].sum()
        if g["n_attrs"].sum() > 0 else np.nan, include_groups=False
    ).reset_index(name="weighted_precision_attr_llm")
    wrec_attr = df.groupby(group_cols).apply(
        lambda g: (g["recall_attr_llm"] * g["n_attrs"]).sum() / g["n_attrs"].sum()
        if g["n_attrs"].sum() > 0 else np.nan, include_groups=False
    ).reset_index(name="weighted_recall_attr_llm")
    wf1_attr = df.groupby(group_cols).apply(
        lambda g: (g["f1_attr_llm"] * g["n_attrs"]).sum() / g["n_attrs"].sum()
        if g["n_attrs"].sum() > 0 else np.nan, include_groups=False
    ).reset_index(name="weighted_f1_attr_llm")

    # Weighted F1 inh (weight = n_inh across all rows)
    wf1_inh = df.groupby(group_cols).apply(
        lambda g: (g["f1_inh"] * g["n_inh"]).sum() / g["n_inh"].sum()
        if g["n_inh"].sum() > 0 else np.nan, include_groups=False
    ).reset_index(name="weighted_f1_inh")

    merged = (
        wf1_class
        .merge(wf1_rel, on=group_cols)
        .merge(score_agg[group_cols + ["weighted_score_relative"]], on=group_cols)
        .merge(wprec_attr_naive, on=group_cols)
        .merge(wrec_attr_naive, on=group_cols)
        .merge(wf1_attr_naive, on=group_cols)
        .merge(wprec_attr_syn, on=group_cols)
        .merge(wrec_attr_syn, on=group_cols)
        .merge(wf1_attr_syn, on=group_cols)
        .merge(wprec_attr, on=group_cols)
        .merge(wrec_attr, on=group_cols)
        .merge(wf1_attr, on=group_cols)
        .merge(wf1_inh, on=group_cols)
    )
    merged = merged.fillna(0.0)
    merged["f1_aggregated"] = (
        merged["weighted_f1_class"] + merged["weighted_f1_rel"]
        + merged["weighted_score_relative"] + merged["weighted_f1_attr_llm"]
        + merged["weighted_f1_inh"]
    ) / 5

    for col in ["weighted_f1_class", "weighted_f1_rel", "weighted_score_relative",
                "weighted_precision_attr_naive", "weighted_recall_attr_naive", "weighted_f1_attr_naive",
                "weighted_precision_attr_syn", "weighted_recall_attr_syn", "weighted_f1_attr_syn",
                "weighted_precision_attr_llm", "weighted_recall_attr_llm", "weighted_f1_attr_llm",
                "weighted_f1_inh", "f1_aggregated"]:
        merged[col] = merged[col].round(4)

    merged = merged.sort_values("f1_aggregated", ascending=False).reset_index(drop=True)

    # Drop models where every aggregated metric is 0
    perf_metric_cols = ["weighted_f1_class", "weighted_f1_rel", "weighted_score_relative",
                        "weighted_f1_attr_llm", "weighted_f1_inh"]
    all_zero_mask = (merged[perf_metric_cols] == 0).all(axis=1)
    dropped = merged.loc[all_zero_mask, "base_model_name"].unique()
    if len(dropped):
        print(f"  [filter] performances.csv: dropping all-zero models: {sorted(dropped)}")
    merged = merged[~all_zero_mask].reset_index(drop=True)

    os.makedirs(graph_dir, exist_ok=True)
    out_path = os.path.join(graph_dir, "performances.csv")
    merged.to_csv(out_path, index=False)
    print(f"Performances CSV saved → {out_path}")
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Best-models bubble chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_bubble_best_models(csv_file: str, extract, graph_dir: str, show: bool, dpi: int,
                            open_source_models: list | None = None,
                            overall_model: str | None = None,
                            overall_tech: str | None = None):
    """Bubble chart: best overall model vs best open-source model.

    X axis — 8 positions: 4 metric categories × 2 models (overall left, open-source right).
    Y axis — per-case score; one bubble per sub_folder at each x position.
    Bubble size — encodes diagram complexity for that metric.
    Follows the same scatter-per-case style as plot_bubble_f1_class_ etc.
    """
    # ── Load & prepare ──────────────────────────────────────────────────────────
    df = pd.read_csv(csv_file)
    df["model_name"] = df["model_name"].str.replace("mlx-community-", "", regex=False)
    df = _apply_extract(df, extract)
    df["score_rel_norm"] = df["score_rel"] / df["max_score"].replace(0, float("nan"))

    # ── Compute global weighted F1 per (base_model_name, prefix) ────────────────
    grp = ["base_model_name", "prefix"]

    wf1_cls = df.groupby(grp).apply(
        lambda g: (g["f1_class"] * g["n_classes"]).sum() / g["n_classes"].sum()
        if g["n_classes"].sum() > 0 else np.nan, include_groups=False
    ).reset_index(name="wf1_cls")

    wf1_rel = df.groupby(grp).apply(
        lambda g: (g["f1_rel"] * g["n_rel"]).sum() / g["n_rel"].sum()
        if g["n_rel"].sum() > 0 else np.nan, include_groups=False
    ).reset_index(name="wf1_rel")

    wf1_attr = df.groupby(grp).apply(
        lambda g: (g["f1_attr_llm"] * g["n_attrs"]).sum() / g["n_attrs"].sum()
        if g["n_attrs"].sum() > 0 else np.nan, include_groups=False
    ).reset_index(name="wf1_attr")

    score_agg = df.groupby(grp).agg(
        score_sum=("score_rel", "sum"), max_score_sum=("max_score", "sum")
    ).reset_index()
    score_agg["wf1_score"] = (
        score_agg["score_sum"] / score_agg["max_score_sum"].replace(0, float("nan"))
    )

    perf = (wf1_cls
            .merge(wf1_rel, on=grp)
            .merge(wf1_attr, on=grp)
            .merge(score_agg[grp + ["wf1_score"]], on=grp))
    perf = perf.fillna(0.0)
    perf["f1_agg"] = (
        perf["wf1_cls"] + perf["wf1_rel"] + perf["wf1_attr"] + perf["wf1_score"]
    ) / 4

    # Best technique per model (highest aggregated F1)
    best_idx = perf.groupby("base_model_name")["f1_agg"].idxmax()
    best = perf.loc[best_idx].reset_index(drop=True)

    # Classify open-source: either from the explicit list or by heuristic
    if open_source_models:
        os_set = set(open_source_models)
        best["is_open_source"] = best["base_model_name"].isin(os_set)
    else:
        _CLOSED = re.compile(r"^(gpt|o1[-_. ]|o3[-_. ]|claude|gemini)", re.IGNORECASE)
        best["is_open_source"] = ~best["base_model_name"].str.match(_CLOSED)

    # ── Pick winners ─────────────────────────────────────────────────────────────
    if overall_model and overall_tech:
        override = perf[(perf["base_model_name"] == overall_model) & (perf["prefix"] == overall_tech)]
        if override.empty:
            print(f"  [warn] override ({overall_model}, {overall_tech}) not found in data; falling back to auto-select")
            best_overall_row = best.loc[best["f1_agg"].idxmax()]
        else:
            best_overall_row = override.iloc[0]
    else:
        best_overall_row = best.loc[best["f1_agg"].idxmax()]
    os_candidates = best[best["is_open_source"]]
    if os_candidates.empty:
        print("  [warn] no open-source models detected; skipping best-models chart")
        return
    best_os_row = os_candidates.loc[os_candidates["f1_agg"].idxmax()]

    best_overall_model = best_overall_row["base_model_name"]
    best_overall_tech  = best_overall_row["prefix"]
    best_os_model      = best_os_row["base_model_name"]
    best_os_tech       = best_os_row["prefix"]

    print(f"  Best overall:     {best_overall_model} ({best_overall_tech})"
          f"  F1-agg={best_overall_row['f1_agg']:.3f}")
    print(f"  Best open-source: {best_os_model} ({best_os_tech})"
          f"  F1-agg={best_os_row['f1_agg']:.3f}")

    # ── Per-case data for each winner ─────────────────────────────────────────────
    df_b = df[(df["base_model_name"] == best_overall_model) & (df["prefix"] == best_overall_tech)].copy()
    df_o = df[(df["base_model_name"] == best_os_model)      & (df["prefix"] == best_os_tech)].copy()

    if df_b.empty or df_o.empty:
        print("  [warn] one of the winner subsets is empty; skipping best-models chart")
        return

    def _short(name, maxlen=22):
        return name if len(name) <= maxlen else name[:maxlen] + "…"

    # X-axis labels: 8 positions in order (overall, os) per metric
    label_b = _short(best_overall_model)
    label_o = _short(best_os_model)

    # (y_col, size_col, x_label_overall, x_label_os, metric_group_label)
    metrics = [
        ("f1_class",       "n_classes", f"F1 Class\n{label_b}", f"F1 Class\n{label_o}", "F1 Class"),
        ("f1_rel",         "n_rel",     f"F1 Assoc.\n{label_b}", f"F1 Assoc.\n{label_o}", "F1 Assoc."),
        ("score_rel_norm", "max_score", f"Card. Score\n{label_b}", f"Card. Score\n{label_o}", "Card. Score"),
        ("f1_attr_llm",    "n_attrs",   f"F1 Attr.\n{label_b}", f"F1 Attr.\n{label_o}", "F1 Attr."),
    ]

    # Build ordered x-tick list and colour map
    x_positions = []
    for _, _, xl_b, xl_o, _ in metrics:
        x_positions += [xl_b, xl_o]

    x_map = {lbl: i for i, lbl in enumerate(x_positions)}
    n = len(x_positions)

    # Two colours: one per winner model
    color_b = "#4C72B0"   # blue — best overall
    color_o = "#DD8452"   # orange — best open-source

    # ── Build long-form dataframe for sns.scatterplot ─────────────────────────────
    rows = []
    for y_col, size_col, xl_b, xl_o, _ in metrics:
        for case_df, xl, color in [(df_b, xl_b, color_b), (df_o, xl_o, color_o)]:
            for _, row in case_df.iterrows():
                rows.append({
                    "x_label": xl,
                    "x_pos":   x_map[xl],
                    "score":   row[y_col],
                    "count":   row[size_col] if pd.notna(row[size_col]) else 0,
                    "color":   color,
                })
    long_df = pd.DataFrame(rows)

    # Normalise bubble sizes within each metric pair so sizes are comparable
    # across the two models but each metric uses its own scale.
    long_df["size_norm"] = 50.0
    for y_col, size_col, xl_b, xl_o, _ in metrics:
        mask = long_df["x_label"].isin([xl_b, xl_o])
        vals = long_df.loc[mask, "count"]
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            long_df.loc[mask, "size_norm"] = 20 + (vals - vmin) / (vmax - vmin) * 230
        else:
            long_df.loc[mask, "size_norm"] = 80

    # ── Plot ─────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    sns.set_theme(style="whitegrid")

    for color, sub in long_df.groupby("color"):
        ax.scatter(sub["x_pos"], sub["score"], s=sub["size_norm"],
                   color=color, alpha=0.72, edgecolors="black", linewidths=0.4, zorder=3)

    # Vertical separators between metric groups
    for sep in [1.5, 3.5, 5.5]:
        ax.axvline(sep, color="lightgray", linewidth=0.8, linestyle="--", zorder=1)

    ax.set_xticks(range(n))
    ax.set_xticklabels(x_positions, rotation=30, ha="right", fontsize=7)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("")
    ax.set_ylabel("Score", fontsize=9)
    ax.set_title(
        f"Best Overall ({_short(best_overall_model, 28)}, {best_overall_tech})  vs  "
        f"Best Open-Source ({_short(best_os_model, 28)}, {best_os_tech})",
        fontsize=9, pad=6,
    )
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_axisbelow(True)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_b,
               markeredgecolor="black", markersize=8, label=f"● {label_b}  (overall)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_o,
               markeredgecolor="black", markersize=8, label=f"● {label_o}  (open-source)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7,
              framealpha=0.9, title="Model", title_fontsize=7,
              bbox_to_anchor=(1.01, 1), borderaxespad=0)

    fig.subplots_adjust(bottom=0.22, top=0.88, left=0.06, right=0.78)
    save_fig(None, "bubble_chart_best_models.svg", graph_dir, show, dpi)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate text2uml model outputs.")
    parser.add_argument("--config", default="eval_config.yaml", help="Path to YAML config file")
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip evaluation, regenerate plots from the existing CSV")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Resolve paths relative to config file location
    config_dir = os.path.dirname(os.path.abspath(args.config)) if os.path.exists(args.config) else os.getcwd()
    dataset_path = cfg["dataset_path"] if os.path.isabs(cfg["dataset_path"]) else os.path.join(config_dir, cfg["dataset_path"])
    grammar_path = cfg["grammar_path"] if os.path.isabs(cfg["grammar_path"]) else os.path.join(config_dir, cfg["grammar_path"])
    graph_dir = cfg["graph_dir"] if os.path.isabs(cfg["graph_dir"]) else os.path.join(config_dir, cfg["graph_dir"])
    csv_file = cfg["output_csv"] if os.path.isabs(cfg["output_csv"]) else os.path.join(dataset_path, cfg["output_csv"])

    os.makedirs(graph_dir, exist_ok=True)
    plt.rcParams["figure.dpi"] = cfg["dpi"]
    load_dotenv()

    extract = make_extractor(cfg["prefixes"])
    ignore_list = set(cfg.get("ignore_list", []))
    selected_models = cfg.get("selected_models", [])
    show = cfg.get("show_plots", False)
    dpi = cfg.get("dpi", 300)
    strip_cfg = cfg.get("strip_prompt", {"enabled": False})

    # Apply LangSmith tracing preference from config (env vars already default to false above)
    if cfg.get("langsmith_tracing", False):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_TRACING"] = "true"

    # ── Evaluation ──
    run_eval = cfg.get("run_evaluation", True) and not args.plots_only
    if run_eval:
        print(f"Running evaluation on {dataset_path} …")
        if strip_cfg.get("enabled"):
            print(f"  Prompt stripping: enabled ({len(strip_cfg.get('patterns', []))} patterns)")
        (crash_pct, exc_summary), skipped, all_tasks = process_folders(
            root_folder=dataset_path,
            csv_file=csv_file,
            grammar_path=grammar_path,
            ignore_list=ignore_list,
            strip_cfg=strip_cfg,
            max_workers=cfg.get("max_workers", 12),
            stall_timeout_s=cfg.get("stall_timeout", 300),
            poll_interval_s=cfg.get("poll_interval", 1.0),
            prefixes=cfg.get("prefixes", {}),
        )
        print("Crash percentages:", crash_pct)
        print("Exception summary:", exc_summary)
        report_skipped(skipped, all_tasks, cfg.get("prefixes", {}), graph_dir)
    else:
        print(f"Skipping evaluation, using existing CSV: {csv_file}")

    # ── Blank output report (always runs when dataset is available) ──
    count_blank_outputs(
        root_folder=dataset_path,
        strip_cfg=strip_cfg,
        prefixes=cfg.get("prefixes", {}),
        graph_dir=graph_dir,
    )

    if not os.path.exists(csv_file):
        print(f"CSV not found: {csv_file}. Skipping plots.")
        return

    # ── Plots ──
    if cfg.get("run_plots", True):
        print(f"Generating plots → {graph_dir}")

        plot_crash_data(csv_file, extract, graph_dir, show, dpi)
        plot_data_class(csv_file, extract, graph_dir, show, dpi)
        plot_data_rel(csv_file, extract, graph_dir, show, dpi)
        plot_data_attr(csv_file, extract, graph_dir, show, dpi)
        plot_data_score_perc(csv_file, extract, graph_dir, show, dpi)
        plot_aggregated_f1(csv_file, extract, graph_dir, show, dpi)
        plot_data_score(csv_file, extract, graph_dir, show, dpi)
        plot_data_class_pc(csv_file, extract, graph_dir, show, dpi)
        plot_data_rel_pc(csv_file, extract, graph_dir, show, dpi)
        plot_data_score_pc(csv_file, extract, graph_dir, show, dpi)
        plot_distributions_bubble_sizes(csv_file, graph_dir, show, dpi)
        plot_distributions_no_aggregation_size_by_overlap(csv_file, graph_dir, show, dpi)

        if selected_models:
            plot_bubble_f1_class_(csv_file, extract, selected_models, graph_dir, show, dpi)
            plot_bubble_f1_rel_(csv_file, extract, selected_models, graph_dir, show, dpi)
            plot_bubble_f1_score_(csv_file, extract, selected_models, graph_dir, show, dpi)
            plot_bubble_f1_attr_(csv_file, extract, selected_models, graph_dir, show, dpi)

        subfolder_model = cfg.get("subfolder_model", "")
        subfolder_technique = cfg.get("subfolder_technique", "two-shots")
        if subfolder_model:
            plot_bubble_f1_subfolders_(csv_file, extract, subfolder_model,
                                       subfolder_technique, graph_dir, show, dpi)

        plot_bubble_best_models(
            csv_file, extract, graph_dir, show, dpi,
            open_source_models=cfg.get("open_source_models") or None,
            overall_model=cfg.get("best_overall_model") or None,
            overall_tech=cfg.get("best_overall_tech") or None,
        )

        generate_latex_table(csv_file, extract, graph_dir, selected_models or None)
        generate_performances_csv(csv_file, extract, graph_dir)

        print(f"All plots saved to {graph_dir}/")


if __name__ == "__main__":
    main()
