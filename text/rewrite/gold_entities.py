"""Extract gold class/attribute names from a case's PlantUML.

Feeds ``text.rewrite.recall_loop``'s level-minus-one objective. Loads
``src/eval.py``'s grammar-based PlantUML parser by file path (mirrors
``experiments.levels.generate.load_runner`` -- ``src`` is not a package).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import List, Tuple

from ..config import TextConfig


def _load_eval_module(tconf: TextConfig) -> ModuleType:
    path = tconf.repo_root / "src" / "eval.py"
    spec = importlib.util.spec_from_file_location("t2u_eval", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load src/eval.py from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["t2u_eval"] = module
    spec.loader.exec_module(module)
    return module


def build_uml_parser(tconf: TextConfig) -> Tuple[ModuleType, object]:
    """(eval module, Lark parser) for PlantUML, built from the repo's grammar.ebnf."""
    eval_mod = _load_eval_module(tconf)
    grammar_path = tconf.repo_root / "grammar.ebnf"
    return eval_mod, eval_mod.init_parser(str(grammar_path))


def gold_entity_names(gold_path: Path, eval_mod: ModuleType, parser: object) -> List[str]:
    """Lowercased class names + attribute var names from a gold PlantUML file."""
    text = gold_path.read_text(encoding="utf-8")
    tree = eval_mod.parse_text(parser, text)
    names = [c.lower() for c in eval_mod.get_from_parsed(tree, "class")]
    for _cls, var in eval_mod.get_attributes_from_parsed(tree):
        attr_name = var.split(":")[0].strip().lower()
        if attr_name:
            names.append(attr_name)
    return names
