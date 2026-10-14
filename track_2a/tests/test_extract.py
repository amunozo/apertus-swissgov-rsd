"""The dev and test extraction scripts must run the same code (rsd.extract.main).

No torch needed for the script checks. The gate check needs torch and is
skipped without it (the uv environment has no torch; Kaggle and the CPU dry
run do).
"""
import ast
from pathlib import Path

import pytest

KAGGLE = Path(__file__).resolve().parents[1] / "kaggle"


def _main_calls(script: str) -> list[dict]:
    """Keyword arguments of every top-level call to main(...) in a script."""
    tree = ast.parse((KAGGLE / script).read_text())
    calls = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and getattr(node.value.func, "id", "") == "main":
            assert not node.value.args, "call main() with keyword arguments only"
            calls.append({k.arg: ast.unparse(k.value) for k in node.value.keywords})
    return calls


def _imports(script: str) -> set[str]:
    tree = ast.parse((KAGGLE / script).read_text())
    return {f"{n.module}.{a.name}" for n in tree.body if isinstance(n, ast.ImportFrom) for a in n.names}


def test_dev_and_test_scripts_share_main():
    for script in ("01_extract_dev.py", "04_extract_test.py"):
        assert "rsd.extract.main" in _imports(script)
    (dev,) = _main_calls("01_extract_dev.py")
    (test,) = _main_calls("04_extract_test.py")
    assert dev == {"split": "'dev'", "final": "False", "state_layers": "STATE_LAYERS"}
    assert test == {"split": "'test'", "final": "True", "state_layers": "[]"}


def test_test_split_needs_final():
    pytest.importorskip("torch")
    from rsd.extract import main

    with pytest.raises(PermissionError):
        main(split="test", final=False, state_layers=[], out=Path("/nonexistent/never_written"))
