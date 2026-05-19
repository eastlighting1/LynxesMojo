"""
Smoke tests for shipped user-facing examples.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = REPO_ROOT / "examples"
PYTHON_TUTORIALS_DIR = EXAMPLES_DIR / "python" / "tutorials"
PYTHON_RECIPES_DIR = EXAMPLES_DIR / "python" / "recipes"


def test_examples_layout_exists():
    assert (EXAMPLES_DIR / "data").is_dir()
    assert (EXAMPLES_DIR / "python").is_dir()
    assert (EXAMPLES_DIR / "python" / "tutorials").is_dir()
    assert (EXAMPLES_DIR / "python" / "recipes").is_dir()
    assert (EXAMPLES_DIR / "cli").is_dir()
    assert (EXAMPLES_DIR / "rust" / "tutorials").is_dir()


def test_phase1_example_graphs_exist():
    expected = {
        "example_simple.gf",
        "example_weighted.gf",
        "example_complex.gf",
    }
    existing = {path.name for path in (EXAMPLES_DIR / "data").glob("*.gf")}
    assert expected.issubset(existing)


def test_python_examples_smoke():
    scripts = [
        PYTHON_TUTORIALS_DIR / "01_read_and_inspect.py",
        PYTHON_TUTORIALS_DIR / "02_lazy_expand.py",
        PYTHON_TUTORIALS_DIR / "03_first_algorithm.py",
        PYTHON_RECIPES_DIR / "pagerank.py",
        PYTHON_RECIPES_DIR / "community_detection.py",
        PYTHON_RECIPES_DIR / "io_roundtrip.py",
    ]
    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"{script.name} failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
