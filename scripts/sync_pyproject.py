"""
Generate py-lynxes/pyproject.toml from the root pyproject.toml,
adjusting relative paths for the py-lynxes/ working directory.

Usage:
    python scripts/sync_pyproject.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "pyproject.toml"
DST = ROOT / "py-lynxes" / "pyproject.toml"

REPLACEMENTS = [
    # maturin manifest-path
    (r'manifest-path = "crates/lynxes-python/Cargo\.toml"',
     'manifest-path = "../crates/lynxes-python/Cargo.toml"'),
    # maturin python-source
    (r'python-source = "py-lynxes/src"',
     'python-source = "src"'),
    # pytest testpaths
    (r'testpaths = \["py-lynxes/tests/unit"\]',
     'testpaths = ["tests/unit"]'),
]

def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    for pattern, replacement in REPLACEMENTS:
        text = re.sub(pattern, replacement, text)
    DST.write_text(text, encoding="utf-8")
    print(f"synced {SRC} → {DST}")

if __name__ == "__main__":
    main()
