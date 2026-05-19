"""
Shared pytest fixtures for Lynxes Python binding tests (TST-009).
"""

from pathlib import Path

import pytest

import lynxes as gf

REPO_ROOT = Path(__file__).resolve().parents[3]
TINY_SOCIAL_GF = REPO_ROOT / "examples" / "data" / "example_simple.gf"


@pytest.fixture(scope="session")
def gf_path():
    """Return the shared tiny social graph fixture path."""
    return str(TINY_SOCIAL_GF)


@pytest.fixture(scope="session")
def graph(gf_path):
    """Loaded GraphFrame from the social fixture."""
    return gf.read_gf(gf_path)


@pytest.fixture(scope="session")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("output")
