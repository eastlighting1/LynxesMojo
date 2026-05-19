# CI Workflows

## Overview

| Workflow | Trigger | Purpose |
|---|---|---|
| `ci.yml` | PR → main, push → main/dev | Rust lint + tests, Python tests |
| `bench.yml` | push → main (engine files), manual | Smoke benchmarks on push, full benchmarks manually |

---

## `ci.yml` — Continuous Integration

Jobs run in parallel:

- **`rust-lint`** — `cargo fmt --check` + `cargo clippy -D warnings`
- **`rust-test`** — build Mojo kernels, then `cargo test --workspace --exclude lynxes-python`
- **`python-test`** — matrix [3.10, 3.11, 3.12, 3.13]: build Mojo kernels, `maturin develop` → `pytest`
- **`python-lint`** — `ruff check` + `ruff format --check`
- **`ci-pass`** — single required status check for branch protection

Local commits should enable `.githooks/pre-commit` with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_git_hooks.ps1
```

The hook invokes `scripts/check_mojo_kernels.sh` directly on Linux or through WSL
from Git for Windows.

## `bench.yml` — Benchmarks

### Automatic (push to main, engine code only)
Runs smoke Rust Criterion benchmarks + Python benchmarks at size 1k.
Results are uploaded as artifacts (30-day retention).

### Manual
```
Actions → Benchmarks → Run workflow
  save-baseline: true   # optional, to save as new reference
```

Manual runs set `LYNXES_FULL_BENCH=1`, enabling the larger Rust benchmark inputs and Python sizes 1k + 10k. The full 100k-node Python benchmark should still be run locally:
```bash
cd py-lynxes
uv run python tests/benchmark/bench_vs_networkx.py --sizes 100000
```
