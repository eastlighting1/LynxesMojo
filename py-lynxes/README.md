# LynxesMojo Python Package

This package directory keeps the Python-facing side of the LynxesMojo toy
prototype. The import name remains `lynxes` so the original Python API shape can
be reused, but this repository is not prepared as a PyPI release package.

The experiment adds a Linux-only Mojo shared library for structural graph
features:

```python
import lynxes as lx

g = lx.read_gf("examples/data/example_simple.gf")
features = g.structural_features(edge_type=None)
```

`structural_features()` returns a `NodeFrame` with `out_degree`, `in_degree`,
and `total_degree` columns computed by the Mojo kernel. The lazy
`aggregate_neighbors(..., lx.count())` path uses the same kernel.

Build the Mojo artifact before using those paths from a source checkout:

```bash
bash scripts/build_mojo_kernels.sh
export LYNXES_MOJO_LIB="$PWD/py-lynxes/src/lynxes/.libs/liblynxes_mojo_kernels.so"
uv run maturin develop --release
```

For the project rationale, architecture, and commit-time Mojo verification flow,
see the root `README.md`.
