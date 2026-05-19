"""
Lynxes: graph analytics engine built on Apache Arrow.

Import the native extension for core functionality.
Public API symbols are re-exported here for stable user-facing access.
"""

import os
from pathlib import Path

from lynxes._csv import read_csv
from lynxes._lynxes import (
    AggExpr,
    Any,
    Bool,
    Date,
    DateTime,
    Duration,
    EdgeFrame,
    Expr,
    Float,
    GraphFrame,
    Int,
    LazyGraphFrame,
    MutableGraphFrame,
    NodeFrame,
    PartitionedGraph,
    PatternEdge,
    PatternNode,
    SampledSubgraph,
    String,
    StringExprNamespace,
    StringView,
    __version__,
    _configure_mojo_runtime,
    col,
    count,
    edge,
    first,
    graph,
    last,
    list,
    mean,
    node,
    partition_graph,
    read_arangodb,
    read_gf,
    read_gfb,
    read_neo4j,
    read_parquet_graph,
    read_sparql,
    sum,
    write_gf,
    write_gfb,
    write_owl,
    write_parquet_graph,
    write_rdf,
)


def _configure_bundled_mojo_runtime() -> None:
    override = os.environ.get("LYNXES_MOJO_LIB")
    if override:
        _configure_mojo_runtime(override)
        return

    candidate = Path(__file__).parent / ".libs" / "liblynxes_mojo_kernels.so"
    if candidate.exists():
        _configure_mojo_runtime(str(candidate))


_configure_bundled_mojo_runtime()

__all__ = [
    "__version__",
    "NodeFrame",
    "EdgeFrame",
    "GraphFrame",
    "MutableGraphFrame",
    "LazyGraphFrame",
    "Expr",
    "AggExpr",
    "PartitionedGraph",
    "StringExprNamespace",
    "PatternNode",
    "PatternEdge",
    "SampledSubgraph",
    "col",
    "node",
    "edge",
    "graph",
    "partition_graph",
    "read_csv",
    "read_arangodb",
    "read_gf",
    "read_gfb",
    "read_neo4j",
    "read_parquet_graph",
    "read_sparql",
    "write_gf",
    "write_gfb",
    "write_parquet_graph",
    "write_rdf",
    "write_owl",
    "count",
    "sum",
    "mean",
    "list",
    "first",
    "last",
    "String",
    "StringView",
    "Int",
    "Float",
    "Bool",
    "Date",
    "DateTime",
    "Duration",
    "Any",
]
