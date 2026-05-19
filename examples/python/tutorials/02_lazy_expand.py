from pathlib import Path

import lynxes as lx

ROOT = Path(__file__).resolve().parents[3]
GRAPH_PATH = ROOT / "examples" / "data" / "example_simple.gf"


def main() -> None:
    # This toy repository assumes a local checkout. Full Lynxes docs live in the
    # upstream Lynxes repository.
    graph = lx.read_gf(GRAPH_PATH)

    result = (
        graph.lazy()
        .filter_nodes(lx.col("_id") == "alice")
        # The lazy API builds a traversal plan first; no graph work happens until
        # collect() materializes the resulting subgraph.
        .expand(edge_type="KNOWS", hops=2, direction="out")
        .collect()
    )

    print(f"graph: {GRAPH_PATH.name}")
    print(f"expanded nodes: {result.node_count()}")
    print(f"expanded edges: {result.edge_count()}")
    print(f"node ids: {result.nodes().ids()}")


if __name__ == "__main__":
    main()
