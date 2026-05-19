from pathlib import Path

import lynxes as lx

ROOT = Path(__file__).resolve().parents[3]
GRAPH_PATH = ROOT / "examples" / "data" / "example_simple.gf"


def main() -> None:
    # This toy repository assumes a local checkout. Full Lynxes docs live in the
    # upstream Lynxes repository.
    graph = lx.read_gf(GRAPH_PATH)

    # Eager algorithms run immediately on a materialized GraphFrame. The output
    # is still frame-shaped, so it can be inspected without leaving the Lynxes
    # surface or inventing a separate result wrapper.
    path = graph.shortest_path("alice", "charlie")
    print(f"graph: {GRAPH_PATH.name}")
    print(f"shortest path: {path}")

    ranks = graph.pagerank()
    print("pagerank preview:")
    print(ranks.head(5, sort_by="pagerank", descending=True))


if __name__ == "__main__":
    main()
