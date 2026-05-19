from pathlib import Path

import lynxes as lx

ROOT = Path(__file__).resolve().parents[3]
GRAPH_PATH = ROOT / "examples" / "data" / "example_simple.gf"


def main() -> None:
    # This toy repository assumes a local checkout. Full Lynxes docs live in the
    # upstream Lynxes repository.
    graph = lx.read_gf(GRAPH_PATH)

    # Eager algorithms execute immediately and return a frame-like result that can
    # be inspected with the same column-oriented methods as other Lynxes outputs.
    ranks = graph.pagerank()

    print(f"graph: {GRAPH_PATH.name}")
    print(f"columns: {ranks.column_names()}")
    print(f"rows: {ranks.len()}")
    print(ranks.head(5, sort_by="pagerank", descending=True))


if __name__ == "__main__":
    main()
