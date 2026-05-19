import tempfile
from pathlib import Path

import lynxes as lx

ROOT = Path(__file__).resolve().parents[3]
GRAPH_PATH = ROOT / "examples" / "data" / "example_simple.gf"


def main() -> None:
    # This toy repository assumes a local checkout. Full Lynxes docs live in the
    # upstream Lynxes repository.
    with tempfile.TemporaryDirectory() as tmp:
        source = lx.read_gf(GRAPH_PATH)
        gfb_path = Path(tmp) / "example_simple.gfb"

        # Round-trip into .gfb to show the native binary format without leaving
        # generated artifacts behind in the repository checkout.
        source.write_gfb(gfb_path)
        restored = lx.read_gfb(gfb_path)

        print(f"source graph: {GRAPH_PATH.name}")
        print(f"restored graph: {gfb_path.name}")
        print(f"source counts: nodes={source.node_count()} edges={source.edge_count()}")
        print(f"restored counts: nodes={restored.node_count()} edges={restored.edge_count()}")


if __name__ == "__main__":
    main()
