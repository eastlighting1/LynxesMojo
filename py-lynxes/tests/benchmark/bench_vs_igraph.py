"""
TST-011: Graphframe vs igraph performance benchmark.

Compares Graphframe and igraph on three operations:
  1. 2-hop BFS expand from a single seed node
  2. PageRank (damping=0.85, 100 iterations)
  3. Weakly Connected Components

Graph sizes: 1 000, 10 000, 100 000 nodes (complex LFR structure)

Usage:
    uv run python python/benchmarks/bench_vs_igraph.py

Results are printed to stdout in a markdown table. This toy repository no
longer keeps generated benchmark documents.
"""

import argparse
import os
import pathlib
import random
import statistics
import sys
import tempfile
import time
from collections.abc import Callable

import numpy as np

# ── dependency check ──────────────────────────────────────────────────────────

try:
    import igraph as ig
except ImportError:
    print("igraph not installed — install with: uv sync --group benchmark")
    print("Skipping TST-011.")
    sys.exit(0)

try:
    import lynxes as gf
except ImportError:
    print("lynxes not installed — run: uv run maturin develop")
    sys.exit(1)

# ── graph generation helpers ──────────────────────────────────────────────────


def generate_lfr_edges(
    n: int, mu: float = 0.1, max_degree: int = 1000, seed: int = 42
) -> np.ndarray:
    """Generate scale-free complex graph edges using numpy."""
    np.random.seed(seed)
    random.seed(seed)

    degrees = np.random.zipf(2.5, n)
    degrees = np.clip(degrees, 1, max_degree)

    num_communities = max(10, int(n / 1000))
    comm_sizes = np.random.zipf(2.0, num_communities)
    comm_probs = comm_sizes / comm_sizes.sum()
    communities = np.random.choice(num_communities, size=n, p=comm_probs)

    external_degrees = np.random.binomial(degrees, mu)
    internal_degrees = degrees - external_degrees
    nodes = np.arange(n)

    def match_stubs(stub_list):
        np.random.shuffle(stub_list)
        if len(stub_list) % 2 != 0:
            stub_list = stub_list[:-1]
        if len(stub_list) == 0:
            return np.empty((0, 2), dtype=np.int32)
        return stub_list.reshape(-1, 2)

    edges_list = []
    for c in range(num_communities):
        c_nodes = nodes[communities == c]
        c_internal_degs = internal_degrees[communities == c]
        c_stubs = np.repeat(c_nodes, c_internal_degs)
        edges_list.append(match_stubs(c_stubs))

    external_stubs = np.repeat(nodes, external_degrees)
    edges_list.append(match_stubs(external_stubs))

    return np.vstack(edges_list)


def lfr_complex_gf(n: int, edges: np.ndarray) -> gf.GraphFrame:
    """Generate a complex LFR graph as a GraphFrame using pre-computed edges."""
    random.seed(42)
    np.random.seed(42)

    cities = ["Seoul", "Busan", "Incheon", "Daegu", "Daejeon"]
    person_statuses = ["active", "away", "inactive"]
    company_statuses = ["hiring", "stable", "scaling"]
    roles = ["Engineer", "Designer", "Manager", "Analyst"]
    projects = ["graph-runtime", "ai-core", "data-pipeline", "frontend-v2"]
    channels = ["coffee-chat", "study-group", "team-sync", "alumni"]

    is_company = np.random.rand(n) < 0.1
    batch_size = 500_000

    with tempfile.NamedTemporaryFile(suffix=".gf", mode="w", delete=False, encoding="utf-8") as f:
        path = f.name
        node_buffer = []
        for i in range(n):
            if is_company[i]:
                city = random.choice(cities)
                status = random.choice(company_statuses)
                founded = random.randint(1990, 2024)
                node_buffer.append(
                    f'(n{i}: Company {{ founded: {founded}, city: "{city}", status: "{status}" }})\n'
                )
            else:
                city = random.choice(cities)
                status = random.choice(person_statuses)
                age = random.randint(20, 65)
                score = round(random.uniform(0.5, 0.99), 2)
                node_buffer.append(
                    f'(n{i}: Person {{ age: {age}, score: {score}, city: "{city}", status: "{status}" }})\n'
                )

            if len(node_buffer) >= batch_size:
                f.writelines(node_buffer)
                node_buffer.clear()
        if node_buffer:
            f.writelines(node_buffer)
            node_buffer.clear()

        f.write("\n")

        edge_buffer = []
        for src, dst in edges:
            src_is_comp = is_company[src]
            dst_is_comp = is_company[dst]

            if not src_is_comp and not dst_is_comp:
                if random.random() < 0.8:
                    channel = random.choice(channels)
                    weight = round(random.uniform(0.1, 1.0), 2)
                    since = random.randint(2015, 2025)
                    edge_buffer.append(
                        f'n{src} -[KNOWS]-> n{dst} {{ since: {since}, weight: {weight}, channel: "{channel}" }}\n'
                    )
                else:
                    cohort = f"bootcamp-{random.randint(1, 10)}"
                    edge_buffer.append(
                        f'n{src} -[MENTORED_THROUGH_BOOTCAMP]-> n{dst} {{ since: {random.randint(2018, 2024)}, cohort: "{cohort}" }}\n'
                    )
            elif not src_is_comp and dst_is_comp:
                if random.random() < 0.7:
                    role = random.choice(roles)
                    edge_buffer.append(
                        f'n{src} -[WORKS_AT]-> n{dst} {{ role: "{role}", since: {random.randint(2010, 2025)}, status: "full-time" }}\n'
                    )
                else:
                    project = random.choice(projects)
                    edge_buffer.append(
                        f'n{src} -[COLLABORATES_ON]-> n{dst} {{ project: "{project}", status: "pilot" }}\n'
                    )
            elif src_is_comp and not dst_is_comp:
                role = random.choice(roles)
                edge_buffer.append(
                    f'n{dst} -[WORKS_AT]-> n{src} {{ role: "{role}", since: {random.randint(2010, 2025)}, status: "contract" }}\n'
                )
            else:
                edge_buffer.append(
                    f'n{src} -[PARTNERS_WITH]-> n{dst} {{ since: {random.randint(2000, 2025)}, tier: "strategic" }}\n'
                )

            if len(edge_buffer) >= batch_size:
                f.writelines(edge_buffer)
                edge_buffer.clear()

        if edge_buffer:
            f.writelines(edge_buffer)
            edge_buffer.clear()

    try:
        return gf.read_gf(path)
    finally:
        os.unlink(path)


def lfr_complex_igraph(n: int, edges: np.ndarray) -> ig.Graph:
    """Generate the equivalent complex LFR graph as an igraph Graph."""
    return ig.Graph(n=n, edges=edges.tolist(), directed=True)


# ── timing helpers ────────────────────────────────────────────────────────────

REPS = 3  # default repetitions per cell; overridden at runtime by --reps


def time_fn(fn: Callable, reps: int | None = None) -> float:
    """Return median wall-clock time (seconds) over `reps` calls."""
    n = REPS if reps is None else reps
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


# ── benchmarks ────────────────────────────────────────────────────────────────


def bench_expand_gf(graph: gf.GraphFrame) -> float:
    seed_id = "n0"
    return time_fn(
        lambda: (
            graph.lazy()
            .filter_nodes(gf.col("_id") == seed_id)
            .expand(hops=2, direction="out")
            .collect()
        )
    )


def bench_expand_igraph(g: ig.Graph) -> float:
    def _run():
        g.bfsiter(0, mode="out", advanced=False)
        visited = set()
        queue = [(0, 0)]
        while queue:
            v, depth = queue.pop(0)
            if v in visited or depth > 2:
                continue
            visited.add(v)
            if depth < 2:
                queue.extend((nb, depth + 1) for nb in g.successors(v))

    return time_fn(_run)


def bench_pagerank_gf(graph: gf.GraphFrame) -> float:
    return time_fn(lambda: graph.pagerank())


def bench_pagerank_igraph(g: ig.Graph) -> float:
    return time_fn(lambda: g.pagerank(damping=0.85, implementation="prpack"))


def bench_cc_gf(graph: gf.GraphFrame) -> float:
    return time_fn(lambda: graph.connected_components())


def bench_cc_igraph(g: ig.Graph) -> float:
    return time_fn(lambda: g.clusters(mode="weak"))


# ── main ──────────────────────────────────────────────────────────────────────

_DEFAULT_SIZES = [1_000, 10_000, 100_000]
_DEFAULT_REPS = 3


def fmt(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def speedup(gf_t: float, ig_t: float) -> str:
    if gf_t == 0:
        return "∞×"
    ratio = ig_t / gf_t
    return f"{ratio:.1f}×" if ratio >= 1 else f"1/{1 / ratio:.1f}×"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lynxes vs igraph benchmark")
    p.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=_DEFAULT_SIZES,
        metavar="N",
        help="graph sizes to benchmark (default: 1000 10000 100000)",
    )
    p.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        metavar="FILE",
        help="write markdown results to FILE instead of the default docs path",
    )
    p.add_argument(
        "--reps",
        type=int,
        default=_DEFAULT_REPS,
        metavar="R",
        help="repetitions per benchmark cell (default: 3)",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    sizes = args.sizes
    global REPS  # noqa: PLW0603
    REPS = args.reps

    rows = []
    print()
    print(f"{'N':>8}  {'Operation':<30}  {'Graphframe':>12}  {'igraph':>12}  {'Speedup':>10}")
    print("-" * 82)

    for n in sizes:
        print(f"  Building graphs n={n:,} ...", end="", flush=True)
        edges = generate_lfr_edges(n)
        graph_gf = lfr_complex_gf(n, edges)
        graph_ig = lfr_complex_igraph(n, edges)
        print(" done")

        ops = [
            ("2-hop BFS expand", bench_expand_gf, bench_expand_igraph),
            ("PageRank", bench_pagerank_gf, bench_pagerank_igraph),
            ("Connected Comps.", bench_cc_gf, bench_cc_igraph),
        ]

        for op_name, gf_fn, ig_fn in ops:
            t_gf = gf_fn(graph_gf)
            t_ig = ig_fn(graph_ig)
            sp = speedup(t_gf, t_ig)
            print(f"{n:>8,}  {op_name:<30}  {fmt(t_gf):>12}  {fmt(t_ig):>12}  {sp:>10}")
            rows.append((n, op_name, t_gf, t_ig))

    # ── optional: write markdown output ──────────────────────────────────────
    if args.output is not None:
        out_path = args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        should_write = True
    else:
        out_dir = pathlib.Path(__file__).parents[2] / "docs" / "benchmarks"
        out_path = out_dir / "bench_vs_igraph.md"
        should_write = out_dir.exists()
    if should_write:
        lines = [
            "# Graphframe vs igraph Benchmark (TST-011)\n",
            f"| {'N':>8} | {'Operation':<30} | {'Graphframe':>12} | {'igraph':>12} | {'Speedup':>10} |\n",
            f"|{'-' * 10}|{'-' * 32}|{'-' * 14}|{'-' * 14}|{'-' * 12}|\n",
        ]
        for n, op, t_gf, t_ig in rows:
            lines.append(
                f"| {n:>8,} | {op:<30} | {fmt(t_gf):>12} | {fmt(t_ig):>12} | {speedup(t_gf, t_ig):>10} |\n"
            )
        out_path.write_text("".join(lines))
        print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
