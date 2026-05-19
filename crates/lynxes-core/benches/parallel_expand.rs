// Benchmark: serial vs. Rayon-parallel BFS expansion  (OPT-003)
//
// Measures `LazyGraphFrame::expand().collect_with_options(partition_parallel=true)`
// against the default serial collect on a ring graph of N nodes.
//
// The ring gives every node exactly one out-edge, so there is zero redundant
// work between shards.  This shows the overhead-free upper bound of speedup
// from frontier partitioning.

use std::sync::Arc;
use std::time::Duration;

use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::{Int8Array, ListArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use lynxes_core::{
    Direction, EdgeFrame, EdgeTypeSpec, GraphFrame, NodeFrame, OptimizerOptions,
    COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
};
use lynxes_lazy::LazyGraphFrame;

// ── Graph builder ─────────────────────────────────────────────────────────────

fn labels_array(n: usize) -> ListArray {
    let mut builder = ListBuilder::new(StringBuilder::new());
    for _ in 0..n {
        builder.values().append_value("Node");
        builder.append(true);
    }
    builder.finish()
}

/// Ring graph: node i → node (i+1) % n.  Every node is in the frontier when
/// we call expand from the full scan, giving maximum parallelism potential.
fn make_graph(n: u32) -> GraphFrame {
    let node_ids: Vec<String> = (0..n).map(|i| i.to_string()).collect();
    let node_strs: Vec<&str> = node_ids.iter().map(String::as_str).collect();

    let node_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        Field::new(
            COL_NODE_LABEL,
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            false,
        ),
    ]));
    let nodes = NodeFrame::from_record_batch(
        RecordBatch::try_new(
            node_schema,
            vec![
                Arc::new(StringArray::from(node_strs.clone())) as Arc<dyn arrow_array::Array>,
                Arc::new(labels_array(n as usize)) as Arc<dyn arrow_array::Array>,
            ],
        )
        .unwrap(),
    )
    .unwrap();

    let srcs: Vec<String> = (0..n).map(|i| i.to_string()).collect();
    let dsts: Vec<String> = (0..n).map(|i| ((i + 1) % n).to_string()).collect();
    let src_strs: Vec<&str> = srcs.iter().map(String::as_str).collect();
    let dst_strs: Vec<&str> = dsts.iter().map(String::as_str).collect();
    let edge_count = n as usize;

    let edge_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_EDGE_SRC, DataType::Utf8, false),
        Field::new(COL_EDGE_DST, DataType::Utf8, false),
        Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
        Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
    ]));
    let edges = EdgeFrame::from_record_batch(
        RecordBatch::try_new(
            edge_schema,
            vec![
                Arc::new(StringArray::from(src_strs)) as Arc<dyn arrow_array::Array>,
                Arc::new(StringArray::from(dst_strs)) as Arc<dyn arrow_array::Array>,
                Arc::new(StringArray::from(vec!["E"; edge_count])) as Arc<dyn arrow_array::Array>,
                Arc::new(Int8Array::from(vec![0i8; edge_count])) as Arc<dyn arrow_array::Array>,
            ],
        )
        .unwrap(),
    )
    .unwrap();

    GraphFrame::new(nodes, edges).unwrap()
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

fn full_bench_enabled() -> bool {
    std::env::var_os("LYNXES_FULL_BENCH").is_some_and(|value| value == "1")
}

fn bench_parallel_expand(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_expand");
    if !full_bench_enabled() {
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(2));
    }

    let serial_opts = OptimizerOptions {
        predicate_pushdown: false,
        projection_pushdown: false,
        traversal_pruning: false,
        subgraph_caching: false,
        early_termination: false,
        partition_parallel: false,
        pattern_expansion: true,
    };

    let parallel_opts = OptimizerOptions {
        partition_parallel: true,
        ..serial_opts
    };

    let sizes: &[u32] = if full_bench_enabled() {
        &[1_000, 10_000, 50_000]
    } else {
        &[1_000, 10_000]
    };

    for &n in sizes {
        let graph = make_graph(n);

        group.bench_with_input(BenchmarkId::new("serial", n), &graph, |b, g| {
            b.iter(|| {
                let result = LazyGraphFrame::from_graph(black_box(g))
                    .expand(EdgeTypeSpec::Any, 1, Direction::Out)
                    .collect_with_options(serial_opts)
                    .unwrap();
                black_box(result)
            });
        });

        group.bench_with_input(BenchmarkId::new("parallel", n), &graph, |b, g| {
            b.iter(|| {
                let result = LazyGraphFrame::from_graph(black_box(g))
                    .expand(EdgeTypeSpec::Any, 1, Direction::Out)
                    .collect_with_options(parallel_opts)
                    .unwrap();
                black_box(result)
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_parallel_expand);
criterion_main!(benches);
