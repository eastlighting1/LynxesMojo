use std::sync::Arc;
use std::time::Duration;

use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::{ArrayRef, Int8Array, ListArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};

use lynxes_core::{
    Direction, EdgeFrame, EdgeTypeSpec, GraphFrame, NodeFrame, SamplingConfig, COL_EDGE_DIRECTION,
    COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
};

fn node_id(idx: u32) -> String {
    format!("n{idx}")
}

fn labels_array(count: usize) -> ListArray {
    let mut builder = ListBuilder::new(StringBuilder::new());
    for _ in 0..count {
        builder.values().append_value("Node");
        builder.append(true);
    }
    builder.finish()
}

fn graph_with_regular_out_degree(node_count: u32, out_degree: u32) -> GraphFrame {
    let ids = (0..node_count).map(node_id).collect::<Vec<_>>();
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
                Arc::new(StringArray::from(ids)) as ArrayRef,
                Arc::new(labels_array(node_count as usize)) as ArrayRef,
            ],
        )
        .unwrap(),
    )
    .unwrap();

    let edge_count = (node_count * out_degree) as usize;
    let mut srcs = Vec::with_capacity(edge_count);
    let mut dsts = Vec::with_capacity(edge_count);
    let mut types = Vec::with_capacity(edge_count);

    for src in 0..node_count {
        for step in 1..=out_degree {
            srcs.push(node_id(src));
            dsts.push(node_id((src + step) % node_count));
            types.push(if step % 3 == 0 { "ALT" } else { "REL" });
        }
    }

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
                Arc::new(StringArray::from(srcs)) as ArrayRef,
                Arc::new(StringArray::from(dsts)) as ArrayRef,
                Arc::new(StringArray::from(types)) as ArrayRef,
                Arc::new(Int8Array::from(vec![Direction::Out.as_i8(); edge_count])) as ArrayRef,
            ],
        )
        .unwrap(),
    )
    .unwrap();

    GraphFrame::new(nodes, edges).unwrap()
}

fn seed_ids(count: u32) -> Vec<String> {
    (0..count).map(node_id).collect()
}

fn full_bench_enabled() -> bool {
    std::env::var_os("LYNXES_FULL_BENCH").is_some_and(|value| value == "1")
}

fn bench_sample_neighbors(c: &mut Criterion) {
    let (node_count, seed_count) = if full_bench_enabled() {
        (50_000, 1_000)
    } else {
        (10_000, 200)
    };
    let graph = graph_with_regular_out_degree(node_count, 32);
    let seeds = Arc::new(seed_ids(seed_count));
    let seed_refs = Arc::new(seeds.iter().map(|s| s.as_str()).collect::<Vec<_>>());
    let config = SamplingConfig {
        hops: 2,
        fan_out: vec![25, 10],
        direction: Direction::Out,
        edge_type: EdgeTypeSpec::Any,
        replace: false,
    };

    let mut group = c.benchmark_group("gnn_sample_neighbors");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.throughput(Throughput::Elements(seed_refs.len() as u64));
    let bench_id = if full_bench_enabled() {
        "2hop_1000_seeds_25x10"
    } else {
        "2hop_200_seeds_25x10"
    };
    group.bench_function(bench_id, |b| {
        let seed_refs = Arc::clone(&seed_refs);
        b.iter(|| {
            black_box(
                graph
                    .sample_neighbors(black_box(seed_refs.as_slice()), black_box(&config))
                    .unwrap(),
            )
        });
    });
    group.finish();
}

fn bench_to_coo(c: &mut Criterion) {
    let node_count = if full_bench_enabled() {
        100_000
    } else {
        10_000
    };
    let graph = graph_with_regular_out_degree(node_count, 10);

    let mut group = c.benchmark_group("gnn_to_coo");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.throughput(Throughput::Elements(graph.edge_count() as u64));
    let bench_id = if full_bench_enabled() {
        "1m_edges"
    } else {
        "100k_edges"
    };
    group.bench_function(bench_id, |b| {
        b.iter(|| {
            let (src, dst) = graph.to_coo();
            black_box((src.len(), dst.len()))
        });
    });
    group.finish();
}

fn bench_random_walk(c: &mut Criterion) {
    let (node_count, seed_count, length, walks_per_node) = if full_bench_enabled() {
        (50_000, 1_000, 80, 10)
    } else {
        (10_000, 200, 40, 4)
    };
    let graph = graph_with_regular_out_degree(node_count, 32);
    let seeds = Arc::new(seed_ids(seed_count));
    let seed_refs = Arc::new(seeds.iter().map(|s| s.as_str()).collect::<Vec<_>>());

    let mut group = c.benchmark_group("gnn_random_walk");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.throughput(Throughput::Elements(
        (seed_refs.len() * walks_per_node) as u64,
    ));
    let bench_id = if full_bench_enabled() {
        "length_80_walks_10_seeds_1000"
    } else {
        "length_40_walks_4_seeds_200"
    };
    group.bench_function(bench_id, |b| {
        let seed_refs = Arc::clone(&seed_refs);
        b.iter_batched(
            || seed_refs.as_slice(),
            |starts| {
                black_box(
                    graph
                        .random_walk(
                            black_box(starts),
                            black_box(length),
                            black_box(walks_per_node),
                            black_box(Direction::Out),
                            black_box(&EdgeTypeSpec::Any),
                        )
                        .unwrap(),
                )
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_sample_neighbors,
    bench_to_coo,
    bench_random_walk,
);
criterion_main!(benches);
