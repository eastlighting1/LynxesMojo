use std::sync::Arc;
use std::time::Duration;

use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::{ArrayRef, Int8Array, ListArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use lynxes_core::{
    Direction, EdgeFrame, EdgeTypeSpec, GraphFrame, NodeFrame, Pattern, PatternStep,
    COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
};
use lynxes_lazy::LazyGraphFrame;

fn node_id(idx: u32) -> String {
    format!("n{idx}")
}

fn labels_array(count: usize) -> ListArray {
    let mut builder = ListBuilder::new(StringBuilder::new());
    for _ in 0..count {
        builder.values().append_value("Entity");
        builder.append(true);
    }
    builder.finish()
}

fn typed_graph(node_count: u32) -> GraphFrame {
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

    let edge_count = (node_count * 3) as usize;
    let mut srcs = Vec::with_capacity(edge_count);
    let mut dsts = Vec::with_capacity(edge_count);
    let mut types = Vec::with_capacity(edge_count);

    for src in 0..node_count {
        srcs.push(node_id(src));
        dsts.push(node_id((src + 1) % node_count));
        types.push("REL1");

        srcs.push(node_id(src));
        dsts.push(node_id((src + 2) % node_count));
        types.push("REL2");

        srcs.push(node_id(src));
        dsts.push(node_id((src + 3) % node_count));
        types.push("REL3");
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

fn two_hop_pattern() -> Pattern {
    Pattern::new(vec![
        PatternStep {
            from_alias: "a".to_owned(),
            edge_alias: Some("e1".to_owned()),
            edge_type: EdgeTypeSpec::Single("REL1".to_owned()),
            direction: Direction::Out,
            to_alias: "b".to_owned(),
        },
        PatternStep {
            from_alias: "b".to_owned(),
            edge_alias: Some("e2".to_owned()),
            edge_type: EdgeTypeSpec::Single("REL2".to_owned()),
            direction: Direction::Out,
            to_alias: "c".to_owned(),
        },
    ])
}

fn three_hop_pattern() -> Pattern {
    Pattern::new(vec![
        PatternStep {
            from_alias: "a".to_owned(),
            edge_alias: Some("e1".to_owned()),
            edge_type: EdgeTypeSpec::Single("REL1".to_owned()),
            direction: Direction::Out,
            to_alias: "b".to_owned(),
        },
        PatternStep {
            from_alias: "b".to_owned(),
            edge_alias: Some("e2".to_owned()),
            edge_type: EdgeTypeSpec::Single("REL2".to_owned()),
            direction: Direction::Out,
            to_alias: "c".to_owned(),
        },
        PatternStep {
            from_alias: "c".to_owned(),
            edge_alias: Some("e3".to_owned()),
            edge_type: EdgeTypeSpec::Single("REL3".to_owned()),
            direction: Direction::Out,
            to_alias: "d".to_owned(),
        },
    ])
}

fn full_bench_enabled() -> bool {
    std::env::var_os("LYNXES_FULL_BENCH").is_some_and(|value| value == "1")
}

fn bench_pattern_match_two_hop(c: &mut Criterion) {
    let node_count = if full_bench_enabled() { 5_000 } else { 500 };
    let graph = typed_graph(node_count);
    let pattern = two_hop_pattern();

    let mut group = c.benchmark_group("kg_pattern_match");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(graph.node_count() as u64));
    group.bench_function("2hop_typed", |b| {
        b.iter(|| {
            let batch = LazyGraphFrame::from_graph(black_box(&graph))
                .match_pattern(black_box(pattern.clone()), None)
                .collect_pattern_rows()
                .unwrap();
            black_box(batch.num_rows())
        });
    });
    group.finish();
}

fn bench_pattern_match_three_hop(c: &mut Criterion) {
    let node_count = if full_bench_enabled() { 5_000 } else { 500 };
    let graph = typed_graph(node_count);
    let pattern = three_hop_pattern();

    let mut group = c.benchmark_group("kg_pattern_match");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(graph.node_count() as u64));
    group.bench_function("3hop_typed", |b| {
        b.iter(|| {
            let batch = LazyGraphFrame::from_graph(black_box(&graph))
                .match_pattern(black_box(pattern.clone()), None)
                .collect_pattern_rows()
                .unwrap();
            black_box(batch.num_rows())
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_pattern_match_two_hop,
    bench_pattern_match_three_hop,
);
criterion_main!(benches);
