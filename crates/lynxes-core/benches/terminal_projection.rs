use std::sync::Arc;
use std::time::Duration;

use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::{ArrayRef, Int64Array, Int8Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lynxes_core::{
    DisplayOptions, DisplayView, EdgeFrame, GraphFrame, NodeFrame, COL_EDGE_DIRECTION,
    COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
};

fn make_projection_graph(nodes: usize, edges: usize) -> GraphFrame {
    let ids: Vec<String> = (0..nodes).map(|idx| format!("n{idx}")).collect();
    let mut labels = ListBuilder::new(StringBuilder::new());
    let ages: Vec<Option<i64>> = (0..nodes).map(|idx| Some((idx % 97) as i64)).collect();
    for idx in 0..nodes {
        labels
            .values()
            .append_value(if idx % 10 == 0 { "Company" } else { "Person" });
        labels.append(true);
    }
    let node_batch = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            Field::new(COL_NODE_ID, DataType::Utf8, false),
            Field::new(
                COL_NODE_LABEL,
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                false,
            ),
            Field::new("age", DataType::Int64, true),
        ])),
        vec![
            Arc::new(StringArray::from(ids.clone())) as ArrayRef,
            Arc::new(labels.finish()) as ArrayRef,
            Arc::new(Int64Array::from(ages)) as ArrayRef,
        ],
    )
    .unwrap();

    let srcs: Vec<String> = (0..edges).map(|idx| format!("n{}", idx % nodes)).collect();
    let dsts: Vec<String> = (0..edges)
        .map(|idx| format!("n{}", (idx * 7 + 3) % nodes))
        .collect();
    let types: Vec<&str> = (0..edges)
        .map(|idx| if idx % 5 == 0 { "WORKS_AT" } else { "KNOWS" })
        .collect();
    let since: Vec<Option<i64>> = (0..edges)
        .map(|idx| Some(2020 + (idx % 5) as i64))
        .collect();
    let edge_batch = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            Field::new(COL_EDGE_SRC, DataType::Utf8, false),
            Field::new(COL_EDGE_DST, DataType::Utf8, false),
            Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
            Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
            Field::new("since", DataType::Int64, true),
        ])),
        vec![
            Arc::new(StringArray::from(srcs)) as ArrayRef,
            Arc::new(StringArray::from(dsts)) as ArrayRef,
            Arc::new(StringArray::from(types)) as ArrayRef,
            Arc::new(Int8Array::from(vec![0i8; edges])) as ArrayRef,
            Arc::new(Int64Array::from(since)) as ArrayRef,
        ],
    )
    .unwrap();

    let nodes = NodeFrame::from_record_batch(node_batch).unwrap();
    let edges = EdgeFrame::from_record_batch(edge_batch).unwrap();
    GraphFrame::new(nodes, edges).unwrap()
}

fn full_bench_enabled() -> bool {
    std::env::var_os("LYNXES_FULL_BENCH").is_some_and(|value| value == "1")
}

fn bench_terminal_projection(c: &mut Criterion) {
    let (nodes, edges) = if full_bench_enabled() {
        (50_000, 200_000)
    } else {
        (5_000, 20_000)
    };
    let graph = make_projection_graph(nodes, edges);

    let mut group = c.benchmark_group("terminal_projection");
    if !full_bench_enabled() {
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(2));
    }

    group.bench_function("table_preview", |b| {
        b.iter(|| {
            black_box(
                graph
                    .display_slice(DisplayOptions {
                        view: DisplayView::Table,
                        max_rows: 10,
                        width: Some(100),
                        sort_by: None,
                        expand_attrs: true,
                        attrs: vec!["since".to_owned()],
                    })
                    .unwrap(),
            )
        })
    });

    group.bench_function("glimpse", |b| {
        b.iter(|| {
            black_box(
                graph
                    .display_glimpse(DisplayOptions {
                        view: DisplayView::Head,
                        max_rows: 3,
                        width: Some(100),
                        sort_by: None,
                        expand_attrs: true,
                        attrs: vec!["since".to_owned()],
                    })
                    .unwrap(),
            )
        })
    });

    group.finish();
}

criterion_group!(benches, bench_terminal_projection);
criterion_main!(benches);
