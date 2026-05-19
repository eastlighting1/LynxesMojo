use std::sync::Arc;
use std::time::Duration;

use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lynxes_core::{
    Direction, EdgeFrame, EdgeTypeSpec, GraphFrame, NodeFrame, ShortestPathConfig,
    COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
};

fn corridor_graph(length: usize, branch_height: usize) -> GraphFrame {
    let mut node_ids = Vec::new();
    let mut src = Vec::new();
    let mut dst = Vec::new();
    let mut weights = Vec::new();

    for x in 0..length {
        node_ids.push(format!("{x}:0"));
        if x + 1 < length {
            let a = format!("{x}:0");
            let b = format!("{}:0", x + 1);
            src.push(a.clone());
            dst.push(b.clone());
            weights.push(1i64);
            src.push(b);
            dst.push(a);
            weights.push(1i64);
        }

        for y in 1..=branch_height {
            node_ids.push(format!("{x}:{y}"));
            let a = format!("{x}:{}", y - 1);
            let b = format!("{x}:{y}");
            src.push(a.clone());
            dst.push(b.clone());
            weights.push(1i64);
            src.push(b);
            dst.push(a);
            weights.push(1i64);
        }
    }

    let node_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        Field::new(
            COL_NODE_LABEL,
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            false,
        ),
    ]));
    let mut label_builder = ListBuilder::new(StringBuilder::new());
    for _ in &node_ids {
        label_builder.values().append_value("Grid");
        label_builder.append(true);
    }
    let node_labels = label_builder.finish();
    let node_batch = RecordBatch::try_new(
        node_schema,
        vec![
            Arc::new(StringArray::from(node_ids.clone())) as ArrayRef,
            Arc::new(node_labels) as ArrayRef,
        ],
    )
    .unwrap();

    let edge_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_EDGE_SRC, DataType::Utf8, false),
        Field::new(COL_EDGE_DST, DataType::Utf8, false),
        Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
        Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
        Field::new("weight", DataType::Int64, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        edge_schema,
        vec![
            Arc::new(StringArray::from(src)) as ArrayRef,
            Arc::new(StringArray::from(dst)) as ArrayRef,
            Arc::new(StringArray::from(vec!["ROAD"; weights.len()])) as ArrayRef,
            Arc::new(arrow_array::Int8Array::from(vec![0i8; weights.len()])) as ArrayRef,
            Arc::new(Int64Array::from(weights)) as ArrayRef,
        ],
    )
    .unwrap();

    let nodes = NodeFrame::from_record_batch(node_batch).unwrap();
    let edges = EdgeFrame::from_record_batch(edge_batch).unwrap();
    GraphFrame::new(nodes, edges).unwrap()
}

fn manhattan(node: &str, dst: &str) -> f64 {
    fn parse(id: &str) -> (i64, i64) {
        let (x, y) = id.split_once(':').unwrap();
        (x.parse().unwrap(), y.parse().unwrap())
    }

    let (x1, y1) = parse(node);
    let (x2, y2) = parse(dst);
    ((x1 - x2).abs() + (y1 - y2).abs()) as f64
}

fn full_bench_enabled() -> bool {
    std::env::var_os("LYNXES_FULL_BENCH").is_some_and(|value| value == "1")
}

fn bench_astar_vs_dijkstra(c: &mut Criterion) {
    let graph = corridor_graph(1000, 8);
    let config = ShortestPathConfig {
        weight_col: Some("weight".to_owned()),
        edge_type: EdgeTypeSpec::Any,
        direction: Direction::Out,
    };
    let src = "0:0";
    let dst = "999:0";

    let mut group = c.benchmark_group("astar_vs_dijkstra_corridor_1000x8");
    if !full_bench_enabled() {
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(2));
    }
    group.bench_function("dijkstra", |b| {
        b.iter(|| {
            black_box(
                graph
                    .shortest_path(black_box(src), black_box(dst), black_box(&config))
                    .unwrap(),
            )
        });
    });
    group.bench_function("astar", |b| {
        b.iter(|| {
            black_box(
                graph
                    .astar_shortest_path(
                        black_box(src),
                        black_box(dst),
                        black_box(&config),
                        Some(&manhattan),
                    )
                    .unwrap(),
            )
        });
    });
    group.finish();
}

criterion_group!(benches, bench_astar_vs_dijkstra);
criterion_main!(benches);
