mod common;

use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lynxes_core::{
    Direction, EdgeFrame, EdgeTypeSpec, GraphFrame, NodeFrame, ShortestPathConfig,
    COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE, COL_NODE_ID,
};

use common::{label_field, labels_array};

fn nodes(ids: &[&str]) -> NodeFrame {
    let labels: Vec<&[&str]> = ids.iter().map(|_| &["Node"][..]).collect();
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        label_field(),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(ids.to_vec())) as ArrayRef,
            Arc::new(labels_array(&labels)) as ArrayRef,
        ],
    )
    .unwrap();
    NodeFrame::from_record_batch(batch).unwrap()
}

fn weighted_edges(src: &[&str], dst: &[&str], edge_type: &[&str], weights: &[i64]) -> EdgeFrame {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_EDGE_SRC, DataType::Utf8, false),
        Field::new(COL_EDGE_DST, DataType::Utf8, false),
        Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
        Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
        Field::new("weight", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(src.to_vec())) as ArrayRef,
            Arc::new(StringArray::from(dst.to_vec())) as ArrayRef,
            Arc::new(StringArray::from(edge_type.to_vec())) as ArrayRef,
            Arc::new(arrow_array::Int8Array::from(vec![0i8; src.len()])) as ArrayRef,
            Arc::new(Int64Array::from(weights.to_vec())) as ArrayRef,
        ],
    )
    .unwrap();
    EdgeFrame::from_record_batch(batch).unwrap()
}

fn config() -> ShortestPathConfig {
    ShortestPathConfig {
        weight_col: Some("weight".to_owned()),
        edge_type: EdgeTypeSpec::Any,
        direction: Direction::Out,
    }
}

#[test]
fn k_shortest_paths_returns_ordered_paths() {
    let graph = GraphFrame::new(
        nodes(&["a", "b", "c", "d", "e"]),
        weighted_edges(
            &["a", "b", "a", "c", "a", "e"],
            &["b", "d", "c", "d", "e", "d"],
            &["ROAD", "ROAD", "ROAD", "ROAD", "ROAD", "ROAD"],
            &[1, 1, 1, 2, 2, 2],
        ),
    )
    .unwrap();

    let paths = graph
        .k_shortest_paths("a", "d", 3, None, &config())
        .unwrap();

    assert_eq!(
        paths,
        vec![
            vec!["a".to_owned(), "b".to_owned(), "d".to_owned()],
            vec!["a".to_owned(), "c".to_owned(), "d".to_owned()],
            vec!["a".to_owned(), "e".to_owned(), "d".to_owned()],
        ]
    );
}

#[test]
fn k_shortest_paths_respects_max_hops() {
    let graph = GraphFrame::new(
        nodes(&["a", "b", "c", "d"]),
        weighted_edges(
            &["a", "b", "a"],
            &["b", "d", "d"],
            &["ROAD", "ROAD", "ROAD"],
            &[1, 1, 5],
        ),
    )
    .unwrap();

    let paths = graph
        .k_shortest_paths("a", "d", 3, Some(1), &config())
        .unwrap();

    assert_eq!(paths, vec![vec!["a".to_owned(), "d".to_owned()]]);
}

#[test]
fn k_shortest_paths_returns_empty_for_zero_k() {
    let graph = GraphFrame::new(
        nodes(&["a", "b"]),
        weighted_edges(&["a"], &["b"], &["ROAD"], &[1]),
    )
    .unwrap();

    let paths = graph
        .k_shortest_paths("a", "b", 0, None, &config())
        .unwrap();
    assert!(paths.is_empty());
}
