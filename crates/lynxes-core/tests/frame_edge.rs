mod common;

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Float64Array, Int64Array, Int8Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use common::{edge_batch_with_since, graph_node_batch, minimal_edge_schema, three_edge_batch};
use lynxes_core::{
    EdgeFrame, GFError, NodeFrame, COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE,
};

#[test]
fn builds_from_valid_batch() {
    let frame = EdgeFrame::from_record_batch(three_edge_batch()).unwrap();
    assert_eq!(frame.len(), 3);
    assert!(!frame.is_empty());
}

#[test]
fn rejects_missing_src_column() {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_EDGE_DST, DataType::Utf8, false),
        Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
        Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["bob"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["KNOWS"])) as ArrayRef,
            Arc::new(Int8Array::from(vec![0i8])) as ArrayRef,
        ],
    )
    .unwrap();

    let err = EdgeFrame::from_record_batch(batch).unwrap_err();
    assert!(matches!(err, GFError::MissingReservedColumn { column } if column == COL_EDGE_SRC));
}

#[test]
fn rejects_invalid_direction_value() {
    let batch = RecordBatch::try_new(
        minimal_edge_schema(),
        vec![
            Arc::new(StringArray::from(vec!["alice"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["bob"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["KNOWS"])) as ArrayRef,
            Arc::new(Int8Array::from(vec![9i8])) as ArrayRef,
        ],
    )
    .unwrap();

    let err = EdgeFrame::from_record_batch(batch).unwrap_err();
    assert!(matches!(err, GFError::InvalidDirection { value: 9 }));
}

#[test]
fn out_neighbors_and_edge_ids_follow_csr() {
    let frame = EdgeFrame::from_record_batch(three_edge_batch()).unwrap();
    let alice = frame.node_row_idx("alice").unwrap();

    assert_eq!(frame.out_degree(alice), 2);
    assert_eq!(frame.out_neighbors(alice).len(), 2);
    assert_eq!(frame.out_edge_ids(alice).len(), 2);
}

#[test]
fn in_neighbors_use_lazy_reverse_csr() {
    let frame = EdgeFrame::from_record_batch(three_edge_batch()).unwrap();
    let charlie = frame.node_row_idx("charlie").unwrap();

    assert_eq!(frame.in_degree(charlie), 2);
    assert_eq!(frame.in_neighbors(charlie).len(), 2);
    assert_eq!(frame.in_edge_ids(charlie).len(), 2);
}

#[test]
fn filter_by_type_returns_matching_edges() {
    let frame = EdgeFrame::from_record_batch(three_edge_batch()).unwrap();
    let knows = frame.filter_by_type("KNOWS").unwrap();

    assert_eq!(knows.len(), 2);
}

#[test]
fn filter_rebuilds_indexes() {
    let frame = EdgeFrame::from_record_batch(three_edge_batch()).unwrap();
    let filtered = frame
        .filter(&arrow_array::BooleanArray::from(vec![false, true, false]))
        .unwrap();

    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered.node_count(), 2);
    assert!(filtered.node_row_idx("bob").is_some());
    assert!(filtered.node_row_idx("alice").is_none());
}

#[test]
fn select_preserves_reserved_columns() {
    let frame = EdgeFrame::from_record_batch(edge_batch_with_since()).unwrap();
    let selected = frame.select(&["since"]).unwrap();

    assert_eq!(
        selected.column_names(),
        vec![
            COL_EDGE_SRC,
            COL_EDGE_DST,
            COL_EDGE_TYPE,
            COL_EDGE_DIRECTION,
            "since"
        ]
    );
}

#[test]
fn select_rejects_missing_column() {
    let frame = EdgeFrame::from_record_batch(edge_batch_with_since()).unwrap();
    let err = frame.select(&["missing"]).unwrap_err();

    assert!(matches!(err, GFError::ColumnNotFound { column } if column == "missing"));
}

#[test]
fn concat_fills_missing_columns_with_nulls() {
    let a = EdgeFrame::from_record_batch(edge_batch_with_since()).unwrap();
    let b = EdgeFrame::from_record_batch(three_edge_batch()).unwrap();
    let result = EdgeFrame::concat(&[&a, &b]).unwrap();

    let since = result
        .column("since")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(result.len(), 5);
    assert!(!since.is_null(0));
    assert!(since.is_null(4));
}

#[test]
fn concat_rejects_incompatible_types() {
    let a = EdgeFrame::from_record_batch(edge_batch_with_since()).unwrap();
    let schema_b = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_EDGE_SRC, DataType::Utf8, false),
        Field::new(COL_EDGE_DST, DataType::Utf8, false),
        Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
        Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
        Field::new("since", DataType::Float64, true),
    ]));
    let batch_b = RecordBatch::try_new(
        schema_b,
        vec![
            Arc::new(StringArray::from(vec!["x"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["y"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["LINKS"])) as ArrayRef,
            Arc::new(Int8Array::from(vec![0i8])) as ArrayRef,
            Arc::new(Float64Array::from(vec![Some(1.5f64)])) as ArrayRef,
        ],
    )
    .unwrap();
    let b = EdgeFrame::from_record_batch(batch_b).unwrap();

    let err = EdgeFrame::concat(&[&a, &b]).unwrap_err();
    assert!(matches!(err, GFError::TypeMismatch { .. }));
}

#[test]
fn with_nodes_rehydrates_valid_graph() {
    let edges = EdgeFrame::from_record_batch(edge_batch_with_since()).unwrap();
    let nodes = NodeFrame::from_record_batch(graph_node_batch()).unwrap();

    let graph = edges.with_nodes(nodes).unwrap();

    assert_eq!(graph.node_count(), 4);
    assert_eq!(graph.edge_count(), 2);
}
