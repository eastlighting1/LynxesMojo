use std::sync::Arc;

use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::{Array, ArrayRef, BooleanArray, Int64Array, ListArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lynxes_core::{
    EdgeFrame, GFError, NodeFrame, COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE,
    COL_NODE_ID, COL_NODE_LABEL,
};

fn labels_array(values: &[&[&str]]) -> ListArray {
    let value_builder = StringBuilder::new();
    let mut builder = ListBuilder::new(value_builder);
    for labels in values {
        for label in *labels {
            builder.values().append_value(label);
        }
        builder.append(true);
    }
    builder.finish()
}

fn label_field() -> Field {
    Field::new(
        COL_NODE_LABEL,
        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
        false,
    )
}

fn two_node_batch() -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        label_field(),
        Field::new("age", DataType::Int64, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["alice", "bob"])) as ArrayRef,
            Arc::new(labels_array(&[&["Person"], &["Person"]])) as ArrayRef,
            Arc::new(Int64Array::from(vec![Some(30), None])) as ArrayRef,
        ],
    )
    .unwrap()
}

fn minimal_node_batch() -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        label_field(),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["solo"])) as ArrayRef,
            Arc::new(labels_array(&[&["Thing"]])) as ArrayRef,
        ],
    )
    .unwrap()
}

fn another_two_node_batch() -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        label_field(),
        Field::new("age", DataType::Int64, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["charlie", "diana"])) as ArrayRef,
            Arc::new(labels_array(&[&["Person"], &["Animal"]])) as ArrayRef,
            Arc::new(Int64Array::from(vec![Some(25), Some(5)])) as ArrayRef,
        ],
    )
    .unwrap()
}

#[test]
fn from_record_batch_builds_id_index() {
    let frame = NodeFrame::from_record_batch(two_node_batch()).unwrap();
    assert_eq!(frame.row_index("alice"), Some(0));
    assert_eq!(frame.row_index("bob"), Some(1));
    assert_eq!(frame.row_index("missing"), None);
}

#[test]
fn from_record_batch_rejects_missing_id_column() {
    let schema = Arc::new(ArrowSchema::new(vec![label_field()]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(labels_array(&[&["Person"]])) as ArrayRef],
    )
    .unwrap();

    let err = NodeFrame::from_record_batch(batch).unwrap_err();
    assert!(matches!(err, GFError::MissingReservedColumn { column } if column == COL_NODE_ID));
}

#[test]
fn from_record_batch_rejects_missing_label_column() {
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        COL_NODE_ID,
        DataType::Utf8,
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(StringArray::from(vec!["alice"])) as ArrayRef],
    )
    .unwrap();

    let err = NodeFrame::from_record_batch(batch).unwrap_err();
    assert!(matches!(err, GFError::MissingReservedColumn { column } if column == COL_NODE_LABEL));
}

#[test]
fn from_record_batch_rejects_wrong_id_type() {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Int64, false),
        label_field(),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1i64])) as ArrayRef,
            Arc::new(labels_array(&[&["Person"]])) as ArrayRef,
        ],
    )
    .unwrap();

    let err = NodeFrame::from_record_batch(batch).unwrap_err();
    assert!(matches!(err, GFError::ReservedColumnType { column, .. } if column == COL_NODE_ID));
}

#[test]
fn from_record_batch_rejects_null_id_values() {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Utf8, true),
        label_field(),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec![Some("alice"), None])) as ArrayRef,
            Arc::new(labels_array(&[&["Person"], &["Person"]])) as ArrayRef,
        ],
    )
    .unwrap();

    let err = NodeFrame::from_record_batch(batch).unwrap_err();
    assert!(matches!(err, GFError::ReservedColumnType { column, .. } if column == COL_NODE_ID));
}

#[test]
fn from_record_batch_rejects_wrong_label_type() {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        Field::new(COL_NODE_LABEL, DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["alice"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["Person"])) as ArrayRef,
        ],
    )
    .unwrap();

    let err = NodeFrame::from_record_batch(batch).unwrap_err();
    assert!(matches!(err, GFError::ReservedColumnType { column, .. } if column == COL_NODE_LABEL));
}

#[test]
fn from_record_batch_rejects_duplicate_ids() {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        label_field(),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["alice", "alice"])) as ArrayRef,
            Arc::new(labels_array(&[&["Person"], &["Person"]])) as ArrayRef,
        ],
    )
    .unwrap();

    let err = NodeFrame::from_record_batch(batch).unwrap_err();
    assert!(matches!(err, GFError::DuplicateNodeId { id } if id == "alice"));
}

#[test]
fn row_returns_single_row_for_known_id() {
    let frame = NodeFrame::from_record_batch(two_node_batch()).unwrap();
    let row = frame.row("alice").unwrap();
    assert_eq!(row.num_rows(), 1);
}

#[test]
fn row_returns_none_for_unknown_id() {
    let frame = NodeFrame::from_record_batch(two_node_batch()).unwrap();
    assert!(frame.row("nobody").is_none());
}

#[test]
fn gather_rows_returns_requested_rows_in_given_order() {
    let frame = NodeFrame::from_record_batch(two_node_batch()).unwrap();

    let gathered = frame.gather_rows(&[1, 0, 1]).unwrap();
    let ids = gathered
        .column_by_name(COL_NODE_ID)
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    assert_eq!(gathered.num_rows(), 3);
    assert_eq!(
        ids.iter().map(|v| v.unwrap()).collect::<Vec<_>>(),
        vec!["bob", "alice", "bob"]
    );
}

#[test]
fn gather_rows_returns_empty_batch_for_empty_input() {
    let frame = NodeFrame::from_record_batch(two_node_batch()).unwrap();

    let gathered = frame.gather_rows(&[]).unwrap();

    assert_eq!(gathered.num_rows(), 0);
    assert_eq!(
        gathered.num_columns(),
        frame.to_record_batch().num_columns()
    );
    assert_eq!(gathered.schema_ref(), frame.to_record_batch().schema_ref());
}

#[test]
fn gather_rows_rejects_out_of_bounds_indices() {
    let frame = NodeFrame::from_record_batch(two_node_batch()).unwrap();

    let err = frame.gather_rows(&[2]).unwrap_err();

    assert!(matches!(err, GFError::InvalidConfig { message } if message.contains("out of bounds")));
}

#[test]
fn filter_applies_mask_and_rebuilds_index() {
    let frame = NodeFrame::from_record_batch(two_node_batch()).unwrap();
    let filtered = frame
        .filter(&BooleanArray::from(vec![false, true]))
        .unwrap();

    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered.row_index("bob"), Some(0));
    assert_eq!(filtered.row_index("alice"), None);
}

#[test]
fn filter_rejects_length_mismatch() {
    let frame = NodeFrame::from_record_batch(two_node_batch()).unwrap();
    let err = frame.filter(&BooleanArray::from(vec![true])).unwrap_err();

    assert!(matches!(
        err,
        GFError::LengthMismatch {
            expected: 2,
            actual: 1
        }
    ));
}

#[test]
fn select_returns_requested_subset_with_reserved_columns() {
    let frame = NodeFrame::from_record_batch(two_node_batch()).unwrap();
    let selected = frame.select(&["age"]).unwrap();

    assert_eq!(
        selected.column_names(),
        vec![COL_NODE_ID, COL_NODE_LABEL, "age"]
    );
    assert_eq!(selected.row_index("alice"), Some(0));
}

#[test]
fn select_rejects_missing_column() {
    let frame = NodeFrame::from_record_batch(two_node_batch()).unwrap();
    let err = frame.select(&["missing"]).unwrap_err();

    assert!(matches!(err, GFError::ColumnNotFound { column } if column == "missing"));
}

#[test]
fn concat_rejects_duplicate_ids_across_frames() {
    let a = NodeFrame::from_record_batch(two_node_batch()).unwrap();
    let b = NodeFrame::from_record_batch(two_node_batch()).unwrap();

    let err = NodeFrame::concat(&[&a, &b]).unwrap_err();
    assert!(matches!(err, GFError::DuplicateNodeId { id } if id == "alice"));
}

#[test]
fn concat_fills_missing_columns_with_nulls() {
    let a = NodeFrame::from_record_batch(two_node_batch()).unwrap();
    let b = NodeFrame::from_record_batch(minimal_node_batch()).unwrap();
    let result = NodeFrame::concat(&[&a, &b]).unwrap();

    assert_eq!(result.len(), 3);
    let age = result
        .column("age")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert!(age.is_null(2));
}

#[test]
fn concat_keeps_row_order_by_input_frames() {
    let a = NodeFrame::from_record_batch(two_node_batch()).unwrap();
    let b = NodeFrame::from_record_batch(another_two_node_batch()).unwrap();
    let result = NodeFrame::concat(&[&a, &b]).unwrap();

    assert_eq!(result.row_index("alice"), Some(0));
    assert_eq!(result.row_index("bob"), Some(1));
    assert_eq!(result.row_index("charlie"), Some(2));
    assert_eq!(result.row_index("diana"), Some(3));
}

#[test]
fn slice_rebuilds_id_index() {
    let frame = NodeFrame::from_record_batch(two_node_batch()).unwrap();
    let sliced = frame.slice(1, 1);

    assert_eq!(sliced.row_index("bob"), Some(0));
    assert_eq!(sliced.row_index("alice"), None);
}

#[test]
fn difference_rejects_schema_mismatch() {
    let a = NodeFrame::from_record_batch(two_node_batch()).unwrap();
    let b = NodeFrame::from_record_batch(minimal_node_batch()).unwrap();

    let err = a.difference(&b).unwrap_err();
    assert!(matches!(err, GFError::SchemaMismatch { .. }));
}

#[test]
fn intersect_returns_common_ids() {
    let a = NodeFrame::from_record_batch(two_node_batch()).unwrap();
    let b = NodeFrame::from_record_batch(
        RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                Field::new(COL_NODE_ID, DataType::Utf8, false),
                label_field(),
                Field::new("age", DataType::Int64, true),
            ])),
            vec![
                Arc::new(StringArray::from(vec!["bob", "charlie"])) as ArrayRef,
                Arc::new(labels_array(&[&["Person"], &["Person"]])) as ArrayRef,
                Arc::new(Int64Array::from(vec![Some(20), Some(30)])) as ArrayRef,
            ],
        )
        .unwrap(),
    )
    .unwrap();

    let result = a.intersect(&b).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result.row_index("bob"), Some(0));
}

#[test]
fn with_edges_rehydrates_valid_graph() {
    let nodes = NodeFrame::from_record_batch(
        RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                Field::new(COL_NODE_ID, DataType::Utf8, false),
                label_field(),
            ])),
            vec![
                Arc::new(StringArray::from(vec!["alice", "bob"])) as ArrayRef,
                Arc::new(labels_array(&[&["Person"], &["Person"]])) as ArrayRef,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let edges = EdgeFrame::from_record_batch(
        RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                Field::new(COL_EDGE_SRC, DataType::Utf8, false),
                Field::new(COL_EDGE_DST, DataType::Utf8, false),
                Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
                Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
            ])),
            vec![
                Arc::new(StringArray::from(vec!["alice"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["bob"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["KNOWS"])) as ArrayRef,
                Arc::new(arrow_array::Int8Array::from(vec![0i8])) as ArrayRef,
            ],
        )
        .unwrap(),
    )
    .unwrap();

    let graph = nodes.with_edges(edges).unwrap();

    assert_eq!(graph.node_count(), 2);
    assert_eq!(graph.edge_count(), 1);
}
