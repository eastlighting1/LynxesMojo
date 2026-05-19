#![allow(dead_code)]

use std::sync::Arc;

use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::{ArrayRef, Int64Array, Int8Array, ListArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lynxes_core::{
    EdgeFrame, GraphFrame, NodeFrame, COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC,
    COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
};

pub fn labels_array(values: &[&[&str]]) -> ListArray {
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

pub fn label_field() -> Field {
    Field::new(
        COL_NODE_LABEL,
        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
        false,
    )
}

pub fn two_node_batch() -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        label_field(),
        Field::new("age", DataType::Int64, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["alice", "bob"])) as ArrayRef,
            Arc::new(labels_array(&[&["Person", "Employee"], &["Person"]])) as ArrayRef,
            Arc::new(Int64Array::from(vec![Some(30), None])) as ArrayRef,
        ],
    )
    .unwrap()
}

pub fn minimal_node_batch() -> RecordBatch {
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

pub fn another_two_node_batch() -> RecordBatch {
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

pub fn minimal_edge_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new(COL_EDGE_SRC, DataType::Utf8, false),
        Field::new(COL_EDGE_DST, DataType::Utf8, false),
        Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
        Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
    ]))
}

pub fn three_edge_batch() -> RecordBatch {
    RecordBatch::try_new(
        minimal_edge_schema(),
        vec![
            Arc::new(StringArray::from(vec!["alice", "bob", "alice"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["bob", "charlie", "charlie"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["KNOWS", "KNOWS", "LIKES"])) as ArrayRef,
            Arc::new(Int8Array::from(vec![0i8, 0, 0])) as ArrayRef,
        ],
    )
    .unwrap()
}

pub fn edge_batch_with_since() -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_EDGE_SRC, DataType::Utf8, false),
        Field::new(COL_EDGE_DST, DataType::Utf8, false),
        Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
        Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
        Field::new("since", DataType::Int64, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["alice", "bob"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["bob", "charlie"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["KNOWS", "KNOWS"])) as ArrayRef,
            Arc::new(Int8Array::from(vec![0i8, 0])) as ArrayRef,
            Arc::new(Int64Array::from(vec![Some(2020), Some(2021)])) as ArrayRef,
        ],
    )
    .unwrap()
}

pub fn graph_node_batch() -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        label_field(),
        Field::new("age", DataType::Int64, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["alice", "bob", "charlie", "diana"])) as ArrayRef,
            Arc::new(labels_array(&[
                &["Person", "Employee"],
                &["Person"],
                &["Person"],
                &["Animal"],
            ])) as ArrayRef,
            Arc::new(Int64Array::from(vec![
                Some(30),
                Some(20),
                Some(25),
                Some(5),
            ])) as ArrayRef,
        ],
    )
    .unwrap()
}

pub fn graph_edge_batch() -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_EDGE_SRC, DataType::Utf8, false),
        Field::new(COL_EDGE_DST, DataType::Utf8, false),
        Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
        Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
        Field::new("since", DataType::Int64, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["alice", "bob", "alice", "diana"])) as ArrayRef,
            Arc::new(StringArray::from(vec![
                "bob", "charlie", "charlie", "alice",
            ])) as ArrayRef,
            Arc::new(StringArray::from(vec!["KNOWS", "KNOWS", "LIKES", "OWNS"])) as ArrayRef,
            Arc::new(Int8Array::from(vec![0i8, 0, 0, 0])) as ArrayRef,
            Arc::new(Int64Array::from(vec![
                Some(2020),
                Some(2021),
                Some(2022),
                Some(2023),
            ])) as ArrayRef,
        ],
    )
    .unwrap()
}

pub fn sample_graph() -> GraphFrame {
    let nodes = NodeFrame::from_record_batch(graph_node_batch()).unwrap();
    let edges = EdgeFrame::from_record_batch(graph_edge_batch()).unwrap();
    GraphFrame::new(nodes, edges).unwrap()
}
