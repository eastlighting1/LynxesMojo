use std::sync::Arc;

use arrow_array::{
    builder::{ListBuilder, StringBuilder},
    ArrayRef, Int64Array, Int8Array, ListArray, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lynxes_core::{
    Direction, EdgeFrame, GraphFrame, NodeFrame, COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC,
    COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
};
use lynxes_io::{read_gfb, read_gfb_inspect, write_gfb, GfbWriteOptions};

fn labels_array(values: &[&[&str]]) -> ListArray {
    let mut builder = ListBuilder::new(StringBuilder::new());
    for labels in values {
        for label in *labels {
            builder.values().append_value(label);
        }
        builder.append(true);
    }
    builder.finish()
}

fn demo_graph() -> GraphFrame {
    let node_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        Field::new(
            COL_NODE_LABEL,
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            false,
        ),
        Field::new("age", DataType::Int64, true),
    ]));
    let edge_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_EDGE_SRC, DataType::Utf8, false),
        Field::new(COL_EDGE_DST, DataType::Utf8, false),
        Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
        Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
        Field::new("weight", DataType::Int64, true),
    ]));

    let nodes = NodeFrame::from_record_batch(
        RecordBatch::try_new(
            node_schema,
            vec![
                Arc::new(StringArray::from(vec!["alice", "bob"])) as ArrayRef,
                Arc::new(labels_array(&[&["Person"], &["Person", "Admin"]])) as ArrayRef,
                Arc::new(Int64Array::from(vec![Some(30), Some(40)])) as ArrayRef,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let edges = EdgeFrame::from_record_batch(
        RecordBatch::try_new(
            edge_schema,
            vec![
                Arc::new(StringArray::from(vec!["alice"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["bob"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["KNOWS"])) as ArrayRef,
                Arc::new(Int8Array::from(vec![Direction::Out.as_i8()])) as ArrayRef,
                Arc::new(Int64Array::from(vec![Some(1)])) as ArrayRef,
            ],
        )
        .unwrap(),
    )
    .unwrap();

    GraphFrame::new(nodes, edges).unwrap()
}

#[test]
fn gfb_file_round_trip_still_works_for_wasm_byte_helpers_path() {
    let graph = demo_graph();
    let path = std::env::temp_dir().join(format!("lynxes-adv007-bytes-{}.gfb", std::process::id()));

    write_gfb(&graph, &path, &GfbWriteOptions::default()).unwrap();
    let decoded = read_gfb(&path).unwrap();
    let inspect = read_gfb_inspect(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(decoded.node_count(), graph.node_count());
    assert_eq!(decoded.edge_count(), graph.edge_count());
    assert_eq!(inspect.node_count, graph.node_count());
    assert_eq!(inspect.edge_count, graph.edge_count());
}
