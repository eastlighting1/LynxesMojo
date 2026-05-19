use std::sync::Arc;

use arrow_array::{
    builder::{ListBuilder, StringBuilder},
    ArrayRef, Int64Array, Int8Array,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};

use lynxes_core::{
    Direction, EdgeFrame, GFError, GraphFrame, NodeFrame, COL_EDGE_DIRECTION, COL_EDGE_DST,
    COL_EDGE_SRC, COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
};
use lynxes_io::{
    read_parquet_graph, read_parquet_graph_with_options, write_parquet_graph, ParquetReadOptions,
};

fn labels_array(values: &[&[&str]]) -> arrow_array::ListArray {
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
        arrow_array::RecordBatch::try_new(
            node_schema,
            vec![
                Arc::new(arrow_array::StringArray::from(vec!["alice", "bob"])) as ArrayRef,
                Arc::new(labels_array(&[&["Person"], &["Person"]])) as ArrayRef,
                Arc::new(Int64Array::from(vec![Some(30), Some(40)])) as ArrayRef,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let edges = EdgeFrame::from_record_batch(
        arrow_array::RecordBatch::try_new(
            edge_schema,
            vec![
                Arc::new(arrow_array::StringArray::from(vec!["alice"])) as ArrayRef,
                Arc::new(arrow_array::StringArray::from(vec!["bob"])) as ArrayRef,
                Arc::new(arrow_array::StringArray::from(vec!["KNOWS"])) as ArrayRef,
                Arc::new(Int8Array::from(vec![Direction::Out.as_i8()])) as ArrayRef,
                Arc::new(Int64Array::from(vec![Some(7)])) as ArrayRef,
            ],
        )
        .unwrap(),
    )
    .unwrap();

    GraphFrame::new(nodes, edges).unwrap()
}

#[test]
fn write_and_read_parquet_graph_round_trip() {
    let graph = demo_graph();
    let node_path = std::env::temp_dir().join(format!(
        "lynxes-ser007-nodes-{}.parquet",
        std::process::id()
    ));
    let edge_path = std::env::temp_dir().join(format!(
        "lynxes-ser007-edges-{}.parquet",
        std::process::id()
    ));

    write_parquet_graph(&graph, &node_path, &edge_path).unwrap();
    let decoded = read_parquet_graph(&node_path, &edge_path).unwrap();
    let _ = std::fs::remove_file(&node_path);
    let _ = std::fs::remove_file(&edge_path);

    assert_eq!(decoded.node_count(), 2);
    assert_eq!(decoded.edge_count(), 1);
    assert!(decoded.nodes().column("age").is_some());
    assert!(decoded.edges().column("weight").is_some());
}

#[test]
fn read_parquet_graph_supports_projection() {
    let graph = demo_graph();
    let node_path = std::env::temp_dir().join(format!(
        "lynxes-ser007-proj-nodes-{}.parquet",
        std::process::id()
    ));
    let edge_path = std::env::temp_dir().join(format!(
        "lynxes-ser007-proj-edges-{}.parquet",
        std::process::id()
    ));

    write_parquet_graph(&graph, &node_path, &edge_path).unwrap();
    let decoded = read_parquet_graph_with_options(
        &node_path,
        &edge_path,
        &ParquetReadOptions {
            node_columns: Some(vec!["age".to_owned()]),
            edge_columns: Some(vec!["weight".to_owned()]),
        },
    )
    .unwrap();
    let _ = std::fs::remove_file(&node_path);
    let _ = std::fs::remove_file(&edge_path);

    assert_eq!(
        decoded.nodes().column_names(),
        vec![COL_NODE_ID, COL_NODE_LABEL, "age"]
    );
    assert_eq!(
        decoded.edges().column_names(),
        vec![
            COL_EDGE_SRC,
            COL_EDGE_DST,
            COL_EDGE_TYPE,
            COL_EDGE_DIRECTION,
            "weight"
        ]
    );
}

#[test]
fn read_parquet_graph_rejects_unknown_projection_column() {
    let graph = demo_graph();
    let node_path = std::env::temp_dir().join(format!(
        "lynxes-ser007-badproj-nodes-{}.parquet",
        std::process::id()
    ));
    let edge_path = std::env::temp_dir().join(format!(
        "lynxes-ser007-badproj-edges-{}.parquet",
        std::process::id()
    ));

    write_parquet_graph(&graph, &node_path, &edge_path).unwrap();
    let err = read_parquet_graph_with_options(
        &node_path,
        &edge_path,
        &ParquetReadOptions {
            node_columns: Some(vec!["missing".to_owned()]),
            edge_columns: None,
        },
    )
    .unwrap_err();
    let _ = std::fs::remove_file(&node_path);
    let _ = std::fs::remove_file(&edge_path);

    assert!(matches!(err, GFError::ColumnNotFound { .. }));
}
