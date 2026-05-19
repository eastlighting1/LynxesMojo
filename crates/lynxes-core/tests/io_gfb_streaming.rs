use std::future::poll_fn;
use std::pin::Pin;
use std::sync::Arc;

use arrow_array::{
    builder::{ListBuilder, StringBuilder},
    ArrayRef, Int64Array, Int8Array, ListArray, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures_core::Stream;
use lynxes_core::{
    Direction, EdgeFrame, GraphFrame, NodeFrame, COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC,
    COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
};
use lynxes_io::{
    read_gfb_streaming, read_gfb_streaming_with_options, write_gfb, GfbReadOptions, GfbWriteOptions,
};

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
                Arc::new(labels_array(&[&["Person"], &["Person"]])) as ArrayRef,
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
fn read_gfb_streaming_yields_one_graph_then_ends() {
    let graph = demo_graph();
    let path =
        std::env::temp_dir().join(format!("lynxes-adv006-stream-{}.gfb", std::process::id()));
    write_gfb(&graph, &path, &GfbWriteOptions::default()).unwrap();

    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let mut stream = read_gfb_streaming(&path).unwrap();
        let first =
            poll_fn(|cx| -> std::task::Poll<Option<_>> { Pin::new(&mut stream).poll_next(cx) })
                .await
                .unwrap()
                .unwrap();
        assert_eq!(first.node_count(), graph.node_count());
        assert_eq!(first.edge_count(), graph.edge_count());
        assert!(poll_fn(|cx| -> std::task::Poll<Option<_>> {
            Pin::new(&mut stream).poll_next(cx)
        })
        .await
        .is_none());
    });

    let _ = std::fs::remove_file(&path);
}

#[test]
fn read_gfb_streaming_with_projection_uses_same_options_contract() {
    let graph = demo_graph();
    let path = std::env::temp_dir().join(format!(
        "lynxes-adv006-stream-proj-{}.gfb",
        std::process::id()
    ));
    write_gfb(&graph, &path, &GfbWriteOptions::default()).unwrap();

    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let mut stream = read_gfb_streaming_with_options(
            &path,
            &GfbReadOptions {
                node_columns: Some(vec!["age".to_owned()]),
                edge_columns: Some(vec!["weight".to_owned()]),
            },
        )
        .unwrap();

        let batch =
            poll_fn(|cx| -> std::task::Poll<Option<_>> { Pin::new(&mut stream).poll_next(cx) })
                .await
                .unwrap()
                .unwrap();
        assert_eq!(
            batch.nodes().column_names(),
            vec![COL_NODE_ID, COL_NODE_LABEL, "age"]
        );
        assert_eq!(
            batch.edges().column_names(),
            vec![
                COL_EDGE_SRC,
                COL_EDGE_DST,
                COL_EDGE_TYPE,
                COL_EDGE_DIRECTION,
                "weight"
            ]
        );
    });

    let _ = std::fs::remove_file(&path);
}
