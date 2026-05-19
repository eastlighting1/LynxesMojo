mod common;

use std::sync::Arc;

use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::{ArrayRef, Int8Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use common::sample_graph;
use lynxes_core::{
    DisplayOptions, DisplayRowKind, DisplayView, EdgeFrame, GraphFrame, NodeFrame,
    COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
};

#[test]
fn display_slice_projects_edges_and_isolated_nodes() {
    let graph = sample_graph();
    let slice = graph
        .display_slice(DisplayOptions {
            view: DisplayView::Table,
            max_rows: 10,
            width: Some(120),
            sort_by: None,
            expand_attrs: false,
            attrs: vec!["since".to_owned()],
        })
        .unwrap();

    assert_eq!(slice.graph_summary.projected_row_count, 4);
    assert_eq!(slice.graph_summary.isolated_node_count, 0);
    assert_eq!(slice.top_rows.len(), 4);
    assert_eq!(slice.top_rows[0].kind, DisplayRowKind::Edge);
    assert_eq!(slice.top_rows[0].values.get("src").unwrap(), "alice");
    assert_eq!(slice.top_rows[0].values.get("since").unwrap(), "2020");
}

#[test]
fn display_slice_tail_includes_isolated_node_rows() {
    let mut labels = ListBuilder::new(StringBuilder::new());
    for label in ["Person", "Person", "Company"] {
        labels.values().append_value(label);
        labels.append(true);
    }
    let nodes = NodeFrame::from_record_batch(
        RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                Field::new(COL_NODE_ID, DataType::Utf8, false),
                Field::new(
                    COL_NODE_LABEL,
                    DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                    false,
                ),
            ])),
            vec![
                Arc::new(StringArray::from(vec!["alice", "bob", "acme"])) as ArrayRef,
                Arc::new(labels.finish()) as ArrayRef,
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
                Arc::new(Int8Array::from(vec![0i8])) as ArrayRef,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let graph = GraphFrame::new(nodes, edges).unwrap();

    let slice = graph
        .display_slice(DisplayOptions {
            view: DisplayView::Tail,
            max_rows: 1,
            width: Some(80),
            sort_by: None,
            expand_attrs: false,
            attrs: Vec::new(),
        })
        .unwrap();

    assert_eq!(slice.bottom_rows.len(), 1);
    assert_eq!(slice.bottom_rows[0].kind, DisplayRowKind::Node);
    assert_eq!(slice.bottom_rows[0].values.get("src").unwrap(), "acme");
}

#[test]
fn display_schema_reports_reserved_and_user_fields() {
    let graph = sample_graph();
    let schema = graph.display_schema();

    assert!(schema
        .node_fields
        .iter()
        .any(|field| field.name == "_id" && field.reserved));
    assert!(schema
        .edge_fields
        .iter()
        .any(|field| field.name == "since" && !field.reserved));
}

#[test]
fn display_attr_stats_reports_distinct_counts() {
    let graph = sample_graph();
    let stats = graph.display_attr_stats();

    assert!(stats
        .node_attrs
        .iter()
        .any(|stat| stat.qualified_name == "node.age"));
    assert!(stats
        .edge_attrs
        .iter()
        .any(|stat| stat.qualified_name == "edge.since"));
    assert!(
        stats
            .edge_attrs
            .iter()
            .find(|stat| stat.qualified_name == "edge.since")
            .unwrap()
            .distinct_count
            >= 4
    );
}

#[test]
fn display_structure_stats_reports_components() {
    let graph = sample_graph();
    let stats = graph.display_structure_stats().unwrap();

    assert_eq!(stats.connected_components, 1);
    assert!(stats.max_degree >= 1);
}
