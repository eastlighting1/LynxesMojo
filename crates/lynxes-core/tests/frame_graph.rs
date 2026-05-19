mod common;

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Int8Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use common::{graph_node_batch, sample_graph};
use lynxes_core::{
    Direction, EdgeFrame, GFError, GraphFrame, NodeFrame, COL_EDGE_DIRECTION, COL_EDGE_DST,
    COL_EDGE_SRC, COL_EDGE_TYPE,
};

#[test]
fn new_rejects_dangling_edge() {
    let nodes = NodeFrame::from_record_batch(graph_node_batch()).unwrap();
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_EDGE_SRC, DataType::Utf8, false),
        Field::new(COL_EDGE_DST, DataType::Utf8, false),
        Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
        Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
    ]));
    let edges = EdgeFrame::from_record_batch(
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["alice"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["ghost"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["KNOWS"])) as ArrayRef,
                Arc::new(Int8Array::from(vec![0i8])) as ArrayRef,
            ],
        )
        .unwrap(),
    )
    .unwrap();

    let err = GraphFrame::new(nodes, edges).unwrap_err();
    assert!(matches!(err, GFError::DanglingEdge { src, dst } if src == "alice" && dst == "ghost"));
}

#[test]
fn subgraph_keeps_only_requested_nodes_and_internal_edges() {
    let graph = sample_graph();
    let sub = graph.subgraph(&["alice", "charlie"]).unwrap();

    assert_eq!(sub.node_count(), 2);
    assert_eq!(sub.edge_count(), 1);
    assert!(sub.nodes().row_index("alice").is_some());
    assert!(sub.nodes().row_index("charlie").is_some());
    assert!(sub.nodes().row_index("bob").is_none());
}

#[test]
fn subgraph_by_label_and_edge_type_work() {
    let graph = sample_graph();
    let by_label = graph.subgraph_by_label("Person").unwrap();
    let by_type = graph.subgraph_by_edge_type("KNOWS").unwrap();

    assert_eq!(by_label.node_count(), 3);
    assert_eq!(by_label.edge_count(), 3);
    assert_eq!(by_type.edge_count(), 2);
    assert_eq!(by_type.node_count(), 3);
}

#[test]
fn density_uses_directed_formula() {
    let graph = sample_graph();
    assert!((graph.density() - (4.0 / 12.0)).abs() < f64::EPSILON);
}

#[test]
fn neighbor_queries_return_expected_ids() {
    let graph = sample_graph();
    assert_eq!(
        graph.out_neighbors("alice").unwrap(),
        vec!["bob", "charlie"]
    );
    assert_eq!(graph.in_neighbors("charlie").unwrap(), vec!["alice", "bob"]);
    assert_eq!(
        graph.neighbors("alice", Direction::Both).unwrap(),
        vec!["bob", "charlie", "diana"]
    );
}

#[test]
fn degree_queries_use_edge_csr() {
    let graph = sample_graph();
    assert_eq!(graph.out_degree("alice").unwrap(), 2);
    assert_eq!(graph.in_degree("alice").unwrap(), 1);
}

#[cfg(not(target_os = "linux"))]
#[test]
fn structural_features_requires_linux_mojo_runtime() {
    let graph = sample_graph();
    let err = graph.structural_features(None).unwrap_err();
    assert!(matches!(err, GFError::UnsupportedOperation { message } if message.contains("Mojo")));
}

#[test]
fn k_hop_subgraph_collects_expected_nodes() {
    let graph = sample_graph();
    let zero = graph.k_hop_subgraph("alice", 0).unwrap();
    let one = graph.k_hop_subgraph("alice", 1).unwrap();

    assert_eq!(zero.node_count(), 1);
    assert_eq!(zero.edge_count(), 0);
    assert_eq!(one.node_count(), 4);
    assert_eq!(one.edge_count(), 4);
}

#[test]
fn unknown_neighbor_root_is_rejected() {
    let graph = sample_graph();
    let err = graph.out_neighbors("ghost").unwrap_err();
    assert!(matches!(err, GFError::NodeNotFound { id } if id == "ghost"));
}

#[test]
fn to_coo_returns_expected_edge_frame_local_coordinates() {
    let graph = sample_graph();

    let (src, dst) = graph.to_coo();

    assert_eq!(
        src.iter().map(|v| v.unwrap()).collect::<Vec<_>>(),
        vec![0, 0, 1, 3]
    );
    assert_eq!(
        dst.iter().map(|v| v.unwrap()).collect::<Vec<_>>(),
        vec![1, 2, 2, 0]
    );
    assert_eq!(src.len(), graph.edge_count());
    assert_eq!(dst.len(), graph.edge_count());
    assert_eq!(src.offset(), 0);
    assert_eq!(dst.offset(), 0);
}

#[test]
fn to_coo_uses_edge_frame_local_index_space_not_node_rows() {
    let nodes = NodeFrame::from_record_batch(graph_node_batch()).unwrap();
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_EDGE_SRC, DataType::Utf8, false),
        Field::new(COL_EDGE_DST, DataType::Utf8, false),
        Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
        Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
    ]));
    let edges = EdgeFrame::from_record_batch(
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["alice"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["charlie"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["KNOWS"])) as ArrayRef,
                Arc::new(Int8Array::from(vec![0i8])) as ArrayRef,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let graph = GraphFrame::new(nodes, edges).unwrap();

    let (src, dst) = graph.to_coo();

    assert_eq!(src.iter().map(|v| v.unwrap()).collect::<Vec<_>>(), vec![0]);
    assert_eq!(dst.iter().map(|v| v.unwrap()).collect::<Vec<_>>(), vec![1]);
    assert_eq!(src.len(), 1);
    assert_eq!(dst.len(), 1);
}
