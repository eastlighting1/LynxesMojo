use std::sync::Arc;

use arrow_array::{
    builder::{ListBuilder, StringBuilder},
    Int64Array, Int8Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};

use lynxes_core::{
    bfs, BfsConfig, BinaryOp, Direction, EdgeFrame, EdgeTypeSpec, Expr, GFError, GraphFrame,
    NodeFrame, ScalarValue, COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE,
    COL_NODE_ID, COL_NODE_LABEL,
};

// ── Fixtures ──────────────────────────────────────────────────────────────

fn labels_array(values: &[&[&str]]) -> arrow_array::ListArray {
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

/// Four-node graph:
///
/// ```text
/// alice --KNOWS--> bob --KNOWS--> charlie
///   |                                ^
///   +--------LIKES------------------+
/// diana (isolated — no edges)
/// ```
fn make_graph() -> GraphFrame {
    let node_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        label_field(),
        Field::new("age", DataType::Int64, true),
    ]));
    let node_batch = RecordBatch::try_new(
        node_schema,
        vec![
            Arc::new(StringArray::from(vec!["alice", "bob", "charlie", "diana"]))
                as Arc<dyn arrow_array::Array>,
            Arc::new(labels_array(&[
                &["Person"],
                &["Person"],
                &["Person"],
                &["Animal"],
            ])) as Arc<dyn arrow_array::Array>,
            Arc::new(Int64Array::from(vec![
                Some(30),
                Some(25),
                Some(20),
                Some(5),
            ])) as Arc<dyn arrow_array::Array>,
        ],
    )
    .unwrap();

    let edge_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_EDGE_SRC, DataType::Utf8, false),
        Field::new(COL_EDGE_DST, DataType::Utf8, false),
        Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
        Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        edge_schema,
        vec![
            Arc::new(StringArray::from(vec!["alice", "bob", "alice"]))
                as Arc<dyn arrow_array::Array>,
            Arc::new(StringArray::from(vec!["bob", "charlie", "charlie"]))
                as Arc<dyn arrow_array::Array>,
            Arc::new(StringArray::from(vec!["KNOWS", "KNOWS", "LIKES"]))
                as Arc<dyn arrow_array::Array>,
            Arc::new(Int8Array::from(vec![0i8, 0, 0])) as Arc<dyn arrow_array::Array>,
        ],
    )
    .unwrap();

    let nodes = NodeFrame::from_record_batch(node_batch).unwrap();
    let edges = EdgeFrame::from_record_batch(edge_batch).unwrap();
    GraphFrame::new(nodes, edges).unwrap()
}

// ── Basic traversal ───────────────────────────────────────────────────────

#[test]
fn empty_roots_returns_empty_frames() {
    let graph = make_graph();
    let config = BfsConfig::new(3);
    let (nodes, edges) = bfs(&graph, &[], &config).unwrap();

    assert_eq!(nodes.len(), 0);
    assert_eq!(edges.len(), 0);
}

#[test]
fn zero_hops_returns_only_roots() {
    let graph = make_graph();
    let config = BfsConfig::new(0);
    let (nodes, edges) = bfs(&graph, &["alice"], &config).unwrap();

    assert_eq!(nodes.len(), 1);
    assert_eq!(edges.len(), 0);
    assert!(nodes.row_index("alice").is_some());
}

#[test]
fn one_hop_out_from_alice_reaches_bob() {
    let graph = make_graph();
    let config = BfsConfig {
        hops: 1,
        direction: Direction::Out,
        edge_type: EdgeTypeSpec::Any,
        pre_filter: None,
    };
    let (nodes, edges) = bfs(&graph, &["alice"], &config).unwrap();

    assert_eq!(nodes.len(), 3);
    assert!(nodes.row_index("alice").is_some());
    assert!(nodes.row_index("bob").is_some());
    assert!(nodes.row_index("charlie").is_some());
    assert_eq!(edges.len(), 2);
}

#[test]
fn two_hops_out_from_alice_reaches_all_connected() {
    let graph = make_graph();
    let config = BfsConfig {
        hops: 2,
        direction: Direction::Out,
        edge_type: EdgeTypeSpec::Any,
        pre_filter: None,
    };
    let (nodes, edges) = bfs(&graph, &["alice"], &config).unwrap();

    assert_eq!(nodes.len(), 3);
    assert_eq!(edges.len(), 3);
}

#[test]
fn diana_is_isolated_node_included_with_zero_expansion() {
    let graph = make_graph();
    let config = BfsConfig::new(2);
    let (nodes, edges) = bfs(&graph, &["diana"], &config).unwrap();

    assert_eq!(nodes.len(), 1);
    assert!(nodes.row_index("diana").is_some());
    assert_eq!(edges.len(), 0);
}

// ── Direction ─────────────────────────────────────────────────────────────

#[test]
fn in_direction_from_charlie_reaches_alice_and_bob() {
    let graph = make_graph();
    let config = BfsConfig {
        hops: 1,
        direction: Direction::In,
        edge_type: EdgeTypeSpec::Any,
        pre_filter: None,
    };
    let (nodes, edges) = bfs(&graph, &["charlie"], &config).unwrap();

    assert!(nodes.row_index("alice").is_some());
    assert!(nodes.row_index("bob").is_some());
    assert!(nodes.row_index("charlie").is_some());
    assert_eq!(edges.len(), 2);
}

#[test]
fn both_direction_from_bob() {
    let graph = make_graph();
    let config = BfsConfig {
        hops: 1,
        direction: Direction::Both,
        edge_type: EdgeTypeSpec::Any,
        pre_filter: None,
    };
    let (nodes, _) = bfs(&graph, &["bob"], &config).unwrap();

    assert!(nodes.row_index("alice").is_some());
    assert!(nodes.row_index("charlie").is_some());
}

#[test]
fn none_direction_is_equivalent_to_both() {
    let graph = make_graph();
    let both_cfg = BfsConfig {
        hops: 1,
        direction: Direction::Both,
        edge_type: EdgeTypeSpec::Any,
        pre_filter: None,
    };
    let none_cfg = BfsConfig {
        hops: 1,
        direction: Direction::None,
        edge_type: EdgeTypeSpec::Any,
        pre_filter: None,
    };

    let (both_nodes, _) = bfs(&graph, &["bob"], &both_cfg).unwrap();
    let (none_nodes, _) = bfs(&graph, &["bob"], &none_cfg).unwrap();

    assert_eq!(both_nodes.len(), none_nodes.len());
}

// ── Edge type filtering ───────────────────────────────────────────────────

#[test]
fn edge_type_single_knows_only() {
    let graph = make_graph();
    let config = BfsConfig {
        hops: 1,
        direction: Direction::Out,
        edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
        pre_filter: None,
    };
    let (nodes, edges) = bfs(&graph, &["alice"], &config).unwrap();

    assert!(nodes.row_index("bob").is_some());
    assert!(nodes.row_index("charlie").is_none());
    assert_eq!(edges.len(), 1);
}

#[test]
fn edge_type_multiple_knows_and_likes() {
    let graph = make_graph();
    let config = BfsConfig {
        hops: 1,
        direction: Direction::Out,
        edge_type: EdgeTypeSpec::Multiple(vec!["KNOWS".to_owned(), "LIKES".to_owned()]),
        pre_filter: None,
    };
    let (nodes, edges) = bfs(&graph, &["alice"], &config).unwrap();

    assert!(nodes.row_index("bob").is_some());
    assert!(nodes.row_index("charlie").is_some());
    assert_eq!(edges.len(), 2);
}

#[test]
fn edge_type_unknown_yields_only_roots() {
    let graph = make_graph();
    let config = BfsConfig {
        hops: 5,
        direction: Direction::Out,
        edge_type: EdgeTypeSpec::Single("HATES".to_owned()),
        pre_filter: None,
    };
    let (nodes, edges) = bfs(&graph, &["alice"], &config).unwrap();

    assert_eq!(nodes.len(), 1);
    assert_eq!(edges.len(), 0);
}

// ── Multi-source ──────────────────────────────────────────────────────────

#[test]
fn multi_source_bfs_deduplicates_visited_nodes() {
    let graph = make_graph();
    let config = BfsConfig {
        hops: 1,
        direction: Direction::Out,
        edge_type: EdgeTypeSpec::Any,
        pre_filter: None,
    };
    let (nodes, edges) = bfs(&graph, &["alice", "bob"], &config).unwrap();

    assert!(nodes.row_index("charlie").is_some());
    let charlie_count = edges
        .to_record_batch()
        .column_by_name(COL_EDGE_DST)
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .filter(|v| *v == Some("charlie"))
        .count();
    assert_eq!(charlie_count, 2);
}

// ── pre_filter ────────────────────────────────────────────────────────────

#[test]
fn pre_filter_excludes_low_age_nodes() {
    let graph = make_graph();
    let filter = Expr::BinaryOp {
        left: Box::new(Expr::Col {
            name: "age".to_owned(),
        }),
        op: BinaryOp::GtEq,
        right: Box::new(Expr::Literal {
            value: ScalarValue::Int(25),
        }),
    };
    let config = BfsConfig {
        hops: 3,
        direction: Direction::Out,
        edge_type: EdgeTypeSpec::Any,
        pre_filter: Some(&filter),
    };
    let (nodes, _) = bfs(&graph, &["alice"], &config).unwrap();

    assert!(nodes.row_index("alice").is_some());
    assert!(nodes.row_index("bob").is_some());
    assert!(nodes.row_index("charlie").is_none());
}

#[test]
fn pre_filter_list_contains_label() {
    let graph = make_graph();
    let filter = Expr::ListContains {
        expr: Box::new(Expr::Col {
            name: COL_NODE_LABEL.to_owned(),
        }),
        item: Box::new(Expr::Literal {
            value: ScalarValue::String("Person".to_owned()),
        }),
    };
    let config = BfsConfig {
        hops: 2,
        direction: Direction::Out,
        edge_type: EdgeTypeSpec::Any,
        pre_filter: Some(&filter),
    };
    let (nodes, _) = bfs(&graph, &["alice"], &config).unwrap();

    assert!(nodes.row_index("bob").is_some());
    assert!(nodes.row_index("charlie").is_some());
}

#[test]
fn pre_filter_bool_literal_false_blocks_all_expansion() {
    let graph = make_graph();
    let filter = Expr::Literal {
        value: ScalarValue::Bool(false),
    };
    let config = BfsConfig {
        hops: 3,
        direction: Direction::Out,
        edge_type: EdgeTypeSpec::Any,
        pre_filter: Some(&filter),
    };
    let (nodes, edges) = bfs(&graph, &["alice"], &config).unwrap();

    assert_eq!(nodes.len(), 1);
    assert_eq!(edges.len(), 0);
}

// ── Error cases ───────────────────────────────────────────────────────────

#[test]
fn unknown_root_returns_node_not_found_error() {
    let graph = make_graph();
    let config = BfsConfig::new(1);
    let err = bfs(&graph, &["ghost"], &config).unwrap_err();

    assert!(matches!(err, GFError::NodeNotFound { id } if id == "ghost"));
}

#[test]
fn mixed_roots_errors_on_first_unknown() {
    let graph = make_graph();
    let config = BfsConfig::new(1);
    let err = bfs(&graph, &["alice", "ghost"], &config).unwrap_err();

    assert!(matches!(err, GFError::NodeNotFound { id } if id == "ghost"));
}
