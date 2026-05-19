use std::sync::Arc;

use arrow_array::{
    builder::{ListBuilder, StringBuilder},
    ArrayRef, Int64Array, Int8Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};

use lynxes_core::{
    BinaryOp, Direction, EdgeFrame, EdgeTypeSpec, Expr, GraphFrame, NodeFrame, ScalarValue,
    COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
};
use lynxes_io::{
    parse_gf, read_gfb, read_gfb_with_options, write_gfb, GfbReadOptions, GfbWriteOptions,
};
use lynxes_lazy::LazyGraphFrame;

// ── Shared fixture ────────────────────────────────────────────────────────────

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
    let mut labels = ListBuilder::new(StringBuilder::new());
    for label in ["Person", "Person", "Company"] {
        labels.values().append_value(label);
        labels.append(true);
    }
    let nodes = NodeFrame::from_record_batch(
        RecordBatch::try_new(
            node_schema,
            vec![
                Arc::new(StringArray::from(vec!["alice", "bob", "acme"])) as ArrayRef,
                Arc::new(labels.finish()) as ArrayRef,
                Arc::new(Int64Array::from(vec![Some(25), Some(40), None])) as ArrayRef,
            ],
        )
        .unwrap(),
    )
    .unwrap();

    let edge_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_EDGE_SRC, DataType::Utf8, false),
        Field::new(COL_EDGE_DST, DataType::Utf8, false),
        Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
        Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
    ]));
    let edges = EdgeFrame::from_record_batch(
        RecordBatch::try_new(
            edge_schema,
            vec![
                Arc::new(StringArray::from(vec!["alice", "bob"])) as Arc<dyn arrow_array::Array>,
                Arc::new(StringArray::from(vec!["bob", "acme"])) as Arc<dyn arrow_array::Array>,
                Arc::new(StringArray::from(vec!["KNOWS", "WORKS_AT"]))
                    as Arc<dyn arrow_array::Array>,
                Arc::new(Int8Array::from(vec![0i8, 0])) as Arc<dyn arrow_array::Array>,
            ],
        )
        .unwrap(),
    )
    .unwrap();

    GraphFrame::new(nodes, edges).unwrap()
}

// ── .gf parse → LazyGraphFrame chain → collect ───────────────────────────────

#[test]
fn gf_parse_filter_collect_nodes_end_to_end() {
    let src = r#"
        (alice :Person { age: 25 })
        (bob   :Person { age: 40 })
        (acme  :Company {})
        alice -[KNOWS]-> bob {}
        bob   -[WORKS_AT]-> acme {}
    "#;
    let doc = parse_gf(src).unwrap();
    assert_eq!(doc.nodes.len(), 3);
    assert_eq!(doc.edges.len(), 2);

    // Now verify the parsed data matches expected
    let alice = doc.nodes.iter().find(|n| n.id == "alice").unwrap();
    assert_eq!(alice.props["age"], lynxes_core::GFValue::Int(25));
    let edge = doc.edges.iter().find(|e| e.edge_type == "KNOWS").unwrap();
    assert_eq!(edge.src_id, "alice");
    assert_eq!(edge.dst_id, "bob");
}

#[test]
fn lazy_filter_expand_collect_returns_correct_subgraph() {
    let graph = LazyGraphFrame::from_graph(&demo_graph())
        .filter_nodes(Expr::BinaryOp {
            left: Box::new(Expr::Col {
                name: COL_NODE_ID.to_owned(),
            }),
            op: BinaryOp::Eq,
            right: Box::new(Expr::Literal {
                value: ScalarValue::String("alice".to_owned()),
            }),
        })
        .expand(EdgeTypeSpec::Single("KNOWS".to_owned()), 1, Direction::Out)
        .collect()
        .unwrap();

    assert_eq!(graph.node_count(), 2);
    assert_eq!(graph.edge_count(), 1);
    assert!(graph.nodes().row_index("alice").is_some());
    assert!(graph.nodes().row_index("bob").is_some());
}

#[test]
fn lazy_filter_aggregate_collect_nodes_returns_count_column() {
    let result = LazyGraphFrame::from_graph(&demo_graph())
        .aggregate_neighbors("KNOWS", lynxes_core::AggExpr::Count)
        .collect_nodes()
        .unwrap();

    assert_eq!(result.len(), 3);
    let count = result
        .column("count")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .unwrap();
    // alice has 1 KNOWS neighbor (bob)
    let alice_row = result.row_index("alice").unwrap() as usize;
    assert_eq!(count.value(alice_row), 1);
    // bob and acme have 0
    let bob_row = result.row_index("bob").unwrap() as usize;
    assert_eq!(count.value(bob_row), 0);
}

#[test]
fn lazy_chain_filter_sort_limit_collect_edges() {
    let edges = LazyGraphFrame::from_graph(&demo_graph())
        .filter_edges(Expr::BinaryOp {
            left: Box::new(Expr::Col {
                name: COL_EDGE_TYPE.to_owned(),
            }),
            op: BinaryOp::Eq,
            right: Box::new(Expr::Literal {
                value: ScalarValue::String("KNOWS".to_owned()),
            }),
        })
        .sort(COL_EDGE_DST, true)
        .limit(1)
        .collect_edges()
        .unwrap();

    assert_eq!(edges.len(), 1);
    let dst = edges
        .column(COL_EDGE_DST)
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(dst.value(0), "bob");
}

// ── .gfb write → read round-trip ─────────────────────────────────────────────

#[test]
fn gfb_write_read_round_trip_preserves_structure() {
    let graph = demo_graph();
    let path = std::env::temp_dir().join(format!("lynxes-e2e-{}.gfb", std::process::id()));

    write_gfb(&graph, &path, &GfbWriteOptions::default()).unwrap();
    let decoded = read_gfb(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(decoded.node_count(), graph.node_count());
    assert_eq!(decoded.edge_count(), graph.edge_count());
    assert!(decoded.nodes().row_index("alice").is_some());
    assert!(decoded.nodes().row_index("bob").is_some());
    assert!(decoded.nodes().row_index("acme").is_some());
}

#[test]
fn gfb_read_with_node_column_projection() {
    let graph = demo_graph();
    let path = std::env::temp_dir().join(format!("lynxes-e2e-proj-{}.gfb", std::process::id()));

    write_gfb(&graph, &path, &GfbWriteOptions::default()).unwrap();
    let decoded = read_gfb_with_options(
        &path,
        &GfbReadOptions {
            node_columns: Some(vec!["age".to_owned()]),
            edge_columns: None,
        },
    )
    .unwrap();
    let _ = std::fs::remove_file(&path);

    assert!(decoded.nodes().column("age").is_some());
    assert_eq!(decoded.node_count(), 3);
}
