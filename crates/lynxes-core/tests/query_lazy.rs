use std::sync::Arc;

use arrow_array::{Int8Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};

use lynxes_core::{
    AggExpr, BinaryOp, Direction, EdgeFrame, EdgeTypeSpec, Expr, GFError, GraphFrame, LogicalPlan,
    NodeFrame, PatternStep, ScalarValue, COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC,
    COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
};
use lynxes_lazy::LazyGraphFrame;

fn age_gt_30() -> Expr {
    Expr::BinaryOp {
        left: Box::new(Expr::Col {
            name: "age".to_owned(),
        }),
        op: BinaryOp::Gt,
        right: Box::new(Expr::Literal {
            value: ScalarValue::Int(30),
        }),
    }
}

fn demo_graph() -> GraphFrame {
    use arrow_array::{
        builder::{ListBuilder, StringBuilder},
        ArrayRef, Int64Array,
    };

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
    for group in [["Person"], ["Person"], ["Company"]] {
        for label in group {
            labels.values().append_value(label);
        }
        labels.append(true);
    }
    let nodes = NodeFrame::from_record_batch(
        RecordBatch::try_new(
            node_schema,
            vec![
                Arc::new(StringArray::from(vec!["alice", "bob", "acme"])) as ArrayRef,
                Arc::new(labels.finish()) as ArrayRef,
                Arc::new(Int64Array::from(vec![Some(30), Some(40), None])) as ArrayRef,
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

#[test]
fn graph_frame_lazy_starts_from_scan() {
    let graph = demo_graph();
    let lazy = LazyGraphFrame::from_graph(&graph);
    assert!(matches!(lazy.plan(), LogicalPlan::Scan { .. }));
}

#[test]
fn builder_methods_accumulate_plan_shape() {
    let graph = demo_graph();
    let lazy = LazyGraphFrame::from_graph(&graph)
        .filter_nodes(age_gt_30())
        .expand(EdgeTypeSpec::Single("KNOWS".to_owned()), 2, Direction::Out)
        .sort("score", true)
        .limit(10);

    assert!(matches!(lazy.plan(), LogicalPlan::Limit { .. }));
}

#[test]
fn explain_runs_optimizer_and_formats_tree() {
    let graph = demo_graph();
    let explain = LazyGraphFrame::from_graph(&graph)
        .filter_nodes(age_gt_30())
        .expand(EdgeTypeSpec::Single("KNOWS".to_owned()), 2, Direction::Out)
        .limit(5)
        .explain();

    assert!(explain.contains("Limit(5)"));
    assert!(explain.contains("Expand(edge_type=Single(\"KNOWS\")"));
    assert!(explain.contains("FilterNodes("));
    assert!(explain.contains("Scan(GraphFrame"));
}

#[test]
fn optimized_plan_reflects_default_enabled_passes_only() {
    let graph = demo_graph();
    let optimized = LazyGraphFrame::from_graph(&graph)
        .filter_nodes(age_gt_30())
        .expand(EdgeTypeSpec::Single("KNOWS".to_owned()), 1, Direction::Out)
        .limit(3)
        .optimized_plan();

    match optimized {
        LogicalPlan::Limit { input, n } => {
            assert_eq!(n, 3);
            assert!(matches!(*input, LogicalPlan::Expand { .. }));
        }
        other => panic!("expected Limit, got {other:?}"),
    }
}

#[test]
fn collect_nodes_executes_filter_and_projection() {
    let graph = demo_graph();
    let nodes = LazyGraphFrame::from_graph(&graph)
        .filter_nodes(age_gt_30())
        .select_nodes(vec!["age".to_owned()])
        .collect_nodes()
        .unwrap();

    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes.id_column().value(0), "bob");
    assert_eq!(
        nodes.column_names(),
        vec![COL_NODE_ID, COL_NODE_LABEL, "age"]
    );
}

#[test]
fn collect_edges_executes_filter_sort_and_limit() {
    let graph = demo_graph();
    let edges = LazyGraphFrame::from_graph(&graph)
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

#[test]
fn collect_executes_expand_to_graph_result() {
    let source = demo_graph();
    let graph = LazyGraphFrame::from_graph(&source)
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
fn collect_executes_traverse_graph_result() {
    let source = demo_graph();
    let graph = LazyGraphFrame::from_graph(&source)
        .filter_nodes(Expr::BinaryOp {
            left: Box::new(Expr::Col {
                name: COL_NODE_ID.to_owned(),
            }),
            op: BinaryOp::Eq,
            right: Box::new(Expr::Literal {
                value: ScalarValue::String("alice".to_owned()),
            }),
        })
        .traverse(vec![
            PatternStep {
                from_alias: "a".to_owned(),
                edge_alias: Some("e1".to_owned()),
                edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
                direction: Direction::Out,
                to_alias: "b".to_owned(),
            },
            PatternStep {
                from_alias: "b".to_owned(),
                edge_alias: Some("e2".to_owned()),
                edge_type: EdgeTypeSpec::Single("WORKS_AT".to_owned()),
                direction: Direction::Out,
                to_alias: "c".to_owned(),
            },
        ])
        .collect()
        .unwrap();

    assert_eq!(graph.node_count(), 3);
    assert_eq!(graph.edge_count(), 2);
    assert!(graph.nodes().row_index("alice").is_some());
    assert!(graph.nodes().row_index("bob").is_some());
    assert!(graph.nodes().row_index("acme").is_some());
}

#[test]
fn collect_nodes_executes_aggregate_neighbors() {
    let graph = demo_graph();
    let nodes = LazyGraphFrame::from_graph(&graph)
        .aggregate_neighbors("KNOWS", AggExpr::Count)
        .collect_nodes()
        .unwrap();

    let count = nodes
        .column("count")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .unwrap();
    assert_eq!(count.value(0), 1);
    assert_eq!(count.value(1), 0);
    assert_eq!(count.value(2), 0);
}

#[test]
fn collect_rejects_non_graph_domain_results() {
    let graph = demo_graph();
    let err = LazyGraphFrame::from_graph(&graph)
        .filter_nodes(age_gt_30())
        .collect()
        .unwrap_err();
    assert!(matches!(err, GFError::DomainMismatch { .. }));
    assert!(err
        .to_string()
        .contains("collect() requires a graph-domain plan"));
}
