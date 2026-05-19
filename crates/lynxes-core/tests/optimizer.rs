use std::sync::Arc;

use arrow_array::{
    builder::{ListBuilder, StringBuilder},
    ArrayRef, Int64Array, Int8Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};

use lynxes_core::{
    BinaryOp, Direction, EdgeFrame, EdgeTypeSpec, Expr, GraphFrame, LogicalPlan, NodeFrame,
    ScalarValue, COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE, COL_NODE_ID,
    COL_NODE_LABEL,
};
use lynxes_lazy::LazyGraphFrame;

// ── Fixture ───────────────────────────────────────────────────────────────────

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
    for label in ["Person", "Person"] {
        labels.values().append_value(label);
        labels.append(true);
    }
    let nodes = NodeFrame::from_record_batch(
        RecordBatch::try_new(
            node_schema,
            vec![
                Arc::new(StringArray::from(vec!["alice", "bob"])) as ArrayRef,
                Arc::new(labels.finish()) as ArrayRef,
                Arc::new(Int64Array::from(vec![Some(25), Some(40)])) as ArrayRef,
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
                Arc::new(StringArray::from(vec!["alice"])) as Arc<dyn arrow_array::Array>,
                Arc::new(StringArray::from(vec!["bob"])) as Arc<dyn arrow_array::Array>,
                Arc::new(StringArray::from(vec!["KNOWS"])) as Arc<dyn arrow_array::Array>,
                Arc::new(Int8Array::from(vec![0i8])) as Arc<dyn arrow_array::Array>,
            ],
        )
        .unwrap(),
    )
    .unwrap();

    GraphFrame::new(nodes, edges).unwrap()
}

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

fn age_lt_50() -> Expr {
    Expr::BinaryOp {
        left: Box::new(Expr::Col {
            name: "age".to_owned(),
        }),
        op: BinaryOp::Lt,
        right: Box::new(Expr::Literal {
            value: ScalarValue::Int(50),
        }),
    }
}

// ── TraversalPruning ──────────────────────────────────────────────────────────

#[test]
fn traversal_pruning_absorbs_filter_directly_above_expand() {
    // Chain: Scan → Expand → FilterNodes
    // TraversalPruning should absorb FilterNodes into Expand.pre_filter
    let optimized = LazyGraphFrame::from_graph(&demo_graph())
        .expand(EdgeTypeSpec::Single("KNOWS".to_owned()), 1, Direction::Out)
        .filter_nodes(age_gt_30())
        .optimized_plan();

    match optimized {
        LogicalPlan::Expand { pre_filter, .. } => {
            assert!(
                pre_filter.is_some(),
                "TraversalPruning must absorb FilterNodes into pre_filter"
            );
        }
        other => panic!("expected Expand at root after pruning, got {:?}", other),
    }
}

#[test]
fn traversal_pruning_does_not_absorb_filter_below_expand() {
    // Chain: Scan → FilterNodes → Expand
    // FilterNodes is the INPUT to Expand, not above it — should not become pre_filter
    let optimized = LazyGraphFrame::from_graph(&demo_graph())
        .filter_nodes(age_gt_30())
        .expand(EdgeTypeSpec::Single("KNOWS".to_owned()), 1, Direction::Out)
        .optimized_plan();

    match optimized {
        LogicalPlan::Expand {
            pre_filter, input, ..
        } => {
            assert!(
                pre_filter.is_none(),
                "FilterNodes below Expand must not become pre_filter"
            );
            assert!(matches!(*input, LogicalPlan::FilterNodes { .. }));
        }
        other => panic!("expected Expand at root, got {:?}", other),
    }
}

#[test]
fn traversal_pruning_does_not_absorb_filter_separated_by_limit() {
    // Chain: Scan → Expand → Limit → FilterNodes
    // FilterNodes is NOT directly above Expand (Limit is between them) — must not be absorbed
    let optimized = LazyGraphFrame::from_graph(&demo_graph())
        .expand(EdgeTypeSpec::Single("KNOWS".to_owned()), 1, Direction::Out)
        .limit(10)
        .filter_nodes(age_gt_30())
        .optimized_plan();

    match &optimized {
        LogicalPlan::FilterNodes { input, .. } => {
            assert!(
                matches!(**input, LogicalPlan::Limit { .. }),
                "filter over limit must stay"
            );
        }
        other => panic!(
            "expected FilterNodes at root when separated by Limit, got {:?}",
            other
        ),
    }
}

// ── PredicatePushdown ─────────────────────────────────────────────────────────

#[test]
fn predicate_pushdown_merges_consecutive_filters_into_one() {
    // Two consecutive FilterNodes on same domain should be merged into one (AND)
    let optimized = LazyGraphFrame::from_graph(&demo_graph())
        .filter_nodes(age_gt_30())
        .filter_nodes(age_lt_50())
        .optimized_plan();

    match optimized {
        LogicalPlan::FilterNodes { input, .. } => {
            // The merged filter should be directly above Scan (no nested FilterNodes)
            assert!(
                matches!(*input, LogicalPlan::Scan { .. }),
                "merged filter should be directly above Scan"
            );
        }
        other => panic!("expected single FilterNodes after merge, got {:?}", other),
    }
}

#[test]
fn predicate_pushdown_passes_filter_through_project_nodes_when_column_available() {
    // FilterNodes(ProjectNodes(Scan)) where predicate uses a projected column
    // PredicatePushdown should push FilterNodes below ProjectNodes
    // Chain: lazy → select_nodes(["age"]) → filter_nodes(age_gt_30())
    let optimized = LazyGraphFrame::from_graph(&demo_graph())
        .select_nodes(vec!["age".to_owned()])
        .filter_nodes(age_gt_30())
        .optimized_plan();

    // After PredicatePushdown: ProjectNodes(FilterNodes(Scan))
    // After ProjectionPushdown: ProjectNodes(FilterNodes(Scan { node_columns: Some([...]) }))
    match optimized {
        LogicalPlan::ProjectNodes { input, .. } => {
            assert!(
                matches!(*input, LogicalPlan::FilterNodes { .. }),
                "PredicatePushdown must move FilterNodes below ProjectNodes; got {:?}",
                *input
            );
        }
        other => panic!("expected ProjectNodes at root, got {:?}", other),
    }
}

// ── ProjectionPushdown ────────────────────────────────────────────────────────

#[test]
fn projection_pushdown_injects_node_columns_into_scan() {
    // Chain: lazy → select_nodes(["age"])
    // After ProjectionPushdown: Scan.node_columns should be Some([...])
    let optimized = LazyGraphFrame::from_graph(&demo_graph())
        .select_nodes(vec!["age".to_owned()])
        .optimized_plan();

    // Walk down to find the Scan
    let scan = match optimized {
        LogicalPlan::ProjectNodes { input, .. } => *input,
        other => panic!("expected ProjectNodes at root, got {:?}", other),
    };

    match scan {
        LogicalPlan::Scan { node_columns, .. } => {
            assert!(
                node_columns.is_some(),
                "ProjectionPushdown must inject node_columns into Scan"
            );
            let cols = node_columns.unwrap();
            assert!(
                cols.contains(&"age".to_owned()),
                "Scan.node_columns must include 'age'"
            );
            assert!(
                cols.contains(&COL_NODE_ID.to_owned()),
                "Scan.node_columns must include _id"
            );
        }
        other => panic!("expected Scan below ProjectNodes, got {:?}", other),
    }
}

#[test]
fn projection_pushdown_leaves_scan_unconstrained_without_projection() {
    // Plain filter with no projection — Scan.node_columns must remain None
    let optimized = LazyGraphFrame::from_graph(&demo_graph())
        .filter_nodes(age_gt_30())
        .optimized_plan();

    let scan = match optimized {
        LogicalPlan::FilterNodes { input, .. } => *input,
        other => panic!("expected FilterNodes at root, got {:?}", other),
    };

    match scan {
        LogicalPlan::Scan { node_columns, .. } => {
            assert!(
                node_columns.is_none(),
                "Scan must not be constrained when no projection exists"
            );
        }
        other => panic!("expected Scan below FilterNodes, got {:?}", other),
    }
}

// ── Explain integration ───────────────────────────────────────────────────────

#[test]
fn explain_reflects_traversal_pruning_in_output() {
    let explain = LazyGraphFrame::from_graph(&demo_graph())
        .expand(EdgeTypeSpec::Single("KNOWS".to_owned()), 1, Direction::Out)
        .filter_nodes(age_gt_30())
        .explain();

    // After TraversalPruning the pre_filter is injected into Expand, so Expand line shows it
    assert!(
        explain.contains("pre_filter=Some"),
        "explain must show pre_filter after TraversalPruning"
    );
    // There should be no standalone FilterNodes line
    assert!(
        !explain.contains("FilterNodes("),
        "FilterNodes must be absorbed; should not appear in explain"
    );
}
