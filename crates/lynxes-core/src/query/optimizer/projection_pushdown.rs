use hashbrown::HashSet;

use crate::{
    AggExpr, Expr, LogicalPlan, OptimizerPass, COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC,
    COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
};

#[derive(Debug, Default, Clone, Copy)]
pub struct ProjectionPushdown;

impl OptimizerPass for ProjectionPushdown {
    fn name(&self) -> &'static str {
        "ProjectionPushdown"
    }

    fn optimize(&self, plan: LogicalPlan) -> LogicalPlan {
        optimize_plan(plan, None, None)
    }
}

fn optimize_plan(
    plan: LogicalPlan,
    required_node_columns: Option<HashSet<String>>,
    required_edge_columns: Option<HashSet<String>>,
) -> LogicalPlan {
    match plan {
        LogicalPlan::Scan {
            source,
            node_columns,
            edge_columns,
        } => LogicalPlan::Scan {
            source,
            node_columns: narrow_existing_columns(node_columns, required_node_columns),
            edge_columns: narrow_existing_columns(edge_columns, required_edge_columns),
        },
        LogicalPlan::Cache { input, key } => LogicalPlan::Cache {
            input: Box::new(optimize_plan(
                *input,
                required_node_columns,
                required_edge_columns,
            )),
            key,
        },
        LogicalPlan::Hint { input, hint } => LogicalPlan::Hint {
            input: Box::new(optimize_plan(
                *input,
                required_node_columns,
                required_edge_columns,
            )),
            hint,
        },
        LogicalPlan::FilterNodes { input, predicate } => {
            let input_required =
                union_required(required_node_columns, Some(expr_column_refs(&predicate)));

            LogicalPlan::FilterNodes {
                input: Box::new(optimize_plan(*input, input_required, required_edge_columns)),
                predicate,
            }
        }
        LogicalPlan::FilterEdges { input, predicate } => {
            let input_required =
                union_required(required_edge_columns, Some(expr_column_refs(&predicate)));

            LogicalPlan::FilterEdges {
                input: Box::new(optimize_plan(*input, required_node_columns, input_required)),
                predicate,
            }
        }
        LogicalPlan::ProjectNodes { input, columns } => {
            let available = node_available_columns(&columns);
            let input_required = match required_node_columns {
                Some(required) => Some(intersect_sets(&required, &available)),
                None => Some(available),
            };

            let optimized_input = optimize_plan(*input, input_required, required_edge_columns);

            collapse_project_nodes(optimized_input, columns)
        }
        LogicalPlan::ProjectEdges { input, columns } => {
            let available = edge_available_columns(&columns);
            let input_required = match required_edge_columns {
                Some(required) => Some(intersect_sets(&required, &available)),
                None => Some(available),
            };

            let optimized_input = optimize_plan(*input, required_node_columns, input_required);

            collapse_project_edges(optimized_input, columns)
        }
        LogicalPlan::Expand {
            input,
            edge_type,
            hops,
            direction,
            pre_filter,
        } => {
            let node_required = union_required(
                required_node_columns,
                pre_filter.as_ref().map(expr_column_refs),
            );

            LogicalPlan::Expand {
                input: Box::new(optimize_plan(*input, node_required, required_edge_columns)),
                edge_type,
                hops,
                direction,
                pre_filter,
            }
        }
        LogicalPlan::Traverse { input, pattern } => LogicalPlan::Traverse {
            input: Box::new(optimize_plan(
                *input,
                required_node_columns,
                required_edge_columns,
            )),
            pattern,
        },
        LogicalPlan::PatternMatch {
            input,
            pattern,
            where_,
        } => LogicalPlan::PatternMatch {
            input: Box::new(optimize_plan(*input, None, None)),
            pattern,
            where_,
        },
        LogicalPlan::AggregateNeighbors {
            input,
            edge_type,
            agg,
        } => {
            let edge_required = include_operator_required(
                required_edge_columns,
                Some(edge_local_requirements(agg_expr_column_refs(&agg))),
            );

            LogicalPlan::AggregateNeighbors {
                input: Box::new(optimize_plan(*input, required_node_columns, edge_required)),
                edge_type,
                agg,
            }
        }
        LogicalPlan::Sort {
            input,
            by,
            descending,
        } => {
            let output_domain = input.output_domain();
            let (node_required, edge_required) = match output_domain {
                crate::PlanDomain::Nodes => (
                    union_required(required_node_columns, Some(singleton_column(&by))),
                    required_edge_columns,
                ),
                crate::PlanDomain::Edges => (
                    required_node_columns,
                    union_required(required_edge_columns, Some(singleton_column(&by))),
                ),
                crate::PlanDomain::Graph | crate::PlanDomain::PatternRows => {
                    (required_node_columns, required_edge_columns)
                }
            };

            LogicalPlan::Sort {
                input: Box::new(optimize_plan(*input, node_required, edge_required)),
                by,
                descending,
            }
        }
        LogicalPlan::Limit { input, n } => LogicalPlan::Limit {
            input: Box::new(optimize_plan(
                *input,
                required_node_columns,
                required_edge_columns,
            )),
            n,
        },
    }
}

fn collapse_project_nodes(input: LogicalPlan, columns: Vec<String>) -> LogicalPlan {
    match input {
        LogicalPlan::ProjectNodes {
            input: inner,
            columns: inner_columns,
        } => LogicalPlan::ProjectNodes {
            input: inner,
            columns: intersect_projection_columns(&inner_columns, &columns),
        },
        other => LogicalPlan::ProjectNodes {
            input: Box::new(other),
            columns,
        },
    }
}

fn collapse_project_edges(input: LogicalPlan, columns: Vec<String>) -> LogicalPlan {
    match input {
        LogicalPlan::ProjectEdges {
            input: inner,
            columns: inner_columns,
        } => LogicalPlan::ProjectEdges {
            input: inner,
            columns: intersect_projection_columns(&inner_columns, &columns),
        },
        other => LogicalPlan::ProjectEdges {
            input: Box::new(other),
            columns,
        },
    }
}

fn narrow_existing_columns(
    existing: Option<Vec<String>>,
    required: Option<HashSet<String>>,
) -> Option<Vec<String>> {
    match (existing, required) {
        (Some(existing), Some(required)) => Some(filter_columns(existing, &required)),
        (Some(existing), None) => Some(existing),
        (None, Some(required)) => Some(sorted_columns(required)),
        (None, None) => None,
    }
}

fn filter_columns(columns: Vec<String>, allowed: &HashSet<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut filtered = Vec::new();

    for column in columns {
        if allowed.contains(&column) && seen.insert(column.clone()) {
            filtered.push(column);
        }
    }

    filtered
}

fn sorted_columns(columns: HashSet<String>) -> Vec<String> {
    let mut columns: Vec<_> = columns.into_iter().collect();
    columns.sort();
    columns
}

fn union_required(
    current: Option<HashSet<String>>,
    extra: Option<HashSet<String>>,
) -> Option<HashSet<String>> {
    match (current, extra) {
        (None, _) => None,
        (Some(current), None) => Some(current),
        (Some(mut current), Some(extra)) => {
            current.extend(extra);
            Some(current)
        }
    }
}

fn include_operator_required(
    current: Option<HashSet<String>>,
    extra: Option<HashSet<String>>,
) -> Option<HashSet<String>> {
    match (current, extra) {
        (None, Some(extra)) => Some(extra),
        (current, extra) => union_required(current, extra),
    }
}

fn intersect_sets(left: &HashSet<String>, right: &HashSet<String>) -> HashSet<String> {
    left.intersection(right).cloned().collect()
}

fn intersect_projection_columns(available: &[String], requested: &[String]) -> Vec<String> {
    let available: HashSet<_> = available.iter().cloned().collect();
    let mut seen = HashSet::new();
    let mut columns = Vec::new();

    for column in requested {
        if available.contains(column) && seen.insert(column.clone()) {
            columns.push(column.clone());
        }
    }

    columns
}

fn singleton_column(name: &str) -> HashSet<String> {
    let mut columns = HashSet::new();
    columns.insert(name.to_owned());
    columns
}

fn expr_column_refs(expr: &Expr) -> HashSet<String> {
    let mut refs = HashSet::new();
    collect_expr_column_refs(expr, &mut refs);
    refs
}

fn collect_expr_column_refs(expr: &Expr, refs: &mut HashSet<String>) {
    match expr {
        Expr::Col { name } => {
            refs.insert(name.clone());
        }
        Expr::Literal { .. } | Expr::PatternCol { .. } => {}
        Expr::BinaryOp { left, right, .. }
        | Expr::And { left, right }
        | Expr::Or { left, right } => {
            collect_expr_column_refs(left, refs);
            collect_expr_column_refs(right, refs);
        }
        Expr::UnaryOp { expr, .. } | Expr::Not { expr } | Expr::Cast { expr, .. } => {
            collect_expr_column_refs(expr, refs);
        }
        Expr::ListContains { expr, item } => {
            collect_expr_column_refs(expr, refs);
            collect_expr_column_refs(item, refs);
        }
        Expr::StringOp { expr, pattern, .. } => {
            collect_expr_column_refs(expr, refs);
            collect_expr_column_refs(pattern, refs);
        }
    }
}

fn agg_expr_column_refs(agg: &AggExpr) -> Option<HashSet<String>> {
    match agg {
        AggExpr::Count => None,
        AggExpr::Sum { expr }
        | AggExpr::Mean { expr }
        | AggExpr::List { expr }
        | AggExpr::First { expr }
        | AggExpr::Last { expr } => Some(expr_column_refs(expr)),
        AggExpr::Alias { expr, .. } => agg_expr_column_refs(expr),
    }
}

fn edge_local_requirements(extra: Option<HashSet<String>>) -> HashSet<String> {
    let mut columns = edge_available_columns(&[]);
    if let Some(extra) = extra {
        columns.extend(extra);
    }
    columns
}

fn node_available_columns(columns: &[String]) -> HashSet<String> {
    let mut available = HashSet::new();
    available.insert(COL_NODE_ID.to_owned());
    available.insert(COL_NODE_LABEL.to_owned());
    available.extend(columns.iter().cloned());
    available
}

fn edge_available_columns(columns: &[String]) -> HashSet<String> {
    let mut available = HashSet::new();
    available.insert(COL_EDGE_SRC.to_owned());
    available.insert(COL_EDGE_DST.to_owned());
    available.insert(COL_EDGE_TYPE.to_owned());
    available.insert(COL_EDGE_DIRECTION.to_owned());
    available.extend(columns.iter().cloned());
    available
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{
        query::{Connector, ScalarValue},
        BinaryOp, Direction,
    };

    use super::*;

    #[derive(Debug)]
    struct DummyConnector;
    impl Connector for DummyConnector {}

    fn scan() -> LogicalPlan {
        LogicalPlan::Scan {
            source: Arc::new(DummyConnector),
            node_columns: None,
            edge_columns: None,
        }
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

    #[test]
    fn collapses_nested_node_projects() {
        let plan = LogicalPlan::ProjectNodes {
            input: Box::new(LogicalPlan::ProjectNodes {
                input: Box::new(scan()),
                columns: vec!["name".to_owned(), "age".to_owned(), "score".to_owned()],
            }),
            columns: vec!["score".to_owned(), "name".to_owned()],
        };

        let optimized = ProjectionPushdown.optimize(plan);

        match optimized {
            LogicalPlan::ProjectNodes { input, columns } => {
                assert_eq!(columns, vec!["score".to_owned(), "name".to_owned()]);
                assert!(matches!(*input, LogicalPlan::Scan { .. }));
            }
            other => panic!("expected ProjectNodes, got {other:?}"),
        }
    }

    #[test]
    fn annotates_scan_with_project_and_filter_columns() {
        let plan = LogicalPlan::ProjectNodes {
            input: Box::new(LogicalPlan::FilterNodes {
                input: Box::new(scan()),
                predicate: age_gt_30(),
            }),
            columns: vec!["name".to_owned()],
        };

        let optimized = ProjectionPushdown.optimize(plan);

        match optimized {
            LogicalPlan::ProjectNodes { input, .. } => match *input {
                LogicalPlan::FilterNodes { input, .. } => match *input {
                    LogicalPlan::Scan {
                        node_columns: Some(node_columns),
                        edge_columns,
                        ..
                    } => {
                        assert_eq!(
                            node_columns,
                            vec![
                                "_id".to_owned(),
                                "_label".to_owned(),
                                "age".to_owned(),
                                "name".to_owned(),
                            ]
                        );
                        assert!(edge_columns.is_none());
                    }
                    other => panic!("expected Scan, got {other:?}"),
                },
                other => panic!("expected FilterNodes, got {other:?}"),
            },
            other => panic!("expected ProjectNodes, got {other:?}"),
        }
    }

    #[test]
    fn preserves_sort_key_in_scan_requirements() {
        let plan = LogicalPlan::Limit {
            input: Box::new(LogicalPlan::Sort {
                input: Box::new(LogicalPlan::ProjectNodes {
                    input: Box::new(scan()),
                    columns: vec!["name".to_owned(), "score".to_owned()],
                }),
                by: "score".to_owned(),
                descending: true,
            }),
            n: 5,
        };

        let optimized = ProjectionPushdown.optimize(plan);

        match optimized {
            LogicalPlan::Limit { input, .. } => match *input {
                LogicalPlan::Sort { input, .. } => match *input {
                    LogicalPlan::ProjectNodes { input, .. } => match *input {
                        LogicalPlan::Scan {
                            node_columns: Some(node_columns),
                            ..
                        } => {
                            assert_eq!(
                                node_columns,
                                vec![
                                    "_id".to_owned(),
                                    "_label".to_owned(),
                                    "name".to_owned(),
                                    "score".to_owned(),
                                ]
                            );
                        }
                        other => panic!("expected Scan, got {other:?}"),
                    },
                    other => panic!("expected ProjectNodes, got {other:?}"),
                },
                other => panic!("expected Sort, got {other:?}"),
            },
            other => panic!("expected Limit, got {other:?}"),
        }
    }

    #[test]
    fn preserves_aggregate_input_edge_columns() {
        let plan = LogicalPlan::AggregateNeighbors {
            input: Box::new(scan()),
            edge_type: "KNOWS".to_owned(),
            agg: AggExpr::Sum {
                expr: Expr::Col {
                    name: "weight".to_owned(),
                },
            },
        };

        let optimized = ProjectionPushdown.optimize(plan);

        match optimized {
            LogicalPlan::AggregateNeighbors { input, .. } => match *input {
                LogicalPlan::Scan {
                    node_columns,
                    edge_columns: Some(edge_columns),
                    ..
                } => {
                    assert!(node_columns.is_none());
                    assert_eq!(
                        edge_columns,
                        vec![
                            "_direction".to_owned(),
                            "_dst".to_owned(),
                            "_src".to_owned(),
                            "_type".to_owned(),
                            "weight".to_owned(),
                        ]
                    );
                }
                other => panic!("expected Scan, got {other:?}"),
            },
            other => panic!("expected AggregateNeighbors, got {other:?}"),
        }
    }

    #[test]
    fn graph_root_expand_pre_filter_stays_fail_closed() {
        let plan = LogicalPlan::Expand {
            input: Box::new(scan()),
            edge_type: crate::EdgeTypeSpec::Any,
            hops: 2,
            direction: Direction::Out,
            pre_filter: Some(age_gt_30()),
        };

        let optimized = ProjectionPushdown.optimize(plan);

        match optimized {
            LogicalPlan::Expand { input, .. } => match *input {
                LogicalPlan::Scan { node_columns, .. } => {
                    assert!(node_columns.is_none());
                }
                other => panic!("expected Scan, got {other:?}"),
            },
            other => panic!("expected Expand, got {other:?}"),
        }
    }

    #[test]
    fn optimization_is_idempotent() {
        let plan = LogicalPlan::ProjectNodes {
            input: Box::new(LogicalPlan::FilterNodes {
                input: Box::new(LogicalPlan::ProjectNodes {
                    input: Box::new(scan()),
                    columns: vec!["name".to_owned(), "age".to_owned(), "score".to_owned()],
                }),
                predicate: age_gt_30(),
            }),
            columns: vec!["name".to_owned()],
        };

        let once = ProjectionPushdown.optimize(plan);
        let twice = ProjectionPushdown.optimize(once.clone());

        assert_eq!(format!("{once:?}"), format!("{twice:?}"));
    }
}
