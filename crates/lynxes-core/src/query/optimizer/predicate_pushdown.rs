use hashbrown::HashSet;

use crate::{
    Expr, LogicalPlan, OptimizerPass, COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC,
    COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
};

#[derive(Debug, Default, Clone, Copy)]
pub struct PredicatePushdown;

impl OptimizerPass for PredicatePushdown {
    fn name(&self) -> &'static str {
        "PredicatePushdown"
    }

    fn optimize(&self, plan: LogicalPlan) -> LogicalPlan {
        optimize_plan(plan)
    }
}

fn optimize_plan(plan: LogicalPlan) -> LogicalPlan {
    match plan {
        LogicalPlan::Scan { .. } => plan,
        LogicalPlan::Cache { input, key } => LogicalPlan::Cache {
            input: Box::new(optimize_plan(*input)),
            key,
        },
        LogicalPlan::Hint { input, hint } => LogicalPlan::Hint {
            input: Box::new(optimize_plan(*input)),
            hint,
        },
        LogicalPlan::FilterNodes { input, predicate } => {
            rewrite_filter_nodes(optimize_plan(*input), predicate)
        }
        LogicalPlan::FilterEdges { input, predicate } => {
            rewrite_filter_edges(optimize_plan(*input), predicate)
        }
        LogicalPlan::ProjectNodes { input, columns } => LogicalPlan::ProjectNodes {
            input: Box::new(optimize_plan(*input)),
            columns,
        },
        LogicalPlan::ProjectEdges { input, columns } => LogicalPlan::ProjectEdges {
            input: Box::new(optimize_plan(*input)),
            columns,
        },
        LogicalPlan::Expand {
            input,
            edge_type,
            hops,
            direction,
            pre_filter,
        } => LogicalPlan::Expand {
            input: Box::new(optimize_plan(*input)),
            edge_type,
            hops,
            direction,
            pre_filter,
        },
        LogicalPlan::Traverse { input, pattern } => LogicalPlan::Traverse {
            input: Box::new(optimize_plan(*input)),
            pattern,
        },
        LogicalPlan::PatternMatch {
            input,
            pattern,
            where_,
        } => LogicalPlan::PatternMatch {
            input: Box::new(optimize_plan(*input)),
            pattern,
            where_,
        },
        LogicalPlan::AggregateNeighbors {
            input,
            edge_type,
            agg,
        } => LogicalPlan::AggregateNeighbors {
            input: Box::new(optimize_plan(*input)),
            edge_type,
            agg,
        },
        LogicalPlan::Sort {
            input,
            by,
            descending,
        } => LogicalPlan::Sort {
            input: Box::new(optimize_plan(*input)),
            by,
            descending,
        },
        LogicalPlan::Limit { input, n } => LogicalPlan::Limit {
            input: Box::new(optimize_plan(*input)),
            n,
        },
    }
}

fn rewrite_filter_nodes(input: LogicalPlan, predicate: Expr) -> LogicalPlan {
    match input {
        LogicalPlan::FilterNodes {
            input: inner,
            predicate: existing,
        } => LogicalPlan::FilterNodes {
            input: inner,
            predicate: and_expr(existing, predicate),
        },
        LogicalPlan::ProjectNodes {
            input: inner,
            columns,
        } if predicate_uses_only_columns(&predicate, &node_available_columns(&columns)) => {
            LogicalPlan::ProjectNodes {
                input: Box::new(rewrite_filter_nodes(*inner, predicate)),
                columns,
            }
        }
        other => LogicalPlan::FilterNodes {
            input: Box::new(other),
            predicate,
        },
    }
}

fn rewrite_filter_edges(input: LogicalPlan, predicate: Expr) -> LogicalPlan {
    match input {
        LogicalPlan::FilterEdges {
            input: inner,
            predicate: existing,
        } => LogicalPlan::FilterEdges {
            input: inner,
            predicate: and_expr(existing, predicate),
        },
        LogicalPlan::ProjectEdges {
            input: inner,
            columns,
        } if predicate_uses_only_columns(&predicate, &edge_available_columns(&columns)) => {
            LogicalPlan::ProjectEdges {
                input: Box::new(rewrite_filter_edges(*inner, predicate)),
                columns,
            }
        }
        other => LogicalPlan::FilterEdges {
            input: Box::new(other),
            predicate,
        },
    }
}

fn and_expr(left: Expr, right: Expr) -> Expr {
    Expr::And {
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn predicate_uses_only_columns(predicate: &Expr, available: &HashSet<String>) -> bool {
    expr_column_refs(predicate)
        .into_iter()
        .all(|name| available.contains(&name))
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
        Expr::Literal { .. } => {}
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
        Expr::PatternCol { .. } => {}
        Expr::StringOp { expr, pattern, .. } => {
            collect_expr_column_refs(expr, refs);
            collect_expr_column_refs(pattern, refs);
        }
    }
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
        query::{Connector, EdgeTypeSpec, Pattern, PatternStep, ScalarValue},
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

    fn score_gt_10() -> Expr {
        Expr::BinaryOp {
            left: Box::new(Expr::Col {
                name: "score".to_owned(),
            }),
            op: BinaryOp::Gt,
            right: Box::new(Expr::Literal {
                value: ScalarValue::Int(10),
            }),
        }
    }

    #[test]
    fn merges_nested_node_filters_with_and() {
        let plan = LogicalPlan::FilterNodes {
            input: Box::new(LogicalPlan::FilterNodes {
                input: Box::new(scan()),
                predicate: age_gt_30(),
            }),
            predicate: score_gt_10(),
        };

        let optimized = PredicatePushdown.optimize(plan);

        match optimized {
            LogicalPlan::FilterNodes { predicate, .. } => {
                assert!(matches!(predicate, Expr::And { .. }));
            }
            other => panic!("expected FilterNodes, got {other:?}"),
        }
    }

    #[test]
    fn merges_nested_edge_filters_with_and() {
        let plan = LogicalPlan::FilterEdges {
            input: Box::new(LogicalPlan::FilterEdges {
                input: Box::new(scan()),
                predicate: Expr::BinaryOp {
                    left: Box::new(Expr::Col {
                        name: "_type".to_owned(),
                    }),
                    op: BinaryOp::Eq,
                    right: Box::new(Expr::Literal {
                        value: ScalarValue::String("KNOWS".to_owned()),
                    }),
                },
            }),
            predicate: Expr::BinaryOp {
                left: Box::new(Expr::Col {
                    name: "weight".to_owned(),
                }),
                op: BinaryOp::Gt,
                right: Box::new(Expr::Literal {
                    value: ScalarValue::Int(1),
                }),
            },
        };

        let optimized = PredicatePushdown.optimize(plan);

        match optimized {
            LogicalPlan::FilterEdges { predicate, .. } => {
                assert!(matches!(predicate, Expr::And { .. }));
            }
            other => panic!("expected FilterEdges, got {other:?}"),
        }
    }

    #[test]
    fn pushes_node_filter_below_project_when_columns_are_preserved() {
        let plan = LogicalPlan::FilterNodes {
            input: Box::new(LogicalPlan::ProjectNodes {
                input: Box::new(scan()),
                columns: vec!["name".to_owned(), "age".to_owned()],
            }),
            predicate: age_gt_30(),
        };

        let optimized = PredicatePushdown.optimize(plan);

        match optimized {
            LogicalPlan::ProjectNodes { input, columns } => {
                assert_eq!(columns, vec!["name".to_owned(), "age".to_owned()]);
                assert!(matches!(*input, LogicalPlan::FilterNodes { .. }));
            }
            other => panic!("expected ProjectNodes, got {other:?}"),
        }
    }

    #[test]
    fn does_not_push_node_filter_below_project_when_column_missing() {
        let plan = LogicalPlan::FilterNodes {
            input: Box::new(LogicalPlan::ProjectNodes {
                input: Box::new(scan()),
                columns: vec!["name".to_owned()],
            }),
            predicate: age_gt_30(),
        };

        let optimized = PredicatePushdown.optimize(plan);
        assert!(matches!(optimized, LogicalPlan::FilterNodes { .. }));
    }

    #[test]
    fn pushes_edge_filter_below_project_when_columns_are_preserved() {
        let predicate = Expr::BinaryOp {
            left: Box::new(Expr::Col {
                name: "weight".to_owned(),
            }),
            op: BinaryOp::Gt,
            right: Box::new(Expr::Literal {
                value: ScalarValue::Int(1),
            }),
        };
        let plan = LogicalPlan::FilterEdges {
            input: Box::new(LogicalPlan::ProjectEdges {
                input: Box::new(scan()),
                columns: vec!["weight".to_owned()],
            }),
            predicate,
        };

        let optimized = PredicatePushdown.optimize(plan);
        assert!(matches!(optimized, LogicalPlan::ProjectEdges { .. }));
    }

    #[test]
    fn respects_barriers_like_expand_sort_and_limit() {
        let barrier_plans = vec![
            LogicalPlan::FilterNodes {
                input: Box::new(LogicalPlan::Expand {
                    input: Box::new(scan()),
                    edge_type: EdgeTypeSpec::Any,
                    hops: 1,
                    direction: Direction::Out,
                    pre_filter: None,
                }),
                predicate: age_gt_30(),
            },
            LogicalPlan::FilterNodes {
                input: Box::new(LogicalPlan::Sort {
                    input: Box::new(scan()),
                    by: "age".to_owned(),
                    descending: false,
                }),
                predicate: age_gt_30(),
            },
            LogicalPlan::FilterNodes {
                input: Box::new(LogicalPlan::Limit {
                    input: Box::new(scan()),
                    n: 5,
                }),
                predicate: age_gt_30(),
            },
            LogicalPlan::FilterNodes {
                input: Box::new(LogicalPlan::Traverse {
                    input: Box::new(scan()),
                    pattern: vec![PatternStep {
                        from_alias: "a".to_owned(),
                        edge_alias: Some("e".to_owned()),
                        edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
                        direction: Direction::Out,
                        to_alias: "b".to_owned(),
                    }],
                }),
                predicate: age_gt_30(),
            },
            LogicalPlan::FilterNodes {
                input: Box::new(LogicalPlan::PatternMatch {
                    input: Box::new(scan()),
                    pattern: Pattern::new(vec![PatternStep {
                        from_alias: "a".to_owned(),
                        edge_alias: Some("e".to_owned()),
                        edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
                        direction: Direction::Out,
                        to_alias: "b".to_owned(),
                    }]),
                    where_: None,
                }),
                predicate: age_gt_30(),
            },
            LogicalPlan::FilterNodes {
                input: Box::new(LogicalPlan::AggregateNeighbors {
                    input: Box::new(scan()),
                    edge_type: "KNOWS".to_owned(),
                    agg: crate::AggExpr::Count,
                }),
                predicate: age_gt_30(),
            },
        ];

        for plan in barrier_plans {
            let optimized = PredicatePushdown.optimize(plan);
            assert!(matches!(optimized, LogicalPlan::FilterNodes { .. }));
        }
    }

    #[test]
    fn optimization_is_idempotent() {
        let plan = LogicalPlan::FilterNodes {
            input: Box::new(LogicalPlan::ProjectNodes {
                input: Box::new(LogicalPlan::FilterNodes {
                    input: Box::new(scan()),
                    predicate: score_gt_10(),
                }),
                columns: vec!["score".to_owned(), "age".to_owned()],
            }),
            predicate: age_gt_30(),
        };

        let once = PredicatePushdown.optimize(plan);
        let twice = PredicatePushdown.optimize(once.clone());

        assert_eq!(format!("{once:?}"), format!("{twice:?}"));
    }
}
