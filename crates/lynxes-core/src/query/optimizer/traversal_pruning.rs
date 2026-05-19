use crate::{
    Expr, LogicalPlan, OptimizerPass, COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE,
};

#[derive(Debug, Default, Clone, Copy)]
pub struct TraversalPruning;

impl OptimizerPass for TraversalPruning {
    fn name(&self) -> &'static str {
        "TraversalPruning"
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
        LogicalPlan::FilterEdges { input, predicate } => LogicalPlan::FilterEdges {
            input: Box::new(optimize_plan(*input)),
            predicate,
        },
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
        LogicalPlan::Expand {
            input,
            edge_type,
            hops,
            direction,
            pre_filter,
        } if is_legal_pre_filter(&predicate) => LogicalPlan::Expand {
            input,
            edge_type,
            hops,
            direction,
            pre_filter: Some(match pre_filter {
                Some(existing) => and_expr(existing, predicate),
                None => predicate,
            }),
        },
        other => LogicalPlan::FilterNodes {
            input: Box::new(other),
            predicate,
        },
    }
}

fn is_legal_pre_filter(expr: &Expr) -> bool {
    match expr {
        Expr::Col { name } => !is_edge_column(name),
        Expr::Literal { .. } => true,
        Expr::BinaryOp { left, right, .. }
        | Expr::And { left, right }
        | Expr::Or { left, right } => is_legal_pre_filter(left) && is_legal_pre_filter(right),
        Expr::UnaryOp { expr, .. } | Expr::Not { expr } | Expr::Cast { expr, .. } => {
            is_legal_pre_filter(expr)
        }
        Expr::ListContains { expr, item } => is_legal_pre_filter(expr) && is_legal_pre_filter(item),
        Expr::PatternCol { .. } => false,
        Expr::StringOp { expr, pattern, .. } => {
            is_legal_pre_filter(expr) && is_legal_pre_filter(pattern)
        }
    }
}

fn is_edge_column(name: &str) -> bool {
    matches!(
        name,
        COL_EDGE_SRC | COL_EDGE_DST | COL_EDGE_TYPE | COL_EDGE_DIRECTION
    )
}

fn and_expr(left: Expr, right: Expr) -> Expr {
    Expr::And {
        left: Box::new(left),
        right: Box::new(right),
    }
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

    fn active_label() -> Expr {
        Expr::ListContains {
            expr: Box::new(Expr::Col {
                name: "_label".to_owned(),
            }),
            item: Box::new(Expr::Literal {
                value: ScalarValue::String("Active".to_owned()),
            }),
        }
    }

    #[test]
    fn absorbs_filter_nodes_directly_above_expand() {
        let plan = LogicalPlan::FilterNodes {
            input: Box::new(LogicalPlan::Expand {
                input: Box::new(scan()),
                edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
                hops: 2,
                direction: Direction::Out,
                pre_filter: None,
            }),
            predicate: age_gt_30(),
        };

        let optimized = TraversalPruning.optimize(plan);

        match optimized {
            LogicalPlan::Expand {
                pre_filter: Some(pre_filter),
                ..
            } => assert_eq!(pre_filter, age_gt_30()),
            other => panic!("expected Expand with pre_filter, got {other:?}"),
        }
    }

    #[test]
    fn merges_existing_pre_filter_with_absorbed_filter() {
        let plan = LogicalPlan::FilterNodes {
            input: Box::new(LogicalPlan::Expand {
                input: Box::new(scan()),
                edge_type: EdgeTypeSpec::Any,
                hops: 1,
                direction: Direction::Out,
                pre_filter: Some(active_label()),
            }),
            predicate: age_gt_30(),
        };

        let optimized = TraversalPruning.optimize(plan);

        match optimized {
            LogicalPlan::Expand {
                pre_filter: Some(Expr::And { left, right }),
                ..
            } => {
                assert_eq!(*left, active_label());
                assert_eq!(*right, age_gt_30());
            }
            other => panic!("expected Expand with merged pre_filter, got {other:?}"),
        }
    }

    #[test]
    fn leaves_edge_attribute_predicates_outside_expand() {
        let plan = LogicalPlan::FilterNodes {
            input: Box::new(LogicalPlan::Expand {
                input: Box::new(scan()),
                edge_type: EdgeTypeSpec::Any,
                hops: 1,
                direction: Direction::Out,
                pre_filter: None,
            }),
            predicate: Expr::BinaryOp {
                left: Box::new(Expr::Col {
                    name: "_type".to_owned(),
                }),
                op: BinaryOp::Eq,
                right: Box::new(Expr::Literal {
                    value: ScalarValue::String("KNOWS".to_owned()),
                }),
            },
        };

        let optimized = TraversalPruning.optimize(plan);
        assert!(matches!(optimized, LogicalPlan::FilterNodes { .. }));
    }

    #[test]
    fn leaves_alias_dependent_pattern_columns_outside_expand() {
        let plan = LogicalPlan::FilterNodes {
            input: Box::new(LogicalPlan::Expand {
                input: Box::new(scan()),
                edge_type: EdgeTypeSpec::Any,
                hops: 1,
                direction: Direction::Out,
                pre_filter: None,
            }),
            predicate: Expr::BinaryOp {
                left: Box::new(Expr::PatternCol {
                    alias: "n".to_owned(),
                    field: "age".to_owned(),
                }),
                op: BinaryOp::Gt,
                right: Box::new(Expr::Literal {
                    value: ScalarValue::Int(30),
                }),
            },
        };

        let optimized = TraversalPruning.optimize(plan);
        assert!(matches!(optimized, LogicalPlan::FilterNodes { .. }));
    }

    #[test]
    fn does_not_rewrite_filters_after_other_graph_operators() {
        let plans = vec![
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
            LogicalPlan::FilterEdges {
                input: Box::new(LogicalPlan::Expand {
                    input: Box::new(scan()),
                    edge_type: EdgeTypeSpec::Any,
                    hops: 1,
                    direction: Direction::Out,
                    pre_filter: None,
                }),
                predicate: age_gt_30(),
            },
        ];

        for plan in plans {
            let optimized = TraversalPruning.optimize(plan);
            assert!(matches!(
                optimized,
                LogicalPlan::FilterNodes { .. } | LogicalPlan::FilterEdges { .. }
            ));
        }
    }

    #[test]
    fn optimization_is_idempotent() {
        let plan = LogicalPlan::FilterNodes {
            input: Box::new(LogicalPlan::Expand {
                input: Box::new(LogicalPlan::ProjectNodes {
                    input: Box::new(scan()),
                    columns: vec!["age".to_owned()],
                }),
                edge_type: EdgeTypeSpec::Any,
                hops: 2,
                direction: Direction::Out,
                pre_filter: Some(active_label()),
            }),
            predicate: age_gt_30(),
        };

        let once = TraversalPruning.optimize(plan);
        let twice = TraversalPruning.optimize(once.clone());

        assert_eq!(format!("{once:?}"), format!("{twice:?}"));
    }
}
