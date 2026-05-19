use crate::{ExecutionHint, LogicalPlan, OptimizerPass};

#[derive(Debug, Default, Clone, Copy)]
pub struct EarlyTermination;

impl OptimizerPass for EarlyTermination {
    fn name(&self) -> &'static str {
        "EarlyTermination"
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
        LogicalPlan::Hint { .. } => plan,
        LogicalPlan::FilterNodes { input, predicate } => LogicalPlan::FilterNodes {
            input: Box::new(optimize_plan(*input)),
            predicate,
        },
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
        LogicalPlan::Limit { input, n } => rewrite_limit(optimize_plan(*input), n),
    }
}

fn rewrite_limit(input: LogicalPlan, n: usize) -> LogicalPlan {
    match input {
        LogicalPlan::Limit {
            input: inner,
            n: inner_n,
        } => rewrite_limit(*inner, n.min(inner_n)),
        LogicalPlan::Hint {
            input: hinted,
            hint,
        } => {
            let rewritten = rewrite_limit(*hinted, n);
            if matches_hint(&hint, &rewritten, n) {
                LogicalPlan::Hint {
                    input: Box::new(rewritten),
                    hint,
                }
            } else {
                rewritten
            }
        }
        plan => LogicalPlan::Limit {
            input: Box::new(annotate_limit_input(plan, n)),
            n,
        },
    }
}

fn annotate_limit_input(plan: LogicalPlan, n: usize) -> LogicalPlan {
    match plan {
        LogicalPlan::Hint { ref hint, .. } if matches_hint_kind(hint, n) => plan,
        LogicalPlan::Sort { .. } => LogicalPlan::Hint {
            input: Box::new(plan),
            hint: ExecutionHint::TopK { n },
        },
        LogicalPlan::Expand { .. }
        | LogicalPlan::Traverse { .. }
        | LogicalPlan::PatternMatch { .. } => LogicalPlan::Hint {
            input: Box::new(plan),
            hint: ExecutionHint::LimitAware { n },
        },
        other => other,
    }
}

fn matches_hint(hint: &ExecutionHint, plan: &LogicalPlan, n: usize) -> bool {
    match hint {
        ExecutionHint::TopK { n: hinted_n } => {
            *hinted_n == n
                && matches!(plan, LogicalPlan::Limit { input, .. } if matches!(input.as_ref(), LogicalPlan::Sort { .. }))
        }
        ExecutionHint::LimitAware { n: hinted_n } => {
            *hinted_n == n
                && matches!(
                    plan,
                    LogicalPlan::Limit { input, .. }
                        if matches!(
                            input.as_ref(),
                            LogicalPlan::Expand { .. }
                                | LogicalPlan::Traverse { .. }
                                | LogicalPlan::PatternMatch { .. }
                        )
                )
        }
        ExecutionHint::PartitionParallel { .. } => false,
    }
}

fn matches_hint_kind(hint: &ExecutionHint, n: usize) -> bool {
    match hint {
        ExecutionHint::TopK { n: hinted_n } | ExecutionHint::LimitAware { n: hinted_n } => {
            *hinted_n == n
        }
        ExecutionHint::PartitionParallel { .. } => false,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{
        query::{Connector, EdgeTypeSpec, PatternStep},
        Direction,
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

    #[test]
    fn collapses_nested_limits_to_the_smallest_bound() {
        let plan = LogicalPlan::Limit {
            input: Box::new(LogicalPlan::Limit {
                input: Box::new(scan()),
                n: 100,
            }),
            n: 10,
        };

        let optimized = EarlyTermination.optimize(plan);

        match optimized {
            LogicalPlan::Limit { input, n } => {
                assert_eq!(n, 10);
                assert!(matches!(*input, LogicalPlan::Scan { .. }));
            }
            other => panic!("expected collapsed Limit, got {other:?}"),
        }
    }

    #[test]
    fn annotates_sort_limit_as_top_k() {
        let plan = LogicalPlan::Limit {
            input: Box::new(LogicalPlan::Sort {
                input: Box::new(scan()),
                by: "score".to_owned(),
                descending: true,
            }),
            n: 25,
        };

        let optimized = EarlyTermination.optimize(plan);

        match optimized {
            LogicalPlan::Limit { input, n } => {
                assert_eq!(n, 25);
                match *input {
                    LogicalPlan::Hint {
                        hint: ExecutionHint::TopK { n },
                        input,
                    } => {
                        assert_eq!(n, 25);
                        assert!(matches!(*input, LogicalPlan::Sort { .. }));
                    }
                    other => panic!("expected TopK hint, got {other:?}"),
                }
            }
            other => panic!("expected Limit, got {other:?}"),
        }
    }

    #[test]
    fn annotates_expand_limit_as_limit_aware() {
        let plan = LogicalPlan::Limit {
            input: Box::new(LogicalPlan::Expand {
                input: Box::new(scan()),
                edge_type: EdgeTypeSpec::Any,
                hops: 2,
                direction: Direction::Out,
                pre_filter: None,
            }),
            n: 5,
        };

        let optimized = EarlyTermination.optimize(plan);

        match optimized {
            LogicalPlan::Limit { input, .. } => match *input {
                LogicalPlan::Hint {
                    hint: ExecutionHint::LimitAware { n },
                    input,
                } => {
                    assert_eq!(n, 5);
                    assert!(matches!(*input, LogicalPlan::Expand { .. }));
                }
                other => panic!("expected LimitAware hint, got {other:?}"),
            },
            other => panic!("expected Limit, got {other:?}"),
        }
    }

    #[test]
    fn annotates_pattern_match_limit_as_limit_aware() {
        let plan = LogicalPlan::Limit {
            input: Box::new(LogicalPlan::PatternMatch {
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
                pattern: crate::Pattern::new(vec![PatternStep {
                    from_alias: "b".to_owned(),
                    edge_alias: Some("e2".to_owned()),
                    edge_type: EdgeTypeSpec::Single("WORKS_AT".to_owned()),
                    direction: Direction::Out,
                    to_alias: "c".to_owned(),
                }]),
                where_: None,
            }),
            n: 3,
        };

        let optimized = EarlyTermination.optimize(plan);

        match optimized {
            LogicalPlan::Limit { input, .. } => {
                assert!(matches!(
                    *input,
                    LogicalPlan::Hint {
                        hint: ExecutionHint::LimitAware { n: 3 },
                        ..
                    }
                ));
            }
            other => panic!("expected Limit, got {other:?}"),
        }
    }

    #[test]
    fn leaves_non_annotatable_inputs_unchanged_under_limit() {
        let plan = LogicalPlan::Limit {
            input: Box::new(LogicalPlan::AggregateNeighbors {
                input: Box::new(scan()),
                edge_type: "KNOWS".to_owned(),
                agg: crate::AggExpr::Count,
            }),
            n: 7,
        };

        let optimized = EarlyTermination.optimize(plan);

        match optimized {
            LogicalPlan::Limit { input, n } => {
                assert_eq!(n, 7);
                assert!(matches!(*input, LogicalPlan::AggregateNeighbors { .. }));
            }
            other => panic!("expected Limit, got {other:?}"),
        }
    }

    #[test]
    fn optimization_is_idempotent() {
        let plan = LogicalPlan::Limit {
            input: Box::new(LogicalPlan::Limit {
                input: Box::new(LogicalPlan::Sort {
                    input: Box::new(LogicalPlan::Expand {
                        input: Box::new(scan()),
                        edge_type: EdgeTypeSpec::Any,
                        hops: 1,
                        direction: Direction::Out,
                        pre_filter: None,
                    }),
                    by: "score".to_owned(),
                    descending: true,
                }),
                n: 50,
            }),
            n: 10,
        };

        let once = EarlyTermination.optimize(plan);
        let twice = EarlyTermination.optimize(once.clone());

        assert_eq!(format!("{once:?}"), format!("{twice:?}"));
    }
}
