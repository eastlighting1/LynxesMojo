use crate::{ExecutionHint, LogicalPlan, OptimizerPass, PartitionStrategy};

#[derive(Debug, Default, Clone, Copy)]
pub struct PartitionParallel;

impl OptimizerPass for PartitionParallel {
    fn name(&self) -> &'static str {
        "PartitionParallel"
    }

    fn optimize(&self, plan: LogicalPlan) -> LogicalPlan {
        optimize_plan(plan)
    }
}

fn optimize_plan(plan: LogicalPlan) -> LogicalPlan {
    let optimized = match plan {
        LogicalPlan::Scan { .. } => plan,
        LogicalPlan::Cache { input, key } => LogicalPlan::Cache {
            input: Box::new(optimize_plan(*input)),
            key,
        },
        LogicalPlan::Hint { .. } if has_partition_hint(&plan) => plan,
        LogicalPlan::Hint { input, hint } => LogicalPlan::Hint {
            input: Box::new(optimize_plan(*input)),
            hint,
        },
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
        LogicalPlan::Limit { input, n } => LogicalPlan::Limit {
            input: Box::new(optimize_plan(*input)),
            n,
        },
    };

    annotate_parallel(optimized)
}

fn annotate_parallel(plan: LogicalPlan) -> LogicalPlan {
    if has_partition_hint(&plan) {
        return plan;
    }

    match partition_strategy(&plan) {
        Some(strategy) => LogicalPlan::Hint {
            input: Box::new(plan),
            hint: ExecutionHint::PartitionParallel { strategy },
        },
        None => plan,
    }
}

fn partition_strategy(plan: &LogicalPlan) -> Option<PartitionStrategy> {
    match plan {
        LogicalPlan::Expand { .. } => Some(PartitionStrategy::ExpandFrontier),
        LogicalPlan::PatternMatch { .. } => Some(PartitionStrategy::PatternRoots),
        _ => None,
    }
}

fn has_partition_hint(plan: &LogicalPlan) -> bool {
    matches!(
        plan,
        LogicalPlan::Hint {
            hint: ExecutionHint::PartitionParallel { .. },
            ..
        }
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{
        query::{Connector, EdgeTypeSpec, Pattern, PatternStep},
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
    fn annotates_expand_with_frontier_partition_strategy() {
        let plan = LogicalPlan::Expand {
            input: Box::new(scan()),
            edge_type: EdgeTypeSpec::Any,
            hops: 2,
            direction: Direction::Out,
            pre_filter: None,
        };

        let optimized = PartitionParallel.optimize(plan);

        match optimized {
            LogicalPlan::Hint {
                hint: ExecutionHint::PartitionParallel { strategy },
                input,
            } => {
                assert_eq!(strategy, PartitionStrategy::ExpandFrontier);
                assert!(matches!(*input, LogicalPlan::Expand { .. }));
            }
            other => panic!("expected partition hint, got {other:?}"),
        }
    }

    #[test]
    fn annotates_pattern_match_with_pattern_root_strategy() {
        let plan = LogicalPlan::PatternMatch {
            input: Box::new(scan()),
            pattern: Pattern::new(vec![PatternStep {
                from_alias: "a".to_owned(),
                edge_alias: Some("e".to_owned()),
                edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
                direction: Direction::Out,
                to_alias: "b".to_owned(),
            }]),
            where_: None,
        };

        let optimized = PartitionParallel.optimize(plan);

        match optimized {
            LogicalPlan::Hint {
                hint: ExecutionHint::PartitionParallel { strategy },
                input,
            } => {
                assert_eq!(strategy, PartitionStrategy::PatternRoots);
                assert!(matches!(*input, LogicalPlan::PatternMatch { .. }));
            }
            other => panic!("expected partition hint, got {other:?}"),
        }
    }

    #[test]
    fn preserves_existing_non_partition_hints_while_annotating_inner_candidate() {
        let plan = LogicalPlan::Hint {
            input: Box::new(LogicalPlan::Expand {
                input: Box::new(scan()),
                edge_type: EdgeTypeSpec::Any,
                hops: 1,
                direction: Direction::Out,
                pre_filter: None,
            }),
            hint: ExecutionHint::LimitAware { n: 10 },
        };

        let optimized = PartitionParallel.optimize(plan);

        match optimized {
            LogicalPlan::Hint {
                hint: ExecutionHint::LimitAware { n },
                input,
            } => {
                assert_eq!(n, 10);
                assert!(matches!(
                    *input,
                    LogicalPlan::Hint {
                        hint: ExecutionHint::PartitionParallel {
                            strategy: PartitionStrategy::ExpandFrontier
                        },
                        ..
                    }
                ));
            }
            other => {
                panic!("expected preserved outer hint with inner partition hint, got {other:?}")
            }
        }
    }

    #[test]
    fn leaves_non_target_nodes_unchanged() {
        let plan = LogicalPlan::ProjectNodes {
            input: Box::new(scan()),
            columns: vec!["name".to_owned()],
        };

        let optimized = PartitionParallel.optimize(plan);
        assert!(matches!(optimized, LogicalPlan::ProjectNodes { .. }));
    }

    #[test]
    fn does_not_duplicate_partition_hints() {
        let plan = LogicalPlan::Hint {
            input: Box::new(LogicalPlan::Expand {
                input: Box::new(scan()),
                edge_type: EdgeTypeSpec::Any,
                hops: 1,
                direction: Direction::Out,
                pre_filter: None,
            }),
            hint: ExecutionHint::PartitionParallel {
                strategy: PartitionStrategy::ExpandFrontier,
            },
        };

        let optimized = PartitionParallel.optimize(plan);

        match optimized {
            LogicalPlan::Hint {
                hint: ExecutionHint::PartitionParallel { strategy },
                input,
            } => {
                assert_eq!(strategy, PartitionStrategy::ExpandFrontier);
                assert!(matches!(
                    *input,
                    LogicalPlan::Expand { .. }
                        | LogicalPlan::Hint {
                            hint: ExecutionHint::PartitionParallel {
                                strategy: PartitionStrategy::ExpandFrontier
                            },
                            ..
                        }
                ));
            }
            other => panic!("expected single partition hint, got {other:?}"),
        }
    }

    #[test]
    fn optimization_is_idempotent() {
        let plan = LogicalPlan::Limit {
            input: Box::new(LogicalPlan::PatternMatch {
                input: Box::new(LogicalPlan::Expand {
                    input: Box::new(scan()),
                    edge_type: EdgeTypeSpec::Any,
                    hops: 2,
                    direction: Direction::Out,
                    pre_filter: None,
                }),
                pattern: Pattern::new(vec![PatternStep {
                    from_alias: "a".to_owned(),
                    edge_alias: Some("e".to_owned()),
                    edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
                    direction: Direction::Out,
                    to_alias: "b".to_owned(),
                }]),
                where_: None,
            }),
            n: 5,
        };

        let once = PartitionParallel.optimize(plan);
        let twice = PartitionParallel.optimize(once.clone());

        assert_eq!(format!("{once:?}"), format!("{twice:?}"));
    }
}
