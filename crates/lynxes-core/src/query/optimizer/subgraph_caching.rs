use crate::LogicalPlan;
use crate::OptimizerPass;

#[derive(Debug, Default, Clone, Copy)]
pub struct SubgraphCaching;

impl OptimizerPass for SubgraphCaching {
    fn name(&self) -> &'static str {
        "SubgraphCaching"
    }

    fn optimize(&self, plan: LogicalPlan) -> LogicalPlan {
        optimize_plan(plan)
    }
}

fn optimize_plan(plan: LogicalPlan) -> LogicalPlan {
    let optimized = match plan {
        LogicalPlan::Scan { .. } => plan,
        LogicalPlan::Cache { .. } => plan,
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

    maybe_wrap_cache(optimized)
}

fn maybe_wrap_cache(plan: LogicalPlan) -> LogicalPlan {
    if matches!(plan, LogicalPlan::Cache { .. }) {
        return plan;
    }

    if !is_cache_candidate(&plan) {
        return plan;
    }

    let Some(key) = fingerprint(&plan) else {
        return plan;
    };

    LogicalPlan::Cache {
        input: Box::new(plan),
        key,
    }
}

fn is_cache_candidate(plan: &LogicalPlan) -> bool {
    matches!(
        plan,
        LogicalPlan::Expand { .. }
            | LogicalPlan::Traverse { .. }
            | LogicalPlan::PatternMatch { .. }
            | LogicalPlan::AggregateNeighbors { .. }
    )
}

fn fingerprint(plan: &LogicalPlan) -> Option<String> {
    match plan {
        LogicalPlan::Scan {
            source,
            node_columns,
            edge_columns,
        } => {
            let source_key = source.cache_source_key()?;
            Some(format!(
                "scan:{source_key}:nodes={:?}:edges={:?}",
                node_columns, edge_columns
            ))
        }
        LogicalPlan::Cache { key, .. } => Some(key.clone()),
        LogicalPlan::Hint { input, hint } => {
            Some(format!("hint({};{:?})", fingerprint(input)?, hint))
        }
        LogicalPlan::FilterNodes { input, predicate } => Some(format!(
            "filter_nodes({};{:?})",
            fingerprint(input)?,
            predicate
        )),
        LogicalPlan::FilterEdges { input, predicate } => Some(format!(
            "filter_edges({};{:?})",
            fingerprint(input)?,
            predicate
        )),
        LogicalPlan::ProjectNodes { input, columns } => Some(format!(
            "project_nodes({};{:?})",
            fingerprint(input)?,
            columns
        )),
        LogicalPlan::ProjectEdges { input, columns } => Some(format!(
            "project_edges({};{:?})",
            fingerprint(input)?,
            columns
        )),
        LogicalPlan::Expand {
            input,
            edge_type,
            hops,
            direction,
            pre_filter,
        } => Some(format!(
            "expand({};{:?};{hops};{:?};{:?})",
            fingerprint(input)?,
            edge_type,
            direction,
            pre_filter
        )),
        LogicalPlan::Traverse { input, pattern } => {
            Some(format!("traverse({};{:?})", fingerprint(input)?, pattern))
        }
        LogicalPlan::PatternMatch {
            input,
            pattern,
            where_,
        } => Some(format!(
            "pattern_match({};{:?};{:?})",
            fingerprint(input)?,
            pattern,
            where_
        )),
        LogicalPlan::AggregateNeighbors {
            input,
            edge_type,
            agg,
        } => Some(format!(
            "aggregate_neighbors({};{edge_type};{:?})",
            fingerprint(input)?,
            agg
        )),
        LogicalPlan::Sort {
            input,
            by,
            descending,
        } => Some(format!("sort({};{by};{descending})", fingerprint(input)?)),
        LogicalPlan::Limit { input, n } => Some(format!("limit({};{n})", fingerprint(input)?)),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{
        query::{Connector, EdgeTypeSpec, Expr, Pattern, PatternStep, ScalarValue},
        AggExpr, BinaryOp, Direction,
    };

    use super::*;

    #[derive(Debug)]
    struct CachedConnector {
        key: &'static str,
    }

    impl Connector for CachedConnector {
        fn cache_source_key(&self) -> Option<String> {
            Some(self.key.to_owned())
        }
    }

    #[derive(Debug)]
    struct UncachedConnector;

    impl Connector for UncachedConnector {}

    fn cached_scan() -> LogicalPlan {
        LogicalPlan::Scan {
            source: Arc::new(CachedConnector { key: "demo-source" }),
            node_columns: None,
            edge_columns: None,
        }
    }

    fn uncached_scan() -> LogicalPlan {
        LogicalPlan::Scan {
            source: Arc::new(UncachedConnector),
            node_columns: None,
            edge_columns: None,
        }
    }

    #[test]
    fn wraps_cacheable_expand_with_deterministic_key() {
        let plan = LogicalPlan::Expand {
            input: Box::new(cached_scan()),
            edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
            hops: 2,
            direction: Direction::Out,
            pre_filter: Some(Expr::BinaryOp {
                left: Box::new(Expr::Col {
                    name: "age".to_owned(),
                }),
                op: BinaryOp::Gt,
                right: Box::new(Expr::Literal {
                    value: ScalarValue::Int(30),
                }),
            }),
        };

        let optimized = SubgraphCaching.optimize(plan);

        match optimized {
            LogicalPlan::Cache { key, input } => {
                assert!(key.contains("expand("));
                assert!(matches!(*input, LogicalPlan::Expand { .. }));
            }
            other => panic!("expected Cache wrapper, got {other:?}"),
        }
    }

    #[test]
    fn equivalent_subplans_receive_same_cache_key() {
        let first = LogicalPlan::AggregateNeighbors {
            input: Box::new(cached_scan()),
            edge_type: "KNOWS".to_owned(),
            agg: AggExpr::Sum {
                expr: Expr::Col {
                    name: "weight".to_owned(),
                },
            },
        };
        let second = LogicalPlan::AggregateNeighbors {
            input: Box::new(cached_scan()),
            edge_type: "KNOWS".to_owned(),
            agg: AggExpr::Sum {
                expr: Expr::Col {
                    name: "weight".to_owned(),
                },
            },
        };

        let first = SubgraphCaching.optimize(first);
        let second = SubgraphCaching.optimize(second);

        match (first, second) {
            (LogicalPlan::Cache { key: left, .. }, LogicalPlan::Cache { key: right, .. }) => {
                assert_eq!(left, right);
            }
            other => panic!("expected Cache wrappers, got {other:?}"),
        }
    }

    #[test]
    fn different_subplans_receive_different_cache_keys() {
        let first = LogicalPlan::Expand {
            input: Box::new(cached_scan()),
            edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
            hops: 1,
            direction: Direction::Out,
            pre_filter: None,
        };
        let second = LogicalPlan::Expand {
            input: Box::new(cached_scan()),
            edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
            hops: 2,
            direction: Direction::Out,
            pre_filter: None,
        };

        let first = SubgraphCaching.optimize(first);
        let second = SubgraphCaching.optimize(second);

        match (first, second) {
            (LogicalPlan::Cache { key: left, .. }, LogicalPlan::Cache { key: right, .. }) => {
                assert_ne!(left, right);
            }
            other => panic!("expected Cache wrappers, got {other:?}"),
        }
    }

    #[test]
    fn leaves_plans_without_cacheable_source_unchanged() {
        let plan = LogicalPlan::Expand {
            input: Box::new(uncached_scan()),
            edge_type: EdgeTypeSpec::Any,
            hops: 1,
            direction: Direction::Out,
            pre_filter: None,
        };

        let optimized = SubgraphCaching.optimize(plan);
        assert!(matches!(optimized, LogicalPlan::Expand { .. }));
    }

    #[test]
    fn does_not_wrap_non_candidate_relational_nodes() {
        let plan = LogicalPlan::ProjectNodes {
            input: Box::new(cached_scan()),
            columns: vec!["name".to_owned()],
        };

        let optimized = SubgraphCaching.optimize(plan);
        assert!(matches!(optimized, LogicalPlan::ProjectNodes { .. }));
    }

    #[test]
    fn optimization_is_idempotent() {
        let plan = LogicalPlan::PatternMatch {
            input: Box::new(LogicalPlan::Traverse {
                input: Box::new(cached_scan()),
                pattern: vec![PatternStep {
                    from_alias: "a".to_owned(),
                    edge_alias: Some("e".to_owned()),
                    edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
                    direction: Direction::Out,
                    to_alias: "b".to_owned(),
                }],
            }),
            pattern: Pattern::new(vec![PatternStep {
                from_alias: "b".to_owned(),
                edge_alias: Some("e2".to_owned()),
                edge_type: EdgeTypeSpec::Single("WORKS_AT".to_owned()),
                direction: Direction::Out,
                to_alias: "c".to_owned(),
            }]),
            where_: None,
        };

        let once = SubgraphCaching.optimize(plan);
        let twice = SubgraphCaching.optimize(once.clone());

        assert_eq!(format!("{once:?}"), format!("{twice:?}"));
    }
}
