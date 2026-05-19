use std::sync::Arc;

use crate::{
    query::{AggExpr, Connector, EdgeTypeSpec, Expr, Pattern, PatternStep},
    Direction,
};

/// Logical result domains that a plan node can produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanDomain {
    Graph,
    Nodes,
    Edges,
    PatternRows,
}

/// Executor-visible optimization hints that do not change logical semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionHint {
    TopK { n: usize },
    LimitAware { n: usize },
    PartitionParallel { strategy: PartitionStrategy },
}

/// Parallel execution strategies available to the physical planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionStrategy {
    ExpandFrontier,
    PatternRoots,
}

/// Canonical lazy plan tree for Lynxes query execution.
///
/// In v0.1 this is a pure tree: every non-`Scan` node has exactly one input.
#[derive(Debug, Clone)]
pub enum LogicalPlan {
    Scan {
        source: Arc<dyn Connector>,
        node_columns: Option<Vec<String>>,
        edge_columns: Option<Vec<String>>,
    },
    Cache {
        input: Box<LogicalPlan>,
        key: String,
    },
    Hint {
        input: Box<LogicalPlan>,
        hint: ExecutionHint,
    },
    FilterNodes {
        input: Box<LogicalPlan>,
        predicate: Expr,
    },
    FilterEdges {
        input: Box<LogicalPlan>,
        predicate: Expr,
    },
    ProjectNodes {
        input: Box<LogicalPlan>,
        columns: Vec<String>,
    },
    ProjectEdges {
        input: Box<LogicalPlan>,
        columns: Vec<String>,
    },
    Expand {
        input: Box<LogicalPlan>,
        edge_type: EdgeTypeSpec,
        hops: u32,
        direction: Direction,
        pre_filter: Option<Expr>,
    },
    Traverse {
        input: Box<LogicalPlan>,
        pattern: Vec<PatternStep>,
    },
    PatternMatch {
        input: Box<LogicalPlan>,
        pattern: Pattern,
        where_: Option<Expr>,
    },
    AggregateNeighbors {
        input: Box<LogicalPlan>,
        edge_type: String,
        agg: AggExpr,
    },
    Sort {
        input: Box<LogicalPlan>,
        by: String,
        descending: bool,
    },
    Limit {
        input: Box<LogicalPlan>,
        n: usize,
    },
}

impl LogicalPlan {
    /// Returns the direct input subtree, if this node is not a `Scan`.
    pub fn input(&self) -> Option<&LogicalPlan> {
        match self {
            Self::Scan { .. } => None,
            Self::Cache { input, .. }
            | Self::Hint { input, .. }
            | Self::FilterNodes { input, .. }
            | Self::FilterEdges { input, .. }
            | Self::ProjectNodes { input, .. }
            | Self::ProjectEdges { input, .. }
            | Self::Expand { input, .. }
            | Self::Traverse { input, .. }
            | Self::PatternMatch { input, .. }
            | Self::AggregateNeighbors { input, .. }
            | Self::Sort { input, .. }
            | Self::Limit { input, .. } => Some(input.as_ref()),
        }
    }

    /// Returns the logical output domain of this node.
    pub fn output_domain(&self) -> PlanDomain {
        match self {
            Self::Scan { .. } => PlanDomain::Graph,
            Self::Cache { input, .. } => input.output_domain(),
            Self::Hint { input, .. } => input.output_domain(),
            Self::FilterNodes { .. } => PlanDomain::Nodes,
            Self::FilterEdges { .. } => PlanDomain::Edges,
            Self::ProjectNodes { .. } => PlanDomain::Nodes,
            Self::ProjectEdges { .. } => PlanDomain::Edges,
            Self::Expand { .. } => PlanDomain::Graph,
            Self::Traverse { .. } => PlanDomain::Graph,
            Self::PatternMatch { .. } => PlanDomain::PatternRows,
            Self::AggregateNeighbors { .. } => PlanDomain::Nodes,
            Self::Sort { input, .. } | Self::Limit { input, .. } => input.output_domain(),
        }
    }
}

#[cfg(test)]
mod tests {
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
    fn scan_is_the_only_leaf() {
        let plan = scan();
        assert!(plan.input().is_none());
        assert_eq!(plan.output_domain(), PlanDomain::Graph);
    }

    #[test]
    fn cache_wrapper_preserves_input_domain() {
        let plan = LogicalPlan::Cache {
            input: Box::new(LogicalPlan::ProjectNodes {
                input: Box::new(scan()),
                columns: vec!["name".to_owned()],
            }),
            key: "cache:demo".to_owned(),
        };

        assert!(plan.input().is_some());
        assert_eq!(plan.output_domain(), PlanDomain::Nodes);
    }

    #[test]
    fn hint_wrapper_preserves_input_domain() {
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

        assert!(plan.input().is_some());
        assert_eq!(plan.output_domain(), PlanDomain::Graph);
    }

    #[test]
    fn relational_nodes_report_expected_output_domains() {
        let nodes = LogicalPlan::FilterNodes {
            input: Box::new(scan()),
            predicate: Expr::Literal {
                value: crate::query::ScalarValue::Bool(true),
            },
        };
        let edges = LogicalPlan::ProjectEdges {
            input: Box::new(scan()),
            columns: vec!["weight".to_owned()],
        };

        assert_eq!(nodes.output_domain(), PlanDomain::Nodes);
        assert_eq!(edges.output_domain(), PlanDomain::Edges);
    }

    #[test]
    fn sort_and_limit_preserve_input_domain() {
        let plan = LogicalPlan::Limit {
            input: Box::new(LogicalPlan::Sort {
                input: Box::new(LogicalPlan::AggregateNeighbors {
                    input: Box::new(scan()),
                    edge_type: "KNOWS".to_owned(),
                    agg: AggExpr::Count,
                }),
                by: "degree".to_owned(),
                descending: true,
            }),
            n: 10,
        };

        assert_eq!(plan.output_domain(), PlanDomain::Nodes);
    }

    #[test]
    fn pattern_match_reports_pattern_row_domain() {
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

        assert_eq!(plan.output_domain(), PlanDomain::PatternRows);
    }
}
