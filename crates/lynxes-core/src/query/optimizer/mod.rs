mod early_termination;
mod partition_parallel;
mod pattern_expansion;
mod predicate_pushdown;
mod projection_pushdown;
mod subgraph_caching;
mod traversal_pruning;

use crate::LogicalPlan;

pub use early_termination::EarlyTermination;
pub use partition_parallel::PartitionParallel;
#[allow(unused_imports)]
pub use pattern_expansion::PatternExpansion;
pub use predicate_pushdown::PredicatePushdown;
pub use projection_pushdown::ProjectionPushdown;
pub use subgraph_caching::SubgraphCaching;
pub use traversal_pruning::TraversalPruning;

/// One semantics-preserving logical rewrite pass.
pub trait OptimizerPass: Send + Sync {
    fn name(&self) -> &'static str;
    fn optimize(&self, plan: LogicalPlan) -> LogicalPlan;
}

/// Pass enable/disable switches for the logical optimizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptimizerOptions {
    pub predicate_pushdown: bool,
    pub projection_pushdown: bool,
    pub traversal_pruning: bool,
    pub subgraph_caching: bool,
    pub early_termination: bool,
    pub partition_parallel: bool,
    pub pattern_expansion: bool,
}

impl Default for OptimizerOptions {
    fn default() -> Self {
        Self {
            predicate_pushdown: true,
            projection_pushdown: true,
            traversal_pruning: true,
            subgraph_caching: false,
            early_termination: false,
            partition_parallel: false,
            pattern_expansion: true,
        }
    }
}

/// Ordered logical optimizer runner.
///
/// Disabled passes are omitted; enabled passes preserve canonical order.
pub struct Optimizer {
    passes: Vec<Box<dyn OptimizerPass>>,
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::new(OptimizerOptions::default())
    }
}

impl Optimizer {
    pub fn new(options: OptimizerOptions) -> Self {
        let mut passes: Vec<Box<dyn OptimizerPass>> = Vec::new();

        if options.predicate_pushdown {
            passes.push(Box::new(PredicatePushdown));
        }
        if options.projection_pushdown {
            passes.push(Box::new(ProjectionPushdown));
        }
        if options.traversal_pruning {
            passes.push(Box::new(TraversalPruning));
        }
        if options.subgraph_caching {
            passes.push(Box::new(SubgraphCaching));
        }
        if options.early_termination {
            passes.push(Box::new(EarlyTermination));
        }
        if options.partition_parallel {
            passes.push(Box::new(PartitionParallel));
        }
        if options.pattern_expansion {
            passes.push(Box::new(PatternExpansion));
        }

        Self { passes }
    }

    pub fn with_passes(passes: Vec<Box<dyn OptimizerPass>>) -> Self {
        Self { passes }
    }

    pub fn run(&self, plan: LogicalPlan) -> LogicalPlan {
        self.passes
            .iter()
            .fold(plan, |plan, pass| pass.optimize(plan))
    }

    pub fn pass_names(&self) -> Vec<&'static str> {
        self.passes.iter().map(|pass| pass.name()).collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use crate::{
        query::{Connector, EdgeTypeSpec, Expr, Pattern, PatternStep, ScalarValue},
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

    #[derive(Debug)]
    struct RecordingPass {
        name: &'static str,
        log: Arc<Mutex<Vec<&'static str>>>,
    }

    impl OptimizerPass for RecordingPass {
        fn name(&self) -> &'static str {
            self.name
        }

        fn optimize(&self, plan: LogicalPlan) -> LogicalPlan {
            self.log.lock().unwrap().push(self.name);
            plan
        }
    }

    #[test]
    fn default_options_match_spec() {
        let opts = OptimizerOptions::default();
        assert!(opts.predicate_pushdown);
        assert!(opts.projection_pushdown);
        assert!(opts.traversal_pruning);
        assert!(!opts.subgraph_caching);
        assert!(!opts.early_termination);
        assert!(!opts.partition_parallel);
        assert!(opts.pattern_expansion);
    }

    #[test]
    fn default_optimizer_uses_canonical_enabled_pass_order() {
        let optimizer = Optimizer::default();
        assert_eq!(
            optimizer.pass_names(),
            vec![
                "PredicatePushdown",
                "ProjectionPushdown",
                "TraversalPruning",
                "PatternExpansion",
            ]
        );
    }

    #[test]
    fn enabling_all_passes_preserves_canonical_order() {
        let optimizer = Optimizer::new(OptimizerOptions {
            predicate_pushdown: true,
            projection_pushdown: true,
            traversal_pruning: true,
            subgraph_caching: true,
            early_termination: true,
            partition_parallel: true,
            pattern_expansion: true,
        });

        assert_eq!(
            optimizer.pass_names(),
            vec![
                "PredicatePushdown",
                "ProjectionPushdown",
                "TraversalPruning",
                "SubgraphCaching",
                "EarlyTermination",
                "PartitionParallel",
                "PatternExpansion",
            ]
        );
    }

    #[test]
    fn disabled_passes_are_skipped_without_reordering_remaining_passes() {
        let optimizer = Optimizer::new(OptimizerOptions {
            predicate_pushdown: false,
            projection_pushdown: true,
            traversal_pruning: false,
            subgraph_caching: true,
            early_termination: false,
            partition_parallel: true,
            pattern_expansion: true,
        });

        assert_eq!(
            optimizer.pass_names(),
            vec![
                "ProjectionPushdown",
                "SubgraphCaching",
                "PartitionParallel",
                "PatternExpansion"
            ]
        );
    }

    #[test]
    fn run_applies_passes_in_order() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let optimizer = Optimizer::with_passes(vec![
            Box::new(RecordingPass {
                name: "first",
                log: Arc::clone(&log),
            }),
            Box::new(RecordingPass {
                name: "second",
                log: Arc::clone(&log),
            }),
            Box::new(RecordingPass {
                name: "third",
                log: Arc::clone(&log),
            }),
        ]);

        let _ = optimizer.run(scan());

        assert_eq!(*log.lock().unwrap(), vec!["first", "second", "third"]);
    }

    #[test]
    fn no_op_default_optimizer_is_idempotent() {
        let plan = LogicalPlan::Limit {
            input: Box::new(LogicalPlan::Sort {
                input: Box::new(LogicalPlan::PatternMatch {
                    input: Box::new(LogicalPlan::Expand {
                        input: Box::new(LogicalPlan::FilterNodes {
                            input: Box::new(scan()),
                            predicate: Expr::BinaryOp {
                                left: Box::new(Expr::Col {
                                    name: "age".to_owned(),
                                }),
                                op: crate::BinaryOp::Gt,
                                right: Box::new(Expr::Literal {
                                    value: ScalarValue::Int(30),
                                }),
                            },
                        }),
                        edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
                        hops: 2,
                        direction: Direction::Out,
                        pre_filter: None,
                    }),
                    pattern: Pattern::new(vec![PatternStep {
                        from_alias: "a".to_owned(),
                        edge_alias: Some("e".to_owned()),
                        edge_type: EdgeTypeSpec::Single("WORKS_AT".to_owned()),
                        direction: Direction::Out,
                        to_alias: "b".to_owned(),
                    }]),
                    where_: None,
                }),
                by: "score".to_owned(),
                descending: true,
            }),
            n: 10,
        };
        let optimizer = Optimizer::default();

        let once = optimizer.run(plan.clone());
        let twice = optimizer.run(once.clone());

        assert_eq!(format!("{once:?}"), format!("{twice:?}"));
    }
}
