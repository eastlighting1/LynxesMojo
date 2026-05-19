pub mod query;

pub use query::{
    AggExpr, BinaryOp, Connector, EarlyTermination, EdgeTypeSpec, ExecutionHint, Expr, LogicalPlan,
    Optimizer, OptimizerOptions, OptimizerPass, PartitionParallel, PartitionStrategy, Pattern,
    PatternExpansion, PatternNodeConstraint, PatternStep, PatternStepConstraint, PlanDomain,
    PredicatePushdown, ProjectionPushdown, SampledSubgraph, SamplingConfig, ScalarValue, StringOp,
    SubgraphCaching, TraversalPruning, UnaryOp,
};

#[cfg(test)]
mod tests {
    use super::{OptimizerPass, PatternExpansion, SampledSubgraph, SamplingConfig};

    #[test]
    fn reexports_pattern_expansion_from_plan_facade() {
        let pass = PatternExpansion;
        assert_eq!(OptimizerPass::name(&pass), "PatternExpansion");
    }

    #[test]
    fn reexports_sampling_types_from_plan_facade() {
        let config = SamplingConfig::default();
        let sampled = SampledSubgraph::default();

        assert_eq!(config.hops, 1);
        assert!(sampled.node_indices.is_empty());
    }
}
