//! Core Rust engine for Lynxes.

mod algo;
mod connector;
mod display;
mod error;
mod frame;
mod mojo_runtime;
mod query;
mod schema;
mod types;

pub use crate::algo::centrality::BetweennessConfig;
pub use crate::algo::community::{CommunityAlgorithm, CommunityConfig};
pub use crate::algo::pagerank::PageRankConfig;
pub use crate::algo::partition::{
    GraphPartitioner, PartitionMethod as GraphPartitionMethod, PartitionStats, PartitionedGraph,
};
pub use crate::algo::sampling;
pub use crate::algo::sampling::{SampledSubgraph, SamplingConfig};
pub use crate::algo::shortest_path::ShortestPathConfig;
pub use crate::algo::traversal::{bfs, BfsConfig};
pub use crate::connector::{Connector, ConnectorFuture, ExpandResult};
pub use crate::display::{
    AttrStatsSummary, AttributeStats, DisplayColumn, DisplayOptions, DisplayRow, DisplayRowKind,
    DisplaySlice, DisplayView, GlimpseColumn, GlimpseSummary, GraphInfo, GraphSummary,
    SchemaFieldSummary, SchemaSummary, StructureStats,
};
pub use crate::error::{GFError, Result, SchemaValidationError};
pub use crate::frame::mutable_graph_frame::MutableGraphFrame;
pub use crate::frame::{CsrIndex, EdgeFrame, GraphFrame, NodeFrame};
pub use crate::mojo_runtime::{configure_mojo_runtime, mojo_runtime_path};
pub use crate::query::optimizer::{
    EarlyTermination, Optimizer, OptimizerOptions, OptimizerPass, PartitionParallel,
    PatternExpansion, PredicatePushdown, ProjectionPushdown, SubgraphCaching, TraversalPruning,
};
pub use crate::query::{
    AggExpr, BinaryOp, EdgeTypeSpec, ExecutionHint, Expr, LogicalPlan, PartitionStrategy, Pattern,
    PatternNodeConstraint, PatternStep, PatternStepConstraint, PlanDomain, ScalarValue, StringOp,
    UnaryOp,
};
pub use crate::schema::{EdgeSchema, FieldDef, GFType, GFValue, NodeSchema, Schema};
pub use crate::types::{
    Direction, EdgeId, NodeId, COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE,
    COL_NODE_ID, COL_NODE_LABEL, EDGE_RESERVED_COLUMNS, NODE_RESERVED_COLUMNS,
};

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
