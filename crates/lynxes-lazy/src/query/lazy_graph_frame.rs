use std::sync::Arc;

use arrow_array::RecordBatch;
use lynxes_core::{Direction, EdgeFrame, GFError, GraphFrame, NodeFrame, Result};
use lynxes_plan::{
    AggExpr, EdgeTypeSpec, Expr, LogicalPlan, Optimizer, OptimizerOptions, Pattern, PatternStep,
};

use crate::query::executor::{execute, ExecutionValue};

/// Lazy graph query builder backed by a logical plan tree.
#[derive(Debug, Clone)]
pub struct LazyGraphFrame {
    plan: LogicalPlan,
    source_graph: Option<Arc<GraphFrame>>,
}

impl LazyGraphFrame {
    pub fn from_graph(graph: &GraphFrame) -> Self {
        Self {
            plan: LogicalPlan::Scan {
                source: Arc::new(InMemoryConnector),
                node_columns: None,
                edge_columns: None,
            },
            source_graph: Some(Arc::new(graph.clone())),
        }
    }

    pub fn from_plan(plan: LogicalPlan) -> Self {
        Self {
            plan,
            source_graph: None,
        }
    }

    /// Build a lazy frame that will be read from the given connector on `.collect()`.
    pub fn from_connector(connector: Arc<dyn lynxes_plan::Connector>) -> Self {
        Self {
            plan: LogicalPlan::Scan {
                source: connector,
                node_columns: None,
                edge_columns: None,
            },
            source_graph: None,
        }
    }

    fn with_source(plan: LogicalPlan, source_graph: Option<Arc<GraphFrame>>) -> Self {
        Self { plan, source_graph }
    }

    pub fn plan(&self) -> &LogicalPlan {
        &self.plan
    }

    pub fn into_plan(self) -> LogicalPlan {
        self.plan
    }

    pub fn filter_nodes(self, expr: Expr) -> Self {
        Self::with_source(
            LogicalPlan::FilterNodes {
                input: Box::new(self.plan),
                predicate: expr,
            },
            self.source_graph,
        )
    }

    pub fn filter_edges(self, expr: Expr) -> Self {
        Self::with_source(
            LogicalPlan::FilterEdges {
                input: Box::new(self.plan),
                predicate: expr,
            },
            self.source_graph,
        )
    }

    pub fn select_nodes(self, columns: Vec<String>) -> Self {
        Self::with_source(
            LogicalPlan::ProjectNodes {
                input: Box::new(self.plan),
                columns,
            },
            self.source_graph,
        )
    }

    pub fn select_edges(self, columns: Vec<String>) -> Self {
        Self::with_source(
            LogicalPlan::ProjectEdges {
                input: Box::new(self.plan),
                columns,
            },
            self.source_graph,
        )
    }

    pub fn expand(self, edge_type: EdgeTypeSpec, hops: u32, direction: Direction) -> Self {
        Self::with_source(
            LogicalPlan::Expand {
                input: Box::new(self.plan),
                edge_type,
                hops,
                direction,
                pre_filter: None,
            },
            self.source_graph,
        )
    }

    pub fn traverse(self, pattern: Vec<PatternStep>) -> Self {
        Self::with_source(
            LogicalPlan::Traverse {
                input: Box::new(self.plan),
                pattern,
            },
            self.source_graph,
        )
    }

    pub fn match_pattern(self, pattern: Pattern, where_: Option<Expr>) -> Self {
        Self::with_source(
            LogicalPlan::PatternMatch {
                input: Box::new(self.plan),
                pattern,
                where_,
            },
            self.source_graph,
        )
    }

    pub fn aggregate_neighbors(self, edge_type: impl Into<String>, agg: AggExpr) -> Self {
        Self::with_source(
            LogicalPlan::AggregateNeighbors {
                input: Box::new(self.plan),
                edge_type: edge_type.into(),
                agg,
            },
            self.source_graph,
        )
    }

    pub fn sort(self, by: impl Into<String>, descending: bool) -> Self {
        Self::with_source(
            LogicalPlan::Sort {
                input: Box::new(self.plan),
                by: by.into(),
                descending,
            },
            self.source_graph,
        )
    }

    pub fn limit(self, n: usize) -> Self {
        Self::with_source(
            LogicalPlan::Limit {
                input: Box::new(self.plan),
                n,
            },
            self.source_graph,
        )
    }

    pub fn optimized_plan(&self) -> LogicalPlan {
        Optimizer::default().run(self.plan.clone())
    }

    pub fn explain(&self) -> String {
        format_plan(&self.optimized_plan(), 0)
    }

    pub fn collect(self) -> Result<GraphFrame> {
        self.collect_with_options(OptimizerOptions::default())
    }

    /// Execute the plan using a custom `OptimizerOptions` configuration.
    ///
    /// Useful when you need to selectively enable passes such as
    /// `partition_parallel` or `early_termination` that are disabled by
    /// default.
    pub fn collect_with_options(self, options: OptimizerOptions) -> Result<GraphFrame> {
        let source_graph = self.require_source_graph("collect")?;
        let plan = Optimizer::new(options).run(self.plan);
        match execute(&plan, source_graph)? {
            ExecutionValue::Graph(graph) => Ok(graph),
            ExecutionValue::Nodes(_) | ExecutionValue::Edges(_) | ExecutionValue::PatternRows(_) => {
                Err(GFError::DomainMismatch {
                    message: "collect() requires a graph-domain plan; use collect_nodes() or collect_edges() for tabular domains".to_owned(),
                })
            }
        }
    }

    pub fn collect_nodes(self) -> Result<NodeFrame> {
        self.collect_nodes_with_options(OptimizerOptions::default())
    }

    pub fn collect_nodes_with_options(self, options: OptimizerOptions) -> Result<NodeFrame> {
        let source_graph = self.require_source_graph("collect_nodes")?;
        let plan = Optimizer::new(options).run(self.plan);
        match execute(&plan, source_graph)? {
            ExecutionValue::Graph(graph) => Ok(graph.nodes().clone()),
            ExecutionValue::Nodes(nodes) => Ok(nodes),
            ExecutionValue::Edges(_) | ExecutionValue::PatternRows(_) => {
                Err(GFError::DomainMismatch {
                    message:
                        "collect_nodes() cannot materialize an edge-domain or pattern-row plan"
                            .to_owned(),
                })
            }
        }
    }

    pub fn collect_edges(self) -> Result<EdgeFrame> {
        self.collect_edges_with_options(OptimizerOptions::default())
    }

    pub fn collect_edges_with_options(self, options: OptimizerOptions) -> Result<EdgeFrame> {
        let source_graph = self.require_source_graph("collect_edges")?;
        let plan = Optimizer::new(options).run(self.plan);
        match execute(&plan, source_graph)? {
            ExecutionValue::Graph(graph) => Ok(graph.edges().clone()),
            ExecutionValue::Edges(edges) => Ok(edges),
            ExecutionValue::Nodes(_) | ExecutionValue::PatternRows(_) => {
                Err(GFError::DomainMismatch {
                    message: "collect_edges() cannot materialize a node-domain or pattern-row plan"
                        .to_owned(),
                })
            }
        }
    }

    #[doc(hidden)]
    pub fn collect_pattern_rows(self) -> Result<RecordBatch> {
        self.collect_pattern_rows_with_options(OptimizerOptions::default())
    }

    #[doc(hidden)]
    pub fn collect_pattern_rows_with_options(
        self,
        options: OptimizerOptions,
    ) -> Result<RecordBatch> {
        let source_graph = self.require_source_graph("collect_pattern_rows")?;
        let plan = Optimizer::new(options).run(self.plan);
        match execute(&plan, source_graph)? {
            ExecutionValue::PatternRows(batch) => Ok(batch),
            ExecutionValue::Graph(_) | ExecutionValue::Nodes(_) | ExecutionValue::Edges(_) => {
                Err(GFError::DomainMismatch {
                    message: "collect_pattern_rows() requires a pattern-row domain plan".to_owned(),
                })
            }
        }
    }

    fn require_source_graph(&self, method: &str) -> Result<Arc<GraphFrame>> {
        self.source_graph
            .clone()
            .ok_or_else(|| executor_not_ready_error(method))
    }
}

#[derive(Debug, Default)]
struct InMemoryConnector;

impl lynxes_plan::Connector for InMemoryConnector {}

fn executor_not_ready_error(method: &str) -> GFError {
    GFError::UnsupportedOperation {
        message: format!(
            "{method} requires physical planning/execution, which is not implemented yet"
        ),
    }
}

fn format_plan(plan: &LogicalPlan, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    let mut lines = vec![format!("{pad}{}", plan_head(plan))];

    if let Some(input) = plan.input() {
        lines.push(format_plan(input, indent + 1));
    }

    lines.join("\n")
}

fn plan_head(plan: &LogicalPlan) -> String {
    match plan {
        LogicalPlan::Scan {
            node_columns,
            edge_columns,
            ..
        } => format!(
            "Scan({}, node_columns={}, edge_columns={})",
            "GraphFrame",
            format_optional_columns(node_columns.as_ref()),
            format_optional_columns(edge_columns.as_ref())
        ),
        LogicalPlan::Cache { key, .. } => format!("Cache(key={key})"),
        LogicalPlan::Hint { hint, .. } => format!("Hint({hint:?})"),
        LogicalPlan::FilterNodes { predicate, .. } => format!("FilterNodes({predicate:?})"),
        LogicalPlan::FilterEdges { predicate, .. } => format!("FilterEdges({predicate:?})"),
        LogicalPlan::ProjectNodes { columns, .. } => format!("ProjectNodes({columns:?})"),
        LogicalPlan::ProjectEdges { columns, .. } => format!("ProjectEdges({columns:?})"),
        LogicalPlan::Expand {
            edge_type,
            hops,
            direction,
            pre_filter,
            ..
        } => format!(
            "Expand(edge_type={edge_type:?}, hops={hops}, direction={direction:?}, pre_filter={pre_filter:?})"
        ),
        LogicalPlan::Traverse { pattern, .. } => format!("Traverse({pattern:?})"),
        LogicalPlan::PatternMatch {
            pattern, where_, ..
        } => format!("PatternMatch(pattern={pattern:?}, where={where_:?})"),
        LogicalPlan::AggregateNeighbors { edge_type, agg, .. } => {
            format!("AggregateNeighbors(edge_type={edge_type}, agg={agg:?})")
        }
        LogicalPlan::Sort { by, descending, .. } => {
            format!("Sort(by={by}, descending={descending})")
        }
        LogicalPlan::Limit { n, .. } => format!("Limit({n})"),
    }
}

fn format_optional_columns(columns: Option<&Vec<String>>) -> String {
    match columns {
        Some(columns) => format!("{columns:?}"),
        None => "all".to_owned(),
    }
}
