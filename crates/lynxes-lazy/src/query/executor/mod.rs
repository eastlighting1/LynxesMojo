use std::sync::Arc;

use arrow_array::RecordBatch;
use hashbrown::HashMap;

use lynxes_core::{EdgeFrame, GFError, GraphFrame, NodeFrame, Result};
use lynxes_plan::{ExecutionHint, LogicalPlan};

mod aggregation;
mod expr_eval;
mod filter;
mod pattern;
mod traversal;

#[allow(unused_imports)]
pub(crate) use aggregation::{
    agg_inner, aggregate_for_node, aggregate_neighbors, evaluate_neighbor_expr,
    evaluate_neighbor_value,
};
#[allow(unused_imports)]
pub(crate) use expr_eval::{
    agg_output_name, agg_output_type, append_node_column, arithmetic_values, boolean_op,
    build_value_array, cast_value, compare_values, compare_values_inclusive, convert_scalar,
    empty_aggregate_value, evaluate_binary, evaluate_binary_values, evaluate_expr, infer_expr_type,
    read_array_value, read_column_value, reduce_agg_values, Value,
};
#[allow(unused_imports)]
pub(crate) use filter::{
    candidate_passes_pre_filter, compare_sort_values, evaluate_edge_predicate,
    evaluate_node_predicate, evaluate_predicate, execute_top_k, extract_node_frontier, int8_array,
    limit_edges, reorder_batch, sort_edges, sort_nodes, string_array, top_k_batch,
};
#[allow(unused_imports)]
pub(crate) use pattern::{
    apply_pattern_where, bind_optional_pattern_step, collect_pattern_aliases,
    evaluate_pattern_expr, execute_pattern_match, execute_pattern_step, execute_pattern_steps,
    matches_edge_type, materialize_pattern_bindings, pattern_candidates,
    pattern_node_matches_constraint, read_pattern_field_value, single_hop_pattern_candidates,
    validate_pattern_support, variable_hop_pattern_candidates,
};
#[allow(unused_imports)]
pub(crate) use traversal::{
    build_edge_node_ids, build_expand_result, execute_limit_aware, execute_partition_parallel,
    expand_frontier_csr, expand_graph, expand_graph_raw, traverse_graph,
};

#[derive(Debug, Clone)]
#[allow(dead_code, clippy::large_enum_variant)]
pub(crate) enum ExecutionValue {
    Graph(GraphFrame),
    Nodes(NodeFrame),
    Edges(EdgeFrame),
    PatternRows(RecordBatch),
}

/// One row of alias bindings produced during `PatternMatch` execution.
///
/// The value space is intentionally fixed to `Option<u32>` so the executor can
/// carry lightweight graph-local identifiers while still representing
/// unmatched optional aliases as `None`. Node aliases use the `EdgeFrame`
/// local compact node index. Edge aliases use edge row ids in the same `u32`
/// slot space.
///
/// Alias collision rule:
/// if an alias is encountered again with the same bound value, the binding row
/// remains valid and execution continues. If the same alias is encountered with
/// a different value, that row is rejected because the pattern would be asking
/// one alias to represent two different graph elements at once.
#[allow(dead_code)]
type PatternBindingRow = HashMap<String, Option<u32>>;

/// The full set of rows emitted by a `PatternMatch` executor pass.
#[allow(dead_code)]
type PatternBindings = Vec<PatternBindingRow>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PatternAliasKind {
    Node,
    Edge,
}

#[allow(dead_code)]
fn bind_pattern_alias(row: &mut PatternBindingRow, alias: &str, value: u32) -> Result<()> {
    bind_pattern_alias_value(row, alias, Some(value))
}

#[allow(dead_code)]
fn bind_pattern_alias_null(row: &mut PatternBindingRow, alias: &str) -> Result<()> {
    bind_pattern_alias_value(row, alias, None)
}

fn bind_pattern_alias_value(
    row: &mut PatternBindingRow,
    alias: &str,
    value: Option<u32>,
) -> Result<()> {
    match row.get(alias).copied() {
        Some(bound) if bound == value => Ok(()),
        Some(bound) => Err(GFError::InvalidConfig {
            message: format!(
                "pattern alias '{alias}' is already bound to {bound:?}, cannot rebind to {value:?}"
            ),
        }),
        None => {
            row.insert(alias.to_owned(), value);
            Ok(())
        }
    }
}

pub(crate) fn execute(plan: &LogicalPlan, source_graph: Arc<GraphFrame>) -> Result<ExecutionValue> {
    match plan {
        LogicalPlan::Scan { .. } => Ok(ExecutionValue::Graph(source_graph.as_ref().clone())),
        LogicalPlan::Cache { input, .. } => execute(input, source_graph),
        LogicalPlan::Hint { hint, input } => execute_hint(hint, input, source_graph),
        LogicalPlan::FilterNodes { input, predicate } => {
            let input = execute(input, source_graph)?;
            let nodes = match input {
                ExecutionValue::Graph(graph) => graph.nodes().clone(),
                ExecutionValue::Nodes(nodes) => nodes,
                ExecutionValue::Edges(_) | ExecutionValue::PatternRows(_) => {
                    return Err(unsupported_plan(
                        "FilterNodes cannot consume an edge or pattern-row domain",
                    ));
                }
            };
            let mask = evaluate_node_predicate(&nodes, predicate)?;
            Ok(ExecutionValue::Nodes(nodes.filter(&mask)?))
        }
        LogicalPlan::FilterEdges { input, predicate } => {
            let input = execute(input, source_graph)?;
            let edges = match input {
                ExecutionValue::Graph(graph) => graph.edges().clone(),
                ExecutionValue::Edges(edges) => edges,
                ExecutionValue::Nodes(_) | ExecutionValue::PatternRows(_) => {
                    return Err(unsupported_plan(
                        "FilterEdges cannot consume a node or pattern-row domain",
                    ));
                }
            };
            let mask = evaluate_edge_predicate(&edges, predicate)?;
            Ok(ExecutionValue::Edges(edges.filter(&mask)?))
        }
        LogicalPlan::ProjectNodes { input, columns } => {
            let input = execute(input, source_graph)?;
            let nodes = match input {
                ExecutionValue::Graph(graph) => graph.nodes().clone(),
                ExecutionValue::Nodes(nodes) => nodes,
                ExecutionValue::Edges(_) | ExecutionValue::PatternRows(_) => {
                    return Err(unsupported_plan(
                        "ProjectNodes cannot consume an edge or pattern-row domain",
                    ));
                }
            };
            let columns: Vec<&str> = columns.iter().map(String::as_str).collect();
            Ok(ExecutionValue::Nodes(nodes.select(&columns)?))
        }
        LogicalPlan::ProjectEdges { input, columns } => {
            let input = execute(input, source_graph)?;
            let edges = match input {
                ExecutionValue::Graph(graph) => graph.edges().clone(),
                ExecutionValue::Edges(edges) => edges,
                ExecutionValue::Nodes(_) | ExecutionValue::PatternRows(_) => {
                    return Err(unsupported_plan(
                        "ProjectEdges cannot consume a node or pattern-row domain",
                    ));
                }
            };
            let columns: Vec<&str> = columns.iter().map(String::as_str).collect();
            Ok(ExecutionValue::Edges(edges.select(&columns)?))
        }
        LogicalPlan::Sort {
            input,
            by,
            descending,
        } => {
            let input = execute(input, source_graph)?;
            match input {
                ExecutionValue::Nodes(nodes) => {
                    Ok(ExecutionValue::Nodes(sort_nodes(&nodes, by, *descending)?))
                }
                ExecutionValue::Edges(edges) => {
                    Ok(ExecutionValue::Edges(sort_edges(&edges, by, *descending)?))
                }
                ExecutionValue::Graph(_) | ExecutionValue::PatternRows(_) => Err(unsupported_plan(
                    "Sort requires a node or edge domain, not a graph or pattern-row domain",
                )),
            }
        }
        LogicalPlan::Limit { input, n } => {
            let input = execute(input, source_graph)?;
            match input {
                ExecutionValue::Nodes(nodes) => {
                    Ok(ExecutionValue::Nodes(nodes.slice(0, (*n).min(nodes.len()))))
                }
                ExecutionValue::Edges(edges) => Ok(ExecutionValue::Edges(limit_edges(&edges, *n)?)),
                ExecutionValue::Graph(graph) => {
                    let node_ids: Vec<&str> = graph
                        .nodes()
                        .id_column()
                        .iter()
                        .take(*n)
                        .flatten()
                        .collect();
                    Ok(ExecutionValue::Graph(graph.subgraph(&node_ids)?))
                }
                ExecutionValue::PatternRows(_) => Err(unsupported_plan(
                    "Limit does not yet support pattern-row domains",
                )),
            }
        }
        LogicalPlan::Expand {
            input,
            edge_type,
            hops,
            direction,
            pre_filter,
        } => {
            let input = execute(input, source_graph.clone())?;
            let frontier = match input {
                ExecutionValue::Graph(graph) => graph.nodes().clone(),
                ExecutionValue::Nodes(nodes) => nodes,
                ExecutionValue::Edges(_) | ExecutionValue::PatternRows(_) => {
                    return Err(unsupported_plan(
                        "Expand cannot consume an edge or pattern-row domain",
                    ));
                }
            };
            Ok(ExecutionValue::Graph(expand_graph(
                source_graph.as_ref(),
                &frontier,
                edge_type,
                *hops as usize,
                *direction,
                pre_filter.as_ref(),
                None,
            )?))
        }
        LogicalPlan::Traverse { input, pattern } => {
            let input = execute(input, source_graph.clone())?;
            let frontier = match input {
                ExecutionValue::Graph(graph) => graph.nodes().clone(),
                ExecutionValue::Nodes(nodes) => nodes,
                ExecutionValue::Edges(_) | ExecutionValue::PatternRows(_) => {
                    return Err(unsupported_plan(
                        "Traverse cannot consume an edge or pattern-row domain",
                    ));
                }
            };
            Ok(ExecutionValue::Graph(traverse_graph(
                source_graph.as_ref(),
                &frontier,
                pattern,
                None,
            )?))
        }
        LogicalPlan::PatternMatch {
            input,
            pattern,
            where_,
        } => {
            let input = execute(input, source_graph.clone())?;
            let anchors = match input {
                ExecutionValue::Graph(graph) => graph.nodes().clone(),
                ExecutionValue::Nodes(nodes) => nodes,
                ExecutionValue::Edges(_) => {
                    return Err(unsupported_plan(
                        "PatternMatch cannot consume an edge domain",
                    ));
                }
                ExecutionValue::PatternRows(_) => {
                    return Err(unsupported_plan(
                        "PatternMatch cannot consume an existing pattern-row domain",
                    ));
                }
            };
            Ok(ExecutionValue::PatternRows(execute_pattern_match(
                source_graph.as_ref(),
                &anchors,
                pattern,
                where_.as_ref(),
            )?))
        }
        LogicalPlan::AggregateNeighbors {
            input,
            edge_type,
            agg,
        } => {
            let input = execute(input, source_graph.clone())?;
            let anchors = match input {
                ExecutionValue::Graph(graph) => graph.nodes().clone(),
                ExecutionValue::Nodes(nodes) => nodes,
                ExecutionValue::Edges(_) | ExecutionValue::PatternRows(_) => {
                    return Err(unsupported_plan(
                        "AggregateNeighbors cannot consume an edge or pattern-row domain",
                    ));
                }
            };
            Ok(ExecutionValue::Nodes(aggregate_neighbors(
                source_graph.as_ref(),
                &anchors,
                edge_type,
                agg,
            )?))
        }
    }
}

fn unsupported_plan(message: &str) -> GFError {
    GFError::UnsupportedOperation {
        message: message.to_owned(),
    }
}

// ???? Hint dispatch ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

fn execute_hint(
    hint: &ExecutionHint,
    input: &LogicalPlan,
    source_graph: Arc<GraphFrame>,
) -> Result<ExecutionValue> {
    match hint {
        ExecutionHint::LimitAware { n } => execute_limit_aware(input, source_graph, *n),
        ExecutionHint::TopK { n } => execute_top_k(input, source_graph, *n),
        ExecutionHint::PartitionParallel { .. } => execute_partition_parallel(input, source_graph),
    }
}

#[cfg(test)]
mod tests;
