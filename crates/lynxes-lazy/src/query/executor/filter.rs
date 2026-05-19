use std::{cmp::Ordering, sync::Arc};

use arrow_array::{ArrayRef, BooleanArray, RecordBatch, StringArray, UInt32Array};

use lynxes_core::{EdgeFrame, GFError, GraphFrame, NodeFrame, Result};
use lynxes_plan::{Expr, LogicalPlan};

use super::{evaluate_expr, execute, read_array_value, unsupported_plan, ExecutionValue, Value};
pub(crate) fn execute_top_k(
    input: &LogicalPlan,
    source_graph: Arc<GraphFrame>,
    n: usize,
) -> Result<ExecutionValue> {
    match input {
        LogicalPlan::Sort {
            input: inner,
            by,
            descending,
        } => {
            let inner_val = execute(inner, source_graph)?;
            match inner_val {
                ExecutionValue::Nodes(nodes) => {
                    let batch = top_k_batch(nodes.to_record_batch(), by, *descending, n)?;
                    Ok(ExecutionValue::Nodes(NodeFrame::from_record_batch(batch)?))
                }
                ExecutionValue::Edges(edges) => {
                    let batch = top_k_batch(edges.to_record_batch(), by, *descending, n)?;
                    Ok(ExecutionValue::Edges(EdgeFrame::from_record_batch(batch)?))
                }
                ExecutionValue::Graph(_) | ExecutionValue::PatternRows(_) => Err(unsupported_plan(
                    "TopK Sort requires a node or edge domain, not a graph or pattern-row domain",
                )),
            }
        }
        // Hint not directly above a Sort ??fall through.
        _ => execute(input, source_graph),
    }
}

/// Extracts a `NodeFrame` frontier from an `ExecutionValue`.
pub(crate) fn extract_node_frontier(val: ExecutionValue, context: &str) -> Result<NodeFrame> {
    match val {
        ExecutionValue::Graph(graph) => Ok(graph.nodes().clone()),
        ExecutionValue::Nodes(nodes) => Ok(nodes),
        ExecutionValue::Edges(_) | ExecutionValue::PatternRows(_) => Err(unsupported_plan(
            &format!("{context} cannot consume an edge or pattern-row domain"),
        )),
    }
}

/// Partial top-K sort of `batch` by column `by`.
///
/// Returns a new `RecordBatch` containing the k rows with the largest (when
/// `descending`) or smallest (when `!descending`) values in the sort column,
/// themselves sorted.
///
/// Complexity: O(n + k log k) average via `select_nth_unstable_by`, versus
/// O(n log n) for a full sort.  Falls back to full sort when `k >= n`.
pub(crate) fn top_k_batch(
    batch: &RecordBatch,
    by: &str,
    descending: bool,
    k: usize,
) -> Result<RecordBatch> {
    let n = batch.num_rows();
    if k >= n {
        // Nothing to save ??just do a regular sort.
        return reorder_batch(batch, by, descending);
    }

    let sort_column = batch
        .column_by_name(by)
        .ok_or_else(|| GFError::ColumnNotFound {
            column: by.to_owned(),
        })?;

    // Collect (row_index, sort_value) pairs.
    let mut indexed: Vec<(usize, Value)> = (0..n)
        .map(|row| {
            let val = read_array_value(sort_column.as_ref(), row, by)?;
            Ok((row, val))
        })
        .collect::<Result<_>>()?;

    // `select_nth_unstable_by` rearranges `indexed` so that elements at
    // positions 0..=k-1 are the k "best" entries (in some order) and the
    // element at position k-1 is in its final sorted position.
    // Average O(n); worst-case O(n吏? but that is rare in practice.
    indexed.select_nth_unstable_by(k - 1, |a, b| {
        // Compare in the direction we want to KEEP (ascending for min-heap,
        // descending for max-heap).  We want the k entries that would appear
        // first in a full sort, so we use the same ordering as the full sort.
        compare_sort_values(&a.1, &b.1, descending).then_with(|| a.0.cmp(&b.0))
    });

    // Sort only the chosen k rows into final order.
    let mut top_k = indexed[..k].to_vec();
    top_k.sort_by(|a, b| compare_sort_values(&a.1, &b.1, descending).then_with(|| a.0.cmp(&b.0)));

    let indices: UInt32Array = top_k.iter().map(|(idx, _)| *idx as u32).collect();
    let reordered: Vec<ArrayRef> = batch
        .columns()
        .iter()
        .map(|col| arrow::compute::take(col.as_ref(), &indices, None))
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| GFError::IoError(std::io::Error::other(e)))?;

    RecordBatch::try_new(batch.schema_ref().clone(), reordered)
        .map_err(|e| GFError::IoError(std::io::Error::other(e)))
}

pub(crate) fn limit_edges(edges: &EdgeFrame, n: usize) -> Result<EdgeFrame> {
    let len = n.min(edges.len());
    let mask: BooleanArray = (0..edges.len()).map(|idx| Some(idx < len)).collect();
    edges.filter(&mask)
}

pub(crate) fn candidate_passes_pre_filter(
    nodes: &NodeFrame,
    id: &str,
    pre_filter: Option<&Expr>,
) -> Result<bool> {
    let Some(pre_filter) = pre_filter else {
        return Ok(true);
    };
    let row = nodes
        .row(id)
        .ok_or_else(|| GFError::NodeNotFound { id: id.to_owned() })?;
    match evaluate_expr(&row, 0, pre_filter)? {
        Value::Bool(value) => Ok(value),
        other => Err(GFError::TypeMismatch {
            message: format!("node pre_filter must evaluate to bool, got {other:?}"),
        }),
    }
}

pub(crate) fn evaluate_node_predicate(nodes: &NodeFrame, expr: &Expr) -> Result<BooleanArray> {
    evaluate_predicate(nodes.to_record_batch(), expr)
}

pub(crate) fn evaluate_edge_predicate(edges: &EdgeFrame, expr: &Expr) -> Result<BooleanArray> {
    evaluate_predicate(edges.to_record_batch(), expr)
}

pub(crate) fn evaluate_predicate(batch: &RecordBatch, expr: &Expr) -> Result<BooleanArray> {
    (0..batch.num_rows())
        .map(|row| match evaluate_expr(batch, row, expr)? {
            Value::Bool(value) => Ok(Some(value)),
            other => Err(GFError::TypeMismatch {
                message: format!("filter predicate must evaluate to bool, got {other:?}"),
            }),
        })
        .collect()
}
pub(crate) fn sort_nodes(nodes: &NodeFrame, by: &str, descending: bool) -> Result<NodeFrame> {
    let batch = reorder_batch(nodes.to_record_batch(), by, descending)?;
    NodeFrame::from_record_batch(batch)
}

pub(crate) fn sort_edges(edges: &EdgeFrame, by: &str, descending: bool) -> Result<EdgeFrame> {
    let batch = reorder_batch(edges.to_record_batch(), by, descending)?;
    EdgeFrame::from_record_batch(batch)
}

pub(crate) fn reorder_batch(
    batch: &RecordBatch,
    by: &str,
    descending: bool,
) -> Result<RecordBatch> {
    let sort_column = batch
        .column_by_name(by)
        .ok_or_else(|| GFError::ColumnNotFound {
            column: by.to_owned(),
        })?;
    let mut row_indices: Vec<usize> = (0..batch.num_rows()).collect();
    row_indices.sort_by(|left, right| {
        let left_value = read_array_value(sort_column.as_ref(), *left, by).unwrap_or(Value::Null);
        let right_value = read_array_value(sort_column.as_ref(), *right, by).unwrap_or(Value::Null);
        compare_sort_values(&left_value, &right_value, descending).then_with(|| left.cmp(right))
    });

    let indices = UInt32Array::from(
        row_indices
            .into_iter()
            .map(|idx| idx as u32)
            .collect::<Vec<_>>(),
    );
    let reordered_columns: Vec<ArrayRef> = batch
        .columns()
        .iter()
        .map(|column| arrow::compute::take(column.as_ref(), &indices, None))
        .collect::<std::result::Result<_, _>>()
        .map_err(|error| GFError::IoError(std::io::Error::other(error)))?;

    RecordBatch::try_new(batch.schema_ref().clone(), reordered_columns)
        .map_err(|error| GFError::IoError(std::io::Error::other(error)))
}

pub(crate) fn compare_sort_values(left: &Value, right: &Value, descending: bool) -> Ordering {
    let ordering = match (left, right) {
        (Value::Null, Value::Null) => Ordering::Equal,
        (Value::Null, _) => Ordering::Greater,
        (_, Value::Null) => Ordering::Less,
        _ => left.partial_cmp(right).unwrap_or(Ordering::Equal),
    };
    if descending {
        ordering.reverse()
    } else {
        ordering
    }
}

pub(crate) fn string_array<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    batch
        .column_by_name(name)
        .ok_or_else(|| GFError::MissingReservedColumn {
            column: name.to_owned(),
        })?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| GFError::ReservedColumnType {
            column: name.to_owned(),
            expected: "Utf8".to_owned(),
            actual: "non-Utf8 array".to_owned(),
        })
}

pub(crate) fn int8_array<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a arrow_array::Int8Array> {
    batch
        .column_by_name(name)
        .ok_or_else(|| GFError::MissingReservedColumn {
            column: name.to_owned(),
        })?
        .as_any()
        .downcast_ref::<arrow_array::Int8Array>()
        .ok_or_else(|| GFError::ReservedColumnType {
            column: name.to_owned(),
            expected: "Int8".to_owned(),
            actual: "non-Int8 array".to_owned(),
        })
}
