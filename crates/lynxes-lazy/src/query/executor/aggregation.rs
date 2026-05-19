use arrow_array::{Int64Array, RecordBatch};

use lynxes_core::{GFError, GraphFrame, NodeFrame, Result, COL_EDGE_DST, COL_EDGE_TYPE};
use lynxes_plan::{AggExpr, Expr, StringOp, UnaryOp};

use super::{
    agg_output_name, agg_output_type, append_node_column, cast_value, convert_scalar,
    empty_aggregate_value, evaluate_binary_values, read_column_value, reduce_agg_values,
    string_array, Value,
};
pub(crate) fn aggregate_neighbors(
    graph: &GraphFrame,
    anchors: &NodeFrame,
    edge_type: &str,
    agg: &AggExpr,
) -> Result<NodeFrame> {
    let output_name = agg_output_name(agg);
    let output_type = agg_output_type(graph, agg)?;
    if matches!(agg_inner(agg), AggExpr::Count) {
        return aggregate_neighbor_count_mojo(
            graph,
            anchors,
            edge_type,
            &output_name,
            &output_type,
        );
    }
    let values: Vec<Value> = anchors
        .id_column()
        .iter()
        .flatten()
        .map(|node_id| aggregate_for_node(graph, node_id, edge_type, agg))
        .collect::<Result<_>>()?;
    append_node_column(anchors, &output_name, &output_type, values)
}

fn aggregate_neighbor_count_mojo(
    graph: &GraphFrame,
    anchors: &NodeFrame,
    edge_type: &str,
    output_name: &str,
    output_type: &arrow_schema::DataType,
) -> Result<NodeFrame> {
    let features = graph.structural_features(Some(edge_type))?;
    let out_degree = features
        .column("out_degree")
        .expect("structural_features returns out_degree")
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("structural_features out_degree is Int64");
    let values: Vec<Value> = anchors
        .id_column()
        .iter()
        .flatten()
        .map(|node_id| {
            features
                .row_index(node_id)
                .map(|row| Value::Int(out_degree.value(row as usize)))
                .unwrap_or(Value::Int(0))
        })
        .collect();
    append_node_column(anchors, output_name, output_type, values)
}

pub(crate) fn agg_inner(agg: &AggExpr) -> &AggExpr {
    match agg {
        AggExpr::Alias { expr, .. } => agg_inner(expr),
        other => other,
    }
}

pub(crate) fn aggregate_for_node(
    graph: &GraphFrame,
    node_id: &str,
    edge_type: &str,
    agg: &AggExpr,
) -> Result<Value> {
    let edge_idx = match graph.edges().node_row_idx(node_id) {
        Some(idx) => idx,
        None => return empty_aggregate_value(agg),
    };
    let edge_rows = graph.edges().out_edge_ids(edge_idx);
    let edge_batch = graph.edges().to_record_batch();
    let type_col = string_array(edge_batch, COL_EDGE_TYPE)?;
    let dst_col = string_array(edge_batch, COL_EDGE_DST)?;

    if matches!(agg_inner(agg), AggExpr::Count) {
        let count = edge_rows
            .iter()
            .filter(|&&edge_row| type_col.value(edge_row as usize) == edge_type)
            .count();
        return Ok(Value::Int(count as i64));
    }

    let mut values = Vec::new();
    for &edge_row in edge_rows {
        let edge_row = edge_row as usize;
        if type_col.value(edge_row) != edge_type {
            continue;
        }
        let neighbor_id = dst_col.value(edge_row);
        let neighbor_row = graph
            .nodes()
            .row(neighbor_id)
            .ok_or_else(|| GFError::NodeNotFound {
                id: neighbor_id.to_owned(),
            })?;
        if let Some(value) = evaluate_neighbor_value(&neighbor_row, edge_batch, edge_row, agg)? {
            values.push(value);
        }
    }

    reduce_agg_values(agg, values)
}

pub(crate) fn evaluate_neighbor_value(
    neighbor_row: &RecordBatch,
    edge_batch: &RecordBatch,
    edge_row: usize,
    agg: &AggExpr,
) -> Result<Option<Value>> {
    match agg {
        AggExpr::Count => Ok(None),
        AggExpr::Sum { expr }
        | AggExpr::Mean { expr }
        | AggExpr::List { expr }
        | AggExpr::First { expr }
        | AggExpr::Last { expr } => {
            let value = evaluate_neighbor_expr(neighbor_row, edge_batch, edge_row, expr)?;
            Ok((value != Value::Null).then_some(value))
        }
        AggExpr::Alias { expr, .. } => {
            evaluate_neighbor_value(neighbor_row, edge_batch, edge_row, expr)
        }
    }
}

pub(crate) fn evaluate_neighbor_expr(
    neighbor_row: &RecordBatch,
    edge_batch: &RecordBatch,
    edge_row: usize,
    expr: &Expr,
) -> Result<Value> {
    match expr {
        Expr::Col { name } => {
            if edge_batch.column_by_name(name).is_some() {
                read_column_value(edge_batch, edge_row, name)
            } else {
                read_column_value(neighbor_row, 0, name)
            }
        }
        Expr::Literal { value } => Ok(convert_scalar(value)),
        Expr::BinaryOp { left, op, right } => {
            let left = evaluate_neighbor_expr(neighbor_row, edge_batch, edge_row, left)?;
            let right = evaluate_neighbor_expr(neighbor_row, edge_batch, edge_row, right)?;
            evaluate_binary_values(left, op, right)
        }
        Expr::UnaryOp { op, expr } => {
            let value = evaluate_neighbor_expr(neighbor_row, edge_batch, edge_row, expr)?;
            match (op, value) {
                (UnaryOp::Neg, Value::Int(value)) => Ok(Value::Int(-value)),
                (UnaryOp::Neg, Value::Float(value)) => Ok(Value::Float(-value)),
                (_, other) => Err(GFError::TypeMismatch {
                    message: format!("unsupported unary expression operand: {other:?}"),
                }),
            }
        }
        Expr::ListContains { expr, item } => {
            let list = evaluate_neighbor_expr(neighbor_row, edge_batch, edge_row, expr)?;
            let item = evaluate_neighbor_expr(neighbor_row, edge_batch, edge_row, item)?;
            match list {
                Value::List(values) => Ok(Value::Bool(values.iter().any(|value| value == &item))),
                other => Err(GFError::TypeMismatch {
                    message: format!("ListContains expects a list operand, got {other:?}"),
                }),
            }
        }
        Expr::Cast { expr, dtype } => cast_value(
            evaluate_neighbor_expr(neighbor_row, edge_batch, edge_row, expr)?,
            dtype,
        ),
        Expr::And { left, right } => {
            let left = evaluate_neighbor_expr(neighbor_row, edge_batch, edge_row, left)?;
            let right = evaluate_neighbor_expr(neighbor_row, edge_batch, edge_row, right)?;
            match (left, right) {
                (Value::Bool(left), Value::Bool(right)) => Ok(Value::Bool(left && right)),
                (left, right) => Err(GFError::TypeMismatch {
                    message: format!(
                        "boolean op expects bool operands, got {left:?} and {right:?}"
                    ),
                }),
            }
        }
        Expr::Or { left, right } => {
            let left = evaluate_neighbor_expr(neighbor_row, edge_batch, edge_row, left)?;
            let right = evaluate_neighbor_expr(neighbor_row, edge_batch, edge_row, right)?;
            match (left, right) {
                (Value::Bool(left), Value::Bool(right)) => Ok(Value::Bool(left || right)),
                (left, right) => Err(GFError::TypeMismatch {
                    message: format!(
                        "boolean op expects bool operands, got {left:?} and {right:?}"
                    ),
                }),
            }
        }
        Expr::Not { expr } => {
            match evaluate_neighbor_expr(neighbor_row, edge_batch, edge_row, expr)? {
                Value::Bool(value) => Ok(Value::Bool(!value)),
                other => Err(GFError::TypeMismatch {
                    message: format!("Not expects bool, got {other:?}"),
                }),
            }
        }
        Expr::PatternCol { alias, field } => Err(GFError::UnsupportedOperation {
            message: format!("PatternCol({alias}.{field}) requires PatternMatch execution"),
        }),
        Expr::StringOp { op, expr, pattern } => {
            let subject = evaluate_neighbor_expr(neighbor_row, edge_batch, edge_row, expr)?;
            let pat = evaluate_neighbor_expr(neighbor_row, edge_batch, edge_row, pattern)?;
            match (subject, pat) {
                (Value::String(s), Value::String(p)) => Ok(Value::Bool(match op {
                    StringOp::Contains => s.contains(p.as_str()),
                    StringOp::StartsWith => s.starts_with(p.as_str()),
                    StringOp::EndsWith => s.ends_with(p.as_str()),
                })),
                (s, p) => Err(GFError::TypeMismatch {
                    message: format!("StringOp expects string operands, got {s:?} and {p:?}"),
                }),
            }
        }
    }
}
