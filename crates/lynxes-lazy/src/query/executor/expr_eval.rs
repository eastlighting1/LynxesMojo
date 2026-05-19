use std::{cmp::Ordering, sync::Arc};

use arrow_array::{
    builder::{
        BooleanBuilder, Float64Builder, Int64Builder, Int8Builder, ListBuilder, StringBuilder,
    },
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, Int8Array, ListArray, RecordBatch,
    StringArray,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};

use lynxes_core::{GFError, GraphFrame, NodeFrame, Result};
use lynxes_plan::{AggExpr, BinaryOp, Expr, ScalarValue, StringOp, UnaryOp};
pub(crate) fn evaluate_binary_values(left: Value, op: &BinaryOp, right: Value) -> Result<Value> {
    match op {
        BinaryOp::Eq => Ok(Value::Bool(left == right)),
        BinaryOp::NotEq => Ok(Value::Bool(left != right)),
        BinaryOp::Gt => compare_values(left, right, Ordering::Greater),
        BinaryOp::GtEq => compare_values_inclusive(left, right, Ordering::Greater),
        BinaryOp::Lt => compare_values(left, right, Ordering::Less),
        BinaryOp::LtEq => compare_values_inclusive(left, right, Ordering::Less),
        BinaryOp::Add => arithmetic_values(left, right, |l, r| l + r, |l, r| l + r),
        BinaryOp::Sub => arithmetic_values(left, right, |l, r| l - r, |l, r| l - r),
        BinaryOp::Mul => arithmetic_values(left, right, |l, r| l * r, |l, r| l * r),
        BinaryOp::Div => arithmetic_values(left, right, |l, r| l / r, |l, r| l / r),
    }
}

pub(crate) fn reduce_agg_values(agg: &AggExpr, values: Vec<Value>) -> Result<Value> {
    match agg {
        AggExpr::Count => Ok(Value::Int(values.len() as i64)),
        AggExpr::Sum { .. } => {
            if values.iter().any(|value| matches!(value, Value::Float(_))) {
                Ok(Value::Float(
                    values
                        .into_iter()
                        .map(|value| match value {
                            Value::Int(value) => Ok(value as f64),
                            Value::Float(value) => Ok(value),
                            other => Err(GFError::TypeMismatch {
                                message: format!("sum expects numeric values, got {other:?}"),
                            }),
                        })
                        .collect::<Result<Vec<_>>>()?
                        .into_iter()
                        .sum(),
                ))
            } else {
                Ok(Value::Int(
                    values
                        .into_iter()
                        .map(|value| match value {
                            Value::Int(value) => Ok(value),
                            other => Err(GFError::TypeMismatch {
                                message: format!("sum expects int values, got {other:?}"),
                            }),
                        })
                        .collect::<Result<Vec<_>>>()?
                        .into_iter()
                        .sum(),
                ))
            }
        }
        AggExpr::Mean { .. } => {
            if values.is_empty() {
                return Ok(Value::Null);
            }
            let count = values.len();
            let total: f64 = values
                .into_iter()
                .map(|value| match value {
                    Value::Int(value) => Ok(value as f64),
                    Value::Float(value) => Ok(value),
                    other => Err(GFError::TypeMismatch {
                        message: format!("mean expects numeric values, got {other:?}"),
                    }),
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .sum();
            Ok(Value::Float(total / count as f64))
        }
        AggExpr::List { .. } => Ok(Value::List(values)),
        AggExpr::First { .. } => Ok(values.into_iter().next().unwrap_or(Value::Null)),
        AggExpr::Last { .. } => Ok(values.into_iter().last().unwrap_or(Value::Null)),
        AggExpr::Alias { expr, .. } => reduce_agg_values(expr, values),
    }
}

pub(crate) fn empty_aggregate_value(agg: &AggExpr) -> Result<Value> {
    match agg {
        AggExpr::Count => Ok(Value::Int(0)),
        AggExpr::Sum { .. } => Ok(Value::Int(0)),
        AggExpr::Mean { .. } | AggExpr::First { .. } | AggExpr::Last { .. } => Ok(Value::Null),
        AggExpr::List { .. } => Ok(Value::List(Vec::new())),
        AggExpr::Alias { expr, .. } => empty_aggregate_value(expr),
    }
}

pub(crate) fn agg_output_name(agg: &AggExpr) -> String {
    match agg {
        AggExpr::Count => "count".to_owned(),
        AggExpr::Sum { .. } => "sum".to_owned(),
        AggExpr::Mean { .. } => "mean".to_owned(),
        AggExpr::List { .. } => "list".to_owned(),
        AggExpr::First { .. } => "first".to_owned(),
        AggExpr::Last { .. } => "last".to_owned(),
        AggExpr::Alias { name, .. } => name.clone(),
    }
}

pub(crate) fn agg_output_type(graph: &GraphFrame, agg: &AggExpr) -> Result<DataType> {
    match agg {
        AggExpr::Count => Ok(DataType::Int64),
        AggExpr::Sum { expr } => infer_expr_type(graph, expr),
        AggExpr::Mean { .. } => Ok(DataType::Float64),
        AggExpr::List { expr } => Ok(DataType::List(Arc::new(Field::new(
            "item",
            infer_expr_type(graph, expr)?,
            true,
        )))),
        AggExpr::First { expr } | AggExpr::Last { expr } => infer_expr_type(graph, expr),
        AggExpr::Alias { expr, .. } => agg_output_type(graph, expr),
    }
}

pub(crate) fn infer_expr_type(graph: &GraphFrame, expr: &Expr) -> Result<DataType> {
    match expr {
        Expr::Col { name } => {
            if let Ok(field) = graph.edges().schema().field_with_name(name) {
                Ok(field.data_type().clone())
            } else if let Ok(field) = graph.nodes().schema().field_with_name(name) {
                Ok(field.data_type().clone())
            } else {
                Err(GFError::ColumnNotFound {
                    column: name.to_owned(),
                })
            }
        }
        Expr::Literal { value } => match value {
            ScalarValue::Null => Ok(DataType::Utf8),
            ScalarValue::String(_) => Ok(DataType::Utf8),
            ScalarValue::Int(_) => Ok(DataType::Int64),
            ScalarValue::Float(_) => Ok(DataType::Float64),
            ScalarValue::Bool(_) => Ok(DataType::Boolean),
            ScalarValue::List(values) => Ok(DataType::List(Arc::new(Field::new(
                "item",
                infer_expr_type(
                    graph,
                    &Expr::Literal {
                        value: values
                            .first()
                            .cloned()
                            .unwrap_or(ScalarValue::String(String::new())),
                    },
                )?,
                true,
            )))),
        },
        Expr::BinaryOp { left, op, right } => match op {
            BinaryOp::Eq
            | BinaryOp::NotEq
            | BinaryOp::Gt
            | BinaryOp::GtEq
            | BinaryOp::Lt
            | BinaryOp::LtEq => Ok(DataType::Boolean),
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
                let left = infer_expr_type(graph, left)?;
                let right = infer_expr_type(graph, right)?;
                if left == DataType::Float64 || right == DataType::Float64 {
                    Ok(DataType::Float64)
                } else {
                    Ok(DataType::Int64)
                }
            }
        },
        Expr::UnaryOp { expr, .. } => infer_expr_type(graph, expr),
        Expr::ListContains { .. } => Ok(DataType::Boolean),
        Expr::Cast { dtype, .. } => Ok(dtype.clone()),
        Expr::And { .. } | Expr::Or { .. } | Expr::Not { .. } => Ok(DataType::Boolean),
        Expr::PatternCol { alias, field } => Err(GFError::UnsupportedOperation {
            message: format!("PatternCol({alias}.{field}) requires PatternMatch execution"),
        }),
        Expr::StringOp { .. } => Ok(DataType::Boolean),
    }
}

pub(crate) fn append_node_column(
    nodes: &NodeFrame,
    column_name: &str,
    data_type: &DataType,
    values: Vec<Value>,
) -> Result<NodeFrame> {
    let new_column = build_value_array(data_type, values)?;
    let mut fields: Vec<Field> = nodes
        .schema()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect();
    fields.push(Field::new(column_name, data_type.clone(), true));
    let mut columns: Vec<ArrayRef> = nodes.to_record_batch().columns().to_vec();
    columns.push(new_column);

    let batch = RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), columns)
        .map_err(|error| GFError::IoError(std::io::Error::other(error)))?;
    NodeFrame::from_record_batch(batch)
}

pub(crate) fn build_value_array(data_type: &DataType, values: Vec<Value>) -> Result<ArrayRef> {
    match data_type {
        DataType::Int8 => {
            let mut builder = Int8Builder::new();
            for value in values {
                match value {
                    Value::Null => builder.append_null(),
                    Value::Int(value) => builder.append_value(value as i8),
                    other => {
                        return Err(GFError::TypeMismatch {
                            message: format!("expected Int8 aggregation result, got {other:?}"),
                        });
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Int64 => {
            let mut builder = Int64Builder::new();
            for value in values {
                match value {
                    Value::Null => builder.append_null(),
                    Value::Int(value) => builder.append_value(value),
                    other => {
                        return Err(GFError::TypeMismatch {
                            message: format!("expected Int64 aggregation result, got {other:?}"),
                        });
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Float64 => {
            let mut builder = Float64Builder::new();
            for value in values {
                match value {
                    Value::Null => builder.append_null(),
                    Value::Int(value) => builder.append_value(value as f64),
                    Value::Float(value) => builder.append_value(value),
                    other => {
                        return Err(GFError::TypeMismatch {
                            message: format!("expected Float64 aggregation result, got {other:?}"),
                        });
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Utf8 => {
            let mut builder = StringBuilder::new();
            for value in values {
                match value {
                    Value::Null => builder.append_null(),
                    Value::String(value) => builder.append_value(value),
                    other => {
                        return Err(GFError::TypeMismatch {
                            message: format!("expected Utf8 aggregation result, got {other:?}"),
                        });
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Boolean => {
            let mut builder = BooleanBuilder::new();
            for value in values {
                match value {
                    Value::Null => builder.append_null(),
                    Value::Bool(value) => builder.append_value(value),
                    other => {
                        return Err(GFError::TypeMismatch {
                            message: format!("expected Boolean aggregation result, got {other:?}"),
                        });
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::List(field) if field.data_type() == &DataType::Utf8 => {
            let mut builder = ListBuilder::new(StringBuilder::new());
            for value in values {
                match value {
                    Value::Null => builder.append(false),
                    Value::List(items) => {
                        for item in items {
                            match item {
                                Value::String(value) => builder.values().append_value(value),
                                Value::Null => builder.values().append_null(),
                                other => {
                                    return Err(GFError::TypeMismatch {
                                        message: format!("expected Utf8 list item, got {other:?}"),
                                    });
                                }
                            }
                        }
                        builder.append(true);
                    }
                    other => {
                        return Err(GFError::TypeMismatch {
                            message: format!(
                                "expected List<Utf8> aggregation result, got {other:?}"
                            ),
                        });
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::List(field) if field.data_type() == &DataType::Int64 => {
            let mut builder = ListBuilder::new(Int64Builder::new());
            for value in values {
                match value {
                    Value::Null => builder.append(false),
                    Value::List(items) => {
                        for item in items {
                            match item {
                                Value::Int(value) => builder.values().append_value(value),
                                Value::Null => builder.values().append_null(),
                                other => {
                                    return Err(GFError::TypeMismatch {
                                        message: format!("expected Int64 list item, got {other:?}"),
                                    });
                                }
                            }
                        }
                        builder.append(true);
                    }
                    other => {
                        return Err(GFError::TypeMismatch {
                            message: format!(
                                "expected List<Int64> aggregation result, got {other:?}"
                            ),
                        });
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::List(field) if field.data_type() == &DataType::Float64 => {
            let mut builder = ListBuilder::new(Float64Builder::new());
            for value in values {
                match value {
                    Value::Null => builder.append(false),
                    Value::List(items) => {
                        for item in items {
                            match item {
                                Value::Int(value) => builder.values().append_value(value as f64),
                                Value::Float(value) => builder.values().append_value(value),
                                Value::Null => builder.values().append_null(),
                                other => {
                                    return Err(GFError::TypeMismatch {
                                        message: format!(
                                            "expected Float64 list item, got {other:?}"
                                        ),
                                    });
                                }
                            }
                        }
                        builder.append(true);
                    }
                    other => {
                        return Err(GFError::TypeMismatch {
                            message: format!(
                                "expected List<Float64> aggregation result, got {other:?}"
                            ),
                        });
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        other => Err(GFError::UnsupportedOperation {
            message: format!("aggregation result type {other:?} is not implemented yet"),
        }),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Value {
    Null,
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    List(Vec<Value>),
}

impl Value {
    pub(crate) fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Self::String(left), Self::String(right)) => Some(left.cmp(right)),
            (Self::Int(left), Self::Int(right)) => Some(left.cmp(right)),
            (Self::Float(left), Self::Float(right)) => left.partial_cmp(right),
            (Self::Bool(left), Self::Bool(right)) => Some(left.cmp(right)),
            (Self::Int(left), Self::Float(right)) => (*left as f64).partial_cmp(right),
            (Self::Float(left), Self::Int(right)) => left.partial_cmp(&(*right as f64)),
            _ => None,
        }
    }
}

pub(crate) fn evaluate_expr(batch: &RecordBatch, row: usize, expr: &Expr) -> Result<Value> {
    match expr {
        Expr::Col { name } => read_column_value(batch, row, name),
        Expr::Literal { value } => Ok(convert_scalar(value)),
        Expr::BinaryOp { left, op, right } => evaluate_binary(batch, row, left, op, right),
        Expr::UnaryOp { op, expr } => {
            let value = evaluate_expr(batch, row, expr)?;
            match (op, value) {
                (UnaryOp::Neg, Value::Int(value)) => Ok(Value::Int(-value)),
                (UnaryOp::Neg, Value::Float(value)) => Ok(Value::Float(-value)),
                (_, other) => Err(GFError::TypeMismatch {
                    message: format!("unsupported unary expression operand: {other:?}"),
                }),
            }
        }
        Expr::ListContains { expr, item } => {
            let list = evaluate_expr(batch, row, expr)?;
            let item = evaluate_expr(batch, row, item)?;
            match list {
                Value::List(values) => Ok(Value::Bool(values.iter().any(|value| value == &item))),
                other => Err(GFError::TypeMismatch {
                    message: format!("ListContains expects a list operand, got {other:?}"),
                }),
            }
        }
        Expr::Cast { expr, dtype } => cast_value(evaluate_expr(batch, row, expr)?, dtype),
        Expr::And { left, right } => boolean_op(batch, row, left, right, |l, r| l && r),
        Expr::Or { left, right } => boolean_op(batch, row, left, right, |l, r| l || r),
        Expr::Not { expr } => match evaluate_expr(batch, row, expr)? {
            Value::Bool(value) => Ok(Value::Bool(!value)),
            other => Err(GFError::TypeMismatch {
                message: format!("Not expects bool, got {other:?}"),
            }),
        },
        Expr::PatternCol { alias, field } => Err(GFError::UnsupportedOperation {
            message: format!("PatternCol({alias}.{field}) requires PatternMatch execution"),
        }),
        Expr::StringOp { op, expr, pattern } => {
            let subject = evaluate_expr(batch, row, expr)?;
            let pat = evaluate_expr(batch, row, pattern)?;
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

pub(crate) fn boolean_op(
    batch: &RecordBatch,
    row: usize,
    left: &Expr,
    right: &Expr,
    f: impl Fn(bool, bool) -> bool,
) -> Result<Value> {
    let left = evaluate_expr(batch, row, left)?;
    let right = evaluate_expr(batch, row, right)?;
    match (left, right) {
        (Value::Bool(left), Value::Bool(right)) => Ok(Value::Bool(f(left, right))),
        (left, right) => Err(GFError::TypeMismatch {
            message: format!("boolean op expects bool operands, got {left:?} and {right:?}"),
        }),
    }
}

pub(crate) fn evaluate_binary(
    batch: &RecordBatch,
    row: usize,
    left: &Expr,
    op: &BinaryOp,
    right: &Expr,
) -> Result<Value> {
    let left = evaluate_expr(batch, row, left)?;
    let right = evaluate_expr(batch, row, right)?;

    match op {
        BinaryOp::Eq => Ok(Value::Bool(left == right)),
        BinaryOp::NotEq => Ok(Value::Bool(left != right)),
        BinaryOp::Gt => compare_values(left, right, Ordering::Greater),
        BinaryOp::GtEq => compare_values_inclusive(left, right, Ordering::Greater),
        BinaryOp::Lt => compare_values(left, right, Ordering::Less),
        BinaryOp::LtEq => compare_values_inclusive(left, right, Ordering::Less),
        BinaryOp::Add => arithmetic_values(left, right, |l, r| l + r, |l, r| l + r),
        BinaryOp::Sub => arithmetic_values(left, right, |l, r| l - r, |l, r| l - r),
        BinaryOp::Mul => arithmetic_values(left, right, |l, r| l * r, |l, r| l * r),
        BinaryOp::Div => arithmetic_values(left, right, |l, r| l / r, |l, r| l / r),
    }
}

pub(crate) fn compare_values(left: Value, right: Value, expected: Ordering) -> Result<Value> {
    if matches!(left, Value::Null) || matches!(right, Value::Null) {
        return Ok(Value::Bool(false));
    }
    let ordering = left
        .partial_cmp(&right)
        .ok_or_else(|| GFError::TypeMismatch {
            message: format!("cannot compare {left:?} and {right:?}"),
        })?;
    Ok(Value::Bool(ordering == expected))
}

pub(crate) fn compare_values_inclusive(
    left: Value,
    right: Value,
    expected: Ordering,
) -> Result<Value> {
    if matches!(left, Value::Null) || matches!(right, Value::Null) {
        return Ok(Value::Bool(false));
    }
    let ordering = left
        .partial_cmp(&right)
        .ok_or_else(|| GFError::TypeMismatch {
            message: format!("cannot compare {left:?} and {right:?}"),
        })?;
    Ok(Value::Bool(
        ordering == expected || ordering == Ordering::Equal,
    ))
}

pub(crate) fn arithmetic_values(
    left: Value,
    right: Value,
    int_op: impl Fn(i64, i64) -> i64,
    float_op: impl Fn(f64, f64) -> f64,
) -> Result<Value> {
    match (left, right) {
        (Value::Int(left), Value::Int(right)) => Ok(Value::Int(int_op(left, right))),
        (Value::Int(left), Value::Float(right)) => Ok(Value::Float(float_op(left as f64, right))),
        (Value::Float(left), Value::Int(right)) => Ok(Value::Float(float_op(left, right as f64))),
        (Value::Float(left), Value::Float(right)) => Ok(Value::Float(float_op(left, right))),
        (left, right) => Err(GFError::TypeMismatch {
            message: format!("arithmetic expects numeric operands, got {left:?} and {right:?}"),
        }),
    }
}

pub(crate) fn cast_value(value: Value, dtype: &arrow_schema::DataType) -> Result<Value> {
    use arrow_schema::DataType;

    match (value, dtype) {
        (Value::Null, _) => Ok(Value::Null),
        (Value::Int(value), DataType::Float64) => Ok(Value::Float(value as f64)),
        (Value::Int(value), DataType::Int64) => Ok(Value::Int(value)),
        (Value::Float(value), DataType::Float64) => Ok(Value::Float(value)),
        (Value::Float(value), DataType::Int64) => Ok(Value::Int(value as i64)),
        (Value::String(value), DataType::Utf8) => Ok(Value::String(value)),
        (Value::Bool(value), DataType::Boolean) => Ok(Value::Bool(value)),
        (value, dtype) => Err(GFError::InvalidCast {
            from: format!("{value:?}"),
            to: format!("{dtype:?}"),
        }),
    }
}

pub(crate) fn convert_scalar(value: &ScalarValue) -> Value {
    match value {
        ScalarValue::Null => Value::Null,
        ScalarValue::String(value) => Value::String(value.clone()),
        ScalarValue::Int(value) => Value::Int(*value),
        ScalarValue::Float(value) => Value::Float(*value),
        ScalarValue::Bool(value) => Value::Bool(*value),
        ScalarValue::List(values) => Value::List(values.iter().map(convert_scalar).collect()),
    }
}

pub(crate) fn read_column_value(batch: &RecordBatch, row: usize, name: &str) -> Result<Value> {
    let column = batch
        .column_by_name(name)
        .ok_or_else(|| GFError::ColumnNotFound {
            column: name.to_owned(),
        })?;
    read_array_value(column.as_ref(), row, name)
}

pub(crate) fn read_array_value(array: &dyn Array, row: usize, name: &str) -> Result<Value> {
    if array.is_null(row) {
        return Ok(Value::Null);
    }

    if let Some(array) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(Value::String(array.value(row).to_owned()));
    }
    if let Some(array) = array.as_any().downcast_ref::<Int8Array>() {
        return Ok(Value::Int(array.value(row) as i64));
    }
    if let Some(array) = array.as_any().downcast_ref::<Int64Array>() {
        return Ok(Value::Int(array.value(row)));
    }
    if let Some(array) = array.as_any().downcast_ref::<Float64Array>() {
        return Ok(Value::Float(array.value(row)));
    }
    if let Some(array) = array.as_any().downcast_ref::<BooleanArray>() {
        return Ok(Value::Bool(array.value(row)));
    }
    if let Some(array) = array.as_any().downcast_ref::<ListArray>() {
        let values = array.value(row);
        let values = values
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| GFError::TypeMismatch {
                message: format!("list column {name} currently supports Utf8 children only"),
            })?;
        return Ok(Value::List(
            values
                .iter()
                .flatten()
                .map(|value| Value::String(value.to_owned()))
                .collect(),
        ));
    }

    Err(GFError::UnsupportedOperation {
        message: format!(
            "executor cannot read column {name} with type {:?}",
            array.data_type()
        ),
    })
}
