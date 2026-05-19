use std::sync::Arc;

use arrow_array::{
    builder::{
        BooleanBuilder, Date32Builder, Float64Builder, Int64Builder, ListBuilder, StringBuilder,
        TimestampMicrosecondBuilder,
    },
    new_null_array, ArrayRef, Int8Array, RecordBatch,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema, TimeUnit};

use lynxes_core::{
    EdgeFrame, FieldDef, GFError, GFType, GFValue, GraphFrame, NodeFrame, Result,
    COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
};

use crate::io::gf_parser::{ParsedGfDocument, ParsedNodeDecl};

impl ParsedGfDocument {
    pub fn to_node_frame(&self) -> Result<NodeFrame> {
        self.validate_nodes_against_schema()?;

        let user_columns = infer_node_columns(self)?;
        let mut fields = vec![
            Field::new(COL_NODE_ID, DataType::Utf8, false),
            Field::new(
                COL_NODE_LABEL,
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                false,
            ),
        ];
        fields.extend(
            user_columns
                .iter()
                .map(|column| Field::new(&column.name, column.dtype.clone(), column.nullable)),
        );

        let mut columns: Vec<ArrayRef> = Vec::with_capacity(fields.len());
        columns.push(Arc::new(arrow_array::StringArray::from(
            self.nodes
                .iter()
                .map(|node| node.id.as_str())
                .collect::<Vec<_>>(),
        )));
        columns.push(Arc::new(build_labels_array(&self.nodes)));
        for column in &user_columns {
            columns.push(build_user_array(
                &column.name,
                &column.dtype,
                column.nullable,
                self.nodes.iter().map(|node| node.props.get(&column.name)),
            )?);
        }

        let batch = RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), columns)
            .map_err(|error| GFError::IoError(std::io::Error::other(error)))?;
        NodeFrame::from_record_batch(batch)
    }

    pub fn to_edge_frame(&self) -> Result<EdgeFrame> {
        self.validate_edges_against_schema()?;

        let user_columns = infer_edge_columns(self)?;
        let mut fields = vec![
            Field::new(COL_EDGE_SRC, DataType::Utf8, false),
            Field::new(COL_EDGE_DST, DataType::Utf8, false),
            Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
            Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
        ];
        fields.extend(
            user_columns
                .iter()
                .map(|column| Field::new(&column.name, column.dtype.clone(), column.nullable)),
        );

        let mut columns: Vec<ArrayRef> = Vec::with_capacity(fields.len());
        columns.push(Arc::new(arrow_array::StringArray::from(
            self.edges
                .iter()
                .map(|edge| edge.src_id.as_str())
                .collect::<Vec<_>>(),
        )));
        columns.push(Arc::new(arrow_array::StringArray::from(
            self.edges
                .iter()
                .map(|edge| edge.dst_id.as_str())
                .collect::<Vec<_>>(),
        )));
        columns.push(Arc::new(arrow_array::StringArray::from(
            self.edges
                .iter()
                .map(|edge| edge.edge_type.as_str())
                .collect::<Vec<_>>(),
        )));
        columns.push(Arc::new(Int8Array::from(
            self.edges
                .iter()
                .map(|edge| edge.direction.as_i8())
                .collect::<Vec<_>>(),
        )));
        for column in &user_columns {
            columns.push(build_user_array(
                &column.name,
                &column.dtype,
                column.nullable,
                self.edges.iter().map(|edge| edge.props.get(&column.name)),
            )?);
        }

        let batch = RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), columns)
            .map_err(|error| GFError::IoError(std::io::Error::other(error)))?;
        EdgeFrame::from_record_batch(batch)
    }

    pub fn to_graph_frame(&self) -> Result<GraphFrame> {
        let schema = (!self.schema.nodes.is_empty() || !self.schema.edges.is_empty())
            .then(|| self.schema.clone());
        GraphFrame::new_with_schema(self.to_node_frame()?, self.to_edge_frame()?, schema, true)
    }

    fn validate_nodes_against_schema(&self) -> Result<()> {
        if self.schema.nodes.is_empty() {
            return Ok(());
        }

        for node in &self.nodes {
            for label in &node.labels {
                // Labels not declared in the schema are unconstrained — skip validation.
                if self.schema.node_schema(label).is_none() {
                    continue;
                }
                let fields = self.schema.resolved_fields(label)?;
                validate_props_against_field_defs(
                    &node.props,
                    &fields,
                    &format!("node {}:{label}", node.id),
                )?;
            }
        }

        Ok(())
    }

    fn validate_edges_against_schema(&self) -> Result<()> {
        if self.schema.edges.is_empty() {
            return Ok(());
        }

        for edge in &self.edges {
            // Edge types not declared in the schema are unconstrained — skip validation.
            let Some(schema) = self.schema.edge_schema(&edge.edge_type) else {
                continue;
            };
            validate_props_against_field_defs(
                &edge.props,
                &schema.fields,
                &format!(
                    "edge {} -[{}]-> {}",
                    edge.src_id, edge.edge_type, edge.dst_id
                ),
            )?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
struct ColumnPlan {
    name: String,
    dtype: DataType,
    nullable: bool,
}

fn infer_node_columns(document: &ParsedGfDocument) -> Result<Vec<ColumnPlan>> {
    infer_columns(
        document
            .nodes
            .iter()
            .flat_map(|node| node.props.keys())
            .cloned()
            .collect(),
        |column| {
            infer_column_plan(
                column,
                document.nodes.iter().map(|node| node.props.get(column)),
                declared_node_field_type(document, column),
            )
        },
    )
}

fn infer_edge_columns(document: &ParsedGfDocument) -> Result<Vec<ColumnPlan>> {
    infer_columns(
        document
            .edges
            .iter()
            .flat_map(|edge| edge.props.keys())
            .cloned()
            .collect(),
        |column| {
            infer_column_plan(
                column,
                document.edges.iter().map(|edge| edge.props.get(column)),
                declared_edge_field_type(document, column),
            )
        },
    )
}

fn infer_columns<F>(ordered_names: Vec<String>, infer: F) -> Result<Vec<ColumnPlan>>
where
    F: Fn(&str) -> Result<ColumnPlan>,
{
    let mut seen = std::collections::HashSet::new();
    let mut plans = Vec::new();
    for column in ordered_names {
        if seen.insert(column.clone()) {
            plans.push(infer(&column)?);
        }
    }
    Ok(plans)
}

fn infer_column_plan<'a, I>(
    column: &str,
    values: I,
    declared_type: Option<GFType>,
) -> Result<ColumnPlan>
where
    I: Clone + IntoIterator<Item = Option<&'a GFValue>>,
{
    let mut inferred = None;
    let mut nullable = false;

    for value in values.clone() {
        match value {
            None | Some(GFValue::Null) => {
                nullable = true;
            }
            Some(value) => {
                let value_type = infer_value_type(value, column)?;
                inferred = Some(match inferred {
                    None => value_type,
                    Some(existing) => merge_data_types(&existing, &value_type, column)?,
                });
            }
        }
    }

    let dtype = match inferred {
        Some(dtype) => dtype,
        None => {
            let declared_type = declared_type.ok_or_else(|| GFError::CannotInferType {
                column: column.to_owned(),
            })?;
            declared_type.to_arrow_dtype()?
        }
    };

    Ok(ColumnPlan {
        name: column.to_owned(),
        dtype,
        nullable,
    })
}

fn declared_node_field_type(document: &ParsedGfDocument, column: &str) -> Option<GFType> {
    let mut found = None;
    for label in document.schema.nodes.keys() {
        let Ok(fields) = document.schema.resolved_fields(label) else {
            return None;
        };
        if let Some(field) = fields.iter().find(|field| field.name == column) {
            match found {
                None => found = Some(field.dtype.clone()),
                Some(ref existing) if existing == &field.dtype => {}
                Some(_) => return None,
            }
        }
    }
    found
}

fn declared_edge_field_type(document: &ParsedGfDocument, column: &str) -> Option<GFType> {
    let mut found = None;
    for schema in document.schema.edges.values() {
        if let Some(field) = schema.fields.iter().find(|field| field.name == column) {
            match found {
                None => found = Some(field.dtype.clone()),
                Some(ref existing) if existing == &field.dtype => {}
                Some(_) => return None,
            }
        }
    }
    found
}

fn validate_props_against_field_defs(
    props: &std::collections::BTreeMap<String, GFValue>,
    fields: &[FieldDef],
    context: &str,
) -> Result<()> {
    for (name, value) in props {
        if let Some(field) = fields.iter().find(|field| field.name == *name) {
            if !gf_value_matches_type(value, &field.dtype) {
                return Err(GFError::TypeMismatch {
                    message: format!(
                        "{context}: property {name} does not match declared type {:?}",
                        field.dtype
                    ),
                });
            }
        }
    }
    Ok(())
}

fn gf_value_matches_type(value: &GFValue, dtype: &GFType) -> bool {
    match (value, dtype) {
        (GFValue::Null, GFType::Optional(_)) => true,
        (GFValue::Null, _) => false,
        (GFValue::String(_), GFType::String | GFType::Any) => true,
        (GFValue::Int(_), GFType::Int | GFType::Any) => true,
        (GFValue::Float(_), GFType::Float | GFType::Any) => true,
        (GFValue::Bool(_), GFType::Bool | GFType::Any) => true,
        (GFValue::Date(_), GFType::Date | GFType::Any) => true,
        (GFValue::DateTime(_), GFType::DateTime | GFType::Any) => true,
        (GFValue::Object(_), GFType::Any) => true,
        (GFValue::List(values), GFType::List(inner)) => values
            .iter()
            .all(|value| matches!(value, GFValue::Null) || gf_value_matches_type(value, inner)),
        (_, GFType::Optional(inner)) => gf_value_matches_type(value, inner),
        _ => false,
    }
}

fn build_labels_array(nodes: &[ParsedNodeDecl]) -> arrow_array::ListArray {
    let mut builder = ListBuilder::new(StringBuilder::new());
    for node in nodes {
        for label in &node.labels {
            builder.values().append_value(label);
        }
        builder.append(true);
    }
    builder.finish()
}

fn build_user_array<'a, I>(
    column: &str,
    dtype: &DataType,
    _nullable: bool,
    values: I,
) -> Result<ArrayRef>
where
    I: Clone + IntoIterator<Item = Option<&'a GFValue>>,
{
    if dtype == &DataType::Null {
        let len = values.into_iter().count();
        return Ok(new_null_array(dtype, len));
    }

    match dtype {
        DataType::Utf8 => {
            let mut builder = StringBuilder::new();
            for value in values {
                match value {
                    None | Some(GFValue::Null) => builder.append_null(),
                    Some(GFValue::String(value)) => builder.append_value(value),
                    other => return type_build_error(column, "Utf8", other),
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Int64 => {
            let mut builder = Int64Builder::new();
            for value in values {
                match value {
                    None | Some(GFValue::Null) => builder.append_null(),
                    Some(GFValue::Int(value)) => builder.append_value(*value),
                    other => return type_build_error(column, "Int64", other),
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Float64 => {
            let mut builder = Float64Builder::new();
            for value in values {
                match value {
                    None | Some(GFValue::Null) => builder.append_null(),
                    Some(GFValue::Int(value)) => builder.append_value(*value as f64),
                    Some(GFValue::Float(value)) => builder.append_value(*value),
                    other => return type_build_error(column, "Float64", other),
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Boolean => {
            let mut builder = BooleanBuilder::new();
            for value in values {
                match value {
                    None | Some(GFValue::Null) => builder.append_null(),
                    Some(GFValue::Bool(value)) => builder.append_value(*value),
                    other => return type_build_error(column, "Boolean", other),
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Date32 => {
            let mut builder = Date32Builder::new();
            for value in values {
                match value {
                    None | Some(GFValue::Null) => builder.append_null(),
                    Some(GFValue::Date(value)) => builder.append_value(parse_date32(value)?),
                    other => return type_build_error(column, "Date32", other),
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Timestamp(TimeUnit::Microsecond, None) => {
            let mut builder = TimestampMicrosecondBuilder::new();
            for value in values {
                match value {
                    None | Some(GFValue::Null) => builder.append_null(),
                    Some(GFValue::DateTime(value)) => {
                        builder.append_value(parse_datetime_micros(value)?)
                    }
                    other => return type_build_error(column, "Timestamp(Microsecond)", other),
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::List(field) if field.data_type() == &DataType::Utf8 => {
            let mut builder = ListBuilder::new(StringBuilder::new());
            for value in values {
                match value {
                    None | Some(GFValue::Null) => builder.append(false),
                    Some(GFValue::List(items)) => {
                        for item in items {
                            match item {
                                GFValue::Null => builder.values().append_null(),
                                GFValue::String(value) => builder.values().append_value(value),
                                other => {
                                    return type_build_error(column, "List<Utf8>", Some(other));
                                }
                            }
                        }
                        builder.append(true);
                    }
                    other => return type_build_error(column, "List<Utf8>", other),
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::List(field) if field.data_type() == &DataType::Int64 => {
            let mut builder = ListBuilder::new(Int64Builder::new());
            for value in values {
                match value {
                    None | Some(GFValue::Null) => builder.append(false),
                    Some(GFValue::List(items)) => {
                        for item in items {
                            match item {
                                GFValue::Null => builder.values().append_null(),
                                GFValue::Int(value) => builder.values().append_value(*value),
                                other => {
                                    return type_build_error(column, "List<Int64>", Some(other));
                                }
                            }
                        }
                        builder.append(true);
                    }
                    other => return type_build_error(column, "List<Int64>", other),
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::List(field) if field.data_type() == &DataType::Float64 => {
            let mut builder = ListBuilder::new(Float64Builder::new());
            for value in values {
                match value {
                    None | Some(GFValue::Null) => builder.append(false),
                    Some(GFValue::List(items)) => {
                        for item in items {
                            match item {
                                GFValue::Null => builder.values().append_null(),
                                GFValue::Int(value) => builder.values().append_value(*value as f64),
                                GFValue::Float(value) => builder.values().append_value(*value),
                                other => {
                                    return type_build_error(column, "List<Float64>", Some(other));
                                }
                            }
                        }
                        builder.append(true);
                    }
                    other => return type_build_error(column, "List<Float64>", other),
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::List(field) if field.data_type() == &DataType::Boolean => {
            let mut builder = ListBuilder::new(BooleanBuilder::new());
            for value in values {
                match value {
                    None | Some(GFValue::Null) => builder.append(false),
                    Some(GFValue::List(items)) => {
                        for item in items {
                            match item {
                                GFValue::Null => builder.values().append_null(),
                                GFValue::Bool(value) => builder.values().append_value(*value),
                                other => {
                                    return type_build_error(column, "List<Boolean>", Some(other));
                                }
                            }
                        }
                        builder.append(true);
                    }
                    other => return type_build_error(column, "List<Boolean>", other),
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        other => Err(GFError::UnsupportedOperation {
            message: format!(
                "SER-004 frame builder does not support column {column} with type {other:?} yet"
            ),
        }),
    }
}

fn type_build_error(column: &str, expected: &str, actual: Option<&GFValue>) -> Result<ArrayRef> {
    Err(GFError::TypeMismatch {
        message: format!(
            "column {column} expected {expected}, got {}",
            actual
                .map(|value| format!("{value:?}"))
                .unwrap_or_else(|| "missing".to_owned())
        ),
    })
}

fn infer_value_type(value: &GFValue, column: &str) -> Result<DataType> {
    match value {
        GFValue::Null => Ok(DataType::Null),
        GFValue::String(_) => Ok(DataType::Utf8),
        GFValue::Int(_) => Ok(DataType::Int64),
        GFValue::Float(_) => Ok(DataType::Float64),
        GFValue::Bool(_) => Ok(DataType::Boolean),
        GFValue::Date(_) => Ok(DataType::Date32),
        GFValue::DateTime(_) => Ok(DataType::Timestamp(TimeUnit::Microsecond, None)),
        GFValue::List(items) => {
            let mut child_type = None;
            for item in items {
                if matches!(item, GFValue::Null) {
                    continue;
                }
                let item_type = infer_value_type(item, column)?;
                child_type = Some(match child_type {
                    None => item_type,
                    Some(existing) => merge_data_types(&existing, &item_type, column)?,
                });
            }
            let child_type = child_type.ok_or_else(|| GFError::CannotInferType {
                column: column.to_owned(),
            })?;
            Ok(DataType::List(Arc::new(Field::new(
                "item", child_type, true,
            ))))
        }
        GFValue::Object(_) => Err(GFError::UnsupportedOperation {
            message: format!("object property columns are not supported in SER-004: {column}"),
        }),
    }
}

fn merge_data_types(left: &DataType, right: &DataType, column: &str) -> Result<DataType> {
    if left == right {
        return Ok(left.clone());
    }

    match (left, right) {
        (DataType::Null, other) | (other, DataType::Null) => Ok(other.clone()),
        (DataType::Int64, DataType::Float64) | (DataType::Float64, DataType::Int64) => {
            Ok(DataType::Float64)
        }
        (DataType::List(left), DataType::List(right)) => Ok(DataType::List(Arc::new(Field::new(
            "item",
            merge_data_types(left.data_type(), right.data_type(), column)?,
            true,
        )))),
        _ => Err(GFError::TypeInferenceFailed {
            column: column.to_owned(),
            message: format!("cannot merge inferred types {left:?} and {right:?}"),
        }),
    }
}

fn parse_date32(value: &str) -> Result<i32> {
    let (year, month, day) = parse_date_parts(value)?;
    Ok(days_from_civil(year, month, day))
}

fn parse_datetime_micros(value: &str) -> Result<i64> {
    if value.len() != 19 {
        return Err(GFError::ParseError {
            message: format!("invalid datetime literal: {value}"),
        });
    }
    let (year, month, day) = parse_date_parts(&value[..10])?;
    if value.as_bytes()[10] != b'T' || value.as_bytes()[13] != b':' || value.as_bytes()[16] != b':'
    {
        return Err(GFError::ParseError {
            message: format!("invalid datetime literal: {value}"),
        });
    }
    let hour = parse_u32(&value[11..13], "hour", value)?;
    let minute = parse_u32(&value[14..16], "minute", value)?;
    let second = parse_u32(&value[17..19], "second", value)?;
    if hour > 23 || minute > 59 || second > 59 {
        return Err(GFError::ParseError {
            message: format!("invalid datetime literal: {value}"),
        });
    }

    let days = days_from_civil(year, month, day) as i64;
    let seconds = hour as i64 * 3600 + minute as i64 * 60 + second as i64;
    Ok(days * 86_400_000_000 + seconds * 1_000_000)
}

fn parse_date_parts(value: &str) -> Result<(i32, u32, u32)> {
    if value.len() != 10 || value.as_bytes()[4] != b'-' || value.as_bytes()[7] != b'-' {
        return Err(GFError::ParseError {
            message: format!("invalid date literal: {value}"),
        });
    }
    let year = value[..4].parse::<i32>().map_err(|_| GFError::ParseError {
        message: format!("invalid date literal: {value}"),
    })?;
    let month = parse_u32(&value[5..7], "month", value)?;
    let day = parse_u32(&value[8..10], "day", value)?;
    if month == 0 || month > 12 {
        return Err(GFError::ParseError {
            message: format!("invalid date literal: {value}"),
        });
    }
    let max_day = days_in_month(year, month);
    if day == 0 || day > max_day {
        return Err(GFError::ParseError {
            message: format!("invalid date literal: {value}"),
        });
    }
    Ok((year, month, day))
}

fn parse_u32(value: &str, _label: &str, original: &str) -> Result<u32> {
    value.parse::<u32>().map_err(|_| GFError::ParseError {
        message: format!("invalid date/datetime literal: {original}"),
    })
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

fn days_from_civil(year: i32, month: u32, day: u32) -> i32 {
    let year = year - i32::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = year - era * 400;
    let month = month as i32;
    let day = day as i32;
    let doy = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, Date32Array, Float64Array, ListArray, TimestampMicrosecondArray};

    #[test]
    fn parsed_document_builds_node_and_edge_frames() {
        let document = crate::parse_gf(
            r#"
            (alice:Person { age: 30, born: 2024-01-01 })
            (bob:Person { age: 31, born: 2024-01-02 })
            alice -[KNOWS]-> bob { weight: 1.5 }
            "#,
        )
        .unwrap();

        let nodes = document.to_node_frame().unwrap();
        let edges = document.to_edge_frame().unwrap();

        assert_eq!(nodes.len(), 2);
        assert_eq!(edges.len(), 1);
        assert!(nodes.column("age").is_some());
        assert!(edges.column("weight").is_some());

        let born = nodes
            .column("born")
            .unwrap()
            .as_any()
            .downcast_ref::<Date32Array>()
            .unwrap();
        assert_eq!(born.value(0), parse_date32("2024-01-01").unwrap());
    }

    #[test]
    fn numeric_inference_promotes_int_and_float_to_float64() {
        let document = crate::parse_gf(
            r#"
            (alice:Person { score: 1 })
            (bob:Person { score: 2.5 })
            "#,
        )
        .unwrap();

        let nodes = document.to_node_frame().unwrap();
        let scores = nodes
            .column("score")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(scores.value(0), 1.0);
        assert_eq!(scores.value(1), 2.5);
    }

    #[test]
    fn bidirectional_edges_remain_two_rows_after_frame_build() {
        let document = crate::parse_gf(
            r#"
            (alice:Person)
            (bob:Person)
            alice <-[FRIEND]-> bob { since: 2024-01-01T09:00:00 }
            "#,
        )
        .unwrap();

        let edges = document.to_edge_frame().unwrap();
        assert_eq!(edges.len(), 2);

        let since = edges
            .column("since")
            .unwrap()
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap();
        assert_eq!(
            since.value(0),
            parse_datetime_micros("2024-01-01T09:00:00").unwrap()
        );
    }

    #[test]
    fn list_string_properties_build_list_utf8_columns() {
        let document = crate::parse_gf(
            r#"
            (alice:Person { tags: ["rust", "arrow"] })
            (bob:Person { tags: ["graph"] })
            "#,
        )
        .unwrap();

        let nodes = document.to_node_frame().unwrap();
        let tags = nodes
            .column("tags")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        assert_eq!(tags.len(), 2);
    }

    #[test]
    fn schema_allows_undeclared_node_labels() {
        // Labels not in the schema are unconstrained — they should load without error.
        let document = crate::parse_gf(
            r#"
            node Person {
                age: Int
            }
            (alice:Person { age: 30 })
            (acme:Organization { founded: 2010 })
            "#,
        )
        .unwrap();

        assert!(document.to_node_frame().is_ok());
    }

    #[test]
    fn schema_guard_rejects_field_type_mismatch_for_declared_labels() {
        // Nodes whose label IS declared must still satisfy the declared field types.
        let document = crate::parse_gf(
            r#"
            node Person {
                age: Int
            }
            (alice:Person { age: "not-an-int" })
            "#,
        )
        .unwrap();

        let err = document.to_node_frame().unwrap_err();
        assert!(matches!(err, GFError::TypeMismatch { .. }));
    }

    #[test]
    fn schema_guard_rejects_declared_type_mismatch() {
        let document = crate::parse_gf(
            r#"
            edge KNOWS {
                since: Date
            }
            (alice:Person)
            (bob:Person)
            alice -[KNOWS]-> bob { since: "today" }
            "#,
        )
        .unwrap();

        let err = document.to_edge_frame().unwrap_err();
        assert!(matches!(err, GFError::TypeMismatch { .. }));
    }

    #[test]
    fn graph_frame_conversion_checks_dangling_edges() {
        let document = crate::parse_gf(
            r#"
            (alice:Person)
            alice -[KNOWS]-> bob
            "#,
        )
        .unwrap();

        let err = document.to_graph_frame().unwrap_err();
        assert!(matches!(err, GFError::DanglingEdge { .. }));
    }

    #[test]
    fn graph_frame_conversion_runs_schema_validator() {
        let document = crate::parse_gf(
            r#"
            node Person {
                age: Int
            }
            (alice:Person)
            "#,
        )
        .unwrap();

        let err = document.to_graph_frame().unwrap_err();
        assert!(matches!(err, GFError::SchemaValidation { .. }));
    }
}
