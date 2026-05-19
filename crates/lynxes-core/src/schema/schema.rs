use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::{Array, ListArray, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use serde::{Deserialize, Serialize};

use crate::schema::types::FieldDef;
use crate::{
    EdgeFrame, GFError, GraphFrame, NodeFrame, Result, SchemaValidationError, COL_EDGE_DIRECTION,
    COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
};

/// Node schema definition parsed from `.gf`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeSchema {
    pub label: String,
    pub fields: Vec<FieldDef>,
    pub extends: Option<String>,
}

impl NodeSchema {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            fields: Vec::new(),
            extends: None,
        }
    }

    pub fn with_extends(mut self, parent: impl Into<String>) -> Self {
        self.extends = Some(parent.into());
        self
    }

    pub fn with_fields(mut self, fields: Vec<FieldDef>) -> Self {
        self.fields = fields;
        self
    }
}

/// Edge schema definition parsed from `.gf`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EdgeSchema {
    pub type_name: String,
    pub fields: Vec<FieldDef>,
}

impl EdgeSchema {
    pub fn new(type_name: impl Into<String>) -> Self {
        Self {
            type_name: type_name.into(),
            fields: Vec::new(),
        }
    }

    pub fn with_fields(mut self, fields: Vec<FieldDef>) -> Self {
        self.fields = fields;
        self
    }
}

/// Schema bundle carried by a `.gf` document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Schema {
    pub nodes: HashMap<String, NodeSchema>,
    pub edges: HashMap<String, EdgeSchema>,
    pub namespaces: HashMap<String, String>,
}

impl Schema {
    pub fn node_schema(&self, label: &str) -> Option<&NodeSchema> {
        self.nodes.get(label)
    }

    pub fn edge_schema(&self, type_name: &str) -> Option<&EdgeSchema> {
        self.edges.get(type_name)
    }

    pub fn resolved_fields(&self, label: &str) -> Result<Vec<FieldDef>> {
        Ok(self.resolved_node_schema(label)?.fields)
    }

    pub fn resolved_node_schema(&self, label: &str) -> Result<NodeSchema> {
        let mut stack = HashSet::new();
        let mut path = Vec::new();
        let fields = self.resolve_node_fields(label, &mut stack, &mut path)?;
        Ok(NodeSchema {
            label: label.to_owned(),
            extends: None,
            fields,
        })
    }

    pub fn to_arrow_node_schema(&self, label: &str) -> Result<ArrowSchema> {
        let mut fields = node_reserved_fields();
        fields.extend(
            self.resolved_fields(label)?
                .into_iter()
                .map(|field| field.to_arrow_field())
                .collect::<Result<Vec<_>>>()?,
        );
        Ok(ArrowSchema::new(fields))
    }

    pub fn to_arrow_edge_schema(&self, type_name: &str) -> Result<ArrowSchema> {
        let schema = self
            .edge_schema(type_name)
            .ok_or_else(|| GFError::SchemaMismatch {
                message: format!("edge type {type_name} is not declared in schema"),
            })?;
        let mut fields = edge_reserved_fields();
        fields.extend(
            schema
                .fields
                .iter()
                .map(FieldDef::to_arrow_field)
                .collect::<Result<Vec<_>>>()?,
        );
        Ok(ArrowSchema::new(fields))
    }

    pub fn to_arrow_schema(&self, label: &str) -> Result<ArrowSchema> {
        self.to_arrow_node_schema(label)
    }

    pub fn validate_graph(&self, graph: &GraphFrame) -> Vec<SchemaValidationError> {
        let mut errors = Vec::new();
        errors.extend(self.validate_nodes(graph.nodes()));
        errors.extend(self.validate_edges(graph.edges()));
        errors
    }

    fn resolve_node_fields(
        &self,
        label: &str,
        stack: &mut HashSet<String>,
        path: &mut Vec<String>,
    ) -> Result<Vec<FieldDef>> {
        if !stack.insert(label.to_owned()) {
            let cycle_start = path.iter().position(|entry| entry == label).unwrap_or(0);
            let mut cycle = path[cycle_start..].to_vec();
            cycle.push(label.to_owned());
            return Err(GFError::CircularInheritance {
                path: cycle.join(" -> "),
            });
        }
        path.push(label.to_owned());

        let schema = self
            .node_schema(label)
            .ok_or_else(|| GFError::SchemaMismatch {
                message: format!("node label {label} is not declared in schema"),
            })?;

        let mut fields = if let Some(parent) = &schema.extends {
            self.resolve_node_fields(parent, stack, path)?
        } else {
            Vec::new()
        };

        for field in &schema.fields {
            field.dtype.validate()?;
            field.validate_default(field.default.as_ref())?;
            if let Some(existing) = fields
                .iter_mut()
                .find(|existing| existing.name == field.name)
            {
                *existing = merge_inherited_field(existing, field, label)?;
            } else {
                fields.push(field.clone());
            }
        }

        path.pop();
        stack.remove(label);
        Ok(fields)
    }
}

fn merge_inherited_field(parent: &FieldDef, child: &FieldDef, label: &str) -> Result<FieldDef> {
    if parent.dtype != child.dtype {
        return Err(GFError::SchemaMismatch {
            message: format!(
                "node label {label} cannot override field {} from {:?} to {:?}",
                child.name, parent.dtype, child.dtype
            ),
        });
    }

    let mut merged = child.clone();
    merged.unique = parent.unique || child.unique;
    merged.indexed = parent.indexed || child.indexed;
    if merged.default.is_none() {
        merged.default = parent.default.clone();
    }
    merged.validate_default(merged.default.as_ref())?;
    Ok(merged)
}

fn node_reserved_fields() -> Vec<Field> {
    vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        Field::new(
            COL_NODE_LABEL,
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            false,
        ),
    ]
}

fn edge_reserved_fields() -> Vec<Field> {
    vec![
        Field::new(COL_EDGE_SRC, DataType::Utf8, false),
        Field::new(COL_EDGE_DST, DataType::Utf8, false),
        Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
        Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
    ]
}

impl Schema {
    fn validate_nodes(&self, nodes: &NodeFrame) -> Vec<SchemaValidationError> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        let mut errors = Vec::new();
        let id_column = nodes.id_column();
        let label_column = nodes
            .column(COL_NODE_LABEL)
            .expect("_label must exist")
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("_label must be List<Utf8>");

        let mut label_fields = HashMap::new();
        for label in self.nodes.keys() {
            match self.resolved_fields(label) {
                Ok(fields) => {
                    label_fields.insert(label.clone(), fields);
                }
                Err(err) => errors.push(SchemaValidationError::NodeFieldTypeMismatch {
                    label: label.clone(),
                    field: "<schema>".to_owned(),
                    expected: "resolved schema".to_owned(),
                    actual: err.to_string(),
                }),
            }
        }

        let mut unique_trackers: HashMap<(String, String), HashSet<String>> = HashMap::new();

        for row in 0..nodes.len() {
            let node_id = id_column.value(row).to_owned();
            for label in node_labels(label_column, row) {
                let Some(fields) = label_fields.get(&label) else {
                    errors.push(SchemaValidationError::UndefinedNodeLabel {
                        node_id: node_id.clone(),
                        label: label.to_owned(),
                    });
                    continue;
                };

                for field in fields {
                    let column = nodes.column(&field.name);
                    match column {
                        None if field.default.is_none() && !field.nullable() => {
                            errors.push(SchemaValidationError::MissingRequiredNodeField {
                                node_id: node_id.clone(),
                                label: label.to_owned(),
                                field: field.name.clone(),
                            });
                        }
                        None => {}
                        Some(array) => {
                            let expected_dtype =
                                field.dtype.to_arrow_dtype().unwrap_or(DataType::Null);
                            if array.data_type() != &expected_dtype {
                                errors.push(SchemaValidationError::NodeFieldTypeMismatch {
                                    label: label.to_owned(),
                                    field: field.name.clone(),
                                    expected: format!("{expected_dtype:?}"),
                                    actual: format!("{:?}", array.data_type()),
                                });
                                continue;
                            }

                            if !field.nullable() && field.default.is_none() && array.is_null(row) {
                                errors.push(SchemaValidationError::MissingRequiredNodeField {
                                    node_id: node_id.clone(),
                                    label: label.to_owned(),
                                    field: field.name.clone(),
                                });
                            }

                            if field.unique && !array.is_null(row) {
                                let value = scalar_value_repr(array.as_ref(), row);
                                let key = (label.to_owned(), field.name.clone());
                                let seen = unique_trackers.entry(key).or_default();
                                if !seen.insert(value.clone()) {
                                    errors.push(SchemaValidationError::UniqueViolation {
                                        scope: format!("node label {label}"),
                                        field: field.name.clone(),
                                        value,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        errors
    }

    fn validate_edges(&self, edges: &EdgeFrame) -> Vec<SchemaValidationError> {
        if self.edges.is_empty() {
            return Vec::new();
        }

        let mut errors = Vec::new();
        let src_column = edge_string_column(edges, COL_EDGE_SRC);
        let dst_column = edge_string_column(edges, COL_EDGE_DST);
        let type_column = edge_string_column(edges, COL_EDGE_TYPE);
        let mut unique_trackers: HashMap<(String, String), HashSet<String>> = HashMap::new();

        for row in 0..edges.len() {
            let src = src_column.value(row).to_owned();
            let dst = dst_column.value(row).to_owned();
            let edge_type = type_column.value(row);
            let Some(schema) = self.edge_schema(edge_type) else {
                errors.push(SchemaValidationError::UndefinedEdgeType {
                    src,
                    dst,
                    edge_type: edge_type.to_owned(),
                });
                continue;
            };

            for field in &schema.fields {
                let column = edges.column(&field.name);
                match column {
                    None if field.default.is_none() && !field.nullable() => {
                        errors.push(SchemaValidationError::MissingRequiredEdgeField {
                            src: src.clone(),
                            dst: dst.clone(),
                            edge_type: edge_type.to_owned(),
                            field: field.name.clone(),
                        });
                    }
                    None => {}
                    Some(array) => {
                        let expected_dtype = field.dtype.to_arrow_dtype().unwrap_or(DataType::Null);
                        if array.data_type() != &expected_dtype {
                            errors.push(SchemaValidationError::EdgeFieldTypeMismatch {
                                edge_type: edge_type.to_owned(),
                                field: field.name.clone(),
                                expected: format!("{expected_dtype:?}"),
                                actual: format!("{:?}", array.data_type()),
                            });
                            continue;
                        }

                        if !field.nullable() && field.default.is_none() && array.is_null(row) {
                            errors.push(SchemaValidationError::MissingRequiredEdgeField {
                                src: src.clone(),
                                dst: dst.clone(),
                                edge_type: edge_type.to_owned(),
                                field: field.name.clone(),
                            });
                        }

                        if field.unique && !array.is_null(row) {
                            let value = scalar_value_repr(array.as_ref(), row);
                            let key = (edge_type.to_owned(), field.name.clone());
                            let seen = unique_trackers.entry(key).or_default();
                            if !seen.insert(value.clone()) {
                                errors.push(SchemaValidationError::UniqueViolation {
                                    scope: format!("edge type {edge_type}"),
                                    field: field.name.clone(),
                                    value,
                                });
                            }
                        }
                    }
                }
            }
        }

        errors
    }
}

fn node_labels(label_column: &ListArray, row: usize) -> Vec<String> {
    label_column
        .value(row)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("_label values must be Utf8")
        .iter()
        .flatten()
        .map(str::to_owned)
        .collect()
}

fn edge_string_column<'a>(edges: &'a EdgeFrame, name: &str) -> &'a StringArray {
    edges
        .column(name)
        .expect("validated EdgeFrame column must exist")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("validated EdgeFrame column must be Utf8")
}

fn scalar_value_repr(array: &dyn Array, row: usize) -> String {
    arrow::util::display::array_value_to_string(array, row)
        .unwrap_or_else(|_| "<invalid>".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow_array::builder::{ListBuilder, StringBuilder};
    use arrow_array::{ArrayRef, Int8Array, ListArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};

    use crate::schema::{GFType, GFValue};
    use crate::{
        EdgeFrame, GraphFrame, NodeFrame, SchemaValidationError, COL_EDGE_DIRECTION, COL_EDGE_DST,
        COL_EDGE_SRC, COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
    };

    fn labels_array(values: &[&[&str]]) -> ListArray {
        let mut builder = ListBuilder::new(StringBuilder::new());
        for labels in values {
            for label in *labels {
                builder.values().append_value(label);
            }
            builder.append(true);
        }
        builder.finish()
    }

    fn node_batch_with_columns(
        extra_fields: Vec<Field>,
        extra_columns: Vec<ArrayRef>,
    ) -> RecordBatch {
        let mut fields = vec![
            Field::new(COL_NODE_ID, DataType::Utf8, false),
            Field::new(
                COL_NODE_LABEL,
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                false,
            ),
        ];
        fields.extend(extra_fields);

        let mut columns: Vec<ArrayRef> = vec![
            Arc::new(arrow_array::StringArray::from(vec!["alice", "bob"])),
            Arc::new(labels_array(&[&["Person"], &["Person", "Ghost"]])),
        ];
        columns.extend(extra_columns);

        RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), columns).unwrap()
    }

    fn edge_batch_with_columns(
        extra_fields: Vec<Field>,
        extra_columns: Vec<ArrayRef>,
    ) -> RecordBatch {
        let mut fields = vec![
            Field::new(COL_EDGE_SRC, DataType::Utf8, false),
            Field::new(COL_EDGE_DST, DataType::Utf8, false),
            Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
            Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
        ];
        fields.extend(extra_fields);

        let mut columns: Vec<ArrayRef> = vec![
            Arc::new(arrow_array::StringArray::from(vec!["alice", "alice"])),
            Arc::new(arrow_array::StringArray::from(vec!["bob", "bob"])),
            Arc::new(arrow_array::StringArray::from(vec!["KNOWS", "MISSING"])),
            Arc::new(Int8Array::from(vec![0i8, 0])),
        ];
        columns.extend(extra_columns);

        RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), columns).unwrap()
    }

    #[test]
    fn resolved_fields_inherit_parent_order() {
        let mut schema = Schema::default();
        schema.nodes.insert(
            "Person".to_owned(),
            NodeSchema::new("Person").with_fields(vec![
                FieldDef::new("name", GFType::String).unwrap(),
                FieldDef::new("age", GFType::Int).unwrap(),
            ]),
        );
        schema.nodes.insert(
            "Employee".to_owned(),
            NodeSchema::new("Employee")
                .with_extends("Person")
                .with_fields(vec![
                    FieldDef::new("age", GFType::Int).unwrap(),
                    FieldDef::new("team", GFType::String).unwrap(),
                ]),
        );

        let fields = schema.resolved_fields("Employee").unwrap();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].name, "name");
        assert_eq!(fields[1].name, "age");
        assert_eq!(fields[1].dtype, GFType::Int);
        assert_eq!(fields[2].name, "team");
    }

    #[test]
    fn resolved_fields_detect_cycle() {
        let mut schema = Schema::default();
        schema
            .nodes
            .insert("A".to_owned(), NodeSchema::new("A").with_extends("B"));
        schema
            .nodes
            .insert("B".to_owned(), NodeSchema::new("B").with_extends("A"));

        let err = schema.resolved_fields("A").unwrap_err();
        assert!(matches!(err, GFError::CircularInheritance { .. }));
    }

    #[test]
    fn arrow_schema_uses_resolved_fields() {
        let mut schema = Schema::default();
        schema.nodes.insert(
            "Person".to_owned(),
            NodeSchema::new("Person").with_fields(vec![
                FieldDef::new("name", GFType::String).unwrap(),
                FieldDef::new("age", GFType::Optional(Box::new(GFType::Int)))
                    .unwrap()
                    .with_default(GFValue::Null)
                    .unwrap(),
            ]),
        );

        let arrow = schema.to_arrow_node_schema("Person").unwrap();
        assert_eq!(arrow.fields().len(), 4);
        assert_eq!(arrow.field(0).name(), COL_NODE_ID);
        assert_eq!(arrow.field(1).name(), COL_NODE_LABEL);
        assert_eq!(arrow.field(2).name(), "name");
        assert!(arrow.field(3).is_nullable());
    }

    #[test]
    fn edge_arrow_schema_preserves_fields() {
        let mut schema = Schema::default();
        schema.edges.insert(
            "KNOWS".to_owned(),
            EdgeSchema::new("KNOWS")
                .with_fields(vec![FieldDef::new("since", GFType::Date).unwrap()]),
        );

        let arrow = schema.to_arrow_edge_schema("KNOWS").unwrap();
        assert_eq!(arrow.fields().len(), 5);
        assert_eq!(arrow.field(0).name(), COL_EDGE_SRC);
        assert_eq!(arrow.field(1).name(), COL_EDGE_DST);
        assert_eq!(arrow.field(2).name(), COL_EDGE_TYPE);
        assert_eq!(arrow.field(3).name(), COL_EDGE_DIRECTION);
        assert_eq!(arrow.field(4).name(), "since");
    }

    #[test]
    fn to_arrow_schema_aliases_node_arrow_schema() {
        let mut schema = Schema::default();
        schema.nodes.insert(
            "Person".to_owned(),
            NodeSchema::new("Person")
                .with_fields(vec![FieldDef::new("name", GFType::String).unwrap()]),
        );

        assert_eq!(
            schema.to_arrow_schema("Person").unwrap(),
            schema.to_arrow_node_schema("Person").unwrap()
        );
    }

    #[test]
    fn resolved_fields_reject_type_changing_override() {
        let mut schema = Schema::default();
        schema.nodes.insert(
            "Person".to_owned(),
            NodeSchema::new("Person")
                .with_fields(vec![FieldDef::new("name", GFType::String).unwrap()]),
        );
        schema.nodes.insert(
            "Robot".to_owned(),
            NodeSchema::new("Robot")
                .with_extends("Person")
                .with_fields(vec![FieldDef::new("name", GFType::Int).unwrap()]),
        );

        let err = schema.resolved_fields("Robot").unwrap_err();
        assert!(matches!(err, GFError::SchemaMismatch { .. }));
    }

    #[test]
    fn resolved_fields_inherit_monotonic_directives_and_default() {
        let mut schema = Schema::default();
        schema.nodes.insert(
            "Person".to_owned(),
            NodeSchema::new("Person").with_fields(vec![FieldDef::new("name", GFType::String)
                .unwrap()
                .with_unique(true)
                .with_indexed(true)
                .with_default(GFValue::String("anon".to_owned()))
                .unwrap()]),
        );
        schema.nodes.insert(
            "Employee".to_owned(),
            NodeSchema::new("Employee")
                .with_extends("Person")
                .with_fields(vec![FieldDef::new("name", GFType::String).unwrap()]),
        );

        let fields = schema.resolved_fields("Employee").unwrap();
        assert_eq!(fields.len(), 1);
        assert!(fields[0].unique);
        assert!(fields[0].indexed);
        assert_eq!(fields[0].default, Some(GFValue::String("anon".to_owned())));
    }

    #[test]
    fn resolved_fields_allow_same_type_default_override() {
        let mut schema = Schema::default();
        schema.nodes.insert(
            "Person".to_owned(),
            NodeSchema::new("Person").with_fields(vec![FieldDef::new("active", GFType::Bool)
                .unwrap()
                .with_default(GFValue::Bool(true))
                .unwrap()]),
        );
        schema.nodes.insert(
            "Contractor".to_owned(),
            NodeSchema::new("Contractor")
                .with_extends("Person")
                .with_fields(vec![FieldDef::new("active", GFType::Bool)
                    .unwrap()
                    .with_default(GFValue::Bool(false))
                    .unwrap()]),
        );

        let fields = schema.resolved_fields("Contractor").unwrap();
        assert_eq!(fields[0].default, Some(GFValue::Bool(false)));
    }

    #[test]
    fn validate_graph_collects_missing_unique_and_undefined_errors() {
        let mut schema = Schema::default();
        schema.nodes.insert(
            "Person".to_owned(),
            NodeSchema::new("Person").with_fields(vec![
                FieldDef::new("name", GFType::String)
                    .unwrap()
                    .with_unique(true),
                FieldDef::new("age", GFType::Int).unwrap(),
            ]),
        );
        schema.edges.insert(
            "KNOWS".to_owned(),
            EdgeSchema::new("KNOWS")
                .with_fields(vec![FieldDef::new("since", GFType::Int).unwrap()]),
        );

        let nodes = NodeFrame::from_record_batch(node_batch_with_columns(
            vec![Field::new("name", DataType::Utf8, true)],
            vec![Arc::new(arrow_array::StringArray::from(vec![
                Some("dup"),
                Some("dup"),
            ]))],
        ))
        .unwrap();
        let edges = EdgeFrame::from_record_batch(edge_batch_with_columns(vec![], vec![])).unwrap();
        let graph = GraphFrame::new_unchecked(nodes, edges);

        let errors = schema.validate_graph(&graph);
        assert!(errors.iter().any(|err| matches!(
            err,
            SchemaValidationError::MissingRequiredNodeField { node_id, field, .. }
            if node_id == "alice" && field == "age"
        )));
        assert!(errors.iter().any(|err| matches!(
            err,
            SchemaValidationError::UndefinedNodeLabel { node_id, label }
            if node_id == "bob" && label == "Ghost"
        )));
        assert!(errors.iter().any(|err| matches!(
            err,
            SchemaValidationError::UniqueViolation { scope, field, value }
            if scope == "node label Person" && field == "name" && value == "dup"
        )));
        assert!(errors.iter().any(|err| matches!(
            err,
            SchemaValidationError::MissingRequiredEdgeField { edge_type, field, .. }
            if edge_type == "KNOWS" && field == "since"
        )));
        assert!(errors.iter().any(|err| matches!(
            err,
            SchemaValidationError::UndefinedEdgeType { edge_type, .. }
            if edge_type == "MISSING"
        )));
    }

    #[test]
    fn validate_graph_reports_type_mismatch_from_arrow_columns() {
        let mut schema = Schema::default();
        schema.nodes.insert(
            "Person".to_owned(),
            NodeSchema::new("Person").with_fields(vec![FieldDef::new("age", GFType::Int).unwrap()]),
        );

        let nodes = NodeFrame::from_record_batch(node_batch_with_columns(
            vec![Field::new("age", DataType::Utf8, true)],
            vec![Arc::new(arrow_array::StringArray::from(vec![
                Some("30"),
                Some("31"),
            ]))],
        ))
        .unwrap();
        let edges = EdgeFrame::empty(&ArrowSchema::new(vec![
            Field::new(COL_EDGE_SRC, DataType::Utf8, false),
            Field::new(COL_EDGE_DST, DataType::Utf8, false),
            Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
            Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
        ]));
        let graph = GraphFrame::new_unchecked(nodes, edges);

        let errors = schema.validate_graph(&graph);
        assert!(errors.iter().any(|err| matches!(
            err,
            SchemaValidationError::NodeFieldTypeMismatch { label, field, expected, actual }
            if label == "Person" && field == "age" && expected.contains("Int64") && actual.contains("Utf8")
        )));
    }
}
