use arrow_schema::DataType;
use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value as JsonValue;

use crate::{Direction, GFError};

/// Symbolic expression tree used by lazy filters, sorting, traversal pruning,
/// aggregation input selection, and pattern predicates.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Col {
        name: String,
    },
    Literal {
        value: ScalarValue,
    },
    BinaryOp {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
    },
    UnaryOp {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    ListContains {
        expr: Box<Expr>,
        item: Box<Expr>,
    },
    Cast {
        expr: Box<Expr>,
        dtype: DataType,
    },
    And {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Or {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Not {
        expr: Box<Expr>,
    },
    PatternCol {
        alias: String,
        field: String,
    },
    /// String operations: `col.str.contains(pat)`, `.startswith()`, `.endswith()`
    StringOp {
        op: StringOp,
        expr: Box<Expr>,
        pattern: Box<Expr>,
    },
}

/// String predicate operations exposed through the `.str` accessor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StringOp {
    Contains,
    StartsWith,
    EndsWith,
}

/// Neighbor aggregation expression used by `AggregateNeighbors`.
#[derive(Debug, Clone, PartialEq)]
pub enum AggExpr {
    Count,
    Sum {
        expr: Expr,
    },
    Mean {
        expr: Expr,
    },
    List {
        expr: Expr,
    },
    First {
        expr: Expr,
    },
    Last {
        expr: Expr,
    },
    /// Wrap any `AggExpr` and override the output column name.
    Alias {
        expr: Box<AggExpr>,
        name: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    Eq,
    NotEq,
    Gt,
    GtEq,
    Lt,
    LtEq,
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    Neg,
}

/// Scalar literal payload carried by `Expr::Literal`.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarValue {
    Null,
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    List(Vec<ScalarValue>),
}

/// Edge-type selection used by traversal-oriented logical nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EdgeTypeSpec {
    Single(String),
    Multiple(Vec<String>),
    Any,
}

/// Declarative pattern shape for `PatternMatch`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pattern {
    pub steps: Vec<PatternStep>,
    pub node_constraints: BTreeMap<String, PatternNodeConstraint>,
    pub step_constraints: Vec<PatternStepConstraint>,
}

/// One traversal step inside a fixed pattern/traversal sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatternStep {
    pub from_alias: String,
    pub edge_alias: Option<String>,
    pub edge_type: EdgeTypeSpec,
    pub direction: Direction,
    pub to_alias: String,
}

/// Alias-level node restrictions captured during pattern lowering.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PatternNodeConstraint {
    pub label: Option<String>,
}

/// Step-level traversal restrictions captured during pattern lowering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatternStepConstraint {
    pub optional: bool,
    pub min_hops: u32,
    pub max_hops: u32,
}

impl Default for PatternStepConstraint {
    fn default() -> Self {
        Self {
            optional: false,
            min_hops: 1,
            max_hops: 1,
        }
    }
}

impl Pattern {
    pub fn new(steps: Vec<PatternStep>) -> Self {
        let step_constraints = vec![PatternStepConstraint::default(); steps.len()];
        Self {
            steps,
            node_constraints: BTreeMap::new(),
            step_constraints,
        }
    }

    pub fn with_constraints(
        steps: Vec<PatternStep>,
        node_constraints: BTreeMap<String, PatternNodeConstraint>,
        step_constraints: Vec<PatternStepConstraint>,
    ) -> crate::Result<Self> {
        if step_constraints.len() != steps.len() {
            return Err(GFError::InvalidConfig {
                message: format!(
                    "pattern step constraint count ({}) must match step count ({})",
                    step_constraints.len(),
                    steps.len()
                ),
            });
        }

        Ok(Self {
            steps,
            node_constraints,
            step_constraints,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    pub fn node_constraint(&self, alias: &str) -> Option<&PatternNodeConstraint> {
        self.node_constraints.get(alias)
    }

    pub fn step_constraint(&self, index: usize) -> &PatternStepConstraint {
        &self.step_constraints[index]
    }
}

impl Serialize for Expr {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        ExprDef::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Expr {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let def = ExprDef::deserialize(deserializer)?;
        Expr::try_from(def).map_err(serde::de::Error::custom)
    }
}

impl Serialize for AggExpr {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        AggExprDef::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for AggExpr {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let def = AggExprDef::deserialize(deserializer)?;
        AggExpr::try_from(def).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
enum ExprDef {
    #[serde(rename = "col")]
    Col { name: String },
    #[serde(rename = "literal")]
    Literal { dtype: String, value: JsonValue },
    #[serde(rename = "binary_op")]
    BinaryOp {
        left: Box<ExprDef>,
        op: BinaryOpDef,
        right: Box<ExprDef>,
    },
    #[serde(rename = "unary_op")]
    UnaryOp { op: UnaryOpDef, expr: Box<ExprDef> },
    #[serde(rename = "list_contains")]
    ListContains {
        expr: Box<ExprDef>,
        item: Box<ExprDef>,
    },
    #[serde(rename = "cast")]
    Cast { expr: Box<ExprDef>, dtype: String },
    #[serde(rename = "and")]
    And {
        left: Box<ExprDef>,
        right: Box<ExprDef>,
    },
    #[serde(rename = "or")]
    Or {
        left: Box<ExprDef>,
        right: Box<ExprDef>,
    },
    #[serde(rename = "not")]
    Not { expr: Box<ExprDef> },
    #[serde(rename = "pattern_col")]
    PatternCol { alias: String, field: String },
    #[serde(rename = "string_op")]
    StringOp {
        op: StringOpDef,
        expr: Box<ExprDef>,
        pattern: Box<ExprDef>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum StringOpDef {
    Contains,
    StartsWith,
    EndsWith,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
enum AggExprDef {
    #[serde(rename = "count")]
    Count,
    #[serde(rename = "sum")]
    Sum { expr: ExprDef },
    #[serde(rename = "mean")]
    Mean { expr: ExprDef },
    #[serde(rename = "list")]
    List { expr: ExprDef },
    #[serde(rename = "first")]
    First { expr: ExprDef },
    #[serde(rename = "last")]
    Last { expr: ExprDef },
    #[serde(rename = "alias")]
    Alias { expr: Box<AggExprDef>, name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum BinaryOpDef {
    Eq,
    NotEq,
    Gt,
    GtEq,
    Lt,
    LtEq,
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum UnaryOpDef {
    Neg,
}

impl From<&Expr> for ExprDef {
    fn from(value: &Expr) -> Self {
        match value {
            Expr::Col { name } => Self::Col { name: name.clone() },
            Expr::Literal { value } => Self::Literal {
                dtype: scalar_dtype_name(value),
                value: scalar_to_json(value),
            },
            Expr::BinaryOp { left, op, right } => Self::BinaryOp {
                left: Box::new(ExprDef::from(left.as_ref())),
                op: BinaryOpDef::from(op),
                right: Box::new(ExprDef::from(right.as_ref())),
            },
            Expr::UnaryOp { op, expr } => Self::UnaryOp {
                op: UnaryOpDef::from(op),
                expr: Box::new(ExprDef::from(expr.as_ref())),
            },
            Expr::ListContains { expr, item } => Self::ListContains {
                expr: Box::new(ExprDef::from(expr.as_ref())),
                item: Box::new(ExprDef::from(item.as_ref())),
            },
            Expr::Cast { expr, dtype } => Self::Cast {
                expr: Box::new(ExprDef::from(expr.as_ref())),
                dtype: dtype_to_name(dtype),
            },
            Expr::And { left, right } => Self::And {
                left: Box::new(ExprDef::from(left.as_ref())),
                right: Box::new(ExprDef::from(right.as_ref())),
            },
            Expr::Or { left, right } => Self::Or {
                left: Box::new(ExprDef::from(left.as_ref())),
                right: Box::new(ExprDef::from(right.as_ref())),
            },
            Expr::Not { expr } => Self::Not {
                expr: Box::new(ExprDef::from(expr.as_ref())),
            },
            Expr::PatternCol { alias, field } => Self::PatternCol {
                alias: alias.clone(),
                field: field.clone(),
            },
            Expr::StringOp { op, expr, pattern } => Self::StringOp {
                op: StringOpDef::from(op),
                expr: Box::new(ExprDef::from(expr.as_ref())),
                pattern: Box::new(ExprDef::from(pattern.as_ref())),
            },
        }
    }
}

impl TryFrom<ExprDef> for Expr {
    type Error = String;

    fn try_from(value: ExprDef) -> std::result::Result<Self, Self::Error> {
        Ok(match value {
            ExprDef::Col { name } => Self::Col { name },
            ExprDef::Literal { dtype, value } => Self::Literal {
                value: json_to_scalar(&dtype, value)?,
            },
            ExprDef::BinaryOp { left, op, right } => Self::BinaryOp {
                left: Box::new(Expr::try_from(*left)?),
                op: BinaryOp::from(op),
                right: Box::new(Expr::try_from(*right)?),
            },
            ExprDef::UnaryOp { op, expr } => Self::UnaryOp {
                op: UnaryOp::from(op),
                expr: Box::new(Expr::try_from(*expr)?),
            },
            ExprDef::ListContains { expr, item } => Self::ListContains {
                expr: Box::new(Expr::try_from(*expr)?),
                item: Box::new(Expr::try_from(*item)?),
            },
            ExprDef::Cast { expr, dtype } => Self::Cast {
                expr: Box::new(Expr::try_from(*expr)?),
                dtype: name_to_dtype(&dtype)?,
            },
            ExprDef::And { left, right } => Self::And {
                left: Box::new(Expr::try_from(*left)?),
                right: Box::new(Expr::try_from(*right)?),
            },
            ExprDef::Or { left, right } => Self::Or {
                left: Box::new(Expr::try_from(*left)?),
                right: Box::new(Expr::try_from(*right)?),
            },
            ExprDef::Not { expr } => Self::Not {
                expr: Box::new(Expr::try_from(*expr)?),
            },
            ExprDef::PatternCol { alias, field } => Self::PatternCol { alias, field },
            ExprDef::StringOp { op, expr, pattern } => Self::StringOp {
                op: StringOp::from(op),
                expr: Box::new(Expr::try_from(*expr)?),
                pattern: Box::new(Expr::try_from(*pattern)?),
            },
        })
    }
}

impl From<&StringOp> for StringOpDef {
    fn from(value: &StringOp) -> Self {
        match value {
            StringOp::Contains => Self::Contains,
            StringOp::StartsWith => Self::StartsWith,
            StringOp::EndsWith => Self::EndsWith,
        }
    }
}

impl From<StringOpDef> for StringOp {
    fn from(value: StringOpDef) -> Self {
        match value {
            StringOpDef::Contains => Self::Contains,
            StringOpDef::StartsWith => Self::StartsWith,
            StringOpDef::EndsWith => Self::EndsWith,
        }
    }
}

impl From<&AggExpr> for AggExprDef {
    fn from(value: &AggExpr) -> Self {
        match value {
            AggExpr::Count => Self::Count,
            AggExpr::Sum { expr } => Self::Sum {
                expr: ExprDef::from(expr),
            },
            AggExpr::Mean { expr } => Self::Mean {
                expr: ExprDef::from(expr),
            },
            AggExpr::List { expr } => Self::List {
                expr: ExprDef::from(expr),
            },
            AggExpr::First { expr } => Self::First {
                expr: ExprDef::from(expr),
            },
            AggExpr::Last { expr } => Self::Last {
                expr: ExprDef::from(expr),
            },
            AggExpr::Alias { expr, name } => Self::Alias {
                expr: Box::new(AggExprDef::from(expr.as_ref())),
                name: name.clone(),
            },
        }
    }
}

impl TryFrom<AggExprDef> for AggExpr {
    type Error = String;

    fn try_from(value: AggExprDef) -> std::result::Result<Self, Self::Error> {
        Ok(match value {
            AggExprDef::Count => Self::Count,
            AggExprDef::Sum { expr } => Self::Sum {
                expr: Expr::try_from(expr)?,
            },
            AggExprDef::Mean { expr } => Self::Mean {
                expr: Expr::try_from(expr)?,
            },
            AggExprDef::List { expr } => Self::List {
                expr: Expr::try_from(expr)?,
            },
            AggExprDef::First { expr } => Self::First {
                expr: Expr::try_from(expr)?,
            },
            AggExprDef::Last { expr } => Self::Last {
                expr: Expr::try_from(expr)?,
            },
            AggExprDef::Alias { expr, name } => Self::Alias {
                expr: Box::new(AggExpr::try_from(*expr)?),
                name,
            },
        })
    }
}

impl From<&BinaryOp> for BinaryOpDef {
    fn from(value: &BinaryOp) -> Self {
        match value {
            BinaryOp::Eq => Self::Eq,
            BinaryOp::NotEq => Self::NotEq,
            BinaryOp::Gt => Self::Gt,
            BinaryOp::GtEq => Self::GtEq,
            BinaryOp::Lt => Self::Lt,
            BinaryOp::LtEq => Self::LtEq,
            BinaryOp::Add => Self::Add,
            BinaryOp::Sub => Self::Sub,
            BinaryOp::Mul => Self::Mul,
            BinaryOp::Div => Self::Div,
        }
    }
}

impl From<BinaryOpDef> for BinaryOp {
    fn from(value: BinaryOpDef) -> Self {
        match value {
            BinaryOpDef::Eq => Self::Eq,
            BinaryOpDef::NotEq => Self::NotEq,
            BinaryOpDef::Gt => Self::Gt,
            BinaryOpDef::GtEq => Self::GtEq,
            BinaryOpDef::Lt => Self::Lt,
            BinaryOpDef::LtEq => Self::LtEq,
            BinaryOpDef::Add => Self::Add,
            BinaryOpDef::Sub => Self::Sub,
            BinaryOpDef::Mul => Self::Mul,
            BinaryOpDef::Div => Self::Div,
        }
    }
}

impl From<&UnaryOp> for UnaryOpDef {
    fn from(value: &UnaryOp) -> Self {
        match value {
            UnaryOp::Neg => Self::Neg,
        }
    }
}

impl From<UnaryOpDef> for UnaryOp {
    fn from(value: UnaryOpDef) -> Self {
        match value {
            UnaryOpDef::Neg => Self::Neg,
        }
    }
}

fn scalar_dtype_name(value: &ScalarValue) -> String {
    match value {
        ScalarValue::Null => "Null".to_owned(),
        ScalarValue::String(_) => "String".to_owned(),
        ScalarValue::Int(_) => "Int".to_owned(),
        ScalarValue::Float(_) => "Float".to_owned(),
        ScalarValue::Bool(_) => "Bool".to_owned(),
        ScalarValue::List(values) => {
            let item_dtype = values
                .first()
                .map(scalar_dtype_name)
                .unwrap_or_else(|| "Null".to_owned());
            format!("List<{item_dtype}>")
        }
    }
}

fn scalar_to_json(value: &ScalarValue) -> JsonValue {
    match value {
        ScalarValue::Null => JsonValue::Null,
        ScalarValue::String(value) => JsonValue::String(value.clone()),
        ScalarValue::Int(value) => JsonValue::from(*value),
        ScalarValue::Float(value) => JsonValue::from(*value),
        ScalarValue::Bool(value) => JsonValue::from(*value),
        ScalarValue::List(values) => JsonValue::Array(values.iter().map(scalar_to_json).collect()),
    }
}

fn json_to_scalar(dtype: &str, value: JsonValue) -> std::result::Result<ScalarValue, String> {
    match dtype {
        "Null" => Ok(ScalarValue::Null),
        "String" => value
            .as_str()
            .map(|value| ScalarValue::String(value.to_owned()))
            .ok_or_else(|| "literal dtype String requires string value".to_owned()),
        "Int" => value
            .as_i64()
            .map(ScalarValue::Int)
            .ok_or_else(|| "literal dtype Int requires integer value".to_owned()),
        "Float" => value
            .as_f64()
            .map(ScalarValue::Float)
            .ok_or_else(|| "literal dtype Float requires float value".to_owned()),
        "Bool" => value
            .as_bool()
            .map(ScalarValue::Bool)
            .ok_or_else(|| "literal dtype Bool requires boolean value".to_owned()),
        list if list.starts_with("List<") && list.ends_with('>') => {
            let inner = &list[5..list.len() - 1];
            let array = value
                .as_array()
                .ok_or_else(|| format!("literal dtype {dtype} requires array value"))?;
            let mut items = Vec::with_capacity(array.len());
            for item in array {
                items.push(json_to_scalar(inner, item.clone())?);
            }
            Ok(ScalarValue::List(items))
        }
        other => Err(format!("unsupported literal dtype tag: {other}")),
    }
}

fn dtype_to_name(dtype: &DataType) -> String {
    match dtype {
        DataType::Utf8 => "String".to_owned(),
        DataType::Int64 => "Int".to_owned(),
        DataType::Float64 => "Float".to_owned(),
        DataType::Boolean => "Bool".to_owned(),
        DataType::Null => "Null".to_owned(),
        DataType::List(field) => format!("List<{}>", dtype_to_name(field.data_type())),
        other => format!("{other:?}"),
    }
}

fn name_to_dtype(name: &str) -> std::result::Result<DataType, String> {
    match name {
        "String" => Ok(DataType::Utf8),
        "Int" => Ok(DataType::Int64),
        "Float" => Ok(DataType::Float64),
        "Bool" => Ok(DataType::Boolean),
        "Null" => Ok(DataType::Null),
        list if list.starts_with("List<") && list.ends_with('>') => {
            let inner = &list[5..list.len() - 1];
            Ok(DataType::List(std::sync::Arc::new(
                arrow_schema::Field::new("item", name_to_dtype(inner)?, true),
            )))
        }
        other => Err(format!("unsupported dtype tag: {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn expr_tree_supports_nested_boolean_predicates() {
        let expr = Expr::And {
            left: Box::new(Expr::BinaryOp {
                left: Box::new(Expr::Col {
                    name: "age".to_owned(),
                }),
                op: BinaryOp::GtEq,
                right: Box::new(Expr::Literal {
                    value: ScalarValue::Int(18),
                }),
            }),
            right: Box::new(Expr::ListContains {
                expr: Box::new(Expr::Col {
                    name: "_label".to_owned(),
                }),
                item: Box::new(Expr::Literal {
                    value: ScalarValue::String("Person".to_owned()),
                }),
            }),
        };

        assert!(matches!(expr, Expr::And { .. }));
    }

    #[test]
    fn edge_type_spec_supports_single_many_and_any() {
        assert_eq!(
            EdgeTypeSpec::Single("KNOWS".to_owned()),
            EdgeTypeSpec::Single("KNOWS".to_owned())
        );
        assert_eq!(
            EdgeTypeSpec::Multiple(vec!["KNOWS".to_owned(), "LIKES".to_owned()]),
            EdgeTypeSpec::Multiple(vec!["KNOWS".to_owned(), "LIKES".to_owned()])
        );
        assert_eq!(EdgeTypeSpec::Any, EdgeTypeSpec::Any);
    }

    #[test]
    fn pattern_new_preserves_step_order() {
        let pattern = Pattern::new(vec![
            PatternStep {
                from_alias: "a".to_owned(),
                edge_alias: Some("e1".to_owned()),
                edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
                direction: Direction::Out,
                to_alias: "b".to_owned(),
            },
            PatternStep {
                from_alias: "b".to_owned(),
                edge_alias: Some("e2".to_owned()),
                edge_type: EdgeTypeSpec::Single("WORKS_AT".to_owned()),
                direction: Direction::Out,
                to_alias: "c".to_owned(),
            },
        ]);

        assert_eq!(pattern.steps.len(), 2);
        assert_eq!(pattern.steps[0].from_alias, "a");
        assert_eq!(pattern.steps[1].to_alias, "c");
    }

    #[test]
    fn expr_serializes_to_canonical_tagged_shape() {
        let expr = Expr::BinaryOp {
            left: Box::new(Expr::Col {
                name: "age".to_owned(),
            }),
            op: BinaryOp::Gt,
            right: Box::new(Expr::Literal {
                value: ScalarValue::Int(30),
            }),
        };

        let value = serde_json::to_value(&expr).unwrap();
        assert_eq!(
            value,
            json!({
                "kind": "binary_op",
                "left": { "kind": "col", "name": "age" },
                "op": "gt",
                "right": { "kind": "literal", "dtype": "Int", "value": 30 }
            })
        );
    }

    #[test]
    fn expr_round_trips_through_tagged_json() {
        let expr = Expr::And {
            left: Box::new(Expr::BinaryOp {
                left: Box::new(Expr::Col {
                    name: "age".to_owned(),
                }),
                op: BinaryOp::GtEq,
                right: Box::new(Expr::Literal {
                    value: ScalarValue::Int(18),
                }),
            }),
            right: Box::new(Expr::ListContains {
                expr: Box::new(Expr::Col {
                    name: "_label".to_owned(),
                }),
                item: Box::new(Expr::Literal {
                    value: ScalarValue::String("Person".to_owned()),
                }),
            }),
        };

        let json = serde_json::to_string(&expr).unwrap();
        let decoded: Expr = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, expr);
    }

    #[test]
    fn agg_expr_round_trips_through_tagged_json() {
        let agg = AggExpr::Mean {
            expr: Expr::Cast {
                expr: Box::new(Expr::Col {
                    name: "weight".to_owned(),
                }),
                dtype: DataType::Float64,
            },
        };

        let json = serde_json::to_string(&agg).unwrap();
        let decoded: AggExpr = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, agg);
    }

    #[test]
    fn agg_expr_alias_round_trips_through_tagged_json() {
        let agg = AggExpr::Alias {
            expr: Box::new(AggExpr::Count),
            name: "follower_count".to_owned(),
        };

        let json = serde_json::to_string(&agg).unwrap();
        let decoded: AggExpr = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, agg);

        // Inner alias should also round-trip nested (alias of sum).
        let agg2 = AggExpr::Alias {
            expr: Box::new(AggExpr::Sum {
                expr: Expr::Col {
                    name: "weight".to_owned(),
                },
            }),
            name: "total_weight".to_owned(),
        };
        let json2 = serde_json::to_string(&agg2).unwrap();
        let decoded2: AggExpr = serde_json::from_str(&json2).unwrap();
        assert_eq!(decoded2, agg2);
    }

    #[test]
    fn literal_list_serialization_uses_list_dtype() {
        let expr = Expr::Literal {
            value: ScalarValue::List(vec![
                ScalarValue::Int(1),
                ScalarValue::Int(2),
                ScalarValue::Int(3),
            ]),
        };

        let value = serde_json::to_value(&expr).unwrap();
        assert_eq!(
            value,
            json!({
                "kind": "literal",
                "dtype": "List<Int>",
                "value": [1, 2, 3]
            })
        );
    }

    #[test]
    fn malformed_literal_dtype_is_rejected() {
        let err = serde_json::from_value::<Expr>(json!({
            "kind": "literal",
            "dtype": "Timestamp",
            "value": 1
        }))
        .unwrap_err();

        assert!(err.to_string().contains("unsupported literal dtype tag"));
    }
}
