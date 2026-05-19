use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_schema::{DataType, Field, TimeUnit};
use serde::{Deserialize, Serialize};

use crate::{GFError, Result};

/// Canonical Lynxes type surface used by schema declarations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GFType {
    String,
    Int,
    Float,
    Bool,
    Date,
    DateTime,
    Duration,
    Any,
    List(Box<GFType>),
    Optional(Box<GFType>),
}

impl GFType {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::List(inner) => {
                inner.validate()?;
                if matches!(inner.as_ref(), Self::Optional(_)) {
                    return Err(GFError::SchemaMismatch {
                        message: "List(Optional(T)) is not supported".to_owned(),
                    });
                }
                Ok(())
            }
            Self::Optional(inner) => {
                inner.validate()?;
                if matches!(inner.as_ref(), Self::Optional(_)) {
                    return Err(GFError::SchemaMismatch {
                        message: "Optional(Optional(T)) is not supported".to_owned(),
                    });
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    pub fn is_optional(&self) -> bool {
        matches!(self, Self::Optional(_))
    }

    pub fn is_list(&self) -> bool {
        matches!(self, Self::List(_))
    }

    pub fn inner(&self) -> Option<&GFType> {
        match self {
            Self::List(inner) | Self::Optional(inner) => Some(inner.as_ref()),
            _ => None,
        }
    }

    pub fn nullable(&self) -> bool {
        self.is_optional()
    }

    pub fn to_arrow_dtype(&self) -> Result<DataType> {
        self.validate()?;
        Ok(match self {
            Self::String | Self::Any => DataType::Utf8,
            Self::Int => DataType::Int64,
            Self::Float => DataType::Float64,
            Self::Bool => DataType::Boolean,
            Self::Date => DataType::Date32,
            Self::DateTime => DataType::Timestamp(TimeUnit::Microsecond, None),
            Self::Duration => DataType::Duration(TimeUnit::Microsecond),
            Self::List(inner) => {
                DataType::List(Arc::new(Field::new("item", inner.to_arrow_dtype()?, true)))
            }
            Self::Optional(inner) => inner.to_arrow_dtype()?,
        })
    }
}

/// Untyped literal value surface used by `.gf` metadata, properties, and defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GFValue {
    Null,
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Date(String),
    DateTime(String),
    List(Vec<GFValue>),
    Object(BTreeMap<String, GFValue>),
}

/// One declared schema field with optional directives.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldDef {
    pub name: String,
    pub dtype: GFType,
    pub unique: bool,
    pub indexed: bool,
    pub default: Option<GFValue>,
}

impl FieldDef {
    pub fn new(name: impl Into<String>, dtype: GFType) -> Result<Self> {
        dtype.validate()?;
        Ok(Self {
            name: name.into(),
            dtype,
            unique: false,
            indexed: false,
            default: None,
        })
    }

    pub fn with_unique(mut self, unique: bool) -> Self {
        self.unique = unique;
        self
    }

    pub fn with_indexed(mut self, indexed: bool) -> Self {
        self.indexed = indexed;
        self
    }

    pub fn with_default(mut self, default: GFValue) -> Result<Self> {
        self.validate_default(Some(&default))?;
        self.default = Some(default);
        Ok(self)
    }

    pub fn nullable(&self) -> bool {
        self.dtype.nullable()
    }

    pub fn validate_default(&self, default: Option<&GFValue>) -> Result<()> {
        if let Some(default) = default {
            self.dtype.validate()?;
            if !gf_value_matches_type(default, &self.dtype) {
                return Err(GFError::TypeMismatch {
                    message: format!(
                        "default for field {} does not match declared type {:?}",
                        self.name, self.dtype
                    ),
                });
            }
        }
        Ok(())
    }

    pub fn to_arrow_field(&self) -> Result<Field> {
        self.dtype.validate()?;
        self.validate_default(self.default.as_ref())?;
        Ok(Field::new(
            &self.name,
            self.dtype.to_arrow_dtype()?,
            self.nullable(),
        ))
    }
}

pub fn gf_value_matches_type(value: &GFValue, dtype: &GFType) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_nested_optional() {
        let err = GFType::Optional(Box::new(GFType::Optional(Box::new(GFType::Int))))
            .validate()
            .unwrap_err();
        assert!(matches!(err, GFError::SchemaMismatch { .. }));
    }

    #[test]
    fn rejects_list_optional_child() {
        let err = GFType::List(Box::new(GFType::Optional(Box::new(GFType::String))))
            .validate()
            .unwrap_err();
        assert!(matches!(err, GFError::SchemaMismatch { .. }));
    }

    #[test]
    fn arrow_dtype_maps_canonical_types() {
        assert_eq!(GFType::Date.to_arrow_dtype().unwrap(), DataType::Date32);
        assert_eq!(
            GFType::DateTime.to_arrow_dtype().unwrap(),
            DataType::Timestamp(TimeUnit::Microsecond, None)
        );
        assert_eq!(
            GFType::Duration.to_arrow_dtype().unwrap(),
            DataType::Duration(TimeUnit::Microsecond)
        );
    }

    #[test]
    fn field_default_must_match_type() {
        let field = FieldDef::new("age", GFType::Int).unwrap();
        let err = field.validate_default(Some(&GFValue::String("bad".to_owned())));
        assert!(matches!(err.unwrap_err(), GFError::TypeMismatch { .. }));
    }

    #[test]
    fn optional_default_allows_null() {
        let field = FieldDef::new("age", GFType::Optional(Box::new(GFType::Int))).unwrap();
        field.validate_default(Some(&GFValue::Null)).unwrap();
    }

    #[test]
    fn to_arrow_field_uses_optional_nullability() {
        let field = FieldDef::new("name", GFType::Optional(Box::new(GFType::String))).unwrap();
        let arrow = field.to_arrow_field().unwrap();
        assert!(arrow.is_nullable());
        assert_eq!(arrow.data_type(), &DataType::Utf8);
    }
}
