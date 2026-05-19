use std::io;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaValidationError {
    UndefinedNodeLabel {
        node_id: String,
        label: String,
    },
    UndefinedEdgeType {
        src: String,
        dst: String,
        edge_type: String,
    },
    MissingRequiredNodeField {
        node_id: String,
        label: String,
        field: String,
    },
    MissingRequiredEdgeField {
        src: String,
        dst: String,
        edge_type: String,
        field: String,
    },
    NodeFieldTypeMismatch {
        label: String,
        field: String,
        expected: String,
        actual: String,
    },
    EdgeFieldTypeMismatch {
        edge_type: String,
        field: String,
        expected: String,
        actual: String,
    },
    UniqueViolation {
        scope: String,
        field: String,
        value: String,
    },
}

impl std::fmt::Display for SchemaValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UndefinedNodeLabel { node_id, label } => {
                write!(f, "node {node_id} uses undefined label {label}")
            }
            Self::UndefinedEdgeType {
                src,
                dst,
                edge_type,
            } => {
                write!(
                    f,
                    "edge {src} -[{edge_type}]-> {dst} uses undefined edge type"
                )
            }
            Self::MissingRequiredNodeField {
                node_id,
                label,
                field,
            } => write!(
                f,
                "node {node_id}:{label} is missing required field {field}"
            ),
            Self::MissingRequiredEdgeField {
                src,
                dst,
                edge_type,
                field,
            } => write!(
                f,
                "edge {src} -[{edge_type}]-> {dst} is missing required field {field}"
            ),
            Self::NodeFieldTypeMismatch {
                label,
                field,
                expected,
                actual,
            } => write!(
                f,
                "node label {label} field {field} expected {expected}, got {actual}"
            ),
            Self::EdgeFieldTypeMismatch {
                edge_type,
                field,
                expected,
                actual,
            } => write!(
                f,
                "edge type {edge_type} field {field} expected {expected}, got {actual}"
            ),
            Self::UniqueViolation {
                scope,
                field,
                value,
            } => write!(f, "{scope} violates unique field {field}={value}"),
        }
    }
}

/// Canonical library error surface for Lynxes.
#[derive(Debug, thiserror::Error)]
pub enum GFError {
    #[error("missing reserved column: {column}")]
    MissingReservedColumn { column: String },

    #[error("reserved column has wrong type: {column}, expected {expected}, got {actual}")]
    ReservedColumnType {
        column: String,
        expected: String,
        actual: String,
    },

    #[error("reserved column name is not allowed for user data: {column}")]
    ReservedColumnName { column: String },

    #[error("duplicate node id: {id}")]
    DuplicateNodeId { id: String },

    #[error("node not found: {id}")]
    NodeNotFound { id: String },

    #[error("edge not found: {id}")]
    EdgeNotFound { id: String },

    #[error("dangling edge: {src} -> {dst}")]
    DanglingEdge { src: String, dst: String },

    #[error("invalid direction value: {value}")]
    InvalidDirection { value: i8 },

    #[error("column not found: {column}")]
    ColumnNotFound { column: String },

    #[error("length mismatch: expected {expected}, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },

    #[error("schema mismatch: {message}")]
    SchemaMismatch { message: String },

    #[error("type mismatch: {message}")]
    TypeMismatch { message: String },

    #[error("cannot infer type for column: {column}")]
    CannotInferType { column: String },

    #[error("type inference failed for column {column}: {message}")]
    TypeInferenceFailed { column: String, message: String },

    #[error("invalid type: {message}")]
    InvalidType { message: String },

    #[error("invalid cast: {from} -> {to}")]
    InvalidCast { from: String, to: String },

    #[error("default value type mismatch for field {field}: {message}")]
    DefaultTypeMismatch { field: String, message: String },

    #[error("missing required field: {field}")]
    MissingRequiredField { field: String },

    #[error("unique constraint violation: {field}={value}")]
    UniqueViolation { field: String, value: String },

    #[error("circular inheritance detected: {path}")]
    CircularInheritance { path: String },

    #[error("{message}")]
    SchemaValidation {
        message: String,
        errors: Vec<SchemaValidationError>,
    },

    #[error("invalid pattern alias: {alias}")]
    InvalidPatternAlias { alias: String },

    #[error("parse error: {message}")]
    ParseError { message: String },

    #[error("invalid config: {message}")]
    InvalidConfig { message: String },

    #[error("negative weight encountered in column: {column}")]
    NegativeWeight { column: String },

    #[error("connector error: {message}")]
    ConnectorError { message: String },

    #[error("domain mismatch: {message}")]
    DomainMismatch { message: String },

    #[error("unsupported operation: {message}")]
    UnsupportedOperation { message: String },

    #[error(transparent)]
    IoError(#[from] io::Error),
}

/// Standard result alias for library APIs.
pub type Result<T> = std::result::Result<T, GFError>;

impl GFError {
    /// Build a [`GFError::SchemaValidation`] with a human-readable summary of
    /// all errors. Up to `MAX_SHOWN` individual errors are printed inline;
    /// any remainder is counted in a trailing "… and N more" line.
    pub fn schema_validation(errors: Vec<SchemaValidationError>) -> Self {
        const MAX_SHOWN: usize = 10;
        let total = errors.len();

        let mut message = format!("schema validation failed with {total} error(s):\n");
        for (i, e) in errors.iter().take(MAX_SHOWN).enumerate() {
            message.push_str(&format!("  [{:>2}] {e}\n", i + 1));
        }
        if total > MAX_SHOWN {
            message.push_str(&format!("  ... and {} more", total - MAX_SHOWN));
        }

        GFError::SchemaValidation { message, errors }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn io_error_converts_via_from() {
        let err = io::Error::new(io::ErrorKind::NotFound, "missing file");
        let gf_err: GFError = err.into();

        assert!(matches!(gf_err, GFError::IoError(_)));
        assert!(gf_err.to_string().contains("missing file"));
    }

    #[test]
    fn structured_error_formats_context() {
        let err = GFError::NodeNotFound {
            id: "alice".to_string(),
        };

        assert_eq!(err.to_string(), "node not found: alice");
    }

    #[test]
    fn domain_mismatch_formats_context() {
        let err = GFError::DomainMismatch {
            message: "collect() requires a graph-domain plan".to_string(),
        };

        assert_eq!(
            err.to_string(),
            "domain mismatch: collect() requires a graph-domain plan"
        );
    }
}
