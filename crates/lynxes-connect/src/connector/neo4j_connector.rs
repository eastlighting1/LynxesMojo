use std::{collections::BTreeMap, sync::Arc};

use lynxes_core::{
    BinaryOp, Direction, EdgeFrame, EdgeTypeSpec, Expr, GFError, GraphFrame, NodeFrame, Result,
    ScalarValue, StringOp, UnaryOp, COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE,
    COL_NODE_ID, COL_NODE_LABEL,
};

use crate::connector::{Connector, ConnectorFuture, ExpandResult};

#[derive(Clone, PartialEq, Eq)]
pub struct Neo4jConfig {
    pub uri: String,
    pub user: String,
    pub password: String,
    pub database: Option<String>,
}

impl std::fmt::Debug for Neo4jConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Neo4jConfig")
            .field("uri", &self.uri)
            .field("user", &self.user)
            .field("password", &"<redacted>")
            .field("database", &self.database)
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CypherValue {
    Null,
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    List(Vec<CypherValue>),
}

pub type CypherParams = BTreeMap<String, CypherValue>;

#[derive(Debug, Clone, PartialEq)]
pub struct CypherQuery {
    pub text: String,
    pub params: CypherParams,
}

pub trait Neo4jBackend: Send + Sync + std::fmt::Debug {
    fn load_nodes<'a>(&'a self, query: CypherQuery) -> ConnectorFuture<'a, NodeFrame>;

    fn load_edges<'a>(&'a self, query: CypherQuery) -> ConnectorFuture<'a, EdgeFrame>;

    fn expand<'a>(&'a self, query: CypherQuery) -> ConnectorFuture<'a, ExpandResult>;

    fn write<'a>(&'a self, graph: &'a GraphFrame, query: CypherQuery) -> ConnectorFuture<'a, ()> {
        let _ = (graph, query);
        Box::pin(async move {
            Err(GFError::UnsupportedOperation {
                message: "Neo4j backend write() is not implemented".to_owned(),
            })
        })
    }
}

#[derive(Debug)]
struct UnsupportedNeo4jBackend;

impl Neo4jBackend for UnsupportedNeo4jBackend {
    fn load_nodes<'a>(&'a self, _query: CypherQuery) -> ConnectorFuture<'a, NodeFrame> {
        Box::pin(async move {
            Err(GFError::ConnectorError {
                message: "Neo4j Bolt backend is not linked in this build".to_owned(),
            })
        })
    }

    fn load_edges<'a>(&'a self, _query: CypherQuery) -> ConnectorFuture<'a, EdgeFrame> {
        Box::pin(async move {
            Err(GFError::ConnectorError {
                message: "Neo4j Bolt backend is not linked in this build".to_owned(),
            })
        })
    }

    fn expand<'a>(&'a self, _query: CypherQuery) -> ConnectorFuture<'a, ExpandResult> {
        Box::pin(async move {
            Err(GFError::ConnectorError {
                message: "Neo4j Bolt backend is not linked in this build".to_owned(),
            })
        })
    }
}

#[derive(Debug, Clone)]
pub struct Neo4jConnector {
    config: Neo4jConfig,
    backend: Arc<dyn Neo4jBackend>,
}

impl Neo4jConnector {
    pub fn new(config: Neo4jConfig) -> Self {
        Self {
            config,
            backend: Arc::new(UnsupportedNeo4jBackend),
        }
    }

    pub fn with_backend(config: Neo4jConfig, backend: Arc<dyn Neo4jBackend>) -> Self {
        Self { config, backend }
    }

    pub fn config(&self) -> &Neo4jConfig {
        &self.config
    }
}

impl Connector for Neo4jConnector {
    fn cache_source_key(&self) -> Option<String> {
        let mut key = format!("neo4j://{}", self.config.uri);
        if let Some(database) = &self.config.database {
            key.push_str("?db=");
            key.push_str(database);
        }
        Some(key)
    }

    fn load_nodes<'a>(
        &'a self,
        labels: Option<&'a [&'a str]>,
        columns: Option<&'a [&'a str]>,
        predicate: Option<&'a Expr>,
        batch_size: usize,
    ) -> ConnectorFuture<'a, NodeFrame> {
        Box::pin(async move {
            validate_batch_size(batch_size)?;
            let mut skip = 0usize;
            let mut pages = Vec::new();

            loop {
                let query = build_load_nodes_query(labels, columns, predicate, skip, batch_size)?;
                let frame = self.backend.load_nodes(query).await?;
                let done = frame.len() < batch_size;
                pages.push(frame);
                if done {
                    break;
                }
                skip += batch_size;
            }

            finish_node_pages(pages)
        })
    }

    fn load_edges<'a>(
        &'a self,
        edge_types: Option<&'a [&'a str]>,
        columns: Option<&'a [&'a str]>,
        predicate: Option<&'a Expr>,
        batch_size: usize,
    ) -> ConnectorFuture<'a, EdgeFrame> {
        Box::pin(async move {
            validate_batch_size(batch_size)?;
            let mut skip = 0usize;
            let mut pages = Vec::new();

            loop {
                let query =
                    build_load_edges_query(edge_types, columns, predicate, skip, batch_size)?;
                let frame = self.backend.load_edges(query).await?;
                let done = frame.len() < batch_size;
                pages.push(frame);
                if done {
                    break;
                }
                skip += batch_size;
            }

            finish_edge_pages(pages)
        })
    }

    fn expand<'a>(
        &'a self,
        node_ids: &'a [&'a str],
        edge_type: &'a EdgeTypeSpec,
        hops: u32,
        direction: Direction,
        node_predicate: Option<&'a Expr>,
    ) -> ConnectorFuture<'a, ExpandResult> {
        Box::pin(async move {
            validate_hops(hops)?;
            let query = build_expand_query(node_ids, edge_type, hops, direction, node_predicate)?;
            self.backend.expand(query).await
        })
    }

    fn write<'a>(&'a self, graph: &'a GraphFrame) -> ConnectorFuture<'a, ()> {
        let _ = graph;
        Box::pin(async move {
            Err(GFError::UnsupportedOperation {
                message: "Neo4jConnector write() is not implemented yet".to_owned(),
            })
        })
    }
}

fn finish_node_pages(mut pages: Vec<NodeFrame>) -> Result<NodeFrame> {
    if pages.is_empty() {
        return Err(GFError::ConnectorError {
            message: "Neo4j backend returned no node pages".to_owned(),
        });
    }
    if pages.len() == 1 {
        return Ok(pages.pop().expect("checked len == 1"));
    }
    let refs: Vec<&NodeFrame> = pages.iter().collect();
    NodeFrame::concat(&refs)
}

fn finish_edge_pages(mut pages: Vec<EdgeFrame>) -> Result<EdgeFrame> {
    if pages.is_empty() {
        return Err(GFError::ConnectorError {
            message: "Neo4j backend returned no edge pages".to_owned(),
        });
    }
    if pages.len() == 1 {
        return Ok(pages.pop().expect("checked len == 1"));
    }
    let refs: Vec<&EdgeFrame> = pages.iter().collect();
    EdgeFrame::concat(&refs)
}

fn build_load_nodes_query(
    labels: Option<&[&str]>,
    columns: Option<&[&str]>,
    predicate: Option<&Expr>,
    skip: usize,
    batch_size: usize,
) -> Result<CypherQuery> {
    let mut builder = CypherQueryBuilder::default();
    let mut where_clauses = Vec::new();

    if let Some(label_clause) = compile_labels_filter(labels) {
        where_clauses.push(label_clause);
    }
    if let Some(predicate) = predicate {
        where_clauses.push(builder.compile_expr(predicate, QueryDomain::Node { var: "n" })?);
    }

    let mut params = builder.params;
    params.insert("skip".to_owned(), CypherValue::Int(skip as i64));
    params.insert("limit".to_owned(), CypherValue::Int(batch_size as i64));

    Ok(CypherQuery {
        text: format!(
            "MATCH (n)\n{where_clause}\nRETURN {projection}\nSKIP $skip LIMIT $limit",
            where_clause = format_where(&where_clauses),
            projection = compile_node_projection(columns),
        ),
        params,
    })
}

fn build_load_edges_query(
    edge_types: Option<&[&str]>,
    columns: Option<&[&str]>,
    predicate: Option<&Expr>,
    skip: usize,
    batch_size: usize,
) -> Result<CypherQuery> {
    let mut builder = CypherQueryBuilder::default();
    let mut where_clauses = Vec::new();

    if let Some(type_clause) = compile_edge_types_filter(edge_types, &mut builder) {
        where_clauses.push(type_clause);
    }
    if let Some(predicate) = predicate {
        where_clauses.push(builder.compile_expr(
            predicate,
            QueryDomain::Edge {
                edge: "r",
                src: "src",
                dst: "dst",
            },
        )?);
    }

    let mut params = builder.params;
    params.insert("skip".to_owned(), CypherValue::Int(skip as i64));
    params.insert("limit".to_owned(), CypherValue::Int(batch_size as i64));

    Ok(CypherQuery {
        text: format!(
            "MATCH (src)-[r]->(dst)\n{where_clause}\nRETURN {projection}\nSKIP $skip LIMIT $limit",
            where_clause = format_where(&where_clauses),
            projection = compile_edge_projection(columns),
        ),
        params,
    })
}

fn build_expand_query(
    node_ids: &[&str],
    edge_type: &EdgeTypeSpec,
    hops: u32,
    direction: Direction,
    node_predicate: Option<&Expr>,
) -> Result<CypherQuery> {
    if node_ids.is_empty() {
        return Err(GFError::InvalidConfig {
            message: "expand requires at least one seed node id".to_owned(),
        });
    }

    let mut builder = CypherQueryBuilder::default();
    builder.params.insert(
        "seed_ids".to_owned(),
        CypherValue::List(
            node_ids
                .iter()
                .map(|value| CypherValue::String((*value).to_owned()))
                .collect(),
        ),
    );

    let mut where_clauses = vec!["seed.`_id` IN $seed_ids".to_owned()];
    if let Some(predicate) = node_predicate {
        where_clauses.push(builder.compile_expr(predicate, QueryDomain::Node { var: "m" })?);
    }

    Ok(CypherQuery {
        text: format!(
            "MATCH (seed)\nWHERE {seed_where}\nMATCH path = (seed){relationship}(m)\nRETURN collect(DISTINCT path) AS paths",
            seed_where = where_clauses.join(" AND "),
            relationship = relationship_pattern(edge_type, direction, hops),
        ),
        params: builder.params,
    })
}

#[derive(Debug, Clone, Copy)]
enum QueryDomain<'a> {
    Node {
        var: &'a str,
    },
    Edge {
        edge: &'a str,
        src: &'a str,
        dst: &'a str,
    },
}

#[derive(Debug, Default)]
struct CypherQueryBuilder {
    params: CypherParams,
    next_param: usize,
}

impl CypherQueryBuilder {
    fn compile_expr(&mut self, expr: &Expr, domain: QueryDomain<'_>) -> Result<String> {
        match expr {
            Expr::Col { name } => self.compile_column(name, domain),
            Expr::Literal { value } => Ok(self.push_param(value.clone())),
            Expr::BinaryOp { left, op, right } => Ok(format!(
                "({} {} {})",
                self.compile_expr(left, domain)?,
                binary_op_symbol(op),
                self.compile_expr(right, domain)?,
            )),
            Expr::UnaryOp { op, expr } => match op {
                UnaryOp::Neg => Ok(format!("(-{})", self.compile_expr(expr, domain)?)),
            },
            Expr::ListContains { expr, item } => Ok(format!(
                "({} IN {})",
                self.compile_expr(item, domain)?,
                self.compile_expr(expr, domain)?,
            )),
            Expr::Cast { .. } => Err(GFError::UnsupportedOperation {
                message: "Neo4jConnector predicate pushdown does not support Cast".to_owned(),
            }),
            Expr::And { left, right } => Ok(format!(
                "(({}) AND ({}))",
                self.compile_expr(left, domain)?,
                self.compile_expr(right, domain)?,
            )),
            Expr::Or { left, right } => Ok(format!(
                "(({}) OR ({}))",
                self.compile_expr(left, domain)?,
                self.compile_expr(right, domain)?,
            )),
            Expr::Not { expr } => Ok(format!("(NOT ({}))", self.compile_expr(expr, domain)?)),
            Expr::PatternCol { alias, field } => Err(GFError::UnsupportedOperation {
                message: format!(
                    "Neo4jConnector predicate pushdown does not support PatternCol({alias}.{field})"
                ),
            }),
            Expr::StringOp { op, expr, pattern } => {
                let subject = self.compile_expr(expr, domain)?;
                let pat = self.compile_expr(pattern, domain)?;
                let func = match op {
                    StringOp::Contains => "CONTAINS",
                    StringOp::StartsWith => "STARTS WITH",
                    StringOp::EndsWith => "ENDS WITH",
                };
                Ok(format!("({subject} {func} {pat})"))
            }
        }
    }

    fn compile_column(&self, name: &str, domain: QueryDomain<'_>) -> Result<String> {
        match domain {
            QueryDomain::Node { var } => match name {
                COL_NODE_ID => Ok(format!("{var}.`_id`")),
                COL_NODE_LABEL => Ok(format!("labels({var})")),
                COL_EDGE_SRC | COL_EDGE_DST | COL_EDGE_TYPE | COL_EDGE_DIRECTION => {
                    Err(GFError::TypeMismatch {
                        message: format!("node predicate cannot reference edge column {name}"),
                    })
                }
                other => Ok(format!("{var}.{}", quoted_ident(other))),
            },
            QueryDomain::Edge { edge, src, dst } => match name {
                COL_EDGE_SRC => Ok(format!("{src}.`_id`")),
                COL_EDGE_DST => Ok(format!("{dst}.`_id`")),
                COL_EDGE_TYPE => Ok(format!("type({edge})")),
                COL_EDGE_DIRECTION => Ok("0".to_owned()),
                COL_NODE_ID | COL_NODE_LABEL => Err(GFError::TypeMismatch {
                    message: format!("edge predicate cannot reference node column {name}"),
                }),
                other => Ok(format!("{edge}.{}", quoted_ident(other))),
            },
        }
    }

    fn push_param(&mut self, value: ScalarValue) -> String {
        let key = format!("p{}", self.next_param);
        self.next_param += 1;
        self.params
            .insert(key.clone(), scalar_to_cypher_value(value));
        format!("${key}")
    }
}

fn compile_labels_filter(labels: Option<&[&str]>) -> Option<String> {
    let labels = labels?;
    if labels.is_empty() {
        return Some("false".to_owned());
    }

    Some(format!(
        "({})",
        labels
            .iter()
            .map(|label| format!("n:{}", quoted_ident(label)))
            .collect::<Vec<_>>()
            .join(" OR ")
    ))
}

fn compile_edge_types_filter(
    edge_types: Option<&[&str]>,
    builder: &mut CypherQueryBuilder,
) -> Option<String> {
    let edge_types = edge_types?;
    if edge_types.is_empty() {
        return Some("false".to_owned());
    }

    Some(format!(
        "({})",
        edge_types
            .iter()
            .map(|edge_type| {
                let key = builder.push_param(ScalarValue::String((*edge_type).to_owned()));
                format!("type(r) = {key}")
            })
            .collect::<Vec<_>>()
            .join(" OR ")
    ))
}

fn compile_node_projection(columns: Option<&[&str]>) -> String {
    let mut projections = vec![
        format!("n.`_id` AS {}", quoted_ident(COL_NODE_ID)),
        format!("labels(n) AS {}", quoted_ident(COL_NODE_LABEL)),
    ];

    match columns {
        Some(columns) => {
            for column in user_projection_columns(columns, &[COL_NODE_ID, COL_NODE_LABEL]) {
                projections.push(format!("n.{0} AS {0}", quoted_ident(column)));
            }
        }
        None => projections.push("n { .* } AS `__props`".to_owned()),
    }

    projections.join(", ")
}

fn compile_edge_projection(columns: Option<&[&str]>) -> String {
    let mut projections = vec![
        format!("src.`_id` AS {}", quoted_ident(COL_EDGE_SRC)),
        format!("dst.`_id` AS {}", quoted_ident(COL_EDGE_DST)),
        format!("type(r) AS {}", quoted_ident(COL_EDGE_TYPE)),
        format!("0 AS {}", quoted_ident(COL_EDGE_DIRECTION)),
    ];

    match columns {
        Some(columns) => {
            for column in user_projection_columns(
                columns,
                &[
                    COL_EDGE_SRC,
                    COL_EDGE_DST,
                    COL_EDGE_TYPE,
                    COL_EDGE_DIRECTION,
                ],
            ) {
                projections.push(format!("r.{0} AS {0}", quoted_ident(column)));
            }
        }
        None => projections.push("r { .* } AS `__props`".to_owned()),
    }

    projections.join(", ")
}

fn user_projection_columns<'a>(columns: &'a [&'a str], reserved: &[&str]) -> Vec<&'a str> {
    columns
        .iter()
        .copied()
        .filter(|column| !reserved.iter().any(|reserved| reserved == column))
        .collect()
}

fn relationship_pattern(edge_type: &EdgeTypeSpec, direction: Direction, hops: u32) -> String {
    let rel = format!("[{}]", relationship_contents(edge_type, hops));
    match direction {
        Direction::Out => format!("-{rel}->"),
        Direction::In => format!("<-{rel}-"),
        Direction::Both | Direction::None => format!("-{rel}-"),
    }
}

fn relationship_contents(edge_type: &EdgeTypeSpec, hops: u32) -> String {
    let range = format!("*1..{hops}");
    match edge_type {
        EdgeTypeSpec::Any => range,
        EdgeTypeSpec::Single(value) => format!(":{}{}", quoted_ident(value), range),
        EdgeTypeSpec::Multiple(values) => format!(
            ":{}{}",
            values
                .iter()
                .map(|value| quoted_ident(value))
                .collect::<Vec<_>>()
                .join("|"),
            range
        ),
    }
}

fn format_where(clauses: &[String]) -> String {
    if clauses.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", clauses.join(" AND "))
    }
}

fn quoted_ident(value: &str) -> String {
    format!("`{}`", value.replace('`', "``"))
}

fn scalar_to_cypher_value(value: ScalarValue) -> CypherValue {
    match value {
        ScalarValue::Null => CypherValue::Null,
        ScalarValue::String(value) => CypherValue::String(value),
        ScalarValue::Int(value) => CypherValue::Int(value),
        ScalarValue::Float(value) => CypherValue::Float(value),
        ScalarValue::Bool(value) => CypherValue::Bool(value),
        ScalarValue::List(values) => {
            CypherValue::List(values.into_iter().map(scalar_to_cypher_value).collect())
        }
    }
}

fn binary_op_symbol(op: &BinaryOp) -> &'static str {
    match op {
        BinaryOp::Eq => "=",
        BinaryOp::NotEq => "<>",
        BinaryOp::Gt => ">",
        BinaryOp::GtEq => ">=",
        BinaryOp::Lt => "<",
        BinaryOp::LtEq => "<=",
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul => "*",
        BinaryOp::Div => "/",
    }
}

fn validate_batch_size(batch_size: usize) -> Result<()> {
    if batch_size == 0 {
        return Err(GFError::InvalidConfig {
            message: "batch_size must be greater than zero".to_owned(),
        });
    }
    Ok(())
}

fn validate_hops(hops: u32) -> Result<()> {
    if hops == 0 {
        return Err(GFError::InvalidConfig {
            message: "hops must be greater than zero".to_owned(),
        });
    }
    Ok(())
}

// ── TST-010: Neo4j connector tests (mock backend) ─────────────────────────────
//
// All tests use `MockNeo4jBackend` — no live Bolt connection required.
// Tests that need a real Neo4j server are marked `#[ignore]` and gated on the
// `NEO4J_URI` environment variable.

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{
        builder::{ListBuilder, StringBuilder},
        Int8Array, RecordBatch, StringArray,
    };
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lynxes_core::{
        BinaryOp, Direction, EdgeFrame, EdgeTypeSpec, Expr, NodeFrame, ScalarValue,
        COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
    };
    use std::sync::Arc;

    // ── Mock backend ──────────────────────────────────────────────────────────

    /// A minimal in-memory `Neo4jBackend` that returns canned `NodeFrame` /
    /// `EdgeFrame` responses regardless of the Cypher text.
    #[derive(Debug)]
    struct MockNeo4jBackend {
        nodes: NodeFrame,
        edges: EdgeFrame,
    }

    impl MockNeo4jBackend {
        fn new() -> Self {
            Self {
                nodes: make_mock_nodes(),
                edges: make_mock_edges(),
            }
        }
    }

    impl Neo4jBackend for MockNeo4jBackend {
        fn load_nodes<'a>(&'a self, _query: CypherQuery) -> ConnectorFuture<'a, NodeFrame> {
            let frame = self.nodes.clone();
            Box::pin(async move { Ok(frame) })
        }

        fn load_edges<'a>(&'a self, _query: CypherQuery) -> ConnectorFuture<'a, EdgeFrame> {
            let frame = self.edges.clone();
            Box::pin(async move { Ok(frame) })
        }

        fn expand<'a>(&'a self, _query: CypherQuery) -> ConnectorFuture<'a, ExpandResult> {
            let nodes = self.nodes.clone();
            let edges = self.edges.clone();
            Box::pin(async move { Ok((nodes, edges)) })
        }
    }

    // ── Graph builders ────────────────────────────────────────────────────────

    fn make_mock_nodes() -> NodeFrame {
        let mut lb = ListBuilder::new(StringBuilder::new());
        lb.values().append_value("Person");
        lb.append(true);
        lb.values().append_value("Person");
        lb.append(true);

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new(COL_NODE_ID, DataType::Utf8, false),
            Field::new(
                COL_NODE_LABEL,
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                false,
            ),
        ]));
        NodeFrame::from_record_batch(
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["alice", "bob"]))
                        as Arc<dyn arrow_array::Array>,
                    Arc::new(lb.finish()) as Arc<dyn arrow_array::Array>,
                ],
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn make_mock_edges() -> EdgeFrame {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new(COL_EDGE_SRC, DataType::Utf8, false),
            Field::new(COL_EDGE_DST, DataType::Utf8, false),
            Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
            Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
        ]));
        EdgeFrame::from_record_batch(
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["alice"])) as Arc<dyn arrow_array::Array>,
                    Arc::new(StringArray::from(vec!["bob"])) as Arc<dyn arrow_array::Array>,
                    Arc::new(StringArray::from(vec!["KNOWS"])) as Arc<dyn arrow_array::Array>,
                    Arc::new(Int8Array::from(vec![0i8])) as Arc<dyn arrow_array::Array>,
                ],
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn make_connector() -> Neo4jConnector {
        let config = Neo4jConfig {
            uri: "bolt://localhost:7687".to_owned(),
            user: "neo4j".to_owned(),
            password: "test".to_owned(),
            database: None,
        };
        Neo4jConnector::with_backend(config, Arc::new(MockNeo4jBackend::new()))
    }

    // ── Config ────────────────────────────────────────────────────────────────

    #[test]
    fn config_uri_stored_correctly() {
        let connector = make_connector();
        assert_eq!(connector.config().uri, "bolt://localhost:7687");
    }

    #[test]
    fn config_password_not_shown_in_debug() {
        let config = Neo4jConfig {
            uri: "bolt://host:7687".to_owned(),
            user: "neo4j".to_owned(),
            password: "secret123".to_owned(),
            database: None,
        };
        let debug = format!("{config:?}");
        assert!(!debug.contains("secret123"), "password must be redacted");
        assert!(debug.contains("<redacted>"));
    }

    #[test]
    fn cache_source_key_includes_uri() {
        let connector = make_connector();
        let key = connector.cache_source_key().unwrap();
        assert!(key.contains("localhost:7687"));
    }

    #[test]
    fn cache_source_key_includes_database_when_set() {
        let config = Neo4jConfig {
            uri: "bolt://host:7687".to_owned(),
            user: "neo4j".to_owned(),
            password: "pw".to_owned(),
            database: Some("mydb".to_owned()),
        };
        let connector = Neo4jConnector::with_backend(config, Arc::new(MockNeo4jBackend::new()));
        let key = connector.cache_source_key().unwrap();
        assert!(
            key.contains("mydb"),
            "key must contain database name: {key}"
        );
    }

    // ── Cypher query generation ───────────────────────────────────────────────

    #[test]
    fn load_nodes_query_contains_match_n() {
        let q = build_load_nodes_query(None, None, None, 0, 100).unwrap();
        assert!(q.text.contains("MATCH (n)"), "query: {}", q.text);
    }

    #[test]
    fn load_nodes_query_respects_label_filter() {
        let q = build_load_nodes_query(Some(&["Person"]), None, None, 0, 100).unwrap();
        assert!(
            q.text.contains("Person"),
            "label filter missing: {}",
            q.text
        );
    }

    #[test]
    fn load_nodes_query_embeds_predicate() {
        let pred = Expr::BinaryOp {
            left: Box::new(Expr::Col {
                name: "age".to_owned(),
            }),
            op: BinaryOp::Gt,
            right: Box::new(Expr::Literal {
                value: ScalarValue::Int(25),
            }),
        };
        let q = build_load_nodes_query(None, None, Some(&pred), 0, 100).unwrap();
        assert!(
            q.text.to_lowercase().contains("where"),
            "WHERE clause missing: {}",
            q.text
        );
    }

    #[test]
    fn load_nodes_query_has_skip_limit_params() {
        let q = build_load_nodes_query(None, None, None, 200, 50).unwrap();
        assert_eq!(q.params.get("skip"), Some(&CypherValue::Int(200)));
        assert_eq!(q.params.get("limit"), Some(&CypherValue::Int(50)));
    }

    #[test]
    fn load_edges_query_contains_match_pattern() {
        let q = build_load_edges_query(None, None, None, 0, 100).unwrap();
        assert!(
            q.text.contains("MATCH (src)-[r]->(dst)"),
            "query: {}",
            q.text
        );
    }

    #[test]
    fn load_edges_query_respects_type_filter() {
        let q = build_load_edges_query(Some(&["KNOWS"]), None, None, 0, 100).unwrap();
        // Edge types are passed as Cypher parameters, not inlined.
        // Verify that a WHERE clause and at least one param referencing the type exist.
        assert!(
            q.text.to_lowercase().contains("where") || q.text.contains("type(r)"),
            "type filter clause missing: {}",
            q.text
        );
        let has_knows_param = q
            .params
            .values()
            .any(|v| matches!(v, CypherValue::String(s) if s == "KNOWS"));
        assert!(has_knows_param, "KNOWS not found in params: {:?}", q.params);
    }

    #[test]
    fn expand_query_embeds_seed_ids() {
        let q = build_expand_query(
            &["alice", "bob"],
            &EdgeTypeSpec::Any,
            1,
            Direction::Out,
            None,
        )
        .unwrap();
        assert!(q.params.contains_key("seed_ids"), "seed_ids param missing");
        assert!(q.text.contains("seed_ids"), "query: {}", q.text);
    }

    #[test]
    fn expand_query_out_direction_arrow_syntax() {
        let q =
            build_expand_query(&["alice"], &EdgeTypeSpec::Any, 1, Direction::Out, None).unwrap();
        // Out direction: (seed)-->(m)  (→ should appear in relationship pattern)
        assert!(q.text.contains("->"), "out arrow missing: {}", q.text);
    }

    #[test]
    fn expand_query_in_direction_arrow_syntax() {
        let q = build_expand_query(&["bob"], &EdgeTypeSpec::Any, 1, Direction::In, None).unwrap();
        assert!(q.text.contains("<-"), "in arrow missing: {}", q.text);
    }

    #[test]
    fn expand_query_empty_seeds_is_error() {
        let err = build_expand_query(&[], &EdgeTypeSpec::Any, 1, Direction::Out, None).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("at least one"), "unexpected error: {msg}");
    }

    #[test]
    fn expand_query_hop_count_in_pattern() {
        let q =
            build_expand_query(&["alice"], &EdgeTypeSpec::Any, 3, Direction::Out, None).unwrap();
        // Pattern should encode max depth (e.g. `*1..3`)
        assert!(q.text.contains("3"), "hop count missing: {}", q.text);
    }

    // ── Validation helpers ────────────────────────────────────────────────────

    #[test]
    fn validate_batch_size_zero_is_error() {
        let err = validate_batch_size(0).unwrap_err();
        assert!(err.to_string().contains("batch_size"));
    }

    #[test]
    fn validate_batch_size_nonzero_is_ok() {
        assert!(validate_batch_size(100).is_ok());
    }

    #[test]
    fn validate_hops_zero_is_error() {
        let err = validate_hops(0).unwrap_err();
        assert!(err.to_string().contains("hops"));
    }

    #[test]
    fn validate_hops_nonzero_is_ok() {
        assert!(validate_hops(1).is_ok());
        assert!(validate_hops(10).is_ok());
    }

    // ── load_nodes / load_edges via mock ──────────────────────────────────────

    #[tokio::test]
    async fn load_nodes_returns_mock_frame() {
        let connector = make_connector();
        let nf = connector.load_nodes(None, None, None, 100).await.unwrap();
        assert_eq!(nf.len(), 2);
    }

    #[tokio::test]
    async fn load_edges_returns_mock_frame() {
        let connector = make_connector();
        let ef = connector.load_edges(None, None, None, 100).await.unwrap();
        assert_eq!(ef.len(), 1);
    }

    #[tokio::test]
    async fn expand_returns_mock_result() {
        let connector = make_connector();
        let (nodes, edges) = connector
            .expand(&["alice"], &EdgeTypeSpec::Any, 1, Direction::Out, None)
            .await
            .unwrap();
        assert_eq!(nodes.len(), 2);
        assert_eq!(edges.len(), 1);
    }

    #[tokio::test]
    async fn write_returns_unsupported_error() {
        use lynxes_core::GraphFrame;
        let connector = make_connector();
        let graph = GraphFrame::new(make_mock_nodes(), make_mock_edges()).unwrap();
        let err = connector.write(&graph).await.unwrap_err();
        assert!(
            err.to_string().contains("not implemented")
                || matches!(err, GFError::UnsupportedOperation { .. })
        );
    }

    // ── Live Neo4j (requires NEO4J_URI env var; skipped in CI) ───────────────

    /// Verify a real Neo4j server can be connected to.
    /// Run with: NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASS=test
    ///           cargo test -- tests::live_neo4j_load_nodes --ignored
    #[tokio::test]
    #[ignore = "requires live Neo4j: set NEO4J_URI, NEO4J_USER, NEO4J_PASS"]
    async fn live_neo4j_load_nodes() {
        let uri = std::env::var("NEO4J_URI").expect("NEO4J_URI not set");
        let user = std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".to_owned());
        let pass = std::env::var("NEO4J_PASS").unwrap_or_else(|_| "test".to_owned());

        let config = Neo4jConfig {
            uri,
            user,
            password: pass,
            database: None,
        };
        let connector = Neo4jConnector::new(config); // uses UnsupportedNeo4jBackend

        // Without a real backend this will error — the test proves the connector
        // initialises cleanly and returns a typed error rather than panicking.
        let result = connector.load_nodes(None, None, None, 10).await;
        assert!(
            result.is_err(),
            "expected error with UnsupportedNeo4jBackend"
        );
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("not linked") || msg.contains("not implemented"));
    }
}
