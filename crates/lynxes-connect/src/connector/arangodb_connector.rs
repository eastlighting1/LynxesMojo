use std::{collections::BTreeMap, sync::Arc};

use lynxes_core::{
    BinaryOp, Direction, EdgeFrame, EdgeTypeSpec, Expr, GFError, GraphFrame, NodeFrame, Result,
    ScalarValue, StringOp, UnaryOp, COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE,
    COL_NODE_ID, COL_NODE_LABEL,
};

use crate::connector::{Connector, ConnectorFuture, ExpandResult};

#[derive(Clone, PartialEq, Eq)]
pub struct ArangoConfig {
    pub endpoint: String,
    pub database: String,
    pub graph: String,
    pub vertex_collection: String,
    pub edge_collection: String,
    pub username: String,
    pub password: String,
}

impl std::fmt::Debug for ArangoConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArangoConfig")
            .field("endpoint", &self.endpoint)
            .field("database", &self.database)
            .field("graph", &self.graph)
            .field("vertex_collection", &self.vertex_collection)
            .field("edge_collection", &self.edge_collection)
            .field("username", &self.username)
            .field("password", &"<redacted>")
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AqlValue {
    Null,
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    List(Vec<AqlValue>),
}

pub type AqlBindVars = BTreeMap<String, AqlValue>;

#[derive(Debug, Clone, PartialEq)]
pub struct AqlQuery {
    pub text: String,
    pub bind_vars: AqlBindVars,
}

pub trait ArangoBackend: Send + Sync + std::fmt::Debug {
    fn load_nodes<'a>(&'a self, query: AqlQuery) -> ConnectorFuture<'a, NodeFrame>;

    fn load_edges<'a>(&'a self, query: AqlQuery) -> ConnectorFuture<'a, EdgeFrame>;

    fn expand<'a>(&'a self, query: AqlQuery) -> ConnectorFuture<'a, ExpandResult>;

    fn write<'a>(&'a self, graph: &'a GraphFrame, query: AqlQuery) -> ConnectorFuture<'a, ()> {
        let _ = (graph, query);
        Box::pin(async move {
            Err(GFError::UnsupportedOperation {
                message: "Arango backend write() is not implemented".to_owned(),
            })
        })
    }
}

#[derive(Debug)]
struct UnsupportedArangoBackend;

impl ArangoBackend for UnsupportedArangoBackend {
    fn load_nodes<'a>(&'a self, _query: AqlQuery) -> ConnectorFuture<'a, NodeFrame> {
        Box::pin(async move {
            Err(GFError::ConnectorError {
                message: "ArangoDB HTTP backend is not linked in this build".to_owned(),
            })
        })
    }

    fn load_edges<'a>(&'a self, _query: AqlQuery) -> ConnectorFuture<'a, EdgeFrame> {
        Box::pin(async move {
            Err(GFError::ConnectorError {
                message: "ArangoDB HTTP backend is not linked in this build".to_owned(),
            })
        })
    }

    fn expand<'a>(&'a self, _query: AqlQuery) -> ConnectorFuture<'a, ExpandResult> {
        Box::pin(async move {
            Err(GFError::ConnectorError {
                message: "ArangoDB HTTP backend is not linked in this build".to_owned(),
            })
        })
    }
}

#[derive(Debug, Clone)]
pub struct ArangoConnector {
    config: ArangoConfig,
    backend: Arc<dyn ArangoBackend>,
}

impl ArangoConnector {
    pub fn new(config: ArangoConfig) -> Self {
        Self {
            config,
            backend: Arc::new(UnsupportedArangoBackend),
        }
    }

    pub fn with_backend(config: ArangoConfig, backend: Arc<dyn ArangoBackend>) -> Self {
        Self { config, backend }
    }

    pub fn config(&self) -> &ArangoConfig {
        &self.config
    }
}

impl Connector for ArangoConnector {
    fn cache_source_key(&self) -> Option<String> {
        Some(format!(
            "arangodb://{}?db={}&graph={}",
            self.config.endpoint, self.config.database, self.config.graph
        ))
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
            let mut offset = 0usize;
            let mut pages = Vec::new();

            loop {
                let query = build_load_nodes_query(
                    &self.config,
                    labels,
                    columns,
                    predicate,
                    offset,
                    batch_size,
                )?;
                let frame = self.backend.load_nodes(query).await?;
                let done = frame.len() < batch_size;
                pages.push(frame);
                if done {
                    break;
                }
                offset += batch_size;
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
            let mut offset = 0usize;
            let mut pages = Vec::new();

            loop {
                let query = build_load_edges_query(
                    &self.config,
                    edge_types,
                    columns,
                    predicate,
                    offset,
                    batch_size,
                )?;
                let frame = self.backend.load_edges(query).await?;
                let done = frame.len() < batch_size;
                pages.push(frame);
                if done {
                    break;
                }
                offset += batch_size;
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
            let query = build_expand_query(
                &self.config,
                node_ids,
                edge_type,
                hops,
                direction,
                node_predicate,
            )?;
            self.backend.expand(query).await
        })
    }

    fn write<'a>(&'a self, graph: &'a GraphFrame) -> ConnectorFuture<'a, ()> {
        let _ = graph;
        Box::pin(async move {
            Err(GFError::UnsupportedOperation {
                message: "ArangoConnector write() is not implemented yet".to_owned(),
            })
        })
    }
}

fn finish_node_pages(mut pages: Vec<NodeFrame>) -> Result<NodeFrame> {
    if pages.is_empty() {
        return Err(GFError::ConnectorError {
            message: "Arango backend returned no node pages".to_owned(),
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
            message: "Arango backend returned no edge pages".to_owned(),
        });
    }
    if pages.len() == 1 {
        return Ok(pages.pop().expect("checked len == 1"));
    }
    let refs: Vec<&EdgeFrame> = pages.iter().collect();
    EdgeFrame::concat(&refs)
}

fn build_load_nodes_query(
    config: &ArangoConfig,
    labels: Option<&[&str]>,
    columns: Option<&[&str]>,
    predicate: Option<&Expr>,
    offset: usize,
    batch_size: usize,
) -> Result<AqlQuery> {
    let mut builder = AqlQueryBuilder::default();
    builder.bind_vars.insert(
        "@vertex_collection".to_owned(),
        AqlValue::String(config.vertex_collection.clone()),
    );
    builder
        .bind_vars
        .insert("offset".to_owned(), AqlValue::Int(offset as i64));
    builder
        .bind_vars
        .insert("limit".to_owned(), AqlValue::Int(batch_size as i64));

    let mut filters = Vec::new();
    if let Some(label_filter) = compile_labels_filter(labels, &mut builder) {
        filters.push(label_filter);
    }
    if let Some(predicate) = predicate {
        filters.push(format!(
            "FILTER {}",
            builder.compile_expr(predicate, QueryDomain::Node { var: "v" })?
        ));
    }

    Ok(AqlQuery {
        text: format!(
            "FOR v IN @@vertex_collection\n{filters}\nLIMIT @offset, @limit\nRETURN {projection}",
            filters = format_filters(&filters),
            projection = compile_node_projection(columns),
        ),
        bind_vars: builder.bind_vars,
    })
}

fn build_load_edges_query(
    config: &ArangoConfig,
    edge_types: Option<&[&str]>,
    columns: Option<&[&str]>,
    predicate: Option<&Expr>,
    offset: usize,
    batch_size: usize,
) -> Result<AqlQuery> {
    let mut builder = AqlQueryBuilder::default();
    builder.bind_vars.insert(
        "@edge_collection".to_owned(),
        AqlValue::String(config.edge_collection.clone()),
    );
    builder
        .bind_vars
        .insert("offset".to_owned(), AqlValue::Int(offset as i64));
    builder
        .bind_vars
        .insert("limit".to_owned(), AqlValue::Int(batch_size as i64));

    let mut filters = Vec::new();
    if let Some(type_filter) = compile_edge_types_filter(edge_types, &mut builder) {
        filters.push(type_filter);
    }
    if let Some(predicate) = predicate {
        filters.push(format!(
            "FILTER {}",
            builder.compile_expr(predicate, QueryDomain::Edge { edge: "e" })?
        ));
    }

    Ok(AqlQuery {
        text: format!(
            "FOR e IN @@edge_collection\n{filters}\nLIMIT @offset, @limit\nRETURN {projection}",
            filters = format_filters(&filters),
            projection = compile_edge_projection(columns),
        ),
        bind_vars: builder.bind_vars,
    })
}

fn build_expand_query(
    config: &ArangoConfig,
    node_ids: &[&str],
    edge_type: &EdgeTypeSpec,
    hops: u32,
    direction: Direction,
    node_predicate: Option<&Expr>,
) -> Result<AqlQuery> {
    if node_ids.is_empty() {
        return Err(GFError::InvalidConfig {
            message: "expand requires at least one seed node id".to_owned(),
        });
    }

    let mut builder = AqlQueryBuilder::default();
    builder.bind_vars.insert(
        "@vertex_collection".to_owned(),
        AqlValue::String(config.vertex_collection.clone()),
    );
    builder
        .bind_vars
        .insert("graph".to_owned(), AqlValue::String(config.graph.clone()));
    builder.bind_vars.insert(
        "seed_ids".to_owned(),
        AqlValue::List(
            node_ids
                .iter()
                .map(|node_id| AqlValue::String((*node_id).to_owned()))
                .collect(),
        ),
    );

    let mut vertex_filters = Vec::new();
    if let Some(type_filter) = compile_expand_edge_types_filter(edge_type, &mut builder) {
        vertex_filters.push(type_filter);
    }
    if let Some(predicate) = node_predicate {
        vertex_filters.push(format!(
            "FILTER {}",
            builder.compile_expr(predicate, QueryDomain::Node { var: "vertex" })?
        ));
    }

    Ok(AqlQuery {
        text: format!(
            "FOR seed IN @@vertex_collection\nFILTER seed._id IN @seed_ids\nFOR vertex, edge, path IN 1..{hops} {dir} seed GRAPH @graph\n{vertex_filters}\nRETURN {{ vertex: vertex, edge: edge, path: path }}",
            hops = hops,
            dir = traversal_direction(direction),
            vertex_filters = format_filters(&vertex_filters),
        ),
        bind_vars: builder.bind_vars,
    })
}

#[derive(Debug, Clone, Copy)]
enum QueryDomain<'a> {
    Node { var: &'a str },
    Edge { edge: &'a str },
}

#[derive(Debug, Default)]
struct AqlQueryBuilder {
    bind_vars: AqlBindVars,
    next_param: usize,
}

impl AqlQueryBuilder {
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
                message: "ArangoConnector predicate pushdown does not support Cast".to_owned(),
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
            Expr::Not { expr } => Ok(format!("(!({}))", self.compile_expr(expr, domain)?)),
            Expr::PatternCol { alias, field } => Err(GFError::UnsupportedOperation {
                message: format!(
                    "ArangoConnector predicate pushdown does not support PatternCol({alias}.{field})"
                ),
            }),
            Expr::StringOp { op, expr, pattern } => {
                let subject = self.compile_expr(expr, domain)?;
                let pat = self.compile_expr(pattern, domain)?;
                // AQL: CONTAINS(str, pat), STARTS_WITH(str, pat), LIKE(str, CONCAT(pat, '%'))
                let aql = match op {
                    StringOp::Contains => format!("CONTAINS({subject}, {pat})"),
                    StringOp::StartsWith => format!("STARTS_WITH({subject}, {pat})"),
                    StringOp::EndsWith => {
                        format!("(SUBSTRING({subject}, LENGTH({subject}) - LENGTH({pat})) == {pat})")
                    }
                };
                Ok(format!("({aql})"))
            }
        }
    }

    fn compile_column(&self, name: &str, domain: QueryDomain<'_>) -> Result<String> {
        match domain {
            QueryDomain::Node { var } => match name {
                COL_NODE_ID => Ok(format!("{var}._id")),
                COL_NODE_LABEL => Ok(format!("{var}.{}", quoted_ident(COL_NODE_LABEL))),
                COL_EDGE_SRC | COL_EDGE_DST | COL_EDGE_TYPE | COL_EDGE_DIRECTION => {
                    Err(GFError::TypeMismatch {
                        message: format!("node predicate cannot reference edge column {name}"),
                    })
                }
                other => Ok(format!("{var}.{}", quoted_ident(other))),
            },
            QueryDomain::Edge { edge } => match name {
                COL_EDGE_SRC => Ok(format!("{edge}._from")),
                COL_EDGE_DST => Ok(format!("{edge}._to")),
                COL_EDGE_TYPE => Ok(format!("{edge}.{}", quoted_ident(COL_EDGE_TYPE))),
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
        self.bind_vars
            .insert(key.clone(), scalar_to_aql_value(value));
        format!("@{key}")
    }
}

fn compile_labels_filter(labels: Option<&[&str]>, builder: &mut AqlQueryBuilder) -> Option<String> {
    let labels = labels?;
    if labels.is_empty() {
        return Some("FILTER false".to_owned());
    }

    let mut clauses = Vec::new();
    for label in labels {
        let key = builder.push_param(ScalarValue::String((*label).to_owned()));
        clauses.push(format!("{key} IN v.`_label`"));
    }
    Some(format!("FILTER {}", clauses.join(" OR ")))
}

fn compile_edge_types_filter(
    edge_types: Option<&[&str]>,
    builder: &mut AqlQueryBuilder,
) -> Option<String> {
    let edge_types = edge_types?;
    if edge_types.is_empty() {
        return Some("FILTER false".to_owned());
    }

    let mut clauses = Vec::new();
    for edge_type in edge_types {
        let key = builder.push_param(ScalarValue::String((*edge_type).to_owned()));
        clauses.push(format!("e.`_type` == {key}"));
    }
    Some(format!("FILTER {}", clauses.join(" OR ")))
}

fn compile_expand_edge_types_filter(
    edge_type: &EdgeTypeSpec,
    builder: &mut AqlQueryBuilder,
) -> Option<String> {
    match edge_type {
        EdgeTypeSpec::Any => None,
        EdgeTypeSpec::Single(value) => {
            let key = builder.push_param(ScalarValue::String(value.clone()));
            Some(format!("FILTER edge.`_type` == {key}"))
        }
        EdgeTypeSpec::Multiple(values) => {
            if values.is_empty() {
                Some("FILTER false".to_owned())
            } else {
                let mut clauses = Vec::new();
                for value in values {
                    let key = builder.push_param(ScalarValue::String(value.clone()));
                    clauses.push(format!("edge.`_type` == {key}"));
                }
                Some(format!("FILTER {}", clauses.join(" OR ")))
            }
        }
    }
}

fn compile_node_projection(columns: Option<&[&str]>) -> String {
    let mut projections = vec![
        format!("{0}: v._id", quoted_ident(COL_NODE_ID)),
        format!("{0}: v.{0}", quoted_ident(COL_NODE_LABEL)),
    ];

    match columns {
        Some(columns) => {
            for column in user_projection_columns(columns, &[COL_NODE_ID, COL_NODE_LABEL]) {
                projections.push(format!("{0}: v.{0}", quoted_ident(column)));
            }
        }
        None => projections.push("props: UNSET(v, ['_id', '_key', '_rev'])".to_owned()),
    }

    format!("{{ {} }}", projections.join(", "))
}

fn compile_edge_projection(columns: Option<&[&str]>) -> String {
    let mut projections = vec![
        format!("{0}: e._from", quoted_ident(COL_EDGE_SRC)),
        format!("{0}: e._to", quoted_ident(COL_EDGE_DST)),
        format!("{0}: e.{0}", quoted_ident(COL_EDGE_TYPE)),
        format!("{0}: 0", quoted_ident(COL_EDGE_DIRECTION)),
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
                projections.push(format!("{0}: e.{0}", quoted_ident(column)));
            }
        }
        None => {
            projections.push("props: UNSET(e, ['_id', '_key', '_rev', '_from', '_to'])".to_owned())
        }
    }

    format!("{{ {} }}", projections.join(", "))
}

fn user_projection_columns<'a>(columns: &'a [&'a str], reserved: &[&str]) -> Vec<&'a str> {
    columns
        .iter()
        .copied()
        .filter(|column| !reserved.iter().any(|reserved| reserved == column))
        .collect()
}

fn format_filters(filters: &[String]) -> String {
    if filters.is_empty() {
        String::new()
    } else {
        filters.join("\n")
    }
}

fn traversal_direction(direction: Direction) -> &'static str {
    match direction {
        Direction::Out => "OUTBOUND",
        Direction::In => "INBOUND",
        Direction::Both | Direction::None => "ANY",
    }
}

fn quoted_ident(value: &str) -> String {
    format!("`{}`", value.replace('`', "``"))
}

fn scalar_to_aql_value(value: ScalarValue) -> AqlValue {
    match value {
        ScalarValue::Null => AqlValue::Null,
        ScalarValue::String(value) => AqlValue::String(value),
        ScalarValue::Int(value) => AqlValue::Int(value),
        ScalarValue::Float(value) => AqlValue::Float(value),
        ScalarValue::Bool(value) => AqlValue::Bool(value),
        ScalarValue::List(values) => {
            AqlValue::List(values.into_iter().map(scalar_to_aql_value).collect())
        }
    }
}

fn binary_op_symbol(op: &BinaryOp) -> &'static str {
    match op {
        BinaryOp::Eq => "==",
        BinaryOp::NotEq => "!=",
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
