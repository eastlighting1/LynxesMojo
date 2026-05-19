use std::{collections::BTreeMap, sync::Arc};

use lynxes_core::{
    BinaryOp, Direction, EdgeFrame, EdgeTypeSpec, Expr, GFError, GraphFrame, NodeFrame, Result,
    ScalarValue, StringOp, UnaryOp, COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE,
    COL_NODE_ID, COL_NODE_LABEL,
};

use crate::connector::{Connector, ConnectorFuture, ExpandResult};

#[derive(Clone, PartialEq, Eq)]
pub struct SparqlConfig {
    pub endpoint: String,
    pub node_template: String,
    pub edge_template: String,
    pub expand_template: Option<String>,
}

impl std::fmt::Debug for SparqlConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SparqlConfig")
            .field("endpoint", &self.endpoint)
            .field("node_template", &self.node_template)
            .field("edge_template", &self.edge_template)
            .field("expand_template", &self.expand_template)
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SparqlValue {
    Null,
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    List(Vec<SparqlValue>),
}

pub type SparqlParams = BTreeMap<String, SparqlValue>;

#[derive(Debug, Clone, PartialEq)]
pub struct SparqlQuery {
    pub text: String,
    pub params: SparqlParams,
}

pub trait SparqlBackend: Send + Sync + std::fmt::Debug {
    fn load_nodes<'a>(&'a self, query: SparqlQuery) -> ConnectorFuture<'a, NodeFrame>;

    fn load_edges<'a>(&'a self, query: SparqlQuery) -> ConnectorFuture<'a, EdgeFrame>;

    fn expand<'a>(&'a self, query: SparqlQuery) -> ConnectorFuture<'a, ExpandResult>;

    fn write<'a>(&'a self, graph: &'a GraphFrame, query: SparqlQuery) -> ConnectorFuture<'a, ()> {
        let _ = (graph, query);
        Box::pin(async move {
            Err(GFError::UnsupportedOperation {
                message: "SPARQL backend write() is not implemented".to_owned(),
            })
        })
    }
}

#[derive(Debug)]
struct UnsupportedSparqlBackend;

impl SparqlBackend for UnsupportedSparqlBackend {
    fn load_nodes<'a>(&'a self, _query: SparqlQuery) -> ConnectorFuture<'a, NodeFrame> {
        Box::pin(async move {
            Err(GFError::ConnectorError {
                message: "SPARQL HTTP backend is not linked in this build".to_owned(),
            })
        })
    }

    fn load_edges<'a>(&'a self, _query: SparqlQuery) -> ConnectorFuture<'a, EdgeFrame> {
        Box::pin(async move {
            Err(GFError::ConnectorError {
                message: "SPARQL HTTP backend is not linked in this build".to_owned(),
            })
        })
    }

    fn expand<'a>(&'a self, _query: SparqlQuery) -> ConnectorFuture<'a, ExpandResult> {
        Box::pin(async move {
            Err(GFError::ConnectorError {
                message: "SPARQL HTTP backend is not linked in this build".to_owned(),
            })
        })
    }
}

#[derive(Debug, Clone)]
pub struct SparqlConnector {
    config: SparqlConfig,
    backend: Arc<dyn SparqlBackend>,
}

impl SparqlConnector {
    pub fn new(config: SparqlConfig) -> Self {
        Self {
            config,
            backend: Arc::new(UnsupportedSparqlBackend),
        }
    }

    pub fn with_backend(config: SparqlConfig, backend: Arc<dyn SparqlBackend>) -> Self {
        Self { config, backend }
    }

    pub fn config(&self) -> &SparqlConfig {
        &self.config
    }
}

impl Connector for SparqlConnector {
    fn cache_source_key(&self) -> Option<String> {
        Some(format!("sparql://{}", self.config.endpoint))
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
                message: "SparqlConnector write() is not implemented yet".to_owned(),
            })
        })
    }
}

fn finish_node_pages(mut pages: Vec<NodeFrame>) -> Result<NodeFrame> {
    if pages.is_empty() {
        return Err(GFError::ConnectorError {
            message: "SPARQL backend returned no node pages".to_owned(),
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
            message: "SPARQL backend returned no edge pages".to_owned(),
        });
    }
    if pages.len() == 1 {
        return Ok(pages.pop().expect("checked len == 1"));
    }
    let refs: Vec<&EdgeFrame> = pages.iter().collect();
    EdgeFrame::concat(&refs)
}

fn build_load_nodes_query(
    config: &SparqlConfig,
    labels: Option<&[&str]>,
    columns: Option<&[&str]>,
    predicate: Option<&Expr>,
    offset: usize,
    batch_size: usize,
) -> Result<SparqlQuery> {
    let mut builder = SparqlQueryBuilder::default();
    let mut filters = Vec::new();

    if let Some(label_filter) = compile_labels_filter(labels, &mut builder)? {
        filters.push(label_filter);
    }
    if let Some(predicate) = predicate {
        filters.push(builder.compile_expr(predicate, QueryDomain::Node { var: "?n" })?);
    }

    let projection = compile_node_projection(columns);
    let text = apply_template(
        &config.node_template,
        &projection,
        &join_filters(&filters),
        batch_size,
        offset,
    )?;

    Ok(SparqlQuery {
        text,
        params: builder.params,
    })
}

fn build_load_edges_query(
    config: &SparqlConfig,
    edge_types: Option<&[&str]>,
    columns: Option<&[&str]>,
    predicate: Option<&Expr>,
    offset: usize,
    batch_size: usize,
) -> Result<SparqlQuery> {
    let mut builder = SparqlQueryBuilder::default();
    let mut filters = Vec::new();

    if let Some(type_filter) = compile_edge_types_filter(edge_types, &mut builder)? {
        filters.push(type_filter);
    }
    if let Some(predicate) = predicate {
        filters.push(builder.compile_expr(predicate, QueryDomain::Edge { var: "?e" })?);
    }

    let projection = compile_edge_projection(columns);
    let text = apply_template(
        &config.edge_template,
        &projection,
        &join_filters(&filters),
        batch_size,
        offset,
    )?;

    Ok(SparqlQuery {
        text,
        params: builder.params,
    })
}

fn build_expand_query(
    config: &SparqlConfig,
    node_ids: &[&str],
    edge_type: &EdgeTypeSpec,
    hops: u32,
    direction: Direction,
    node_predicate: Option<&Expr>,
) -> Result<SparqlQuery> {
    if node_ids.is_empty() {
        return Err(GFError::InvalidConfig {
            message: "expand requires at least one seed node id".to_owned(),
        });
    }

    let template =
        config
            .expand_template
            .as_ref()
            .ok_or_else(|| GFError::UnsupportedOperation {
                message: "SparqlConnector expand() requires expand_template in SparqlConfig"
                    .to_owned(),
            })?;

    let mut builder = SparqlQueryBuilder::default();
    let mut filters = vec![compile_seed_filter(node_ids, &mut builder)];
    if let Some(type_filter) = compile_expand_edge_types_filter(edge_type, &mut builder)? {
        filters.push(type_filter);
    }
    if let Some(predicate) = node_predicate {
        filters.push(builder.compile_expr(predicate, QueryDomain::Node { var: "?m" })?);
    }

    let projection = format!(
        "?seed ?m ?e ?src ?dst ?etype {} \"{}\" AS ?direction",
        direction_path_clause(direction, hops),
        direction_label(direction),
    );
    let text = apply_template(template, &projection, &join_filters(&filters), 0, 0)?;

    Ok(SparqlQuery {
        text,
        params: builder.params,
    })
}

#[derive(Debug, Clone, Copy)]
enum QueryDomain<'a> {
    Node { var: &'a str },
    Edge { var: &'a str },
}

#[derive(Debug, Default)]
struct SparqlQueryBuilder {
    params: SparqlParams,
    next_param: usize,
}

impl SparqlQueryBuilder {
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
                "({} = {})",
                self.compile_expr(item, domain)?,
                self.compile_expr(expr, domain)?,
            )),
            Expr::Cast { .. } => Err(GFError::UnsupportedOperation {
                message: "SparqlConnector predicate pushdown does not support Cast".to_owned(),
            }),
            Expr::And { left, right } => Ok(format!(
                "(({}) && ({}))",
                self.compile_expr(left, domain)?,
                self.compile_expr(right, domain)?,
            )),
            Expr::Or { left, right } => Ok(format!(
                "(({}) || ({}))",
                self.compile_expr(left, domain)?,
                self.compile_expr(right, domain)?,
            )),
            Expr::Not { expr } => Ok(format!("(!({}))", self.compile_expr(expr, domain)?)),
            Expr::PatternCol { alias, field } => Err(GFError::UnsupportedOperation {
                message: format!(
                    "SparqlConnector predicate pushdown does not support PatternCol({alias}.{field})"
                ),
            }),
            Expr::StringOp { op, expr, pattern } => {
                let subject = self.compile_expr(expr, domain)?;
                let pat = self.compile_expr(pattern, domain)?;
                // SPARQL 1.1 string functions
                let sparql = match op {
                    StringOp::Contains => format!("CONTAINS({subject}, {pat})"),
                    StringOp::StartsWith => format!("STRSTARTS({subject}, {pat})"),
                    StringOp::EndsWith => format!("STRENDS({subject}, {pat})"),
                };
                Ok(format!("({sparql})"))
            }
        }
    }

    fn compile_column(&self, name: &str, domain: QueryDomain<'_>) -> Result<String> {
        match domain {
            QueryDomain::Node { var } => match name {
                COL_NODE_ID => Ok(format!("{var}Id")),
                COL_NODE_LABEL => Ok(format!("{var}Label")),
                COL_EDGE_SRC | COL_EDGE_DST | COL_EDGE_TYPE | COL_EDGE_DIRECTION => {
                    Err(GFError::TypeMismatch {
                        message: format!("node predicate cannot reference edge column {name}"),
                    })
                }
                other => Ok(format!("{var}{}", sparql_ident(other))),
            },
            QueryDomain::Edge { var } => match name {
                COL_EDGE_SRC => Ok("?src".to_owned()),
                COL_EDGE_DST => Ok("?dst".to_owned()),
                COL_EDGE_TYPE => Ok("?etype".to_owned()),
                COL_EDGE_DIRECTION => Ok("\"out\"".to_owned()),
                COL_NODE_ID | COL_NODE_LABEL => Err(GFError::TypeMismatch {
                    message: format!("edge predicate cannot reference node column {name}"),
                }),
                other => Ok(format!("{var}{}", sparql_ident(other))),
            },
        }
    }

    fn push_param(&mut self, value: ScalarValue) -> String {
        let key = format!("p{}", self.next_param);
        self.next_param += 1;
        self.params
            .insert(key.clone(), scalar_to_sparql_value(value.clone()));
        sparql_literal(&value)
    }
}

fn compile_labels_filter(
    labels: Option<&[&str]>,
    builder: &mut SparqlQueryBuilder,
) -> Result<Option<String>> {
    let labels = match labels {
        Some(labels) => labels,
        None => return Ok(None),
    };
    if labels.is_empty() {
        return Ok(Some("false".to_owned()));
    }

    let mut clauses = Vec::new();
    for label in labels {
        let value = builder.push_param(ScalarValue::String((*label).to_owned()));
        clauses.push(format!("?nLabel = {value}"));
    }
    Ok(Some(clauses.join(" || ")))
}

fn compile_edge_types_filter(
    edge_types: Option<&[&str]>,
    builder: &mut SparqlQueryBuilder,
) -> Result<Option<String>> {
    let edge_types = match edge_types {
        Some(edge_types) => edge_types,
        None => return Ok(None),
    };
    if edge_types.is_empty() {
        return Ok(Some("false".to_owned()));
    }

    let mut clauses = Vec::new();
    for edge_type in edge_types {
        let value = builder.push_param(ScalarValue::String((*edge_type).to_owned()));
        clauses.push(format!("?etype = {value}"));
    }
    Ok(Some(clauses.join(" || ")))
}

fn compile_expand_edge_types_filter(
    edge_type: &EdgeTypeSpec,
    builder: &mut SparqlQueryBuilder,
) -> Result<Option<String>> {
    match edge_type {
        EdgeTypeSpec::Any => Ok(None),
        EdgeTypeSpec::Single(value) => {
            let value = builder.push_param(ScalarValue::String(value.clone()));
            Ok(Some(format!("?etype = {value}")))
        }
        EdgeTypeSpec::Multiple(values) => {
            if values.is_empty() {
                Ok(Some("false".to_owned()))
            } else {
                let clauses = values
                    .iter()
                    .map(|value| {
                        let value = builder.push_param(ScalarValue::String(value.clone()));
                        format!("?etype = {value}")
                    })
                    .collect::<Vec<_>>();
                Ok(Some(clauses.join(" || ")))
            }
        }
    }
}

fn compile_seed_filter(node_ids: &[&str], builder: &mut SparqlQueryBuilder) -> String {
    let clauses = node_ids
        .iter()
        .map(|node_id| {
            let value = builder.push_param(ScalarValue::String((*node_id).to_owned()));
            format!("?seedId = {value}")
        })
        .collect::<Vec<_>>();
    clauses.join(" || ")
}

fn compile_node_projection(columns: Option<&[&str]>) -> String {
    let mut projections = vec!["?nId AS ?_id".to_owned(), "?nLabel AS ?_label".to_owned()];
    if let Some(columns) = columns {
        for column in user_projection_columns(columns, &[COL_NODE_ID, COL_NODE_LABEL]) {
            projections.push(format!(
                "?n{} AS ?{}",
                sparql_ident(column),
                sparql_ident(column)
            ));
        }
    }
    projections.join(" ")
}

fn compile_edge_projection(columns: Option<&[&str]>) -> String {
    let mut projections = vec![
        "?src AS ?_src".to_owned(),
        "?dst AS ?_dst".to_owned(),
        "?etype AS ?_type".to_owned(),
        "\"out\" AS ?_direction".to_owned(),
    ];
    if let Some(columns) = columns {
        for column in user_projection_columns(
            columns,
            &[
                COL_EDGE_SRC,
                COL_EDGE_DST,
                COL_EDGE_TYPE,
                COL_EDGE_DIRECTION,
            ],
        ) {
            projections.push(format!(
                "?e{} AS ?{}",
                sparql_ident(column),
                sparql_ident(column)
            ));
        }
    }
    projections.join(" ")
}

fn user_projection_columns<'a>(columns: &'a [&'a str], reserved: &[&str]) -> Vec<&'a str> {
    columns
        .iter()
        .copied()
        .filter(|column| !reserved.iter().any(|reserved| reserved == column))
        .collect()
}

fn join_filters(filters: &[String]) -> String {
    if filters.is_empty() {
        String::new()
    } else {
        filters
            .iter()
            .map(|filter| format!("FILTER ({filter})"))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn apply_template(
    template: &str,
    projection: &str,
    filters: &str,
    limit: usize,
    offset: usize,
) -> Result<String> {
    if !template.contains("{{projection}}") || !template.contains("{{filters}}") {
        return Err(GFError::InvalidConfig {
            message: "SPARQL template must contain {{projection}} and {{filters}} placeholders"
                .to_owned(),
        });
    }

    Ok(template
        .replace("{{projection}}", projection)
        .replace("{{filters}}", filters)
        .replace("{{limit}}", &limit.to_string())
        .replace("{{offset}}", &offset.to_string()))
}

fn direction_path_clause(direction: Direction, hops: u32) -> String {
    let path = format!("?p{{1,{hops}}}");
    match direction {
        Direction::Out => format!("?seed {path} ?m ."),
        Direction::In => format!("?m {path} ?seed ."),
        Direction::Both | Direction::None => {
            format!("{{ ?seed {path} ?m . }} UNION {{ ?m {path} ?seed . }}")
        }
    }
}

fn direction_label(direction: Direction) -> &'static str {
    match direction {
        Direction::Out => "out",
        Direction::In => "in",
        Direction::Both | Direction::None => "both",
    }
}

fn sparql_ident(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
}

fn sparql_literal(value: &ScalarValue) -> String {
    match value {
        ScalarValue::Null => "UNDEF".to_owned(),
        ScalarValue::String(value) => format!("\"{}\"", value.replace('"', "\\\"")),
        ScalarValue::Int(value) => value.to_string(),
        ScalarValue::Float(value) => value.to_string(),
        ScalarValue::Bool(value) => value.to_string(),
        ScalarValue::List(values) => {
            let rendered = values
                .iter()
                .map(sparql_literal)
                .collect::<Vec<_>>()
                .join(", ");
            format!("({rendered})")
        }
    }
}

fn scalar_to_sparql_value(value: ScalarValue) -> SparqlValue {
    match value {
        ScalarValue::Null => SparqlValue::Null,
        ScalarValue::String(value) => SparqlValue::String(value),
        ScalarValue::Int(value) => SparqlValue::Int(value),
        ScalarValue::Float(value) => SparqlValue::Float(value),
        ScalarValue::Bool(value) => SparqlValue::Bool(value),
        ScalarValue::List(values) => {
            SparqlValue::List(values.into_iter().map(scalar_to_sparql_value).collect())
        }
    }
}

fn binary_op_symbol(op: &BinaryOp) -> &'static str {
    match op {
        BinaryOp::Eq => "=",
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
