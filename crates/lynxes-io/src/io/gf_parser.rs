use std::collections::{BTreeMap, HashMap, HashSet};

use pest::iterators::Pair;
use pest::Parser;
use pest_derive::Parser;

use lynxes_core::{
    Direction, EdgeSchema, FieldDef, GFError, GFType, GFValue, NodeSchema, Result, Schema,
};

#[derive(Parser)]
#[grammar = "io/gf.pest"]
struct GfParser;

/// Parsed node declaration from a `.gf` document.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedNodeDecl {
    pub id: String,
    pub labels: Vec<String>,
    pub props: BTreeMap<String, GFValue>,
}

/// Parsed edge row after direction normalization/materialization.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedEdgeDecl {
    pub src_id: String,
    pub dst_id: String,
    pub edge_type: String,
    pub direction: Direction,
    pub props: BTreeMap<String, GFValue>,
}

/// Full `.gf` parse result used by later frame-building stages.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ParsedGfDocument {
    pub meta: HashMap<String, GFValue>,
    pub imports: Vec<String>,
    pub schema: Schema,
    pub nodes: Vec<ParsedNodeDecl>,
    pub edges: Vec<ParsedEdgeDecl>,
}

pub fn parse_gf(source: &str) -> Result<ParsedGfDocument> {
    let mut pairs = GfParser::parse(Rule::file, source).map_err(parse_error)?;
    let file = pairs.next().ok_or_else(|| GFError::ParseError {
        message: "missing file root".to_owned(),
    })?;

    let mut document = ParsedGfDocument::default();

    for item in file.into_inner() {
        match item.as_rule() {
            Rule::meta_block => merge_meta_block(&mut document.meta, item)?,
            Rule::namespace_block => merge_namespace_block(&mut document.schema.namespaces, item)?,
            Rule::import_stmt => document.imports.push(parse_import(item)?),
            Rule::node_schema => {
                let schema = parse_node_schema(item)?;
                document.schema.nodes.insert(schema.label.clone(), schema);
            }
            Rule::edge_schema => {
                let schema = parse_edge_schema(item)?;
                document
                    .schema
                    .edges
                    .insert(schema.type_name.clone(), schema);
            }
            Rule::node_decl => document.nodes.push(parse_node_decl(item)?),
            Rule::edge_decl => document.edges.extend(parse_edge_decl(item)?),
            Rule::EOI | Rule::comment | Rule::bom => {}
            other => {
                return Err(GFError::ParseError {
                    message: format!("unexpected top-level rule: {other:?}"),
                });
            }
        }
    }

    Ok(document)
}

fn merge_meta_block(meta: &mut HashMap<String, GFValue>, pair: Pair<'_, Rule>) -> Result<()> {
    let object = pair
        .into_inner()
        .find(|inner| inner.as_rule() == Rule::object)
        .ok_or_else(|| GFError::ParseError {
            message: "meta block missing object".to_owned(),
        })?;

    for (key, value) in parse_object(object)? {
        meta.insert(key, value);
    }

    Ok(())
}

fn merge_namespace_block(
    namespaces: &mut HashMap<String, String>,
    pair: Pair<'_, Rule>,
) -> Result<()> {
    let object = pair
        .into_inner()
        .find(|inner| inner.as_rule() == Rule::object)
        .ok_or_else(|| GFError::ParseError {
            message: "namespace block missing object".to_owned(),
        })?;

    for (key, value) in parse_object(object)? {
        match value {
            GFValue::String(value) => {
                namespaces.insert(key, value);
            }
            other => {
                return Err(GFError::ParseError {
                    message: format!("namespace value for {key} must be string, got {other:?}"),
                });
            }
        }
    }

    Ok(())
}

fn parse_import(pair: Pair<'_, Rule>) -> Result<String> {
    let string = pair
        .into_inner()
        .find(|inner| inner.as_rule() == Rule::string)
        .ok_or_else(|| GFError::ParseError {
            message: "import missing string literal".to_owned(),
        })?;
    parse_string(string)
}

fn parse_node_schema(pair: Pair<'_, Rule>) -> Result<NodeSchema> {
    let mut label = None;
    let mut extends = None;
    let mut fields = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::label => label = Some(inner.as_str().to_owned()),
            Rule::extends_clause => extends = Some(parse_extends_clause(inner)?),
            Rule::field_def => fields.push(parse_field_def(inner)?),
            _ => {}
        }
    }

    Ok(NodeSchema {
        label: required(label, "node schema label")?,
        fields,
        extends,
    })
}

fn parse_edge_schema(pair: Pair<'_, Rule>) -> Result<EdgeSchema> {
    let mut type_name = None;
    let mut fields = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::edge_type => type_name = Some(inner.as_str().to_owned()),
            Rule::field_def => fields.push(parse_field_def(inner)?),
            _ => {}
        }
    }

    Ok(EdgeSchema {
        type_name: required(type_name, "edge schema type")?,
        fields,
    })
}

fn parse_extends_clause(pair: Pair<'_, Rule>) -> Result<String> {
    pair.into_inner()
        .find(|inner| inner.as_rule() == Rule::label)
        .map(|inner| inner.as_str().to_owned())
        .ok_or_else(|| GFError::ParseError {
            message: "extends clause missing label".to_owned(),
        })
}

fn parse_field_def(pair: Pair<'_, Rule>) -> Result<FieldDef> {
    let mut name = None;
    let mut dtype = None;
    let mut unique = false;
    let mut indexed = false;
    let mut default = None;
    let mut seen_directives = HashSet::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::ident => name = Some(inner.as_str().to_owned()),
            Rule::type_expr => dtype = Some(parse_type_expr(inner)?),
            Rule::unique_directive => {
                ensure_unique_directive(&mut seen_directives, "unique", inner.as_str())?;
                unique = true;
            }
            Rule::index_directive => {
                ensure_unique_directive(&mut seen_directives, "index", inner.as_str())?;
                indexed = true;
            }
            Rule::default_directive => {
                ensure_unique_directive(&mut seen_directives, "default", inner.as_str())?;
                default = Some(parse_default_directive(inner)?);
            }
            _ => {}
        }
    }

    Ok(FieldDef {
        name: required(name, "field name")?,
        dtype: required(dtype, "field type")?,
        unique,
        indexed,
        default,
    })
}

fn ensure_unique_directive(
    seen: &mut HashSet<&'static str>,
    kind: &'static str,
    raw: &str,
) -> Result<()> {
    if !seen.insert(kind) {
        return Err(GFError::ParseError {
            message: format!("duplicate directive: {raw}"),
        });
    }
    Ok(())
}

fn parse_default_directive(pair: Pair<'_, Rule>) -> Result<GFValue> {
    let value = pair
        .into_inner()
        .next()
        .ok_or_else(|| GFError::ParseError {
            message: "default directive missing value".to_owned(),
        })?;
    parse_value(value)
}

fn parse_type_expr(pair: Pair<'_, Rule>) -> Result<GFType> {
    let text = pair.as_str().trim();
    parse_type_expr_text(text)
}

fn parse_type_expr_text(text: &str) -> Result<GFType> {
    if let Some(inner) = text.strip_suffix('?') {
        let inner_type = parse_type_expr_text(inner.trim())?;
        if matches!(inner_type, GFType::Optional(_)) {
            return Err(GFError::InvalidType {
                message: format!("nested optional is not allowed: {text}"),
            });
        }
        return Ok(GFType::Optional(Box::new(inner_type)));
    }

    if text == "List" {
        return Ok(GFType::List(Box::new(GFType::Any)));
    }

    if text.starts_with("List<") && text.ends_with('>') {
        let inner = &text[5..text.len() - 1];
        let inner_type = parse_type_expr_text(inner.trim())?;
        if matches!(inner_type, GFType::Optional(_)) {
            return Err(GFError::InvalidType {
                message: format!("List<Optional<T>> is not allowed: {text}"),
            });
        }
        return Ok(GFType::List(Box::new(inner_type)));
    }

    match text {
        "String" => Ok(GFType::String),
        "Int" => Ok(GFType::Int),
        "Float" => Ok(GFType::Float),
        "Bool" => Ok(GFType::Bool),
        "Date" => Ok(GFType::Date),
        "DateTime" => Ok(GFType::DateTime),
        "Duration" => Ok(GFType::Duration),
        "Any" => Ok(GFType::Any),
        other => Err(GFError::InvalidType {
            message: format!("unknown type expression: {other}"),
        }),
    }
}

fn parse_node_decl(pair: Pair<'_, Rule>) -> Result<ParsedNodeDecl> {
    let mut id = None;
    let mut labels = Vec::new();
    let mut props = BTreeMap::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::node_id => id = Some(inner.as_str().to_owned()),
            Rule::node_label_clause => labels = parse_label_clause(inner),
            Rule::props => props = parse_props(inner)?,
            _ => {}
        }
    }

    Ok(ParsedNodeDecl {
        id: required(id, "node id")?,
        labels,
        props,
    })
}

fn parse_label_clause(pair: Pair<'_, Rule>) -> Vec<String> {
    pair.into_inner()
        .filter(|inner| inner.as_rule() == Rule::label_list)
        .flat_map(|list| {
            list.into_inner()
                .filter(|label| label.as_rule() == Rule::label)
                .map(|label| label.as_str().to_owned())
                .collect::<Vec<_>>()
        })
        .collect()
}

fn parse_edge_decl(pair: Pair<'_, Rule>) -> Result<Vec<ParsedEdgeDecl>> {
    let mut node_refs = Vec::with_capacity(2);
    let mut edge_type = None;
    let mut edge_kind = None;
    let mut props = BTreeMap::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::node_ref => node_refs.push(inner.as_str().to_owned()),
            Rule::out_edge | Rule::in_edge | Rule::bi_edge | Rule::undirected_edge => {
                let (kind, ty) = parse_edge_body(inner)?;
                edge_kind = Some(kind);
                edge_type = Some(ty);
            }
            Rule::props => props = parse_props(inner)?,
            _ => {}
        }
    }

    if node_refs.len() != 2 {
        return Err(GFError::ParseError {
            message: format!(
                "edge declaration expected 2 node refs, got {}",
                node_refs.len()
            ),
        });
    }

    let left = node_refs[0].clone();
    let right = node_refs[1].clone();
    let edge_type = required(edge_type, "edge type")?;
    let edge_kind = required(edge_kind, "edge direction")?;

    Ok(match edge_kind {
        EdgeKind::Out => vec![ParsedEdgeDecl {
            src_id: left,
            dst_id: right,
            edge_type,
            direction: Direction::Out,
            props,
        }],
        EdgeKind::In => vec![ParsedEdgeDecl {
            src_id: right,
            dst_id: left,
            edge_type,
            direction: Direction::In,
            props,
        }],
        EdgeKind::Bi => vec![
            ParsedEdgeDecl {
                src_id: left.clone(),
                dst_id: right.clone(),
                edge_type: edge_type.clone(),
                direction: Direction::Out,
                props: props.clone(),
            },
            ParsedEdgeDecl {
                src_id: right,
                dst_id: left,
                edge_type,
                direction: Direction::In,
                props,
            },
        ],
        EdgeKind::Undirected => vec![ParsedEdgeDecl {
            src_id: left,
            dst_id: right,
            edge_type,
            direction: Direction::None,
            props,
        }],
    })
}

fn parse_edge_body(pair: Pair<'_, Rule>) -> Result<(EdgeKind, String)> {
    let kind = match pair.as_rule() {
        Rule::out_edge => EdgeKind::Out,
        Rule::in_edge => EdgeKind::In,
        Rule::bi_edge => EdgeKind::Bi,
        Rule::undirected_edge => EdgeKind::Undirected,
        other => {
            return Err(GFError::ParseError {
                message: format!("unexpected edge body rule: {other:?}"),
            });
        }
    };

    let edge_type = pair
        .into_inner()
        .find(|part| part.as_rule() == Rule::edge_type)
        .map(|part| part.as_str().to_owned())
        .ok_or_else(|| GFError::ParseError {
            message: "edge body missing edge type".to_owned(),
        })?;

    Ok((kind, edge_type))
}

fn parse_props(pair: Pair<'_, Rule>) -> Result<BTreeMap<String, GFValue>> {
    let object = pair
        .into_inner()
        .find(|inner| inner.as_rule() == Rule::object)
        .ok_or_else(|| GFError::ParseError {
            message: "props missing object".to_owned(),
        })?;
    parse_object(object)
}

fn parse_object(pair: Pair<'_, Rule>) -> Result<BTreeMap<String, GFValue>> {
    let mut entries = BTreeMap::new();

    for inner in pair.into_inner() {
        if inner.as_rule() != Rule::pair_list {
            continue;
        }
        for pair in inner.into_inner() {
            if pair.as_rule() != Rule::pair {
                continue;
            }

            let mut inner = pair.into_inner();
            let key = inner
                .next()
                .ok_or_else(|| GFError::ParseError {
                    message: "object pair missing key".to_owned(),
                })?
                .as_str()
                .to_owned();
            let value = parse_value(inner.next().ok_or_else(|| GFError::ParseError {
                message: format!("object pair missing value for key {key}"),
            })?)?;
            entries.insert(key, value);
        }
    }

    Ok(entries)
}

fn parse_value(pair: Pair<'_, Rule>) -> Result<GFValue> {
    Ok(match pair.as_rule() {
        Rule::value => {
            let inner = pair
                .into_inner()
                .next()
                .ok_or_else(|| GFError::ParseError {
                    message: "value missing literal".to_owned(),
                })?;
            return parse_value(inner);
        }
        Rule::string => GFValue::String(parse_string(pair)?),
        Rule::integer => {
            GFValue::Int(pair.as_str().parse().map_err(|err| GFError::ParseError {
                message: format!("invalid integer {}: {err}", pair.as_str()),
            })?)
        }
        Rule::float => {
            GFValue::Float(pair.as_str().parse().map_err(|err| GFError::ParseError {
                message: format!("invalid float {}: {err}", pair.as_str()),
            })?)
        }
        Rule::boolean => GFValue::Bool(pair.as_str() == "true"),
        Rule::null => GFValue::Null,
        Rule::date => GFValue::Date(pair.as_str().to_owned()),
        Rule::datetime => GFValue::DateTime(pair.as_str().to_owned()),
        Rule::list => {
            let mut values = Vec::new();
            for inner in pair.into_inner() {
                if inner.as_rule() != Rule::value_list {
                    continue;
                }
                for item in inner.into_inner() {
                    values.push(parse_value(item)?);
                }
            }
            GFValue::List(values)
        }
        Rule::object => GFValue::Object(parse_object(pair)?),
        other => {
            return Err(GFError::ParseError {
                message: format!("unexpected value rule: {other:?}"),
            });
        }
    })
}

fn parse_string(pair: Pair<'_, Rule>) -> Result<String> {
    serde_json::from_str(pair.as_str()).map_err(|err| GFError::ParseError {
        message: format!("invalid string literal {}: {err}", pair.as_str()),
    })
}

fn required<T>(value: Option<T>, label: &str) -> Result<T> {
    value.ok_or_else(|| GFError::ParseError {
        message: format!("missing {label}"),
    })
}

fn parse_error(err: pest::error::Error<Rule>) -> GFError {
    GFError::ParseError {
        message: err.to_string(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EdgeKind {
    Out,
    In,
    Bi,
    Undirected,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_meta_namespace_and_imports() {
        let doc = parse_gf(
            r#"
            @meta { owner: "team-a", version: 1 }
            @meta { version: 2, active: true }
            @namespace { ex: "https://example.com/" }
            @import "common.gf"
            "#,
        )
        .unwrap();

        assert_eq!(
            doc.meta.get("owner"),
            Some(&GFValue::String("team-a".to_owned()))
        );
        assert_eq!(doc.meta.get("version"), Some(&GFValue::Int(2)));
        assert_eq!(doc.meta.get("active"), Some(&GFValue::Bool(true)));
        assert_eq!(
            doc.schema.namespaces.get("ex"),
            Some(&"https://example.com/".to_owned())
        );
        assert_eq!(doc.imports, vec!["common.gf".to_owned()]);
    }

    #[test]
    fn parses_schema_blocks_with_directives_and_extends() {
        let doc = parse_gf(
            r#"
            node Person extends Entity {
                name: String @unique
                age: Int @index @default(42)
                tags: List<String>
            }

            edge KNOWS {
                since: Date
                weight: Float?
            }
            "#,
        )
        .unwrap();

        let person = doc.schema.nodes.get("Person").unwrap();
        assert_eq!(person.extends.as_deref(), Some("Entity"));
        assert_eq!(person.fields.len(), 3);
        assert!(person.fields[0].unique);
        assert!(person.fields[1].indexed);
        assert_eq!(person.fields[1].default, Some(GFValue::Int(42)));

        let knows = doc.schema.edges.get("KNOWS").unwrap();
        assert_eq!(knows.fields.len(), 2);
        assert_eq!(
            knows.fields[1].dtype,
            GFType::Optional(Box::new(GFType::Float))
        );
    }

    #[test]
    fn rejects_duplicate_field_directives() {
        let err = parse_gf(
            r#"
            node Person {
                name: String @index @index
            }
            "#,
        )
        .unwrap_err();

        assert!(matches!(err, GFError::ParseError { .. }));
        assert!(err.to_string().contains("duplicate directive"));
    }

    #[test]
    fn parses_node_declarations() {
        let doc = parse_gf(
            r#"
            (alice:Person|Employee { age: 30, active: true, tags: ["rust", "arrow"] })
            "#,
        )
        .unwrap();

        assert_eq!(doc.nodes.len(), 1);
        assert_eq!(doc.nodes[0].id, "alice");
        assert_eq!(
            doc.nodes[0].labels,
            vec!["Person".to_owned(), "Employee".to_owned()]
        );
        assert_eq!(doc.nodes[0].props.get("age"), Some(&GFValue::Int(30)));
    }

    #[test]
    fn parses_all_edge_directions() {
        let doc = parse_gf(
            r#"
            alice -[KNOWS]-> bob
            alice <-[LIKES]- bob
            alice <-[FRIEND]-> bob
            alice --[PEERS]-- bob
            "#,
        )
        .unwrap();

        assert_eq!(doc.edges.len(), 5);
        assert_eq!(doc.edges[0].direction, Direction::Out);
        assert_eq!(doc.edges[0].src_id, "alice");
        assert_eq!(doc.edges[0].dst_id, "bob");

        assert_eq!(doc.edges[1].direction, Direction::In);
        assert_eq!(doc.edges[1].src_id, "bob");
        assert_eq!(doc.edges[1].dst_id, "alice");

        assert_eq!(doc.edges[2].direction, Direction::Out);
        assert_eq!(doc.edges[3].direction, Direction::In);
        assert_eq!(doc.edges[4].direction, Direction::None);
    }

    #[test]
    fn bidirectional_edges_split_into_two_rows_with_shared_props() {
        let doc = parse_gf(
            r#"
            alice <-[FRIEND]-> bob { since: 2024-01-01 }
            "#,
        )
        .unwrap();

        assert_eq!(doc.edges.len(), 2);
        assert_eq!(doc.edges[0].src_id, "alice");
        assert_eq!(doc.edges[0].dst_id, "bob");
        assert_eq!(doc.edges[0].direction, Direction::Out);
        assert_eq!(doc.edges[1].src_id, "bob");
        assert_eq!(doc.edges[1].dst_id, "alice");
        assert_eq!(doc.edges[1].direction, Direction::In);
        assert_eq!(
            doc.edges[0].props.get("since"),
            Some(&GFValue::Date("2024-01-01".to_owned()))
        );
        assert_eq!(doc.edges[0].props, doc.edges[1].props);
    }

    #[test]
    fn malformed_input_returns_parse_error() {
        let err = parse_gf("node Person { name: String ").unwrap_err();
        assert!(matches!(err, GFError::ParseError { .. }));
    }
}
