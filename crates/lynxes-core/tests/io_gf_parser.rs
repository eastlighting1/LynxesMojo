use lynxes_core::{Direction, GFError, GFValue};
use lynxes_io::parse_gf;

// ── Minimal parsing (no schema) ───────────────────────────────────────────────

#[test]
fn parses_minimal_nodes_without_schema() {
    let src = r#"
        (alice :Person {})
        (bob   :Person {})
    "#;
    let doc = parse_gf(src).unwrap();
    assert_eq!(doc.nodes.len(), 2);
    assert_eq!(doc.nodes[0].id, "alice");
    assert_eq!(doc.nodes[1].id, "bob");
    assert!(doc.schema.nodes.is_empty());
}

#[test]
fn parses_node_with_integer_and_string_properties() {
    let src = r#"(alice :Person { age: 30, name: "Alice" })"#;
    let doc = parse_gf(src).unwrap();
    let node = &doc.nodes[0];
    assert_eq!(node.id, "alice");
    assert_eq!(node.props["age"], GFValue::Int(30));
    assert_eq!(node.props["name"], GFValue::String("Alice".to_owned()));
}

#[test]
fn parses_multiple_labels_on_node_using_pipe_separator() {
    let src = r#"(ceo :Person|Executive {})"#;
    let doc = parse_gf(src).unwrap();
    let labels = &doc.nodes[0].labels;
    assert!(labels.contains(&"Person".to_owned()));
    assert!(labels.contains(&"Executive".to_owned()));
}

#[test]
fn parses_node_without_label() {
    let src = r#"(lone {})"#;
    let doc = parse_gf(src).unwrap();
    assert_eq!(doc.nodes[0].id, "lone");
    assert!(doc.nodes[0].labels.is_empty());
}

// ── Edge direction parsing ────────────────────────────────────────────────────

#[test]
fn parses_out_directed_edge() {
    let src = r#"
        (a :X {})
        (b :X {})
        a -[KNOWS]-> b {}
    "#;
    let doc = parse_gf(src).unwrap();
    assert_eq!(doc.edges.len(), 1);
    assert_eq!(doc.edges[0].direction, Direction::Out);
    assert_eq!(doc.edges[0].src_id, "a");
    assert_eq!(doc.edges[0].dst_id, "b");
    assert_eq!(doc.edges[0].edge_type, "KNOWS");
}

#[test]
fn parses_in_directed_edge() {
    let src = r#"
        (a :X {})
        (b :X {})
        a <-[KNOWS]- b {}
    "#;
    let doc = parse_gf(src).unwrap();
    assert_eq!(doc.edges.len(), 1);
    assert_eq!(doc.edges[0].direction, Direction::In);
}

#[test]
fn parses_bidirectional_edge_as_two_rows() {
    let src = r#"
        (a :X {})
        (b :X {})
        a <-[KNOWS]-> b {}
    "#;
    let doc = parse_gf(src).unwrap();
    // Bidirectional creates one Out and one In row
    assert_eq!(doc.edges.len(), 2);
    let directions: Vec<Direction> = doc.edges.iter().map(|e| e.direction).collect();
    assert!(directions.contains(&Direction::Out));
    assert!(directions.contains(&Direction::In));
}

#[test]
fn parses_undirected_edge_as_none_direction() {
    let src = r#"
        (a :X {})
        (b :X {})
        a --[KNOWS]-- b {}
    "#;
    let doc = parse_gf(src).unwrap();
    assert_eq!(doc.edges.len(), 1);
    assert_eq!(doc.edges[0].direction, Direction::None);
}

// ── Schema parsing ────────────────────────────────────────────────────────────

#[test]
fn parses_node_schema_block() {
    let src = r#"
        node Person {
            age: Int
            name: String
        }
        (alice :Person { age: 30, name: "Alice" })
    "#;
    let doc = parse_gf(src).unwrap();
    assert!(doc.schema.nodes.contains_key("Person"));
    let schema = &doc.schema.nodes["Person"];
    assert!(schema.fields.iter().any(|f| f.name == "age"));
    assert!(schema.fields.iter().any(|f| f.name == "name"));
}

#[test]
fn parses_edge_schema_block() {
    let src = r#"
        edge KNOWS {
            since: Int
        }
        (a :X {})
        (b :X {})
        a -[KNOWS]-> b { since: 2020 }
    "#;
    let doc = parse_gf(src).unwrap();
    assert!(doc.schema.edges.contains_key("KNOWS"));
    let schema = &doc.schema.edges["KNOWS"];
    assert!(schema.fields.iter().any(|f| f.name == "since"));
}

// ── Multi-edge ────────────────────────────────────────────────────────────────

#[test]
fn parses_multiple_edges_between_same_pair() {
    let src = r#"
        (a :X {})
        (b :X {})
        a -[KNOWS]-> b {}
        a -[LIKES]-> b {}
    "#;
    let doc = parse_gf(src).unwrap();
    assert_eq!(doc.edges.len(), 2);
    let types: Vec<&str> = doc.edges.iter().map(|e| e.edge_type.as_str()).collect();
    assert!(types.contains(&"KNOWS"));
    assert!(types.contains(&"LIKES"));
}

#[test]
fn parses_edge_with_float_and_int_properties() {
    let src = r#"
        (a :X {})
        (b :X {})
        a -[KNOWS]-> b { since: 2020, weight: 1.5 }
    "#;
    let doc = parse_gf(src).unwrap();
    let edge = &doc.edges[0];
    assert_eq!(edge.props["since"], GFValue::Int(2020));
    assert_eq!(edge.props["weight"], GFValue::Float(1.5));
}

// ── Error cases ───────────────────────────────────────────────────────────────

#[test]
fn malformed_input_returns_parse_error() {
    let src = "this is not valid .gf syntax @@@@";
    let err = parse_gf(src).unwrap_err();
    assert!(matches!(err, GFError::ParseError { .. }));
}

#[test]
fn empty_input_parses_to_empty_document() {
    let doc = parse_gf("").unwrap();
    assert!(doc.nodes.is_empty());
    assert!(doc.edges.is_empty());
    assert!(doc.meta.is_empty());
}
