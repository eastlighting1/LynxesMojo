use std::{fs, path::PathBuf};

use lynxes_connect::{Connector, GFConnector};
use lynxes_core::{BinaryOp, EdgeTypeSpec, Expr, GFError, ScalarValue, COL_EDGE_TYPE, COL_NODE_ID};
use lynxes_io::{parse_gf, read_gfb, write_gfb, GfbWriteOptions};

const SAMPLE_GF: &str = r#"
(alice:Person|Employee { age: 30 })
(bob:Person { age: 20 })
(charlie:Person { age: 25 })
(acme:Company { size: 100 })
alice -[KNOWS]-> bob { weight: 1 }
bob -[KNOWS]-> charlie { weight: 2 }
bob -[WORKS_AT]-> acme { weight: 3 }
"#;

#[tokio::test]
async fn gf_connector_loads_and_filters_nodes_from_gf() {
    let path = temp_path("connector-load-nodes.gf");
    fs::write(&path, SAMPLE_GF).unwrap();
    let connector = GFConnector::new(&path).unwrap();

    let predicate = Expr::BinaryOp {
        left: Box::new(Expr::Col {
            name: "age".to_owned(),
        }),
        op: BinaryOp::Gt,
        right: Box::new(Expr::Literal {
            value: ScalarValue::Int(21),
        }),
    };

    let nodes = connector
        .load_nodes(Some(&["Person"]), Some(&["age"]), Some(&predicate), 1024)
        .await
        .unwrap();

    assert_eq!(
        nodes.column_names(),
        vec![COL_NODE_ID, lynxes_core::COL_NODE_LABEL, "age"]
    );
    assert_eq!(nodes.len(), 2);
    assert!(nodes.row_index("alice").is_some());
    assert!(nodes.row_index("charlie").is_some());

    let _ = fs::remove_file(path);
}

#[tokio::test]
async fn gf_connector_loads_edges_from_gf_with_projection() {
    let path = temp_path("connector-load-edges.gf");
    fs::write(&path, SAMPLE_GF).unwrap();
    let connector = GFConnector::new(&path).unwrap();

    let predicate = Expr::BinaryOp {
        left: Box::new(Expr::Col {
            name: COL_EDGE_TYPE.to_owned(),
        }),
        op: BinaryOp::Eq,
        right: Box::new(Expr::Literal {
            value: ScalarValue::String("KNOWS".to_owned()),
        }),
    };

    let edges = connector
        .load_edges(Some(&["KNOWS"]), Some(&["weight"]), Some(&predicate), 1024)
        .await
        .unwrap();

    assert_eq!(
        edges.column_names(),
        vec![
            lynxes_core::COL_EDGE_SRC,
            lynxes_core::COL_EDGE_DST,
            lynxes_core::COL_EDGE_TYPE,
            lynxes_core::COL_EDGE_DIRECTION,
            "weight",
        ]
    );
    assert_eq!(edges.len(), 2);

    let _ = fs::remove_file(path);
}

#[tokio::test]
async fn gf_connector_expand_uses_local_graph_semantics() {
    let path = temp_path("connector-expand.gf");
    fs::write(&path, SAMPLE_GF).unwrap();
    let connector = GFConnector::new(&path).unwrap();

    let (nodes, edges) = connector
        .expand(
            &["alice"],
            &EdgeTypeSpec::Single("KNOWS".to_owned()),
            2,
            lynxes_core::Direction::Out,
            None,
        )
        .await
        .unwrap();

    assert_eq!(nodes.len(), 3);
    assert_eq!(edges.len(), 2);
    assert!(nodes.row_index("alice").is_some());
    assert!(nodes.row_index("bob").is_some());
    assert!(nodes.row_index("charlie").is_some());

    let _ = fs::remove_file(path);
}

#[tokio::test]
async fn gf_connector_reads_gfb_and_writes_gfb() {
    let input = temp_path("connector-roundtrip-input.gf");
    let output = temp_path("connector-roundtrip-output.gfb");
    fs::write(&input, SAMPLE_GF).unwrap();

    let source_graph = parse_gf(SAMPLE_GF).unwrap().to_graph_frame().unwrap();
    write_gfb(&source_graph, &output, &GfbWriteOptions::default()).unwrap();

    let connector = GFConnector::new(&output).unwrap();
    let nodes = connector
        .load_nodes(None, Some(&[]), None, 256)
        .await
        .unwrap();
    assert_eq!(
        nodes.column_names(),
        vec![COL_NODE_ID, lynxes_core::COL_NODE_LABEL]
    );

    let write_target = temp_path("connector-write-target.gfb");
    let writer = GFConnector::from_gfb(&write_target);
    writer.write(&source_graph).await.unwrap();
    let roundtrip = read_gfb(&write_target).unwrap();
    assert_eq!(roundtrip.node_count(), source_graph.node_count());
    assert_eq!(roundtrip.edge_count(), source_graph.edge_count());

    let _ = fs::remove_file(input);
    let _ = fs::remove_file(output);
    let _ = fs::remove_file(write_target);
}

#[tokio::test]
async fn gf_connector_rejects_non_gfb_write_targets() {
    let target = temp_path("connector-write-target.gf");
    let connector = GFConnector::from_gf(&target);
    let graph = parse_gf(SAMPLE_GF).unwrap().to_graph_frame().unwrap();

    let err = connector.write(&graph).await.unwrap_err();
    assert!(matches!(err, GFError::UnsupportedOperation { .. }));
}

fn temp_path(name: &str) -> PathBuf {
    let pid = std::process::id();
    match name.rsplit_once('.') {
        Some((stem, ext)) => std::env::temp_dir().join(format!("lynxes-{stem}-{pid}.{ext}")),
        None => std::env::temp_dir().join(format!("lynxes-{name}-{pid}")),
    }
}
