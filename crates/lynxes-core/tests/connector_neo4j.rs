mod common;

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use arrow_array::BooleanArray;
use lynxes_connect::{
    Connector, ConnectorFuture, CypherQuery, CypherValue, ExpandResult, Neo4jBackend, Neo4jConfig,
    Neo4jConnector,
};
use lynxes_core::{
    BinaryOp, Direction, EdgeFrame, EdgeTypeSpec, Expr, GFError, NodeFrame, ScalarValue,
    COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
};

use common::{another_two_node_batch, edge_batch_with_since, two_node_batch};

#[derive(Debug, Default)]
struct MockNeo4jBackend {
    node_pages: Mutex<VecDeque<NodeFrame>>,
    edge_pages: Mutex<VecDeque<EdgeFrame>>,
    expand_results: Mutex<VecDeque<(NodeFrame, EdgeFrame)>>,
    node_queries: Mutex<Vec<CypherQuery>>,
    edge_queries: Mutex<Vec<CypherQuery>>,
    expand_queries: Mutex<Vec<CypherQuery>>,
}

impl MockNeo4jBackend {
    fn with_node_pages(pages: Vec<NodeFrame>) -> Self {
        Self {
            node_pages: Mutex::new(VecDeque::from(pages)),
            ..Self::default()
        }
    }

    fn with_edge_pages(pages: Vec<EdgeFrame>) -> Self {
        Self {
            edge_pages: Mutex::new(VecDeque::from(pages)),
            ..Self::default()
        }
    }

    fn with_expand_result(result: (NodeFrame, EdgeFrame)) -> Self {
        Self {
            expand_results: Mutex::new(VecDeque::from(vec![result])),
            ..Self::default()
        }
    }
}

impl Neo4jBackend for MockNeo4jBackend {
    fn load_nodes<'a>(&'a self, query: CypherQuery) -> ConnectorFuture<'a, NodeFrame> {
        Box::pin(async move {
            self.node_queries.lock().unwrap().push(query);
            self.node_pages
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| GFError::ConnectorError {
                    message: "mock node page queue is empty".to_owned(),
                })
        })
    }

    fn load_edges<'a>(&'a self, query: CypherQuery) -> ConnectorFuture<'a, EdgeFrame> {
        Box::pin(async move {
            self.edge_queries.lock().unwrap().push(query);
            self.edge_pages
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| GFError::ConnectorError {
                    message: "mock edge page queue is empty".to_owned(),
                })
        })
    }

    fn expand<'a>(&'a self, query: CypherQuery) -> ConnectorFuture<'a, ExpandResult> {
        Box::pin(async move {
            self.expand_queries.lock().unwrap().push(query);
            self.expand_results
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| GFError::ConnectorError {
                    message: "mock expand queue is empty".to_owned(),
                })
        })
    }
}

#[tokio::test]
async fn neo4j_connector_pushes_down_nodes_and_paginates() {
    let first = NodeFrame::from_record_batch(two_node_batch()).unwrap();
    let second = NodeFrame::from_record_batch(another_two_node_batch())
        .unwrap()
        .slice(0, 1);
    let backend = Arc::new(MockNeo4jBackend::with_node_pages(vec![first, second]));
    let connector = Neo4jConnector::with_backend(sample_config(), backend.clone());

    let predicate = Expr::BinaryOp {
        left: Box::new(Expr::Col {
            name: "age".to_owned(),
        }),
        op: BinaryOp::Gt,
        right: Box::new(Expr::Literal {
            value: ScalarValue::Int(18),
        }),
    };

    let nodes = connector
        .load_nodes(Some(&["Person"]), Some(&["age"]), Some(&predicate), 2)
        .await
        .unwrap();

    assert_eq!(nodes.len(), 3);
    assert_eq!(
        nodes.column_names(),
        vec![COL_NODE_ID, COL_NODE_LABEL, "age"]
    );
    assert!(nodes.row_index("alice").is_some());
    assert!(nodes.row_index("bob").is_some());
    assert!(nodes.row_index("charlie").is_some());

    let queries = backend.node_queries.lock().unwrap();
    assert_eq!(queries.len(), 2);
    assert!(queries[0].text.contains("MATCH (n)"));
    assert!(queries[0].text.contains("n:`Person`"));
    assert!(queries[0].text.contains("n.`age` > $p0"));
    assert!(queries[0]
        .text
        .contains("RETURN n.`_id` AS `_id`, labels(n) AS `_label`, n.`age` AS `age`"));
    assert!(queries[0].text.contains("SKIP $skip LIMIT $limit"));
    assert_eq!(queries[0].params.get("skip"), Some(&CypherValue::Int(0)));
    assert_eq!(queries[0].params.get("limit"), Some(&CypherValue::Int(2)));
    assert_eq!(queries[1].params.get("skip"), Some(&CypherValue::Int(2)));
}

#[tokio::test]
async fn neo4j_connector_pushes_down_edges_and_paginates() {
    let first = EdgeFrame::from_record_batch(edge_batch_with_since()).unwrap();
    let mask = BooleanArray::from(vec![true, false]);
    let second = EdgeFrame::from_record_batch(edge_batch_with_since())
        .unwrap()
        .filter(&mask)
        .unwrap();
    let backend = Arc::new(MockNeo4jBackend::with_edge_pages(vec![first, second]));
    let connector = Neo4jConnector::with_backend(sample_config(), backend.clone());

    let predicate = Expr::BinaryOp {
        left: Box::new(Expr::Col {
            name: "since".to_owned(),
        }),
        op: BinaryOp::GtEq,
        right: Box::new(Expr::Literal {
            value: ScalarValue::Int(2020),
        }),
    };

    let edges = connector
        .load_edges(Some(&["KNOWS"]), Some(&["since"]), Some(&predicate), 2)
        .await
        .unwrap();

    assert_eq!(edges.len(), 3);
    assert_eq!(
        edges.column_names(),
        vec![
            COL_EDGE_SRC,
            COL_EDGE_DST,
            COL_EDGE_TYPE,
            COL_EDGE_DIRECTION,
            "since"
        ]
    );

    let queries = backend.edge_queries.lock().unwrap();
    assert_eq!(queries.len(), 2);
    assert!(queries[0].text.contains("MATCH (src)-[r]->(dst)"));
    assert!(queries[0].text.contains("type(r) = $p0"));
    assert!(queries[0].text.contains("r.`since` >= $p1"));
    assert!(queries[0]
        .text
        .contains("RETURN src.`_id` AS `_src`, dst.`_id` AS `_dst`, type(r) AS `_type`, 0 AS `_direction`, r.`since` AS `since`"));
}

#[tokio::test]
async fn neo4j_connector_builds_expand_cypher() {
    let nodes = NodeFrame::from_record_batch(two_node_batch()).unwrap();
    let edges = EdgeFrame::from_record_batch(edge_batch_with_since()).unwrap();
    let backend = Arc::new(MockNeo4jBackend::with_expand_result((nodes, edges)));
    let connector = Neo4jConnector::with_backend(sample_config(), backend.clone());

    let node_predicate = Expr::BinaryOp {
        left: Box::new(Expr::Col {
            name: "age".to_owned(),
        }),
        op: BinaryOp::Gt,
        right: Box::new(Expr::Literal {
            value: ScalarValue::Int(21),
        }),
    };

    let (nodes, edges) = connector
        .expand(
            &["alice", "bob"],
            &EdgeTypeSpec::Multiple(vec!["KNOWS".to_owned(), "LIKES".to_owned()]),
            3,
            Direction::Both,
            Some(&node_predicate),
        )
        .await
        .unwrap();

    assert_eq!(nodes.len(), 2);
    assert_eq!(edges.len(), 2);

    let queries = backend.expand_queries.lock().unwrap();
    assert_eq!(queries.len(), 1);
    assert!(queries[0]
        .text
        .contains("MATCH path = (seed)-[:`KNOWS`|`LIKES`*1..3]-(m)"));
    assert!(queries[0].text.contains("seed.`_id` IN $seed_ids"));
    assert!(queries[0].text.contains("m.`age` > $p0"));
}

#[tokio::test]
async fn neo4j_connector_rejects_domain_invalid_predicate() {
    let backend = Arc::new(MockNeo4jBackend::with_node_pages(vec![
        NodeFrame::from_record_batch(two_node_batch()).unwrap(),
    ]));
    let connector = Neo4jConnector::with_backend(sample_config(), backend.clone());

    let invalid = Expr::BinaryOp {
        left: Box::new(Expr::Col {
            name: COL_EDGE_SRC.to_owned(),
        }),
        op: BinaryOp::Eq,
        right: Box::new(Expr::Literal {
            value: ScalarValue::String("alice".to_owned()),
        }),
    };

    let err = connector
        .load_nodes(None, Some(&["age"]), Some(&invalid), 2)
        .await
        .unwrap_err();

    assert!(matches!(err, GFError::TypeMismatch { .. }));
    assert!(backend.node_queries.lock().unwrap().is_empty());
}

fn sample_config() -> Neo4jConfig {
    Neo4jConfig {
        uri: "bolt://localhost:7687".to_owned(),
        user: "neo4j".to_owned(),
        password: "password".to_owned(),
        database: Some("neo4j".to_owned()),
    }
}
