mod common;

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use arrow_array::BooleanArray;
use lynxes_connect::{
    AqlQuery, AqlValue, ArangoBackend, ArangoConfig, ArangoConnector, Connector, ConnectorFuture,
    ExpandResult,
};
use lynxes_core::{
    BinaryOp, Direction, EdgeFrame, EdgeTypeSpec, Expr, GFError, NodeFrame, ScalarValue,
    COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
};

use common::{another_two_node_batch, edge_batch_with_since, two_node_batch};

#[derive(Debug, Default)]
struct MockArangoBackend {
    node_pages: Mutex<VecDeque<NodeFrame>>,
    edge_pages: Mutex<VecDeque<EdgeFrame>>,
    expand_results: Mutex<VecDeque<(NodeFrame, EdgeFrame)>>,
    node_queries: Mutex<Vec<AqlQuery>>,
    edge_queries: Mutex<Vec<AqlQuery>>,
    expand_queries: Mutex<Vec<AqlQuery>>,
}

impl MockArangoBackend {
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

impl ArangoBackend for MockArangoBackend {
    fn load_nodes<'a>(&'a self, query: AqlQuery) -> ConnectorFuture<'a, NodeFrame> {
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

    fn load_edges<'a>(&'a self, query: AqlQuery) -> ConnectorFuture<'a, EdgeFrame> {
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

    fn expand<'a>(&'a self, query: AqlQuery) -> ConnectorFuture<'a, ExpandResult> {
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
async fn arango_connector_pushes_down_nodes_and_paginates() {
    let first = NodeFrame::from_record_batch(two_node_batch()).unwrap();
    let second = NodeFrame::from_record_batch(another_two_node_batch())
        .unwrap()
        .slice(0, 1);
    let backend = Arc::new(MockArangoBackend::with_node_pages(vec![first, second]));
    let connector = ArangoConnector::with_backend(sample_config(), backend.clone());

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

    let queries = backend.node_queries.lock().unwrap();
    assert_eq!(queries.len(), 2);
    assert!(queries[0].text.contains("FOR v IN @@vertex_collection"));
    assert!(queries[0].text.contains("FILTER @p0 IN v.`_label`"));
    assert!(queries[0].text.contains("FILTER (v.`age` > @p1)"));
    assert!(queries[0]
        .text
        .contains("RETURN { `_id`: v._id, `_label`: v.`_label`, `age`: v.`age` }"));
    assert_eq!(
        queries[0].bind_vars.get("@vertex_collection"),
        Some(&AqlValue::String("vertices".to_owned()))
    );
    assert_eq!(queries[0].bind_vars.get("offset"), Some(&AqlValue::Int(0)));
    assert_eq!(queries[1].bind_vars.get("offset"), Some(&AqlValue::Int(2)));
}

#[tokio::test]
async fn arango_connector_pushes_down_edges_and_paginates() {
    let first = EdgeFrame::from_record_batch(edge_batch_with_since()).unwrap();
    let mask = BooleanArray::from(vec![true, false]);
    let second = EdgeFrame::from_record_batch(edge_batch_with_since())
        .unwrap()
        .filter(&mask)
        .unwrap();
    let backend = Arc::new(MockArangoBackend::with_edge_pages(vec![first, second]));
    let connector = ArangoConnector::with_backend(sample_config(), backend.clone());

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
    assert!(queries[0].text.contains("FOR e IN @@edge_collection"));
    assert!(queries[0].text.contains("FILTER e.`_type` == @p0"));
    assert!(queries[0].text.contains("FILTER (e.`since` >= @p1)"));
    assert!(queries[0]
        .text
        .contains("RETURN { `_src`: e._from, `_dst`: e._to, `_type`: e.`_type`, `_direction`: 0, `since`: e.`since` }"));
}

#[tokio::test]
async fn arango_connector_builds_expand_aql() {
    let nodes = NodeFrame::from_record_batch(two_node_batch()).unwrap();
    let edges = EdgeFrame::from_record_batch(edge_batch_with_since()).unwrap();
    let backend = Arc::new(MockArangoBackend::with_expand_result((nodes, edges)));
    let connector = ArangoConnector::with_backend(sample_config(), backend.clone());

    let node_predicate = Expr::BinaryOp {
        left: Box::new(Expr::Col {
            name: "age".to_owned(),
        }),
        op: BinaryOp::Gt,
        right: Box::new(Expr::Literal {
            value: ScalarValue::Int(21),
        }),
    };

    let (_nodes, _edges) = connector
        .expand(
            &["vertices/alice"],
            &EdgeTypeSpec::Single("KNOWS".to_owned()),
            2,
            Direction::Out,
            Some(&node_predicate),
        )
        .await
        .unwrap();

    let queries = backend.expand_queries.lock().unwrap();
    assert_eq!(queries.len(), 1);
    assert!(queries[0].text.contains("FOR seed IN @@vertex_collection"));
    assert!(queries[0].text.contains("FILTER seed._id IN @seed_ids"));
    assert!(queries[0]
        .text
        .contains("FOR vertex, edge, path IN 1..2 OUTBOUND seed GRAPH @graph"));
    assert!(queries[0].text.contains("FILTER edge.`_type` == @p0"));
    assert!(queries[0].text.contains("FILTER (vertex.`age` > @p1)"));
}

#[tokio::test]
async fn arango_connector_rejects_domain_invalid_predicate() {
    let backend = Arc::new(MockArangoBackend::with_node_pages(vec![
        NodeFrame::from_record_batch(two_node_batch()).unwrap(),
    ]));
    let connector = ArangoConnector::with_backend(sample_config(), backend.clone());

    let invalid = Expr::BinaryOp {
        left: Box::new(Expr::Col {
            name: COL_EDGE_SRC.to_owned(),
        }),
        op: BinaryOp::Eq,
        right: Box::new(Expr::Literal {
            value: ScalarValue::String("vertices/alice".to_owned()),
        }),
    };

    let err = connector
        .load_nodes(None, Some(&["age"]), Some(&invalid), 2)
        .await
        .unwrap_err();

    assert!(matches!(err, GFError::TypeMismatch { .. }));
    assert!(backend.node_queries.lock().unwrap().is_empty());
}

fn sample_config() -> ArangoConfig {
    ArangoConfig {
        endpoint: "http://localhost:8529".to_owned(),
        database: "_system".to_owned(),
        graph: "demo_graph".to_owned(),
        vertex_collection: "vertices".to_owned(),
        edge_collection: "edges".to_owned(),
        username: "root".to_owned(),
        password: "open-sesame".to_owned(),
    }
}
