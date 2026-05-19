mod common;

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use arrow_array::BooleanArray;
use lynxes_connect::{
    Connector, ConnectorFuture, ExpandResult, SparqlBackend, SparqlConfig, SparqlConnector,
    SparqlQuery,
};
use lynxes_core::{
    BinaryOp, Direction, EdgeFrame, EdgeTypeSpec, Expr, GFError, NodeFrame, ScalarValue,
    COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
};

use common::{another_two_node_batch, edge_batch_with_since, two_node_batch};

#[derive(Debug, Default)]
struct MockSparqlBackend {
    node_pages: Mutex<VecDeque<NodeFrame>>,
    edge_pages: Mutex<VecDeque<EdgeFrame>>,
    expand_results: Mutex<VecDeque<(NodeFrame, EdgeFrame)>>,
    node_queries: Mutex<Vec<SparqlQuery>>,
    edge_queries: Mutex<Vec<SparqlQuery>>,
    expand_queries: Mutex<Vec<SparqlQuery>>,
}

impl MockSparqlBackend {
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

impl SparqlBackend for MockSparqlBackend {
    fn load_nodes<'a>(&'a self, query: SparqlQuery) -> ConnectorFuture<'a, NodeFrame> {
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

    fn load_edges<'a>(&'a self, query: SparqlQuery) -> ConnectorFuture<'a, EdgeFrame> {
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

    fn expand<'a>(&'a self, query: SparqlQuery) -> ConnectorFuture<'a, ExpandResult> {
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
async fn sparql_connector_pushes_down_nodes_and_paginates() {
    let first = NodeFrame::from_record_batch(two_node_batch()).unwrap();
    let second = NodeFrame::from_record_batch(another_two_node_batch())
        .unwrap()
        .slice(0, 1);
    let backend = Arc::new(MockSparqlBackend::with_node_pages(vec![first, second]));
    let connector = SparqlConnector::with_backend(sample_config(), backend.clone());

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
    assert!(queries[0]
        .text
        .contains("SELECT ?nId AS ?_id ?nLabel AS ?_label ?nage AS ?age WHERE"));
    assert!(queries[0].text.contains("FILTER (?nLabel = \"Person\")"));
    assert!(queries[0].text.contains("FILTER ((?nage > 18))"));
    assert!(queries[0].text.contains("LIMIT 2"));
    assert!(queries[0].text.contains("OFFSET 0"));
    assert!(queries[1].text.contains("OFFSET 2"));
}

#[tokio::test]
async fn sparql_connector_pushes_down_edges_and_paginates() {
    let first = EdgeFrame::from_record_batch(edge_batch_with_since()).unwrap();
    let mask = BooleanArray::from(vec![true, false]);
    let second = EdgeFrame::from_record_batch(edge_batch_with_since())
        .unwrap()
        .filter(&mask)
        .unwrap();
    let backend = Arc::new(MockSparqlBackend::with_edge_pages(vec![first, second]));
    let connector = SparqlConnector::with_backend(sample_config(), backend.clone());

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
    assert!(queries[0]
        .text
        .contains("SELECT ?src AS ?_src ?dst AS ?_dst ?etype AS ?_type \"out\" AS ?_direction ?esince AS ?since WHERE"));
    assert!(queries[0].text.contains("FILTER (?etype = \"KNOWS\")"));
    assert!(queries[0].text.contains("FILTER ((?esince >= 2020))"));
}

#[tokio::test]
async fn sparql_connector_builds_expand_query_when_template_exists() {
    let nodes = NodeFrame::from_record_batch(two_node_batch()).unwrap();
    let edges = EdgeFrame::from_record_batch(edge_batch_with_since()).unwrap();
    let backend = Arc::new(MockSparqlBackend::with_expand_result((nodes, edges)));
    let connector = SparqlConnector::with_backend(sample_config(), backend.clone());

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
            &["alice"],
            &EdgeTypeSpec::Single("KNOWS".to_owned()),
            2,
            Direction::Out,
            Some(&node_predicate),
        )
        .await
        .unwrap();

    let queries = backend.expand_queries.lock().unwrap();
    assert_eq!(queries.len(), 1);
    assert!(queries[0].text.contains("FILTER (?seedId = \"alice\")"));
    assert!(queries[0].text.contains("FILTER (?etype = \"KNOWS\")"));
    assert!(queries[0].text.contains("FILTER ((?mage > 21))"));
    assert!(queries[0].text.contains("?seed ?p{1,2} ?m ."));
}

#[tokio::test]
async fn sparql_connector_rejects_invalid_node_predicate_domain() {
    let backend = Arc::new(MockSparqlBackend::with_node_pages(vec![
        NodeFrame::from_record_batch(two_node_batch()).unwrap(),
    ]));
    let connector = SparqlConnector::with_backend(sample_config(), backend.clone());

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

fn sample_config() -> SparqlConfig {
    SparqlConfig {
        endpoint: "http://localhost:3030/ds/query".to_owned(),
        node_template: r#"SELECT {{projection}} WHERE {
  ?n a ?nLabel .
  BIND(STR(?n) AS ?nId)
  OPTIONAL { ?n <urn:age> ?nage . }
  {{filters}}
}
LIMIT {{limit}}
OFFSET {{offset}}"#
            .to_owned(),
        edge_template: r#"SELECT {{projection}} WHERE {
  ?src ?p ?dst .
  BIND(STR(?src) AS ?src)
  BIND(STR(?dst) AS ?dst)
  BIND(STR(?p) AS ?etype)
  OPTIONAL { ?edge <urn:since> ?esince . }
  {{filters}}
}
LIMIT {{limit}}
OFFSET {{offset}}"#
            .to_owned(),
        expand_template: Some(
            r#"SELECT {{projection}} WHERE {
  ?seed a ?seedLabel .
  BIND(STR(?seed) AS ?seedId)
  ?m a ?mLabel .
  BIND(STR(?m) AS ?mId)
  OPTIONAL { ?m <urn:age> ?mage . }
  ?src ?e ?dst .
  BIND(STR(?e) AS ?etype)
  {{filters}}
}"#
            .to_owned(),
        ),
    }
}
