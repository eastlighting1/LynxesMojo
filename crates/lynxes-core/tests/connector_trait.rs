use lynxes_core::{Connector, Direction, EdgeTypeSpec, Expr, GFError};

#[derive(Debug, Default)]
struct DummyConnector;

impl Connector for DummyConnector {}

#[tokio::test]
async fn default_load_nodes_rejects_zero_batch_size() {
    let connector = DummyConnector;
    let err = connector.load_nodes(None, None, None, 0).await.unwrap_err();

    assert!(matches!(err, GFError::InvalidConfig { .. }));
}

#[tokio::test]
async fn default_expand_rejects_zero_hops() {
    let connector = DummyConnector;
    let err = connector
        .expand(&[], &EdgeTypeSpec::Any, 0, Direction::Out, None)
        .await
        .unwrap_err();

    assert!(matches!(err, GFError::InvalidConfig { .. }));
}

#[tokio::test]
async fn default_methods_fail_closed_when_unimplemented() {
    let connector = DummyConnector;

    let load_nodes_err = connector
        .load_nodes(
            None,
            None,
            Some(&Expr::Literal {
                value: lynxes_core::ScalarValue::Bool(true),
            }),
            1,
        )
        .await
        .unwrap_err();
    let load_edges_err = connector.load_edges(None, None, None, 1).await.unwrap_err();
    let expand_err = connector
        .expand(&["alice"], &EdgeTypeSpec::Any, 1, Direction::Out, None)
        .await
        .unwrap_err();
    let write_err = connector.write(&crate_graph()).await.unwrap_err();

    assert!(matches!(
        load_nodes_err,
        GFError::UnsupportedOperation { .. }
    ));
    assert!(matches!(
        load_edges_err,
        GFError::UnsupportedOperation { .. }
    ));
    assert!(matches!(expand_err, GFError::UnsupportedOperation { .. }));
    assert!(matches!(write_err, GFError::UnsupportedOperation { .. }));
}

fn crate_graph() -> lynxes_core::GraphFrame {
    let nodes = lynxes_core::NodeFrame::from_record_batch(common::graph_node_batch()).unwrap();
    let edges = lynxes_core::EdgeFrame::from_record_batch(common::graph_edge_batch()).unwrap();
    lynxes_core::GraphFrame::new(nodes, edges).unwrap()
}

mod common;
