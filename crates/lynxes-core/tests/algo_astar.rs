mod common;

use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lynxes_core::{
    Direction, EdgeFrame, EdgeTypeSpec, GFError, GraphFrame, NodeFrame, ShortestPathConfig,
    COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE, COL_NODE_ID,
};

use common::{label_field, labels_array};

fn nodes(ids: &[&str]) -> NodeFrame {
    let labels: Vec<&[&str]> = ids.iter().map(|_| &["Node"][..]).collect();
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        label_field(),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(ids.to_vec())) as ArrayRef,
            Arc::new(labels_array(&labels)) as ArrayRef,
        ],
    )
    .unwrap();
    NodeFrame::from_record_batch(batch).unwrap()
}

fn weighted_edges(src: &[&str], dst: &[&str], weight: &[i64]) -> EdgeFrame {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_EDGE_SRC, DataType::Utf8, false),
        Field::new(COL_EDGE_DST, DataType::Utf8, false),
        Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
        Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
        Field::new("weight", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(src.to_vec())) as ArrayRef,
            Arc::new(StringArray::from(dst.to_vec())) as ArrayRef,
            Arc::new(StringArray::from(vec!["ROAD"; src.len()])) as ArrayRef,
            Arc::new(arrow_array::Int8Array::from(vec![0i8; src.len()])) as ArrayRef,
            Arc::new(Int64Array::from(weight.to_vec())) as ArrayRef,
        ],
    )
    .unwrap();
    EdgeFrame::from_record_batch(batch).unwrap()
}

fn sample_weighted_graph() -> GraphFrame {
    GraphFrame::new(
        nodes(&["a", "b", "c", "d"]),
        weighted_edges(&["a", "b", "a", "c"], &["b", "d", "c", "d"], &[1, 1, 2, 5]),
    )
    .unwrap()
}

#[test]
fn astar_falls_back_to_dijkstra_when_heuristic_is_none() {
    let graph = sample_weighted_graph();
    let config = ShortestPathConfig {
        weight_col: Some("weight".to_owned()),
        edge_type: EdgeTypeSpec::Any,
        direction: Direction::Out,
    };

    let dijkstra = graph.shortest_path("a", "d", &config).unwrap();
    let astar = graph.astar_shortest_path("a", "d", &config, None).unwrap();

    assert_eq!(astar, dijkstra);
    assert_eq!(
        astar,
        Some(vec!["a".to_owned(), "b".to_owned(), "d".to_owned()])
    );
}

#[test]
fn astar_uses_heuristic_and_matches_shortest_path_result() {
    let graph = sample_weighted_graph();
    let config = ShortestPathConfig {
        weight_col: Some("weight".to_owned()),
        edge_type: EdgeTypeSpec::Any,
        direction: Direction::Out,
    };

    let heuristic = |node: &str, dst: &str| -> f64 {
        let coords: (f64, f64) = match node {
            "a" => (0.0, 0.0),
            "b" => (1.0, 0.0),
            "c" => (0.0, 1.0),
            "d" => (2.0, 0.0),
            _ => unreachable!(),
        };
        let target: (f64, f64) = match dst {
            "d" => (2.0, 0.0),
            _ => unreachable!(),
        };
        (coords.0 - target.0).abs() + (coords.1 - target.1).abs()
    };

    let astar = graph
        .astar_shortest_path("a", "d", &config, Some(&heuristic))
        .unwrap();

    assert_eq!(
        astar,
        Some(vec!["a".to_owned(), "b".to_owned(), "d".to_owned()])
    );
}

#[test]
fn astar_rejects_invalid_heuristic_values() {
    let graph = sample_weighted_graph();
    let config = ShortestPathConfig {
        weight_col: Some("weight".to_owned()),
        edge_type: EdgeTypeSpec::Any,
        direction: Direction::Out,
    };

    let err = graph
        .astar_shortest_path("a", "d", &config, Some(&|_, _| f64::NAN))
        .unwrap_err();

    assert!(
        matches!(err, GFError::InvalidConfig { message } if message.contains("heuristic must return"))
    );
}
