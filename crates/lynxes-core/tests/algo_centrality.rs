use std::sync::Arc;

use arrow_array::{
    builder::{ListBuilder, StringBuilder},
    Float64Array, Int8Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};

use lynxes_core::{
    Direction, EdgeFrame, GFError, GraphFrame, NodeFrame, COL_EDGE_DIRECTION, COL_EDGE_DST,
    COL_EDGE_SRC, COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
};

// ── Fixtures ──────────────────────────────────────────────────────────────

fn label_field() -> Field {
    Field::new(
        COL_NODE_LABEL,
        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
        false,
    )
}

fn empty_labels(n: usize) -> arrow_array::ListArray {
    let value_builder = StringBuilder::new();
    let mut builder = ListBuilder::new(value_builder);
    for _ in 0..n {
        builder.append(true);
    }
    builder.finish()
}

fn make_graph(nodes: &[&str], edges: &[(&str, &str)]) -> GraphFrame {
    let node_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        label_field(),
    ]));
    let node_batch = RecordBatch::try_new(
        node_schema,
        vec![
            Arc::new(StringArray::from(nodes.to_vec())) as Arc<dyn arrow_array::Array>,
            Arc::new(empty_labels(nodes.len())) as Arc<dyn arrow_array::Array>,
        ],
    )
    .unwrap();

    let edge_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_EDGE_SRC, DataType::Utf8, false),
        Field::new(COL_EDGE_DST, DataType::Utf8, false),
        Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
        Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
    ]));
    let (srcs, dsts): (Vec<_>, Vec<_>) = edges.iter().copied().unzip();
    let edge_batch = RecordBatch::try_new(
        edge_schema,
        vec![
            Arc::new(StringArray::from(srcs)) as Arc<dyn arrow_array::Array>,
            Arc::new(StringArray::from(dsts)) as Arc<dyn arrow_array::Array>,
            Arc::new(StringArray::from(vec!["E"; edges.len()])) as Arc<dyn arrow_array::Array>,
            Arc::new(Int8Array::from(vec![0i8; edges.len()])) as Arc<dyn arrow_array::Array>,
        ],
    )
    .unwrap();

    let nf = NodeFrame::from_record_batch(node_batch).unwrap();
    let ef = EdgeFrame::from_record_batch(edge_batch).unwrap();
    GraphFrame::new(nf, ef).unwrap()
}

fn get_f64(result: &NodeFrame, id: &str, col: &str) -> f64 {
    let row = result.row_index(id).expect("id present") as usize;
    result
        .column(col)
        .expect("column present")
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Float64 column")
        .value(row)
}

// ── has_path ──────────────────────────────────────────────────────────────

#[test]
fn has_path_direct_edge_is_true() {
    let g = make_graph(&["a", "b"], &[("a", "b")]);
    assert!(g.has_path("a", "b", None).unwrap());
}

#[test]
fn has_path_indirect_two_hops_is_true() {
    let g = make_graph(&["a", "b", "c"], &[("a", "b"), ("b", "c")]);
    assert!(g.has_path("a", "c", None).unwrap());
}

#[test]
fn has_path_reversed_edge_returns_false() {
    let g = make_graph(&["a", "b"], &[("b", "a")]);
    assert!(!g.has_path("a", "b", None).unwrap());
}

#[test]
fn has_path_self_path_always_true() {
    let g = make_graph(&["a", "b"], &[("a", "b")]);
    assert!(g.has_path("a", "a", None).unwrap());
}

#[test]
fn has_path_disconnected_returns_false() {
    let g = make_graph(&["a", "b", "c"], &[("a", "b")]);
    assert!(!g.has_path("a", "c", None).unwrap());
}

#[test]
fn has_path_max_hops_zero_only_self() {
    let g = make_graph(&["a", "b"], &[("a", "b")]);
    assert!(!g.has_path("a", "b", Some(0)).unwrap());
}

#[test]
fn has_path_max_hops_one_finds_direct_edge() {
    let g = make_graph(&["a", "b"], &[("a", "b")]);
    assert!(g.has_path("a", "b", Some(1)).unwrap());
}

#[test]
fn has_path_max_hops_too_small_returns_false() {
    let g = make_graph(&["a", "b", "c"], &[("a", "b"), ("b", "c")]);
    assert!(!g.has_path("a", "c", Some(1)).unwrap());
}

#[test]
fn has_path_max_hops_exact_succeeds() {
    let g = make_graph(&["a", "b", "c"], &[("a", "b"), ("b", "c")]);
    assert!(g.has_path("a", "c", Some(2)).unwrap());
}

#[test]
fn has_path_isolated_src_returns_false() {
    let node_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        label_field(),
    ]));
    let edge_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_EDGE_SRC, DataType::Utf8, false),
        Field::new(COL_EDGE_DST, DataType::Utf8, false),
        Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
        Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
    ]));
    let nb = RecordBatch::try_new(
        node_schema,
        vec![
            Arc::new(StringArray::from(vec!["a", "b"])) as Arc<dyn arrow_array::Array>,
            Arc::new(empty_labels(2)) as Arc<dyn arrow_array::Array>,
        ],
    )
    .unwrap();
    let nf = NodeFrame::from_record_batch(nb).unwrap();
    let ef = EdgeFrame::from_record_batch(RecordBatch::new_empty(edge_schema)).unwrap();
    let g = GraphFrame::new(nf, ef).unwrap();
    assert!(!g.has_path("a", "b", None).unwrap());
}

#[test]
fn has_path_unknown_src_returns_error() {
    let g = make_graph(&["a", "b"], &[("a", "b")]);
    let err = g.has_path("ghost", "b", None).unwrap_err();
    assert!(matches!(err, GFError::NodeNotFound { id } if id == "ghost"));
}

#[test]
fn has_path_unknown_dst_returns_error() {
    let g = make_graph(&["a", "b"], &[("a", "b")]);
    let err = g.has_path("a", "ghost", None).unwrap_err();
    assert!(matches!(err, GFError::NodeNotFound { id } if id == "ghost"));
}

// ── degree_centrality ─────────────────────────────────────────────────────

#[test]
fn degree_centrality_out_star_hub_is_one() {
    let g = make_graph(
        &["hub", "a", "b", "c"],
        &[("hub", "a"), ("hub", "b"), ("hub", "c")],
    );
    let result = g.degree_centrality(Direction::Out).unwrap();
    let dc_hub = get_f64(&result, "hub", "degree_centrality");
    assert!(
        (dc_hub - 1.0).abs() < 1e-10,
        "hub out-centrality = 1.0, got {}",
        dc_hub
    );
}

#[test]
fn degree_centrality_in_star_spokes_are_zero() {
    // in-star: edges flow FROM spokes TO hub; spokes have 0 in-degree
    let g = make_graph(
        &["hub", "a", "b", "c"],
        &[("a", "hub"), ("b", "hub"), ("c", "hub")],
    );
    let result = g.degree_centrality(Direction::In).unwrap();
    for id in &["a", "b", "c"] {
        let dc = get_f64(&result, id, "degree_centrality");
        assert!(
            (dc - 0.0).abs() < 1e-10,
            "{} in-centrality = 0.0, got {}",
            id,
            dc
        );
    }
    let dc_hub = get_f64(&result, "hub", "degree_centrality");
    assert!(
        (dc_hub - 1.0).abs() < 1e-10,
        "hub in-centrality = 1.0, got {}",
        dc_hub
    );
}

#[test]
fn degree_centrality_values_in_zero_one_range() {
    let g = make_graph(
        &["a", "b", "c", "d"],
        &[("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")],
    );
    for dir in &[Direction::Out, Direction::In, Direction::Both] {
        let result = g.degree_centrality(*dir).unwrap();
        for id in &["a", "b", "c", "d"] {
            let v = get_f64(&result, id, "degree_centrality");
            assert!(
                (0.0..=1.0).contains(&v),
                "{:?} centrality of {} = {} out of [0,1]",
                dir,
                id,
                v
            );
        }
    }
}

#[test]
fn degree_centrality_isolated_node_is_zero() {
    let node_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        label_field(),
    ]));
    let edge_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_EDGE_SRC, DataType::Utf8, false),
        Field::new(COL_EDGE_DST, DataType::Utf8, false),
        Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
        Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
    ]));
    let nb = RecordBatch::try_new(
        node_schema,
        vec![
            Arc::new(StringArray::from(vec!["iso", "other"])) as Arc<dyn arrow_array::Array>,
            Arc::new(empty_labels(2)) as Arc<dyn arrow_array::Array>,
        ],
    )
    .unwrap();
    let nf = NodeFrame::from_record_batch(nb).unwrap();
    let ef = EdgeFrame::from_record_batch(RecordBatch::new_empty(edge_schema)).unwrap();
    let g = GraphFrame::new(nf, ef).unwrap();

    let result = g.degree_centrality(Direction::Out).unwrap();
    let dc = get_f64(&result, "iso", "degree_centrality");
    assert!((dc - 0.0).abs() < 1e-10);
}

#[test]
fn degree_centrality_single_node_is_zero() {
    let node_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        label_field(),
    ]));
    let edge_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_EDGE_SRC, DataType::Utf8, false),
        Field::new(COL_EDGE_DST, DataType::Utf8, false),
        Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
        Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
    ]));
    let nb = RecordBatch::try_new(
        node_schema,
        vec![
            Arc::new(StringArray::from(vec!["solo"])) as Arc<dyn arrow_array::Array>,
            Arc::new(empty_labels(1)) as Arc<dyn arrow_array::Array>,
        ],
    )
    .unwrap();
    let nf = NodeFrame::from_record_batch(nb).unwrap();
    let ef = EdgeFrame::from_record_batch(RecordBatch::new_empty(edge_schema)).unwrap();
    let g = GraphFrame::new(nf, ef).unwrap();
    let result = g.degree_centrality(Direction::Out).unwrap();
    assert!((get_f64(&result, "solo", "degree_centrality")).abs() < 1e-10);
}

#[test]
fn degree_centrality_output_schema_has_three_columns() {
    let g = make_graph(&["a", "b"], &[("a", "b")]);
    let result = g.degree_centrality(Direction::Out).unwrap();
    assert_eq!(result.column_names().len(), 3);
    assert!(result.column("degree_centrality").is_some());
}

// ── betweenness_centrality ────────────────────────────────────────────────

#[test]
fn betweenness_endpoints_of_chain_are_zero() {
    let g = make_graph(&["a", "b", "c", "d"], &[("a", "b"), ("b", "c"), ("c", "d")]);
    let result = g.betweenness_centrality().unwrap();
    let bt_a = get_f64(&result, "a", "betweenness");
    let bt_d = get_f64(&result, "d", "betweenness");
    let bt_b = get_f64(&result, "b", "betweenness");
    let bt_c = get_f64(&result, "c", "betweenness");

    assert!((bt_a).abs() < 1e-10, "a betweenness = 0, got {}", bt_a);
    assert!((bt_d).abs() < 1e-10, "d betweenness = 0, got {}", bt_d);
    assert!(bt_b > 0.0, "b betweenness > 0, got {}", bt_b);
    assert!(bt_c > 0.0, "c betweenness > 0, got {}", bt_c);
}

#[test]
fn betweenness_bridge_node_higher_than_leaves() {
    let g = make_graph(
        &["a", "b", "c", "d"],
        &[("a", "b"), ("b", "c"), ("b", "d"), ("c", "b"), ("d", "b")],
    );
    let result = g.betweenness_centrality().unwrap();
    let bt_b = get_f64(&result, "b", "betweenness");
    let bt_a = get_f64(&result, "a", "betweenness");
    let bt_c = get_f64(&result, "c", "betweenness");

    assert!(bt_b > bt_a, "b({}) should beat a({})", bt_b, bt_a);
    assert!(bt_b > bt_c, "b({}) should beat c({})", bt_b, bt_c);
}

#[test]
fn betweenness_values_in_zero_one_range() {
    let g = make_graph(
        &["a", "b", "c", "d", "e"],
        &[("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("b", "d")],
    );
    let result = g.betweenness_centrality().unwrap();
    for id in &["a", "b", "c", "d", "e"] {
        let v = get_f64(&result, id, "betweenness");
        assert!(
            (0.0..=1.0).contains(&v),
            "betweenness of {} = {} out of [0,1]",
            id,
            v
        );
    }
}

#[test]
fn betweenness_single_node_is_zero() {
    let node_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        label_field(),
    ]));
    let edge_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_EDGE_SRC, DataType::Utf8, false),
        Field::new(COL_EDGE_DST, DataType::Utf8, false),
        Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
        Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
    ]));
    let nb = RecordBatch::try_new(
        node_schema,
        vec![
            Arc::new(StringArray::from(vec!["solo"])) as Arc<dyn arrow_array::Array>,
            Arc::new(empty_labels(1)) as Arc<dyn arrow_array::Array>,
        ],
    )
    .unwrap();
    let nf = NodeFrame::from_record_batch(nb).unwrap();
    let ef = EdgeFrame::from_record_batch(RecordBatch::new_empty(edge_schema)).unwrap();
    let g = GraphFrame::new(nf, ef).unwrap();
    let result = g.betweenness_centrality().unwrap();
    assert!((get_f64(&result, "solo", "betweenness")).abs() < 1e-10);
}

#[test]
fn betweenness_two_node_chain_endpoints_are_zero() {
    let g = make_graph(&["a", "b"], &[("a", "b")]);
    let result = g.betweenness_centrality().unwrap();
    let bt_a = get_f64(&result, "a", "betweenness");
    let bt_b = get_f64(&result, "b", "betweenness");
    assert!((bt_a).abs() < 1e-10);
    assert!((bt_b).abs() < 1e-10);
}

#[test]
fn betweenness_isolated_node_is_zero() {
    let node_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        label_field(),
    ]));
    let edge_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_EDGE_SRC, DataType::Utf8, false),
        Field::new(COL_EDGE_DST, DataType::Utf8, false),
        Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
        Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
    ]));
    let nb = RecordBatch::try_new(
        node_schema,
        vec![
            Arc::new(StringArray::from(vec!["iso", "a", "b"])) as Arc<dyn arrow_array::Array>,
            Arc::new(empty_labels(3)) as Arc<dyn arrow_array::Array>,
        ],
    )
    .unwrap();
    let edge_batch = RecordBatch::try_new(
        edge_schema,
        vec![
            Arc::new(StringArray::from(vec!["a"])) as Arc<dyn arrow_array::Array>,
            Arc::new(StringArray::from(vec!["b"])) as Arc<dyn arrow_array::Array>,
            Arc::new(StringArray::from(vec!["E"])) as Arc<dyn arrow_array::Array>,
            Arc::new(Int8Array::from(vec![0i8])) as Arc<dyn arrow_array::Array>,
        ],
    )
    .unwrap();
    let nf = NodeFrame::from_record_batch(nb).unwrap();
    let ef = EdgeFrame::from_record_batch(edge_batch).unwrap();
    let g = GraphFrame::new(nf, ef).unwrap();
    let result = g.betweenness_centrality().unwrap();
    assert!((get_f64(&result, "iso", "betweenness")).abs() < 1e-10);
}

#[test]
fn betweenness_output_schema_has_three_columns() {
    let g = make_graph(&["a", "b", "c"], &[("a", "b"), ("b", "c")]);
    let result = g.betweenness_centrality().unwrap();
    assert_eq!(result.column_names().len(), 3);
    assert!(result.column("betweenness").is_some());
}

#[test]
fn betweenness_known_line_graph_values() {
    // Line: a→b→c  (N=3)
    // b lies on a→c → raw betweenness(b) = 1.
    // Normalisation: (3-1)*(3-2) = 2.  betweenness(b) = 1/2 = 0.5.
    let g = make_graph(&["a", "b", "c"], &[("a", "b"), ("b", "c")]);
    let result = g.betweenness_centrality().unwrap();
    let bt_b = get_f64(&result, "b", "betweenness");
    assert!(
        (bt_b - 0.5).abs() < 1e-10,
        "b betweenness = 0.5, got {}",
        bt_b
    );

    let bt_a = get_f64(&result, "a", "betweenness");
    let bt_c = get_f64(&result, "c", "betweenness");
    assert!((bt_a).abs() < 1e-10, "a betweenness = 0, got {}", bt_a);
    assert!((bt_c).abs() < 1e-10, "c betweenness = 0, got {}", bt_c);
}
