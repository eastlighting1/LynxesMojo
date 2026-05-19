use std::sync::Arc;

use arrow_array::{
    builder::{ListBuilder, StringBuilder},
    Float64Array, Int8Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};

use lynxes_core::{
    Direction, EdgeFrame, EdgeTypeSpec, GFError, GraphFrame, NodeFrame, ShortestPathConfig,
    COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
};

// ── Graph builder helpers ─────────────────────────────────────────────────

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

fn make_weighted_graph(nodes: &[&str], edges: &[(&str, &str, f64)]) -> GraphFrame {
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
        Field::new("cost", DataType::Float64, true),
    ]));
    let (srcs, dsts, weights) = unzip3(edges.iter().map(|&(s, d, w)| (s, d, Some(w))));
    let edge_batch = RecordBatch::try_new(
        edge_schema,
        vec![
            Arc::new(StringArray::from(srcs)) as Arc<dyn arrow_array::Array>,
            Arc::new(StringArray::from(dsts)) as Arc<dyn arrow_array::Array>,
            Arc::new(StringArray::from(vec!["E"; edges.len()])) as Arc<dyn arrow_array::Array>,
            Arc::new(Int8Array::from(vec![0i8; edges.len()])) as Arc<dyn arrow_array::Array>,
            Arc::new(Float64Array::from(weights)) as Arc<dyn arrow_array::Array>,
        ],
    )
    .unwrap();

    let nf = NodeFrame::from_record_batch(node_batch).unwrap();
    let ef = EdgeFrame::from_record_batch(edge_batch).unwrap();
    GraphFrame::new(nf, ef).unwrap()
}

fn make_unweighted_graph(nodes: &[&str], edges: &[(&str, &str)]) -> GraphFrame {
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

fn cfg() -> ShortestPathConfig {
    ShortestPathConfig::default()
}

fn wcfg() -> ShortestPathConfig {
    ShortestPathConfig {
        weight_col: Some("cost".to_owned()),
        ..Default::default()
    }
}

fn unzip3<I, A, B, C>(iter: I) -> (Vec<A>, Vec<B>, Vec<C>)
where
    I: Iterator<Item = (A, B, C)>,
{
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut c = Vec::new();
    for (x, y, z) in iter {
        a.push(x);
        b.push(y);
        c.push(z);
    }
    (a, b, c)
}

// ── shortest_path ─────────────────────────────────────────────────────────

#[test]
fn direct_edge_returns_two_node_path() {
    let g = make_unweighted_graph(&["a", "b"], &[("a", "b")]);
    let path = g.shortest_path("a", "b", &cfg()).unwrap();
    assert_eq!(path, Some(vec!["a".to_owned(), "b".to_owned()]));
}

#[test]
fn two_hop_path_is_returned_correctly() {
    let g = make_unweighted_graph(&["a", "b", "c"], &[("a", "b"), ("b", "c")]);
    let path = g.shortest_path("a", "c", &cfg()).unwrap();
    assert_eq!(
        path,
        Some(vec!["a".to_owned(), "b".to_owned(), "c".to_owned()])
    );
}

#[test]
fn shorter_path_wins_over_longer_one() {
    let g = make_weighted_graph(
        &["a", "b", "c"],
        &[("a", "b", 10.0), ("a", "c", 1.0), ("c", "b", 1.0)],
    );
    let path = g.shortest_path("a", "b", &wcfg()).unwrap().unwrap();
    assert_eq!(path, vec!["a", "c", "b"]);
}

#[test]
fn no_path_returns_none() {
    let g = make_unweighted_graph(&["a", "b"], &[("b", "a")]);
    let path = g.shortest_path("a", "b", &cfg()).unwrap();
    assert_eq!(path, None);
}

#[test]
fn src_equals_dst_returns_single_node_path() {
    let g = make_unweighted_graph(&["a", "b"], &[("a", "b")]);
    let path = g.shortest_path("a", "a", &cfg()).unwrap();
    assert_eq!(path, Some(vec!["a".to_owned()]));
}

#[test]
fn weighted_longer_hop_count_but_cheaper() {
    let g = make_weighted_graph(
        &["a", "b", "c", "d"],
        &[
            ("a", "b", 5.0),
            ("a", "c", 1.0),
            ("c", "d", 1.0),
            ("d", "b", 1.0),
        ],
    );
    let path = g.shortest_path("a", "b", &wcfg()).unwrap().unwrap();
    assert_eq!(path, vec!["a", "c", "d", "b"]);
}

#[test]
fn src_isolated_from_edges_returns_none() {
    let node_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        label_field(),
    ]));
    let node_batch = RecordBatch::try_new(
        node_schema,
        vec![
            Arc::new(StringArray::from(vec!["x", "y"])) as Arc<dyn arrow_array::Array>,
            Arc::new(empty_labels(2)) as Arc<dyn arrow_array::Array>,
        ],
    )
    .unwrap();

    let edge_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_EDGE_SRC, DataType::Utf8, false),
        Field::new(COL_EDGE_DST, DataType::Utf8, false),
        Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
        Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
    ]));
    let edge_batch = RecordBatch::new_empty(edge_schema);
    let nf = NodeFrame::from_record_batch(node_batch).unwrap();
    let ef = EdgeFrame::from_record_batch(edge_batch).unwrap();
    let g = GraphFrame::new(nf, ef).unwrap();

    let path = g.shortest_path("x", "y", &cfg()).unwrap();
    assert_eq!(path, None);
}

// ── Direction filter ──────────────────────────────────────────────────────

#[test]
fn in_direction_follows_incoming_edges() {
    let g = make_unweighted_graph(&["a", "b", "c", "d"], &[("a", "b"), ("a", "c"), ("b", "d")]);
    let config = ShortestPathConfig {
        direction: Direction::In,
        ..Default::default()
    };
    let path = g.shortest_path("d", "a", &config).unwrap();
    assert_eq!(
        path,
        Some(vec!["d".to_owned(), "b".to_owned(), "a".to_owned()])
    );
}

#[test]
fn out_direction_cannot_traverse_incoming_edge() {
    let g = make_unweighted_graph(&["a", "b"], &[("b", "a")]);
    let path = g.shortest_path("a", "b", &cfg()).unwrap();
    assert_eq!(path, None);
}

// ── Edge type filter ──────────────────────────────────────────────────────

#[test]
fn edge_type_filter_excludes_wrong_type() {
    let node_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        label_field(),
    ]));
    let node_batch = RecordBatch::try_new(
        node_schema,
        vec![
            Arc::new(StringArray::from(vec!["a", "b", "c"])) as Arc<dyn arrow_array::Array>,
            Arc::new(empty_labels(3)) as Arc<dyn arrow_array::Array>,
        ],
    )
    .unwrap();

    let edge_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_EDGE_SRC, DataType::Utf8, false),
        Field::new(COL_EDGE_DST, DataType::Utf8, false),
        Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
        Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        edge_schema,
        vec![
            Arc::new(StringArray::from(vec!["a", "b", "a"])) as Arc<dyn arrow_array::Array>,
            Arc::new(StringArray::from(vec!["b", "c", "c"])) as Arc<dyn arrow_array::Array>,
            Arc::new(StringArray::from(vec!["KNOWS", "KNOWS", "LIKES"]))
                as Arc<dyn arrow_array::Array>,
            Arc::new(Int8Array::from(vec![0i8, 0, 0])) as Arc<dyn arrow_array::Array>,
        ],
    )
    .unwrap();

    let nf = NodeFrame::from_record_batch(node_batch).unwrap();
    let ef = EdgeFrame::from_record_batch(edge_batch).unwrap();
    let g = GraphFrame::new(nf, ef).unwrap();

    let config = ShortestPathConfig {
        edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
        ..Default::default()
    };
    let path = g.shortest_path("a", "c", &config).unwrap().unwrap();
    assert_eq!(path, vec!["a", "b", "c"]);
}

// ── Error cases ───────────────────────────────────────────────────────────

#[test]
fn unknown_src_returns_node_not_found() {
    let g = make_unweighted_graph(&["a", "b"], &[("a", "b")]);
    let err = g.shortest_path("ghost", "b", &cfg()).unwrap_err();
    assert!(matches!(err, GFError::NodeNotFound { id } if id == "ghost"));
}

#[test]
fn unknown_dst_returns_node_not_found() {
    let g = make_unweighted_graph(&["a", "b"], &[("a", "b")]);
    let err = g.shortest_path("a", "ghost", &cfg()).unwrap_err();
    assert!(matches!(err, GFError::NodeNotFound { id } if id == "ghost"));
}

#[test]
fn missing_weight_col_returns_column_not_found() {
    let g = make_unweighted_graph(&["a", "b"], &[("a", "b")]);
    let config = ShortestPathConfig {
        weight_col: Some("nonexistent".to_owned()),
        ..Default::default()
    };
    let err = g.shortest_path("a", "b", &config).unwrap_err();
    assert!(matches!(err, GFError::ColumnNotFound { column } if column == "nonexistent"));
}

#[test]
fn negative_weight_returns_negative_weight_error() {
    let g = make_weighted_graph(&["a", "b"], &[("a", "b", -1.0)]);
    let err = g.shortest_path("a", "b", &wcfg()).unwrap_err();
    assert!(matches!(err, GFError::NegativeWeight { .. }));
}

// ── all_shortest_paths ────────────────────────────────────────────────────

#[test]
fn all_paths_single_path_returned() {
    let g = make_unweighted_graph(&["a", "b", "c"], &[("a", "b"), ("b", "c")]);
    let paths = g.all_shortest_paths("a", "c", &cfg()).unwrap();
    assert_eq!(paths.len(), 1);
    assert_eq!(paths[0], vec!["a", "b", "c"]);
}

#[test]
fn all_paths_two_equal_cost_paths() {
    let g = make_weighted_graph(
        &["a", "b", "c"],
        &[("a", "b", 2.0), ("a", "c", 1.0), ("c", "b", 1.0)],
    );
    let mut paths = g.all_shortest_paths("a", "b", &wcfg()).unwrap();
    paths.sort();
    assert_eq!(paths.len(), 2);
    assert!(paths.contains(&vec!["a".to_owned(), "b".to_owned()]));
    assert!(paths.contains(&vec!["a".to_owned(), "c".to_owned(), "b".to_owned()]));
}

#[test]
fn all_paths_no_path_returns_empty() {
    let g = make_unweighted_graph(&["a", "b"], &[("b", "a")]);
    let paths = g.all_shortest_paths("a", "b", &cfg()).unwrap();
    assert!(paths.is_empty());
}

#[test]
fn all_paths_src_equals_dst() {
    let g = make_unweighted_graph(&["a", "b"], &[("a", "b")]);
    let paths = g.all_shortest_paths("a", "a", &cfg()).unwrap();
    assert_eq!(paths, vec![vec!["a".to_owned()]]);
}

#[test]
fn all_paths_three_parallel_routes() {
    let g = make_weighted_graph(
        &["a", "b", "c", "d", "e"],
        &[
            ("a", "b", 1.0),
            ("a", "c", 1.0),
            ("a", "d", 1.0),
            ("b", "e", 1.0),
            ("c", "e", 1.0),
            ("d", "e", 1.0),
        ],
    );
    let paths = g.all_shortest_paths("a", "e", &wcfg()).unwrap();
    assert_eq!(paths.len(), 3);
    for path in &paths {
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], "a");
        assert_eq!(path[2], "e");
    }
}

#[test]
fn all_paths_unknown_src_returns_error() {
    let g = make_unweighted_graph(&["a", "b"], &[("a", "b")]);
    let err = g.all_shortest_paths("ghost", "b", &cfg()).unwrap_err();
    assert!(matches!(err, GFError::NodeNotFound { id } if id == "ghost"));
}
