use std::sync::Arc;

use arrow_array::{
    builder::{ListBuilder, StringBuilder},
    Float64Array, Int8Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};

use lynxes_core::{
    EdgeFrame, GFError, GraphFrame, NodeFrame, PageRankConfig, COL_EDGE_DIRECTION, COL_EDGE_DST,
    COL_EDGE_SRC, COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
};

// ── Test graph builders ───────────────────────────────────────────────────

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
        Field::new("weight", DataType::Float64, true),
    ]));
    let srcs: Vec<&str> = edges.iter().map(|e| e.0).collect();
    let dsts: Vec<&str> = edges.iter().map(|e| e.1).collect();
    let wts: Vec<Option<f64>> = edges.iter().map(|e| Some(e.2)).collect();
    let edge_batch = RecordBatch::try_new(
        edge_schema,
        vec![
            Arc::new(StringArray::from(srcs)) as Arc<dyn arrow_array::Array>,
            Arc::new(StringArray::from(dsts)) as Arc<dyn arrow_array::Array>,
            Arc::new(StringArray::from(vec!["E"; edges.len()])) as Arc<dyn arrow_array::Array>,
            Arc::new(Int8Array::from(vec![0i8; edges.len()])) as Arc<dyn arrow_array::Array>,
            Arc::new(Float64Array::from(wts)) as Arc<dyn arrow_array::Array>,
        ],
    )
    .unwrap();

    let nf = NodeFrame::from_record_batch(node_batch).unwrap();
    let ef = EdgeFrame::from_record_batch(edge_batch).unwrap();
    GraphFrame::new(nf, ef).unwrap()
}

fn wcfg() -> PageRankConfig {
    PageRankConfig {
        weight_col: Some("weight".to_owned()),
        ..Default::default()
    }
}

fn pr_of(result: &NodeFrame, id: &str) -> f64 {
    let row = result.row_index(id).expect("id present in result") as usize;
    result
        .column("pagerank")
        .expect("pagerank column present")
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("pagerank is Float64")
        .value(row)
}

// ── Basic correctness ─────────────────────────────────────────────────────

#[test]
fn empty_graph_returns_empty_nodeframe_with_pagerank_column() {
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
    let nf = NodeFrame::from_record_batch(RecordBatch::new_empty(node_schema)).unwrap();
    let ef = EdgeFrame::from_record_batch(RecordBatch::new_empty(edge_schema)).unwrap();
    let g = GraphFrame::new(nf, ef).unwrap();

    let result = g.pagerank(&PageRankConfig::default()).unwrap();
    assert_eq!(result.len(), 0);
    assert!(result.column("pagerank").is_some());
}

#[test]
fn single_dangling_node_converges_to_one() {
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
    let node_batch = RecordBatch::try_new(
        node_schema,
        vec![
            Arc::new(StringArray::from(vec!["solo"])) as Arc<dyn arrow_array::Array>,
            Arc::new(empty_labels(1)) as Arc<dyn arrow_array::Array>,
        ],
    )
    .unwrap();
    let nf = NodeFrame::from_record_batch(node_batch).unwrap();
    let ef = EdgeFrame::from_record_batch(RecordBatch::new_empty(edge_schema)).unwrap();
    let g = GraphFrame::new(nf, ef).unwrap();

    let result = g.pagerank(&PageRankConfig::default()).unwrap();
    assert_eq!(result.len(), 1);
    let r = pr_of(&result, "solo");
    assert!((r - 1.0).abs() < 1e-9, "expected 1.0, got {}", r);
}

#[test]
fn symmetric_ring_converges_to_equal_ranks() {
    let g = make_unweighted_graph(&["a", "b", "c"], &[("a", "b"), ("b", "c"), ("c", "a")]);
    let result = g.pagerank(&PageRankConfig::default()).unwrap();
    assert_eq!(result.len(), 3);

    let expected = 1.0 / 3.0;
    for id in &["a", "b", "c"] {
        let r = pr_of(&result, id);
        assert!(
            (r - expected).abs() < 1e-5,
            "rank of {} = {} (expected ≈ {:.6})",
            id,
            r,
            expected
        );
    }
}

#[test]
fn hub_node_gets_highest_rank() {
    let g = make_unweighted_graph(
        &["hub", "a", "b", "c"],
        &[("a", "hub"), ("b", "hub"), ("c", "hub"), ("hub", "a")],
    );
    let result = g.pagerank(&PageRankConfig::default()).unwrap();
    let pr_hub = pr_of(&result, "hub");
    let pr_a = pr_of(&result, "a");
    let pr_b = pr_of(&result, "b");
    let pr_c = pr_of(&result, "c");

    assert!(pr_hub > pr_a, "hub({}) should beat a({})", pr_hub, pr_a);
    assert!(pr_a > pr_b, "a({}) should beat b({})", pr_a, pr_b);
    assert!((pr_b - pr_c).abs() < 1e-9, "b and c should be equal");
}

#[test]
fn dangling_node_mass_distributes_to_all() {
    let g = make_unweighted_graph(&["src", "sink"], &[("src", "sink")]);
    let result = g.pagerank(&PageRankConfig::default()).unwrap();
    let pr_sink = pr_of(&result, "sink");
    let pr_src = pr_of(&result, "src");

    assert!(pr_sink > 0.0);
    assert!(pr_src > 0.0);
    let total = pr_sink + pr_src;
    assert!(
        (total - 1.0).abs() < 1e-6,
        "ranks must sum to 1 (got {})",
        total
    );
}

#[test]
fn ranks_sum_to_one() {
    let g = make_unweighted_graph(
        &["a", "b", "c", "d"],
        &[("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("a", "c")],
    );
    let result = g.pagerank(&PageRankConfig::default()).unwrap();
    let sum: f64 = (0..result.len() as u32)
        .map(|r| {
            result
                .column("pagerank")
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .value(r as usize)
        })
        .sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "ranks sum = {} (expected 1.0)",
        sum
    );
}

#[test]
fn output_row_order_matches_input_node_frame() {
    let nodes = &["z", "a", "m"];
    let g = make_unweighted_graph(nodes, &[("a", "z"), ("m", "a"), ("z", "m")]);
    let result = g.pagerank(&PageRankConfig::default()).unwrap();

    assert_eq!(result.row_index("z"), Some(0));
    assert_eq!(result.row_index("a"), Some(1));
    assert_eq!(result.row_index("m"), Some(2));
}

#[test]
fn output_has_id_label_pagerank_columns() {
    let g = make_unweighted_graph(&["a", "b"], &[("a", "b"), ("b", "a")]);
    let result = g.pagerank(&PageRankConfig::default()).unwrap();

    assert!(result.column(COL_NODE_ID).is_some(), "missing _id");
    assert!(result.column(COL_NODE_LABEL).is_some(), "missing _label");
    assert!(result.column("pagerank").is_some(), "missing pagerank");
    assert_eq!(result.column_names().len(), 3);
}

// ── Weighted PageRank ─────────────────────────────────────────────────────

#[test]
fn weighted_high_weight_destination_gets_more_rank() {
    let g = make_weighted_graph(
        &["a", "b", "c"],
        &[
            ("a", "b", 1.0),
            ("a", "c", 9.0),
            ("b", "a", 1.0),
            ("c", "a", 1.0),
        ],
    );
    let result = g.pagerank(&wcfg()).unwrap();
    let pr_b = pr_of(&result, "b");
    let pr_c = pr_of(&result, "c");

    assert!(
        pr_c > pr_b,
        "c({}) should outrank b({}) due to higher edge weight",
        pr_c,
        pr_b
    );
}

#[test]
fn zero_weight_edge_makes_node_dangling() {
    let g = make_weighted_graph(&["a", "b"], &[("a", "b", 0.0), ("b", "a", 1.0)]);
    let result = g.pagerank(&wcfg()).unwrap();
    let sum = pr_of(&result, "a") + pr_of(&result, "b");
    assert!((sum - 1.0).abs() < 1e-6);
}

// ── Iteration control ─────────────────────────────────────────────────────

#[test]
fn max_iter_one_returns_one_step_result() {
    let g = make_unweighted_graph(&["a", "b"], &[("a", "b"), ("b", "a")]);
    let config = PageRankConfig {
        max_iter: 1,
        ..Default::default()
    };
    let result = g.pagerank(&config).unwrap();

    assert_eq!(result.len(), 2);
    assert!(result.column("pagerank").is_some());
}

#[test]
fn converges_faster_than_max_iter() {
    let g = make_unweighted_graph(&["a", "b"], &[("a", "b"), ("b", "a")]);
    let config = PageRankConfig {
        max_iter: 1000,
        epsilon: 1e-12,
        ..Default::default()
    };
    let result = g.pagerank(&config).unwrap();

    let pr_a = pr_of(&result, "a");
    let pr_b = pr_of(&result, "b");
    assert!((pr_a - pr_b).abs() < 1e-10, "expected equal ranks");
    assert!((pr_a + pr_b - 1.0).abs() < 1e-10, "ranks must sum to 1");
}

// ── Config validation ─────────────────────────────────────────────────────

#[test]
fn damping_zero_returns_invalid_config() {
    let g = make_unweighted_graph(&["a", "b"], &[("a", "b")]);
    let config = PageRankConfig {
        damping: 0.0,
        ..Default::default()
    };
    let err = g.pagerank(&config).unwrap_err();
    assert!(matches!(err, GFError::InvalidConfig { .. }));
}

#[test]
fn damping_one_returns_invalid_config() {
    let g = make_unweighted_graph(&["a", "b"], &[("a", "b")]);
    let config = PageRankConfig {
        damping: 1.0,
        ..Default::default()
    };
    let err = g.pagerank(&config).unwrap_err();
    assert!(matches!(err, GFError::InvalidConfig { .. }));
}

#[test]
fn damping_negative_returns_invalid_config() {
    let g = make_unweighted_graph(&["a", "b"], &[("a", "b")]);
    let config = PageRankConfig {
        damping: -0.1,
        ..Default::default()
    };
    let err = g.pagerank(&config).unwrap_err();
    assert!(matches!(err, GFError::InvalidConfig { .. }));
}

#[test]
fn max_iter_zero_returns_invalid_config() {
    let g = make_unweighted_graph(&["a", "b"], &[("a", "b")]);
    let config = PageRankConfig {
        max_iter: 0,
        ..Default::default()
    };
    let err = g.pagerank(&config).unwrap_err();
    assert!(matches!(err, GFError::InvalidConfig { .. }));
}

#[test]
fn epsilon_zero_returns_invalid_config() {
    let g = make_unweighted_graph(&["a", "b"], &[("a", "b")]);
    let config = PageRankConfig {
        epsilon: 0.0,
        ..Default::default()
    };
    let err = g.pagerank(&config).unwrap_err();
    assert!(matches!(err, GFError::InvalidConfig { .. }));
}

#[test]
fn epsilon_negative_returns_invalid_config() {
    let g = make_unweighted_graph(&["a", "b"], &[("a", "b")]);
    let config = PageRankConfig {
        epsilon: -1e-6,
        ..Default::default()
    };
    let err = g.pagerank(&config).unwrap_err();
    assert!(matches!(err, GFError::InvalidConfig { .. }));
}

// ── Weight column errors ──────────────────────────────────────────────────

#[test]
fn missing_weight_col_returns_column_not_found() {
    let g = make_unweighted_graph(&["a", "b"], &[("a", "b")]);
    let config = PageRankConfig {
        weight_col: Some("no_such_column".to_owned()),
        ..Default::default()
    };
    let err = g.pagerank(&config).unwrap_err();
    assert!(matches!(err, GFError::ColumnNotFound { .. }));
}

#[test]
fn negative_weight_returns_negative_weight_error() {
    let g = make_weighted_graph(&["a", "b"], &[("a", "b", -1.0)]);
    let err = g.pagerank(&wcfg()).unwrap_err();
    assert!(matches!(err, GFError::NegativeWeight { .. }));
}
