use std::sync::Arc;

use arrow_array::{
    builder::{ListBuilder, StringBuilder},
    Int8Array, RecordBatch, StringArray, UInt32Array,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};

use lynxes_core::{
    EdgeFrame, GraphFrame, NodeFrame, COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC,
    COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
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

fn cid_of(result: &NodeFrame, id: &str) -> u32 {
    let row = result.row_index(id).expect("id present") as usize;
    result
        .column("component_id")
        .expect("component_id column present")
        .as_any()
        .downcast_ref::<UInt32Array>()
        .expect("component_id is UInt32")
        .value(row)
}

// ── connected_components ──────────────────────────────────────────────────

#[test]
fn empty_graph_returns_empty_nodeframe_with_component_id_column() {
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

    let result = g.connected_components().unwrap();
    assert_eq!(result.len(), 0);
    assert!(result.column("component_id").is_some());
}

#[test]
fn single_isolated_node_gets_component_zero() {
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

    let result = g.connected_components().unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(cid_of(&result, "solo"), 0);
}

#[test]
fn two_connected_nodes_share_component_zero() {
    let g = make_graph(&["a", "b"], &[("a", "b")]);
    let result = g.connected_components().unwrap();
    assert_eq!(cid_of(&result, "a"), 0);
    assert_eq!(cid_of(&result, "b"), 0);
}

#[test]
fn directed_edge_creates_undirected_connectivity() {
    let g = make_graph(&["a", "b", "c"], &[("a", "b"), ("c", "b")]);
    let result = g.connected_components().unwrap();
    let ca = cid_of(&result, "a");
    let cb = cid_of(&result, "b");
    let cc = cid_of(&result, "c");
    assert_eq!(ca, cb);
    assert_eq!(cb, cc);
}

#[test]
fn two_disconnected_pairs_get_different_component_ids() {
    let g = make_graph(&["a", "b", "c", "d"], &[("a", "b"), ("c", "d")]);
    let result = g.connected_components().unwrap();

    let ca = cid_of(&result, "a");
    let cb = cid_of(&result, "b");
    let cc = cid_of(&result, "c");
    let cd = cid_of(&result, "d");

    assert_eq!(ca, cb, "a and b must share a component");
    assert_eq!(cc, cd, "c and d must share a component");
    assert_ne!(ca, cc, "the two components must differ");
}

#[test]
fn component_ids_assigned_in_row_discovery_order() {
    let g = make_graph(&["a", "b", "c", "d"], &[("a", "b"), ("c", "d")]);
    let result = g.connected_components().unwrap();

    assert_eq!(cid_of(&result, "a"), 0);
    assert_eq!(cid_of(&result, "b"), 0);
    assert_eq!(cid_of(&result, "c"), 1);
    assert_eq!(cid_of(&result, "d"), 1);
}

#[test]
fn three_isolated_nodes_get_sequential_component_ids() {
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
            Arc::new(StringArray::from(vec!["x", "y", "z"])) as Arc<dyn arrow_array::Array>,
            Arc::new(empty_labels(3)) as Arc<dyn arrow_array::Array>,
        ],
    )
    .unwrap();
    let nf = NodeFrame::from_record_batch(node_batch).unwrap();
    let ef = EdgeFrame::from_record_batch(RecordBatch::new_empty(edge_schema)).unwrap();
    let g = GraphFrame::new(nf, ef).unwrap();

    let result = g.connected_components().unwrap();
    assert_eq!(cid_of(&result, "x"), 0);
    assert_eq!(cid_of(&result, "y"), 1);
    assert_eq!(cid_of(&result, "z"), 2);
}

#[test]
fn output_columns_are_id_label_component_id_only() {
    let g = make_graph(&["a", "b"], &[("a", "b")]);
    let result = g.connected_components().unwrap();

    assert!(result.column(COL_NODE_ID).is_some());
    assert!(result.column(COL_NODE_LABEL).is_some());
    assert!(result.column("component_id").is_some());
    assert_eq!(result.column_names().len(), 3);
}

#[test]
fn output_row_order_matches_input_node_frame() {
    let g = make_graph(&["z", "a", "m"], &[("z", "a"), ("a", "m")]);
    let result = g.connected_components().unwrap();

    assert_eq!(result.row_index("z"), Some(0));
    assert_eq!(result.row_index("a"), Some(1));
    assert_eq!(result.row_index("m"), Some(2));
}

#[test]
fn all_nodes_in_same_component_get_id_zero() {
    let g = make_graph(&["a", "b", "c", "d"], &[("a", "b"), ("b", "c"), ("c", "d")]);
    let result = g.connected_components().unwrap();
    for id in &["a", "b", "c", "d"] {
        assert_eq!(cid_of(&result, id), 0, "{} should be in component 0", id);
    }
}

#[test]
fn cycle_with_detached_node_produces_two_components() {
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
            Arc::new(StringArray::from(vec!["a", "b", "c", "d"])) as Arc<dyn arrow_array::Array>,
            Arc::new(empty_labels(4)) as Arc<dyn arrow_array::Array>,
        ],
    )
    .unwrap();
    let edge_batch = RecordBatch::try_new(
        edge_schema,
        vec![
            Arc::new(StringArray::from(vec!["a", "b", "c"])) as Arc<dyn arrow_array::Array>,
            Arc::new(StringArray::from(vec!["b", "c", "a"])) as Arc<dyn arrow_array::Array>,
            Arc::new(StringArray::from(vec!["E", "E", "E"])) as Arc<dyn arrow_array::Array>,
            Arc::new(Int8Array::from(vec![0i8, 0, 0])) as Arc<dyn arrow_array::Array>,
        ],
    )
    .unwrap();
    let nf = NodeFrame::from_record_batch(node_batch).unwrap();
    let ef = EdgeFrame::from_record_batch(edge_batch).unwrap();
    let g = GraphFrame::new(nf, ef).unwrap();

    let result = g.connected_components().unwrap();
    let ca = cid_of(&result, "a");
    let cb = cid_of(&result, "b");
    let cc = cid_of(&result, "c");
    let cd = cid_of(&result, "d");

    assert_eq!(ca, cb);
    assert_eq!(cb, cc);
    assert_ne!(ca, cd);
    assert_eq!(ca, 0);
    assert_eq!(cd, 1);
}

// ── largest_connected_component ───────────────────────────────────────────

#[test]
fn largest_cc_empty_graph_returns_empty_graphframe() {
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

    let lcc = g.largest_connected_component().unwrap();
    assert_eq!(lcc.node_count(), 0);
    assert_eq!(lcc.edge_count(), 0);
}

#[test]
fn largest_cc_returns_bigger_of_two_components() {
    let g = make_graph(
        &["a", "b", "c", "d", "e"],
        &[("a", "b"), ("b", "c"), ("d", "e")],
    );
    let lcc = g.largest_connected_component().unwrap();

    assert_eq!(lcc.node_count(), 3);
    assert!(lcc.nodes().row_index("a").is_some());
    assert!(lcc.nodes().row_index("b").is_some());
    assert!(lcc.nodes().row_index("c").is_some());
    assert!(lcc.nodes().row_index("d").is_none());
    assert!(lcc.nodes().row_index("e").is_none());
}

#[test]
fn largest_cc_edges_are_induced_by_result_nodes() {
    let g = make_graph(
        &["a", "b", "c", "d"],
        &[("a", "b"), ("b", "c"), ("c", "d"), ("d", "c")],
    );
    let lcc = g.largest_connected_component().unwrap();
    assert_eq!(lcc.node_count(), 4);
    assert_eq!(lcc.edge_count(), 4);
}

#[test]
fn largest_cc_tie_broken_by_lowest_component_id() {
    let g = make_graph(&["a", "b", "c", "d"], &[("a", "b"), ("c", "d")]);
    let lcc = g.largest_connected_component().unwrap();

    assert_eq!(lcc.node_count(), 2);
    assert!(
        lcc.nodes().row_index("a").is_some(),
        "component 0 (a,b) should win tie"
    );
    assert!(lcc.nodes().row_index("b").is_some());
    assert!(lcc.nodes().row_index("c").is_none());
    assert!(lcc.nodes().row_index("d").is_none());
}

#[test]
fn largest_cc_single_node_graph_returns_that_node() {
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

    let lcc = g.largest_connected_component().unwrap();
    assert_eq!(lcc.node_count(), 1);
    assert!(lcc.nodes().row_index("solo").is_some());
}

#[test]
fn largest_cc_fully_connected_graph_returns_all_nodes() {
    let g = make_graph(
        &["a", "b", "c", "d"],
        &[("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")],
    );
    let lcc = g.largest_connected_component().unwrap();
    assert_eq!(lcc.node_count(), 4);
    assert_eq!(lcc.edge_count(), 4);
}
