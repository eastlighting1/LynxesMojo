mod common;

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Float64Array, Int64Array, ListArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lynxes_core::{
    BetweennessConfig, EdgeFrame, GFError, GraphFrame, NodeFrame, COL_EDGE_DIRECTION, COL_EDGE_DST,
    COL_EDGE_SRC, COL_EDGE_TYPE, COL_NODE_ID,
};

use common::{label_field, labels_array};

fn nodes(ids: &[&str]) -> NodeFrame {
    let labels: Vec<&[&str]> = ids.iter().map(|_| &["Node"][..]).collect();
    let labels_array: ListArray = labels_array(&labels);
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        label_field(),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(ids.to_vec())) as ArrayRef,
            Arc::new(labels_array) as ArrayRef,
        ],
    )
    .unwrap();
    NodeFrame::from_record_batch(batch).unwrap()
}

fn edges_with_optional_weight(
    src: &[&str],
    dst: &[&str],
    edge_type: &[&str],
    weights: Option<&[i64]>,
) -> EdgeFrame {
    let mut fields = vec![
        Field::new(COL_EDGE_SRC, DataType::Utf8, false),
        Field::new(COL_EDGE_DST, DataType::Utf8, false),
        Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
        Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
    ];
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(src.to_vec())) as ArrayRef,
        Arc::new(StringArray::from(dst.to_vec())) as ArrayRef,
        Arc::new(StringArray::from(edge_type.to_vec())) as ArrayRef,
        Arc::new(arrow_array::Int8Array::from(vec![0i8; src.len()])) as ArrayRef,
    ];

    if let Some(weights) = weights {
        fields.push(Field::new("weight", DataType::Int64, false));
        columns.push(Arc::new(Int64Array::from(weights.to_vec())) as ArrayRef);
    }

    let batch = RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), columns).unwrap();
    EdgeFrame::from_record_batch(batch).unwrap()
}

fn value_by_id(frame: &NodeFrame, id: &str, column: &str) -> f64 {
    let row = frame.row_index(id).unwrap() as usize;
    let values = frame
        .column(column)
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    values.value(row)
}

#[test]
fn betweenness_path_graph_marks_middle_node() {
    let graph = GraphFrame::new(
        nodes(&["a", "b", "c"]),
        edges_with_optional_weight(&["a", "b"], &["b", "c"], &["LINK", "LINK"], None),
    )
    .unwrap();

    let result = graph.betweenness_centrality().unwrap();

    assert_eq!(value_by_id(&result, "a", "betweenness"), 0.0);
    assert!((value_by_id(&result, "b", "betweenness") - 0.5).abs() < 1e-12);
    assert_eq!(value_by_id(&result, "c", "betweenness"), 0.0);
}

#[test]
fn betweenness_weighted_graph_changes_shortest_path_contribution() {
    let graph = GraphFrame::new(
        nodes(&["a", "b", "c", "d"]),
        edges_with_optional_weight(
            &["a", "b", "a", "c"],
            &["b", "d", "c", "d"],
            &["LINK", "LINK", "LINK", "LINK"],
            Some(&[1, 1, 1, 5]),
        ),
    )
    .unwrap();

    let unweighted = graph.betweenness_centrality().unwrap();
    let weighted = graph
        .betweenness_centrality_with_config(&BetweennessConfig {
            weight_col: Some("weight".to_owned()),
        })
        .unwrap();

    let b_unweighted = value_by_id(&unweighted, "b", "betweenness");
    let c_unweighted = value_by_id(&unweighted, "c", "betweenness");
    let b_weighted = value_by_id(&weighted, "b", "betweenness");
    let c_weighted = value_by_id(&weighted, "c", "betweenness");

    assert!((b_unweighted - c_unweighted).abs() < 1e-12);
    assert!(b_weighted > b_unweighted);
    assert_eq!(c_weighted, 0.0);
}

#[test]
fn betweenness_rejects_negative_weight_column() {
    let graph = GraphFrame::new(
        nodes(&["a", "b"]),
        edges_with_optional_weight(&["a"], &["b"], &["LINK"], Some(&[-1])),
    )
    .unwrap();

    let err = graph
        .betweenness_centrality_with_config(&BetweennessConfig {
            weight_col: Some("weight".to_owned()),
        })
        .unwrap_err();

    assert!(matches!(err, GFError::NegativeWeight { column } if column == "weight"));
}
