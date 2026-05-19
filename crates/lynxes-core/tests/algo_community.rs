mod common;

use std::sync::Arc;

use arrow_array::{ArrayRef, Int8Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lynxes_core::{
    CommunityAlgorithm, CommunityConfig, EdgeFrame, GFError, GraphFrame, NodeFrame,
    COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE,
};

use common::label_field;

#[test]
fn community_detection_returns_uint32_column() {
    let graph = bridge_graph();
    let communities = graph
        .community_detection(CommunityConfig::default())
        .unwrap();

    assert_eq!(
        communities.column_names(),
        vec![
            lynxes_core::COL_NODE_ID,
            lynxes_core::COL_NODE_LABEL,
            "community_id"
        ]
    );
    assert_eq!(
        communities
            .schema()
            .field_with_name("community_id")
            .unwrap()
            .data_type(),
        &DataType::UInt32
    );
}

#[test]
fn louvain_splits_two_cliques_connected_by_bridge() {
    let graph = bridge_graph();
    let communities = graph
        .community_detection(CommunityConfig::default())
        .unwrap();
    let community_col = communities
        .column("community_id")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow_array::UInt32Array>()
        .unwrap();

    let left = community_col.value(communities.row_index("a").unwrap() as usize);
    let b = community_col.value(communities.row_index("b").unwrap() as usize);
    let c = community_col.value(communities.row_index("c").unwrap() as usize);
    let right = community_col.value(communities.row_index("d").unwrap() as usize);
    let e = community_col.value(communities.row_index("e").unwrap() as usize);
    let f = community_col.value(communities.row_index("f").unwrap() as usize);

    assert_eq!(left, b);
    assert_eq!(b, c);
    assert_eq!(right, e);
    assert_eq!(e, f);
    assert_ne!(left, right);
}

#[test]
fn community_detection_supports_seeded_execution() {
    let graph = bridge_graph();
    let config = CommunityConfig {
        algorithm: CommunityAlgorithm::Louvain,
        resolution: 1.0,
        seed: Some(7),
    };

    let first = graph.community_detection(config.clone()).unwrap();
    let second = graph.community_detection(config).unwrap();

    assert_eq!(first.to_record_batch(), second.to_record_batch());
}

#[test]
fn invalid_resolution_is_rejected() {
    let graph = bridge_graph();
    let err = graph
        .community_detection(CommunityConfig {
            algorithm: CommunityAlgorithm::Louvain,
            resolution: 0.0,
            seed: None,
        })
        .unwrap_err();

    assert!(matches!(err, GFError::InvalidConfig { .. }));
}

fn bridge_graph() -> GraphFrame {
    let node_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(lynxes_core::COL_NODE_ID, DataType::Utf8, false),
        label_field(),
    ]));
    let nodes = NodeFrame::from_record_batch(
        RecordBatch::try_new(
            node_schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e", "f"])) as ArrayRef,
                Arc::new(common::labels_array(&[
                    &["Node"],
                    &["Node"],
                    &["Node"],
                    &["Node"],
                    &["Node"],
                    &["Node"],
                ])) as ArrayRef,
            ],
        )
        .unwrap(),
    )
    .unwrap();

    let edge_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_EDGE_SRC, DataType::Utf8, false),
        Field::new(COL_EDGE_DST, DataType::Utf8, false),
        Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
        Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
    ]));
    let edges = NodeEdgeBuilder::bridge_edges(edge_schema);

    GraphFrame::new(nodes, edges).unwrap()
}

struct NodeEdgeBuilder;

impl NodeEdgeBuilder {
    fn bridge_edges(schema: Arc<ArrowSchema>) -> EdgeFrame {
        let directed_edges = vec![
            ("a", "b"),
            ("b", "a"),
            ("a", "c"),
            ("c", "a"),
            ("b", "c"),
            ("c", "b"),
            ("d", "e"),
            ("e", "d"),
            ("d", "f"),
            ("f", "d"),
            ("e", "f"),
            ("f", "e"),
            ("c", "d"),
            ("d", "c"),
        ];
        let src: Vec<&str> = directed_edges.iter().map(|(src, _)| *src).collect();
        let dst: Vec<&str> = directed_edges.iter().map(|(_, dst)| *dst).collect();
        let len = directed_edges.len();

        EdgeFrame::from_record_batch(
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(src)) as ArrayRef,
                    Arc::new(StringArray::from(dst)) as ArrayRef,
                    Arc::new(StringArray::from(vec!["LINK"; len])) as ArrayRef,
                    Arc::new(Int8Array::from(vec![0i8; len])) as ArrayRef,
                ],
            )
            .unwrap(),
        )
        .unwrap()
    }
}
