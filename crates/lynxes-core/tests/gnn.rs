mod common;

use std::collections::HashSet;

use arrow_array::{Array, StringArray};
use lynxes_core::{Direction, EdgeTypeSpec};

use common::sample_graph;

fn adjacency_from_coo(include_reverse: bool) -> HashSet<(u32, u32)> {
    let graph = sample_graph();
    let (src, dst) = graph.to_coo();
    let mut edges = HashSet::new();

    for (src_idx, dst_idx) in src
        .iter()
        .map(|v| v.unwrap() as u32)
        .zip(dst.iter().map(|v| v.unwrap() as u32))
    {
        edges.insert((src_idx, dst_idx));
        if include_reverse {
            edges.insert((dst_idx, src_idx));
        }
    }

    edges
}

#[test]
fn gnn_to_coo_returns_graph_topology_in_edge_local_index_space() {
    let graph = sample_graph();
    let (src, dst) = graph.to_coo();

    let pairs: Vec<(i64, i64)> = src
        .iter()
        .map(|v| v.unwrap())
        .zip(dst.iter().map(|v| v.unwrap()))
        .collect();

    assert_eq!(pairs, vec![(0, 1), (0, 2), (1, 2), (3, 0)]);
    assert_eq!(pairs.len(), graph.edge_count());
}

#[test]
fn gnn_gather_rows_aligns_with_sampled_node_row_ids() {
    let graph = sample_graph();
    let sampled = graph
        .sample_neighbors(
            &["alice"],
            &lynxes_core::SamplingConfig {
                hops: 1,
                fan_out: vec![8],
                ..Default::default()
            },
        )
        .unwrap();

    let batch = graph.nodes().gather_rows(&sampled.node_row_ids).unwrap();
    let ids = batch
        .column_by_name(lynxes_core::COL_NODE_ID)
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|v| v.unwrap().to_owned())
        .collect::<Vec<_>>();

    assert_eq!(sampled.node_row_ids, vec![0, 1, 2]);
    assert_eq!(ids, vec!["alice", "bob", "charlie"]);
}

#[test]
fn gnn_sample_neighbors_fanout_cap_returns_real_neighbors_only() {
    let graph = sample_graph();
    let sampled = graph
        .sample_neighbors(
            &["alice"],
            &lynxes_core::SamplingConfig {
                hops: 1,
                fan_out: vec![1],
                direction: Direction::Out,
                edge_type: EdgeTypeSpec::Any,
                replace: false,
            },
        )
        .unwrap();

    let adjacency = adjacency_from_coo(false);
    assert_eq!(sampled.node_indices.len(), 2);
    assert_eq!(sampled.node_row_ids.len(), 2);
    assert_eq!(sampled.edge_src.len(), 1);
    assert_eq!(sampled.edge_dst.len(), 1);
    assert!(adjacency.contains(&(sampled.edge_src[0], sampled.edge_dst[0])));
    assert_eq!(sampled.edge_src[0], 0);
    assert!(matches!(sampled.edge_dst[0], 1 | 2));

    let batch = graph.nodes().gather_rows(&sampled.node_row_ids).unwrap();
    let ids = batch
        .column_by_name(lynxes_core::COL_NODE_ID)
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|v| v.unwrap().to_owned())
        .collect::<Vec<_>>();

    assert_eq!(ids[0], "alice");
    assert!(matches!(ids[1].as_str(), "bob" | "charlie"));
}

#[test]
fn gnn_random_walk_returns_length_bounded_valid_paths() {
    let graph = sample_graph();
    let adjacency = adjacency_from_coo(true);

    let walks = graph
        .random_walk(&["alice", "bob"], 3, 4, Direction::Both, &EdgeTypeSpec::Any)
        .unwrap();

    assert_eq!(walks.len(), 8);
    for walk in walks {
        assert!((1..=4).contains(&walk.len()));
        for pair in walk.windows(2) {
            assert!(adjacency.contains(&(pair[0], pair[1])));
        }
    }
}
