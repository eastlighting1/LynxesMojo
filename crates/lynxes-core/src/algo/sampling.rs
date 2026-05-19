use crate::{Direction, EdgeTypeSpec, GFError, GraphFrame, Result};

#[cfg(not(target_arch = "wasm32"))]
use hashbrown::HashSet;
#[cfg(not(target_arch = "wasm32"))]
use rand::{seq::index::sample, thread_rng, Rng};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

/// Configuration for neighborhood sampling and random-walk style GNN helpers.
///
/// `fan_out` is interpreted hop-by-hop. For example, `hops = 2` with
/// `fan_out = vec![25, 10]` means "sample up to 25 neighbors at hop 1, then
/// up to 10 neighbors per retained frontier node at hop 2".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SamplingConfig {
    pub hops: usize,
    pub fan_out: Vec<usize>,
    pub direction: Direction,
    pub edge_type: EdgeTypeSpec,
    pub replace: bool,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            hops: 1,
            fan_out: vec![25],
            direction: Direction::Out,
            edge_type: EdgeTypeSpec::Any,
            replace: false,
        }
    }
}

/// Lightweight graph fragment returned by sampling-oriented utilities.
///
/// All index-bearing vectors use the `EdgeFrame` local compact node index space
/// unless otherwise stated.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SampledSubgraph {
    pub node_indices: Vec<u32>,
    pub edge_src: Vec<u32>,
    pub edge_dst: Vec<u32>,
    pub edge_row_ids: Vec<u32>,
    pub node_row_ids: Vec<u32>,
}

#[cfg(not(target_arch = "wasm32"))]
impl GraphFrame {
    pub fn sample_neighbors(
        &self,
        seeds: &[&str],
        config: &SamplingConfig,
    ) -> Result<SampledSubgraph> {
        validate_sampling_config(config)?;

        let mut frontier = Vec::with_capacity(seeds.len());
        let mut seen_nodes = HashSet::new();
        let mut sampled = SampledSubgraph::default();

        for &seed in seeds {
            let _ = self
                .node_row_by_id(seed)
                .ok_or_else(|| GFError::NodeNotFound {
                    id: seed.to_owned(),
                })?;

            if let Some(edge_idx) = self.edges().node_row_idx(seed) {
                if seen_nodes.insert(edge_idx) {
                    push_sampled_node(self, &mut sampled, edge_idx);
                    frontier.push(edge_idx);
                }
            }
        }

        if frontier.is_empty() || config.hops == 0 {
            return Ok(sampled);
        }

        let mut rng = thread_rng();

        for hop in 0..config.hops {
            if frontier.is_empty() {
                break;
            }

            let fan_out = config.fan_out[hop];
            let mut next_frontier = Vec::new();

            for &src_idx in &frontier {
                let candidates = filtered_neighbor_candidates(self, src_idx, config);
                let picked =
                    sampled_neighbor_positions(candidates.len(), fan_out, config.replace, &mut rng);

                for pos in picked {
                    let (dst_idx, edge_row) = candidates[pos];

                    sampled.edge_src.push(src_idx);
                    sampled.edge_dst.push(dst_idx);
                    sampled.edge_row_ids.push(edge_row);

                    if seen_nodes.insert(dst_idx) {
                        push_sampled_node(self, &mut sampled, dst_idx);
                    }
                    next_frontier.push(dst_idx);
                }
            }

            frontier = next_frontier;
        }

        Ok(sampled)
    }

    /// Runs fixed-length random walks from the given start nodes.
    ///
    /// Each returned walk is expressed in the `EdgeFrame` local compact node
    /// index space. The first element is always the start node when that node
    /// participates in `EdgeFrame`. If a start node exists in `NodeFrame` but
    /// never appears in `_src` / `_dst`, Lynxes cannot assign it an edge-local
    /// index, so that walk is returned as empty and terminates immediately.
    pub fn random_walk(
        &self,
        start_nodes: &[&str],
        length: usize,
        walks_per_node: usize,
        direction: Direction,
        edge_type: &EdgeTypeSpec,
    ) -> Result<Vec<Vec<u32>>> {
        let starts = resolve_walk_starts(self, start_nodes)?;
        if walks_per_node == 0 || starts.is_empty() {
            return Ok(Vec::new());
        }

        let jobs: Vec<Option<u32>> = starts
            .iter()
            .flat_map(|start_idx| std::iter::repeat_n(*start_idx, walks_per_node))
            .collect();

        jobs.into_par_iter()
            .map(|start_idx| {
                let mut walk = Vec::with_capacity(length.saturating_add(1));
                let Some(mut current_idx) = start_idx else {
                    return Ok(walk);
                };

                let mut rng = thread_rng();
                walk.push(current_idx);

                for _ in 0..length {
                    let candidates = filtered_neighbor_candidates_with_spec(
                        self,
                        current_idx,
                        direction,
                        edge_type,
                    );
                    if candidates.is_empty() {
                        break;
                    }

                    let next_pos = rng.gen_range(0..candidates.len());
                    let (next_idx, _) = candidates[next_pos];
                    walk.push(next_idx);
                    current_idx = next_idx;
                }

                Ok(walk)
            })
            .collect()
    }
}

#[cfg(target_arch = "wasm32")]
impl GraphFrame {
    pub fn sample_neighbors(
        &self,
        _seeds: &[&str],
        _config: &SamplingConfig,
    ) -> Result<SampledSubgraph> {
        Err(GFError::UnsupportedOperation {
            message: "sample_neighbors() is not available on wasm32 yet".to_owned(),
        })
    }

    pub fn random_walk(
        &self,
        _start_nodes: &[&str],
        _length: usize,
        _walks_per_node: usize,
        _direction: Direction,
        _edge_type: &EdgeTypeSpec,
    ) -> Result<Vec<Vec<u32>>> {
        Err(GFError::UnsupportedOperation {
            message: "random_walk() is not available on wasm32 yet".to_owned(),
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn validate_sampling_config(config: &SamplingConfig) -> Result<()> {
    if config.fan_out.len() < config.hops {
        return Err(GFError::InvalidConfig {
            message: format!(
                "sample_neighbors requires fan_out.len() >= hops (got fan_out.len() = {}, hops = {})",
                config.fan_out.len(),
                config.hops
            ),
        });
    }
    if config.hops > 0 && config.fan_out.iter().take(config.hops).any(|&n| n == 0) {
        return Err(GFError::InvalidConfig {
            message: "sample_neighbors fan_out entries for active hops must be > 0".to_owned(),
        });
    }

    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn sampled_neighbor_positions(
    degree: usize,
    fan_out: usize,
    replace: bool,
    rng: &mut rand::rngs::ThreadRng,
) -> Vec<usize> {
    if degree == 0 || fan_out == 0 {
        return Vec::new();
    }
    if replace {
        return (0..fan_out).map(|_| rng.gen_range(0..degree)).collect();
    }
    if fan_out >= degree {
        return (0..degree).collect();
    }

    sample(rng, degree, fan_out).into_vec()
}

#[cfg(not(target_arch = "wasm32"))]
fn filtered_neighbor_candidates(
    graph: &GraphFrame,
    src_idx: u32,
    config: &SamplingConfig,
) -> Vec<(u32, u32)> {
    filtered_neighbor_candidates_with_spec(graph, src_idx, config.direction, &config.edge_type)
}

#[cfg(not(target_arch = "wasm32"))]
fn filtered_neighbor_candidates_with_spec(
    graph: &GraphFrame,
    src_idx: u32,
    direction: Direction,
    edge_type: &EdgeTypeSpec,
) -> Vec<(u32, u32)> {
    let edges = graph.edges();
    let mut candidates = Vec::new();
    let mut seen_edge_rows = HashSet::new();

    if matches!(direction, Direction::Out | Direction::Both) {
        for (&dst_idx, &edge_row) in edges
            .out_neighbors(src_idx)
            .iter()
            .zip(edges.out_edge_ids(src_idx).iter())
        {
            if seen_edge_rows.insert(edge_row)
                && matches_edge_type(edges.edge_type_at(edge_row), edge_type)
            {
                candidates.push((dst_idx, edge_row));
            }
        }
    }

    if matches!(direction, Direction::In | Direction::Both) {
        for (&neighbor_idx, &edge_row) in edges
            .in_neighbors(src_idx)
            .iter()
            .zip(edges.in_edge_ids(src_idx).iter())
        {
            if seen_edge_rows.insert(edge_row)
                && matches_edge_type(edges.edge_type_at(edge_row), edge_type)
            {
                candidates.push((neighbor_idx, edge_row));
            }
        }
    }

    candidates
}

#[cfg(not(target_arch = "wasm32"))]
fn matches_edge_type(actual: &str, spec: &EdgeTypeSpec) -> bool {
    match spec {
        EdgeTypeSpec::Any => true,
        EdgeTypeSpec::Single(expected) => actual == expected,
        EdgeTypeSpec::Multiple(expected) => expected.iter().any(|ty| ty == actual),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn push_sampled_node(graph: &GraphFrame, sampled: &mut SampledSubgraph, edge_idx: u32) {
    sampled.node_indices.push(edge_idx);

    let node_row = graph
        .edge_node_id_by_idx(edge_idx)
        .and_then(|node_id| graph.node_row_by_id(node_id))
        .expect("EdgeFrame local node indices must map back to NodeFrame rows");
    sampled.node_row_ids.push(node_row);
}

#[cfg(not(target_arch = "wasm32"))]
fn resolve_walk_starts(graph: &GraphFrame, start_nodes: &[&str]) -> Result<Vec<Option<u32>>> {
    start_nodes
        .iter()
        .map(|&start| {
            let _ = graph
                .node_row_by_id(start)
                .ok_or_else(|| GFError::NodeNotFound {
                    id: start.to_owned(),
                })?;
            Ok(graph.edges().node_row_idx(start))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::builder::{ListBuilder, StringBuilder};
    use arrow_array::{ArrayRef, Int64Array, Int8Array, ListArray, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};

    use crate::{
        EdgeFrame, GraphFrame, NodeFrame, COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC,
        COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
    };

    use super::{SampledSubgraph, SamplingConfig};
    use crate::{Direction, EdgeTypeSpec, GFError};

    #[test]
    fn sampling_config_default_matches_single_hop_outbound_sampling() {
        let config = SamplingConfig::default();

        assert_eq!(config.hops, 1);
        assert_eq!(config.fan_out, vec![25]);
        assert_eq!(config.direction, Direction::Out);
        assert_eq!(config.edge_type, EdgeTypeSpec::Any);
        assert!(!config.replace);
    }

    #[test]
    fn sampled_subgraph_can_be_constructed_with_explicit_buffers() {
        let sampled = SampledSubgraph {
            node_indices: vec![0, 1, 3],
            edge_src: vec![0, 1],
            edge_dst: vec![1, 3],
            edge_row_ids: vec![4, 7],
            node_row_ids: vec![10, 11, 14],
        };

        assert_eq!(sampled.node_indices, vec![0, 1, 3]);
        assert_eq!(sampled.edge_src, vec![0, 1]);
        assert_eq!(sampled.edge_dst, vec![1, 3]);
        assert_eq!(sampled.edge_row_ids, vec![4, 7]);
        assert_eq!(sampled.node_row_ids, vec![10, 11, 14]);
    }

    #[test]
    fn sampling_config_supports_explicit_multi_hop_settings() {
        let config = SamplingConfig {
            hops: 2,
            fan_out: vec![15, 8],
            direction: Direction::Both,
            edge_type: EdgeTypeSpec::Multiple(vec!["KNOWS".to_owned(), "LIKES".to_owned()]),
            replace: true,
        };

        assert_eq!(config.hops, 2);
        assert_eq!(config.fan_out, vec![15, 8]);
        assert_eq!(config.direction, Direction::Both);
        assert_eq!(
            config.edge_type,
            EdgeTypeSpec::Multiple(vec!["KNOWS".to_owned(), "LIKES".to_owned()])
        );
        assert!(config.replace);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn labels_array(values: &[&[&str]]) -> ListArray {
        let value_builder = StringBuilder::new();
        let mut builder = ListBuilder::new(value_builder);
        for labels in values {
            for label in *labels {
                builder.values().append_value(label);
            }
            builder.append(true);
        }
        builder.finish()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn label_field() -> Field {
        Field::new(
            COL_NODE_LABEL,
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            false,
        )
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn sample_graph() -> GraphFrame {
        let node_schema = Arc::new(ArrowSchema::new(vec![
            Field::new(COL_NODE_ID, DataType::Utf8, false),
            label_field(),
            Field::new("age", DataType::Int64, true),
        ]));
        let edge_schema = Arc::new(ArrowSchema::new(vec![
            Field::new(COL_EDGE_SRC, DataType::Utf8, false),
            Field::new(COL_EDGE_DST, DataType::Utf8, false),
            Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
            Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
            Field::new("since", DataType::Int64, true),
        ]));

        let nodes = NodeFrame::from_record_batch(
            RecordBatch::try_new(
                node_schema,
                vec![
                    Arc::new(StringArray::from(vec!["alice", "bob", "charlie", "diana"]))
                        as ArrayRef,
                    Arc::new(labels_array(&[
                        &["Person"],
                        &["Person"],
                        &["Person"],
                        &["Animal"],
                    ])) as ArrayRef,
                    Arc::new(Int64Array::from(vec![
                        Some(30),
                        Some(20),
                        Some(25),
                        Some(5),
                    ])) as ArrayRef,
                ],
            )
            .unwrap(),
        )
        .unwrap();

        let edges = EdgeFrame::from_record_batch(
            RecordBatch::try_new(
                edge_schema,
                vec![
                    Arc::new(StringArray::from(vec!["alice", "bob", "alice", "diana"])) as ArrayRef,
                    Arc::new(StringArray::from(vec![
                        "bob", "charlie", "charlie", "alice",
                    ])) as ArrayRef,
                    Arc::new(StringArray::from(vec!["KNOWS", "KNOWS", "LIKES", "OWNS"]))
                        as ArrayRef,
                    Arc::new(Int8Array::from(vec![0i8, 0, 0, 0])) as ArrayRef,
                    Arc::new(Int64Array::from(vec![
                        Some(2020),
                        Some(2021),
                        Some(2022),
                        Some(2023),
                    ])) as ArrayRef,
                ],
            )
            .unwrap(),
        )
        .unwrap();

        GraphFrame::new(nodes, edges).unwrap()
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn sample_neighbors_returns_all_one_hop_edges_when_fan_out_covers_degree() {
        let graph = sample_graph();
        let config = SamplingConfig {
            hops: 1,
            fan_out: vec![8],
            ..SamplingConfig::default()
        };

        let sampled = graph.sample_neighbors(&["alice"], &config).unwrap();

        assert_eq!(sampled.node_indices, vec![0, 1, 2]);
        assert_eq!(sampled.node_row_ids, vec![0, 1, 2]);
        assert_eq!(sampled.edge_src, vec![0, 0]);
        assert_eq!(sampled.edge_dst, vec![1, 2]);
        assert_eq!(sampled.edge_row_ids, vec![0, 2]);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn sample_neighbors_expands_two_hops_from_frontier_order() {
        let graph = sample_graph();
        let config = SamplingConfig {
            hops: 2,
            fan_out: vec![8, 8],
            ..SamplingConfig::default()
        };

        let sampled = graph.sample_neighbors(&["alice"], &config).unwrap();

        assert_eq!(sampled.node_indices, vec![0, 1, 2]);
        assert_eq!(sampled.node_row_ids, vec![0, 1, 2]);
        assert_eq!(sampled.edge_src, vec![0, 0, 1]);
        assert_eq!(sampled.edge_dst, vec![1, 2, 2]);
        assert_eq!(sampled.edge_row_ids, vec![0, 2, 1]);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn sample_neighbors_caps_result_to_fan_out_per_frontier_node() {
        let graph = sample_graph();
        let config = SamplingConfig {
            hops: 1,
            fan_out: vec![1],
            ..SamplingConfig::default()
        };

        let sampled = graph.sample_neighbors(&["alice"], &config).unwrap();

        assert_eq!(sampled.edge_src.len(), 1);
        assert_eq!(sampled.edge_dst.len(), 1);
        assert_eq!(sampled.edge_row_ids.len(), 1);
        assert_eq!(sampled.node_indices[0], 0);
        assert_eq!(sampled.node_row_ids[0], 0);
        assert!(matches!(sampled.edge_dst[0], 1 | 2));
        assert!(matches!(sampled.edge_row_ids[0], 0 | 2));
        assert_eq!(sampled.node_indices.len(), 2);
        assert_eq!(sampled.node_row_ids.len(), 2);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn sample_neighbors_rejects_unknown_seed_ids() {
        let graph = sample_graph();
        let err = graph
            .sample_neighbors(&["ghost"], &SamplingConfig::default())
            .unwrap_err();

        assert!(matches!(err, GFError::NodeNotFound { id } if id == "ghost"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn sample_neighbors_supports_inbound_direction() {
        let graph = sample_graph();
        let config = SamplingConfig {
            hops: 1,
            fan_out: vec![8],
            direction: Direction::In,
            ..SamplingConfig::default()
        };

        let sampled = graph.sample_neighbors(&["alice"], &config).unwrap();

        assert_eq!(sampled.node_indices, vec![0, 3]);
        assert_eq!(sampled.node_row_ids, vec![0, 3]);
        assert_eq!(sampled.edge_src, vec![0]);
        assert_eq!(sampled.edge_dst, vec![3]);
        assert_eq!(sampled.edge_row_ids, vec![3]);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn sample_neighbors_supports_both_direction_without_duplicate_edges() {
        let graph = sample_graph();
        let config = SamplingConfig {
            hops: 1,
            fan_out: vec![8],
            direction: Direction::Both,
            ..SamplingConfig::default()
        };

        let sampled = graph.sample_neighbors(&["alice"], &config).unwrap();

        assert_eq!(sampled.node_indices, vec![0, 1, 2, 3]);
        assert_eq!(sampled.node_row_ids, vec![0, 1, 2, 3]);
        assert_eq!(sampled.edge_src, vec![0, 0, 0]);
        assert_eq!(sampled.edge_dst, vec![1, 2, 3]);
        assert_eq!(sampled.edge_row_ids, vec![0, 2, 3]);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn sample_neighbors_filters_single_edge_type() {
        let graph = sample_graph();
        let config = SamplingConfig {
            hops: 1,
            fan_out: vec![8],
            direction: Direction::Both,
            edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
            ..SamplingConfig::default()
        };

        let sampled = graph.sample_neighbors(&["alice"], &config).unwrap();

        assert_eq!(sampled.node_indices, vec![0, 1]);
        assert_eq!(sampled.node_row_ids, vec![0, 1]);
        assert_eq!(sampled.edge_src, vec![0]);
        assert_eq!(sampled.edge_dst, vec![1]);
        assert_eq!(sampled.edge_row_ids, vec![0]);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn sample_neighbors_filters_multiple_edge_types() {
        let graph = sample_graph();
        let config = SamplingConfig {
            hops: 1,
            fan_out: vec![8],
            direction: Direction::Both,
            edge_type: EdgeTypeSpec::Multiple(vec!["KNOWS".to_owned(), "OWNS".to_owned()]),
            ..SamplingConfig::default()
        };

        let sampled = graph.sample_neighbors(&["alice"], &config).unwrap();

        assert_eq!(sampled.node_indices, vec![0, 1, 3]);
        assert_eq!(sampled.node_row_ids, vec![0, 1, 3]);
        assert_eq!(sampled.edge_src, vec![0, 0]);
        assert_eq!(sampled.edge_dst, vec![1, 3]);
        assert_eq!(sampled.edge_row_ids, vec![0, 3]);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn sample_neighbors_supports_sampling_with_replacement() {
        let graph = sample_graph();
        let config = SamplingConfig {
            hops: 1,
            fan_out: vec![3],
            direction: Direction::Out,
            edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
            replace: true,
        };

        let sampled = graph.sample_neighbors(&["bob"], &config).unwrap();

        assert_eq!(sampled.node_indices, vec![1, 2]);
        assert_eq!(sampled.node_row_ids, vec![1, 2]);
        assert_eq!(sampled.edge_src, vec![1, 1, 1]);
        assert_eq!(sampled.edge_dst, vec![2, 2, 2]);
        assert_eq!(sampled.edge_row_ids, vec![1, 1, 1]);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn sample_neighbors_direction_none_returns_only_seed_nodes() {
        let graph = sample_graph();
        let config = SamplingConfig {
            hops: 1,
            fan_out: vec![8],
            direction: Direction::None,
            ..SamplingConfig::default()
        };

        let sampled = graph.sample_neighbors(&["alice"], &config).unwrap();

        assert_eq!(sampled.node_indices, vec![0]);
        assert_eq!(sampled.node_row_ids, vec![0]);
        assert!(sampled.edge_src.is_empty());
        assert!(sampled.edge_dst.is_empty());
        assert!(sampled.edge_row_ids.is_empty());
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn graph_with_isolated_node() -> GraphFrame {
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

        let nodes = NodeFrame::from_record_batch(
            RecordBatch::try_new(
                node_schema,
                vec![
                    Arc::new(StringArray::from(vec!["alice", "bob", "eve"])) as ArrayRef,
                    Arc::new(labels_array(&[&["Person"], &["Person"], &["Person"]])) as ArrayRef,
                ],
            )
            .unwrap(),
        )
        .unwrap();

        let edges = EdgeFrame::from_record_batch(
            RecordBatch::try_new(
                edge_schema,
                vec![
                    Arc::new(StringArray::from(vec!["alice"])) as ArrayRef,
                    Arc::new(StringArray::from(vec!["bob"])) as ArrayRef,
                    Arc::new(StringArray::from(vec!["KNOWS"])) as ArrayRef,
                    Arc::new(Int8Array::from(vec![0i8])) as ArrayRef,
                ],
            )
            .unwrap(),
        )
        .unwrap();

        GraphFrame::new(nodes, edges).unwrap()
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn random_walk_follows_only_available_outbound_path() {
        let graph = sample_graph();

        let walks = graph
            .random_walk(
                &["bob"],
                3,
                2,
                Direction::Out,
                &EdgeTypeSpec::Single("KNOWS".to_owned()),
            )
            .unwrap();

        assert_eq!(walks, vec![vec![1, 2], vec![1, 2]]);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn random_walk_supports_inbound_direction() {
        let graph = sample_graph();

        let walks = graph
            .random_walk(
                &["charlie"],
                2,
                2,
                Direction::In,
                &EdgeTypeSpec::Single("KNOWS".to_owned()),
            )
            .unwrap();

        for walk in walks {
            assert!(walk.len() >= 2);
            assert!(walk.len() <= 3);
            assert_eq!(walk[0], 2);
            assert!(matches!(walk[1], 0 | 1));
            if walk.len() == 3 {
                assert_eq!(walk[1], 1);
                assert_eq!(walk[2], 0);
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn random_walk_stops_early_on_isolated_nodes() {
        let graph = graph_with_isolated_node();

        let walks = graph
            .random_walk(&["eve"], 5, 3, Direction::Out, &EdgeTypeSpec::Any)
            .unwrap();

        assert_eq!(
            walks,
            vec![Vec::<u32>::new(), Vec::<u32>::new(), Vec::<u32>::new()]
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn random_walk_respects_walks_per_node_and_start_order() {
        let graph = sample_graph();

        let walks = graph
            .random_walk(&["alice", "bob"], 1, 2, Direction::Out, &EdgeTypeSpec::Any)
            .unwrap();

        assert_eq!(walks.len(), 4);
        assert_eq!(walks[2], vec![1, 2]);
        assert_eq!(walks[3], vec![1, 2]);
        assert_eq!(walks[0][0], 0);
        assert_eq!(walks[1][0], 0);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn random_walk_rejects_unknown_start_nodes() {
        let graph = sample_graph();
        let err = graph
            .random_walk(&["ghost"], 3, 1, Direction::Out, &EdgeTypeSpec::Any)
            .unwrap_err();

        assert!(matches!(err, GFError::NodeNotFound { id } if id == "ghost"));
    }
}
