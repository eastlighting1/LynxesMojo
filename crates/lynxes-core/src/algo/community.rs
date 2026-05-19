//! Community detection algorithms (ADV-002).
//!
//! Provides [`GraphFrame::community_detection`] with a Louvain-based
//! modularity maximisation kernel.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, UInt32Array};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use hashbrown::HashMap;

use crate::{
    frame::graph_frame::GraphFrame, GFError, NodeFrame, Result, COL_NODE_ID, COL_NODE_LABEL,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommunityAlgorithm {
    Louvain,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CommunityConfig {
    pub algorithm: CommunityAlgorithm,
    pub resolution: f64,
    pub seed: Option<u64>,
}

impl Default for CommunityConfig {
    fn default() -> Self {
        Self {
            algorithm: CommunityAlgorithm::Louvain,
            resolution: 1.0,
            seed: None,
        }
    }
}

impl GraphFrame {
    /// Detects graph communities and returns a [`NodeFrame`] with `community_id`.
    ///
    /// The current implementation supports [`CommunityAlgorithm::Louvain`]
    /// and treats the graph as weighted-undirected with unit edge weights.
    pub fn community_detection(&self, config: CommunityConfig) -> Result<NodeFrame> {
        if !(config.resolution.is_finite() && config.resolution > 0.0) {
            return Err(GFError::InvalidConfig {
                message: format!(
                    "community_detection resolution must be finite and > 0.0, got {}",
                    config.resolution
                ),
            });
        }

        match config.algorithm {
            CommunityAlgorithm::Louvain => self.louvain_communities(config),
        }
    }

    fn louvain_communities(&self, config: CommunityConfig) -> Result<NodeFrame> {
        let node_count = self.node_count();
        if node_count == 0 {
            return build_empty_output(self);
        }
        if self.edge_count() == 0 {
            let ids: Vec<u32> = (0..node_count as u32).collect();
            return build_output(self, &ids);
        }

        let mut graph = WeightedGraph::from_graphframe(self)?;
        let mut original_to_current: Vec<usize> = (0..node_count).collect();
        let mut rng = config.seed.map(SimpleRng::new);

        loop {
            let (assignment, moved_any) = louvain_phase1(&graph, config.resolution, rng.as_mut());

            for community in &mut original_to_current {
                *community = assignment[*community];
            }

            let community_count = assignment.iter().copied().max().map(|v| v + 1).unwrap_or(0);
            if !moved_any || community_count == graph.node_count() {
                break;
            }

            graph = graph.compress(&assignment, community_count);
        }

        let canonical = canonicalize_assignments(&original_to_current);
        build_output(self, &canonical)
    }
}

#[derive(Debug, Clone)]
struct WeightedGraph {
    adjacency: Vec<Vec<(usize, f64)>>,
    self_loops: Vec<f64>,
    degree: Vec<f64>,
    total_edge_weight: f64,
}

impl WeightedGraph {
    fn node_count(&self) -> usize {
        self.adjacency.len()
    }

    fn from_graphframe(graph: &GraphFrame) -> Result<Self> {
        let node_count = graph.node_count();
        let mut edge_weights: HashMap<(usize, usize), f64> = HashMap::new();
        let mut self_loops = vec![0.0f64; node_count];
        let mut degree = vec![0.0f64; node_count];
        let mut total_edge_weight = 0.0f64;

        let edge_batch = graph.edges().to_record_batch();
        let src = edge_batch
            .column_by_name(crate::COL_EDGE_SRC)
            .expect("_src exists")
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .expect("_src is Utf8");
        let dst = edge_batch
            .column_by_name(crate::COL_EDGE_DST)
            .expect("_dst exists")
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .expect("_dst is Utf8");

        for row in 0..edge_batch.num_rows() {
            let src_row = graph
                .nodes()
                .row_index(src.value(row))
                .expect("validated graph keeps edge endpoints in nodes")
                as usize;
            let dst_row = graph
                .nodes()
                .row_index(dst.value(row))
                .expect("validated graph keeps edge endpoints in nodes")
                as usize;

            total_edge_weight += 1.0;
            if src_row == dst_row {
                self_loops[src_row] += 1.0;
                degree[src_row] += 2.0;
            } else {
                let key = if src_row < dst_row {
                    (src_row, dst_row)
                } else {
                    (dst_row, src_row)
                };
                *edge_weights.entry(key).or_insert(0.0) += 1.0;
                degree[src_row] += 1.0;
                degree[dst_row] += 1.0;
            }
        }

        let mut adjacency = vec![Vec::new(); node_count];
        for ((left, right), weight) in edge_weights {
            adjacency[left].push((right, weight));
            adjacency[right].push((left, weight));
        }

        Ok(Self {
            adjacency,
            self_loops,
            degree,
            total_edge_weight,
        })
    }

    fn compress(&self, assignment: &[usize], community_count: usize) -> Self {
        let mut pair_weights: HashMap<(usize, usize), f64> = HashMap::new();
        let mut self_loops = vec![0.0f64; community_count];
        let mut degree = vec![0.0f64; community_count];

        for (node, &community) in assignment.iter().enumerate().take(self.node_count()) {
            self_loops[community] += self.self_loops[node];
        }

        for left in 0..self.node_count() {
            for &(right, weight) in &self.adjacency[left] {
                if left >= right {
                    continue;
                }
                let cl = assignment[left];
                let cr = assignment[right];
                if cl == cr {
                    self_loops[cl] += weight;
                    degree[cl] += 2.0 * weight;
                } else {
                    let key = if cl < cr { (cl, cr) } else { (cr, cl) };
                    *pair_weights.entry(key).or_insert(0.0) += weight;
                    degree[cl] += weight;
                    degree[cr] += weight;
                }
            }
        }

        for (community, loop_weight) in self_loops.iter().enumerate() {
            degree[community] += 2.0 * *loop_weight;
        }

        let mut adjacency = vec![Vec::new(); community_count];
        for ((left, right), weight) in pair_weights {
            adjacency[left].push((right, weight));
            adjacency[right].push((left, weight));
        }

        Self {
            adjacency,
            self_loops,
            degree,
            total_edge_weight: self.total_edge_weight,
        }
    }
}

fn louvain_phase1(
    graph: &WeightedGraph,
    resolution: f64,
    mut rng: Option<&mut SimpleRng>,
) -> (Vec<usize>, bool) {
    let node_count = graph.node_count();
    let mut community: Vec<usize> = (0..node_count).collect();
    let mut total_degree = graph.degree.clone();
    let m2 = 2.0 * graph.total_edge_weight.max(f64::EPSILON);
    let mut moved_any = false;

    loop {
        let mut pass_improved = false;
        let mut order: Vec<usize> = (0..node_count).collect();
        if let Some(rng) = rng.as_deref_mut() {
            shuffle(&mut order, rng);
        }

        for node in order {
            let current = community[node];
            let node_degree = graph.degree[node];
            if node_degree == 0.0 {
                continue;
            }

            let mut weights_by_comm: HashMap<usize, f64> = HashMap::new();
            for &(neighbor, weight) in &graph.adjacency[node] {
                *weights_by_comm.entry(community[neighbor]).or_insert(0.0) += weight;
            }
            weights_by_comm
                .entry(current)
                .or_insert(graph.self_loops[node]);

            total_degree[current] -= node_degree;

            let mut best_comm = current;
            let mut best_gain = 0.0f64;
            for (&candidate, &weight_to_candidate) in &weights_by_comm {
                let gain =
                    weight_to_candidate - resolution * total_degree[candidate] * node_degree / m2;
                if gain > best_gain + 1e-12
                    || ((gain - best_gain).abs() <= 1e-12 && candidate < best_comm)
                {
                    best_gain = gain;
                    best_comm = candidate;
                }
            }

            if best_comm != current {
                community[node] = best_comm;
                pass_improved = true;
                moved_any = true;
            }
            total_degree[community[node]] += node_degree;
        }

        if !pass_improved {
            break;
        }
    }

    (normalize_communities(&community), moved_any)
}

fn normalize_communities(raw: &[usize]) -> Vec<usize> {
    let mut remap: HashMap<usize, usize> = HashMap::new();
    let mut next = 0usize;
    raw.iter()
        .map(|community| {
            *remap.entry(*community).or_insert_with(|| {
                let current = next;
                next += 1;
                current
            })
        })
        .collect()
}

fn canonicalize_assignments(raw: &[usize]) -> Vec<u32> {
    let mut remap: HashMap<usize, u32> = HashMap::new();
    let mut next = 0u32;
    raw.iter()
        .map(|community| {
            *remap.entry(*community).or_insert_with(|| {
                let current = next;
                next += 1;
                current
            })
        })
        .collect()
}

#[derive(Debug, Clone)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

fn shuffle(values: &mut [usize], rng: &mut SimpleRng) {
    if values.len() < 2 {
        return;
    }
    for i in (1..values.len()).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        values.swap(i, j);
    }
}

fn build_output(graph: &GraphFrame, community_ids: &[u32]) -> Result<NodeFrame> {
    let nodes = graph.nodes();
    let id_col = nodes
        .column(COL_NODE_ID)
        .expect("NodeFrame has _id")
        .clone();
    let label_col = nodes
        .column(COL_NODE_LABEL)
        .expect("NodeFrame has _label")
        .clone();
    let label_field = nodes
        .schema()
        .field_with_name(COL_NODE_LABEL)
        .expect("NodeFrame schema has _label field")
        .clone();
    let community_col = Arc::new(UInt32Array::from(community_ids.to_vec())) as ArrayRef;

    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        label_field,
        Field::new("community_id", DataType::UInt32, false),
    ]));
    let batch = RecordBatch::try_new(schema, vec![id_col, label_col, community_col])
        .map_err(std::io::Error::other)?;
    NodeFrame::from_record_batch(batch)
}

fn build_empty_output(graph: &GraphFrame) -> Result<NodeFrame> {
    let label_field = graph
        .nodes()
        .schema()
        .field_with_name(COL_NODE_LABEL)
        .expect("NodeFrame schema has _label field")
        .clone();
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        label_field,
        Field::new("community_id", DataType::UInt32, false),
    ]));
    NodeFrame::from_record_batch(RecordBatch::new_empty(schema))
}
