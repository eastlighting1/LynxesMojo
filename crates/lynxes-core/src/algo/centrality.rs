//! Centrality and reachability algorithms.
//!
//! Provides:
//! - [`GraphFrame::has_path`]: BFS reachability
//! - [`GraphFrame::degree_centrality`]: normalized in/out/total degree
//! - [`GraphFrame::betweenness_centrality`]: Brandes algorithm with optional weights

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
    RecordBatch,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use hashbrown::HashSet;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::{
    frame::graph_frame::GraphFrame, Direction, EdgeFrame, GFError, NodeFrame, Result, COL_NODE_ID,
    COL_NODE_LABEL,
};

/// Configuration for betweenness-centrality computation.
#[derive(Default)]
pub struct BetweennessConfig {
    /// Optional numeric edge column used as edge weight.
    ///
    /// `None` means every eligible edge has uniform weight `1`.
    pub weight_col: Option<String>,
}

const COST_EPS: f64 = 1e-10;

#[derive(Debug, Clone, PartialEq)]
struct WeightedState {
    cost: f64,
    node_idx: u32,
}

impl Eq for WeightedState {}

impl PartialOrd for WeightedState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WeightedState {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.node_idx.cmp(&self.node_idx))
    }
}

impl GraphFrame {
    /// Returns `true` if there is a directed path from `src` to `dst`.
    ///
    /// Uses BFS over outgoing edges. `max_hops = None` means unbounded search;
    /// `max_hops = Some(k)` restricts the search to paths with at most `k` hops.
    ///
    /// `has_path(x, x)` always returns `true`.
    pub fn has_path(&self, src: &str, dst: &str, max_hops: Option<usize>) -> Result<bool> {
        if self.nodes().row_index(src).is_none() {
            return Err(GFError::NodeNotFound { id: src.to_owned() });
        }
        if self.nodes().row_index(dst).is_none() {
            return Err(GFError::NodeNotFound { id: dst.to_owned() });
        }

        if src == dst {
            return Ok(true);
        }

        let Some(src_eidx) = self.edges().node_row_idx(src) else {
            return Ok(false);
        };
        let Some(dst_eidx) = self.edges().node_row_idx(dst) else {
            return Ok(false);
        };

        let max = max_hops.unwrap_or(usize::MAX);
        let mut visited: HashSet<u32> = HashSet::new();
        let mut frontier = vec![src_eidx];
        visited.insert(src_eidx);

        for _ in 0..max {
            if frontier.is_empty() {
                break;
            }
            let current = std::mem::take(&mut frontier);
            for node_idx in current {
                for &nb in self.edges().out_neighbors(node_idx) {
                    if nb == dst_eidx {
                        return Ok(true);
                    }
                    if visited.insert(nb) {
                        frontier.push(nb);
                    }
                }
            }
        }

        Ok(false)
    }

    /// Computes normalized degree centrality for every node.
    pub fn degree_centrality(&self, direction: Direction) -> Result<NodeFrame> {
        let n = self.nodes().len();
        let id_col = self.nodes().id_column();

        let values: Vec<f64> = (0..n)
            .map(|row| {
                if n <= 1 {
                    return 0.0;
                }
                let id = id_col.value(row);
                let Some(eidx) = self.edges().node_row_idx(id) else {
                    return 0.0;
                };
                match direction {
                    Direction::Out => self.edges().out_degree(eidx) as f64 / (n - 1) as f64,
                    Direction::In => self.edges().in_degree(eidx) as f64 / (n - 1) as f64,
                    Direction::Both | Direction::None => {
                        let total = self.edges().out_degree(eidx) + self.edges().in_degree(eidx);
                        total as f64 / (2 * (n - 1)) as f64
                    }
                }
            })
            .collect();

        build_f64_column_output(self, values, "degree_centrality")
    }

    /// Computes normalized betweenness centrality for every node.
    ///
    /// This preserves the original unweighted API surface by delegating to the
    /// default [`BetweennessConfig`].
    pub fn betweenness_centrality(&self) -> Result<NodeFrame> {
        self.betweenness_centrality_with_config(&BetweennessConfig::default())
    }

    /// Computes normalized betweenness centrality with optional edge weights.
    ///
    /// For `weight_col = None`, Lynxes uses the unweighted Brandes BFS
    /// kernel. When a weight column is configured, it switches to weighted
    /// Brandes with per-source Dijkstra exploration. Source-node traversals are
    /// parallelized with Rayon.
    pub fn betweenness_centrality_with_config(
        &self,
        config: &BetweennessConfig,
    ) -> Result<NodeFrame> {
        let n = self.nodes().len();
        if n < 2 {
            return build_f64_column_output(self, vec![0.0; n], "betweenness");
        }

        let ec = self.edges().node_count();
        if ec == 0 {
            return build_f64_column_output(self, vec![0.0; n], "betweenness");
        }

        validate_weight_col(self.edges(), config)?;
        let bt_eidx = brandes_parallel(self, ec, config)?;
        let norm = if n > 2 {
            ((n - 1) * (n - 2)) as f64
        } else {
            1.0
        };

        let id_col = self.nodes().id_column();
        let values: Vec<f64> = (0..n)
            .map(|row| {
                let id = id_col.value(row);
                match self.edges().node_row_idx(id) {
                    Some(eidx) => bt_eidx[eidx as usize] / norm,
                    None => 0.0,
                }
            })
            .collect();

        build_f64_column_output(self, values, "betweenness")
    }
}

fn brandes_parallel(graph: &GraphFrame, ec: usize, config: &BetweennessConfig) -> Result<Vec<f64>> {
    #[cfg(not(target_arch = "wasm32"))]
    let partials: Vec<Result<Vec<f64>>> = (0..ec as u32)
        .into_par_iter()
        .map(|source| brandes_from_source(graph, ec, source, config))
        .collect();
    #[cfg(target_arch = "wasm32")]
    let partials: Vec<Result<Vec<f64>>> = (0..ec as u32)
        .map(|source| brandes_from_source(graph, ec, source, config))
        .collect();

    let mut betweenness = vec![0.0f64; ec];
    for partial in partials {
        let partial = partial?;
        for (score, local) in betweenness.iter_mut().zip(partial) {
            *score += local;
        }
    }

    Ok(betweenness)
}

fn brandes_from_source(
    graph: &GraphFrame,
    ec: usize,
    source: u32,
    config: &BetweennessConfig,
) -> Result<Vec<f64>> {
    if config.weight_col.is_some() {
        weighted_brandes_from_source(graph, ec, source, config)
    } else {
        Ok(unweighted_brandes_from_source(graph, ec, source))
    }
}

fn unweighted_brandes_from_source(graph: &GraphFrame, ec: usize, source: u32) -> Vec<f64> {
    let mut local_scores = vec![0.0f64; ec];
    let mut stack: Vec<u32> = Vec::with_capacity(ec);
    let mut pred: Vec<Vec<u32>> = vec![Vec::new(); ec];
    let mut sigma = vec![0.0f64; ec];
    let mut dist: Vec<i32> = vec![-1; ec];
    let mut delta = vec![0.0f64; ec];
    let mut queue: VecDeque<u32> = VecDeque::with_capacity(ec);

    sigma[source as usize] = 1.0;
    dist[source as usize] = 0;
    queue.push_back(source);

    while let Some(v) = queue.pop_front() {
        stack.push(v);
        let dv = dist[v as usize];

        for &w in graph.edges().out_neighbors(v) {
            if dist[w as usize] == -1 {
                queue.push_back(w);
                dist[w as usize] = dv + 1;
            }
            if dist[w as usize] == dv + 1 {
                sigma[w as usize] += sigma[v as usize];
                pred[w as usize].push(v);
            }
        }
    }

    while let Some(w) = stack.pop() {
        for &v in &pred[w as usize] {
            let sw = sigma[w as usize];
            if sw > 0.0 {
                delta[v as usize] += (sigma[v as usize] / sw) * (1.0 + delta[w as usize]);
            }
        }
        if w != source {
            local_scores[w as usize] += delta[w as usize];
        }
    }

    local_scores
}

fn weighted_brandes_from_source(
    graph: &GraphFrame,
    ec: usize,
    source: u32,
    config: &BetweennessConfig,
) -> Result<Vec<f64>> {
    let mut local_scores = vec![0.0f64; ec];
    let mut stack: Vec<u32> = Vec::with_capacity(ec);
    let mut pred: Vec<Vec<u32>> = vec![Vec::new(); ec];
    let mut sigma = vec![0.0f64; ec];
    let mut dist = vec![f64::INFINITY; ec];
    let mut delta = vec![0.0f64; ec];
    let mut heap = BinaryHeap::new();

    sigma[source as usize] = 1.0;
    dist[source as usize] = 0.0;
    heap.push(WeightedState {
        cost: 0.0,
        node_idx: source,
    });

    while let Some(WeightedState { cost, node_idx }) = heap.pop() {
        if cost > dist[node_idx as usize] + COST_EPS {
            continue;
        }

        stack.push(node_idx);

        for (&neighbor, &edge_row) in graph
            .edges()
            .out_neighbors(node_idx)
            .iter()
            .zip(graph.edges().out_edge_ids(node_idx).iter())
        {
            let next_cost = cost + edge_weight(graph.edges(), edge_row, config)?;
            let old_cost = dist[neighbor as usize];

            if next_cost < old_cost - COST_EPS {
                dist[neighbor as usize] = next_cost;
                sigma[neighbor as usize] = sigma[node_idx as usize];
                pred[neighbor as usize].clear();
                pred[neighbor as usize].push(node_idx);
                heap.push(WeightedState {
                    cost: next_cost,
                    node_idx: neighbor,
                });
            } else if (next_cost - old_cost).abs() < COST_EPS {
                sigma[neighbor as usize] += sigma[node_idx as usize];
                pred[neighbor as usize].push(node_idx);
            }
        }
    }

    while let Some(w) = stack.pop() {
        for &v in &pred[w as usize] {
            let sw = sigma[w as usize];
            if sw > 0.0 {
                delta[v as usize] += (sigma[v as usize] / sw) * (1.0 + delta[w as usize]);
            }
        }
        if w != source {
            local_scores[w as usize] += delta[w as usize];
        }
    }

    Ok(local_scores)
}

fn build_f64_column_output(
    graph: &GraphFrame,
    values: Vec<f64>,
    col_name: &str,
) -> Result<NodeFrame> {
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

    let val_col = Arc::new(Float64Array::from(values)) as ArrayRef;

    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        label_field,
        Field::new(col_name, DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(schema, vec![id_col, label_col, val_col])
        .map_err(std::io::Error::other)?;

    NodeFrame::from_record_batch(batch)
}

fn validate_weight_col(edges: &EdgeFrame, config: &BetweennessConfig) -> Result<()> {
    let col_name = match &config.weight_col {
        None => return Ok(()),
        Some(c) => c.as_str(),
    };
    let col = edges
        .column(col_name)
        .ok_or_else(|| GFError::ColumnNotFound {
            column: col_name.to_owned(),
        })?;

    match col.data_type() {
        DataType::Float32
        | DataType::Float64
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64 => Ok(()),
        other => Err(GFError::TypeMismatch {
            message: format!(
                "weight column '{}' must be numeric, got {:?}",
                col_name, other
            ),
        }),
    }
}

fn edge_weight(edges: &EdgeFrame, edge_row: u32, config: &BetweennessConfig) -> Result<f64> {
    let col_name = match &config.weight_col {
        None => return Ok(1.0),
        Some(c) => c.as_str(),
    };

    let col = edges
        .column(col_name)
        .ok_or_else(|| GFError::ColumnNotFound {
            column: col_name.to_owned(),
        })?;

    let row = edge_row as usize;
    if col.is_null(row) {
        return Err(GFError::TypeMismatch {
            message: format!("null weight in column '{}' at edge row {}", col_name, row),
        });
    }

    let weight: f64 = if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
        a.value(row)
    } else if let Some(a) = col.as_any().downcast_ref::<Float32Array>() {
        a.value(row) as f64
    } else if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
        a.value(row) as f64
    } else if let Some(a) = col.as_any().downcast_ref::<Int32Array>() {
        a.value(row) as f64
    } else if let Some(a) = col.as_any().downcast_ref::<Int16Array>() {
        a.value(row) as f64
    } else if let Some(a) = col.as_any().downcast_ref::<Int8Array>() {
        a.value(row) as f64
    } else {
        return Err(GFError::TypeMismatch {
            message: format!(
                "weight column '{}' must be numeric, got {:?}",
                col_name,
                col.data_type()
            ),
        });
    };

    if weight.is_nan() || weight.is_infinite() {
        return Err(GFError::TypeMismatch {
            message: format!(
                "weight column '{}' contains non-finite value at edge row {}",
                col_name, row
            ),
        });
    }
    if weight < 0.0 {
        return Err(GFError::NegativeWeight {
            column: col_name.to_owned(),
        });
    }

    Ok(weight)
}
