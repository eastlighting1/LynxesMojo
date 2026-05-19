//! Dijkstra shortest-path algorithms (ALG-002).
//!
//! Provides [`GraphFrame::shortest_path`] and [`GraphFrame::all_shortest_paths`].

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use arrow_array::{
    Array, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
};
use arrow_schema::DataType;
use hashbrown::HashSet;

use crate::{frame::graph_frame::GraphFrame, Direction, EdgeFrame, EdgeTypeSpec, GFError, Result};

// ── Public configuration ─────────────────────────────────────────────────────

/// Configuration for Dijkstra shortest-path queries.
pub struct ShortestPathConfig {
    /// Optional numeric edge column used as edge weight.
    ///
    /// `None` means every eligible edge has uniform weight `1`.
    pub weight_col: Option<String>,

    /// Edge type restriction applied during traversal.
    pub edge_type: EdgeTypeSpec,

    /// Which direction to follow edges.
    pub direction: Direction,
}

impl Default for ShortestPathConfig {
    fn default() -> Self {
        Self {
            weight_col: None,
            edge_type: EdgeTypeSpec::Any,
            direction: Direction::Out,
        }
    }
}

// ── GraphFrame public API ─────────────────────────────────────────────────────

impl GraphFrame {
    /// Returns the single shortest path from `src` to `dst`, or `None` when no
    /// path exists.
    ///
    /// The path is a `Vec<String>` of node `_id` values from `src` to `dst`,
    /// inclusive.
    ///
    /// # Tie-breaking
    ///
    /// When multiple predecessors achieve the same minimum cost to a node, the
    /// predecessor with the lower internal EdgeFrame compact-node index is chosen.
    /// This is fully deterministic for a fixed graph layout.
    ///
    /// # Errors
    ///
    /// - [`GFError::NodeNotFound`] — `src` or `dst` absent from `nodes`.
    /// - [`GFError::ColumnNotFound`] — `weight_col` absent from `edges`.
    /// - [`GFError::TypeMismatch`] — `weight_col` is non-numeric, or contains
    ///   null / NaN / infinite values.
    /// - [`GFError::NegativeWeight`] — a traversed edge weight is negative.
    ///
    /// # Complexity
    ///
    /// O((N + E) log N) using a binary min-heap.
    pub fn shortest_path(
        &self,
        src: &str,
        dst: &str,
        config: &ShortestPathConfig,
    ) -> Result<Option<Vec<String>>> {
        validate_endpoints(self, src, dst)?;
        validate_weight_col(self.edges(), config)?;

        if src == dst {
            return Ok(Some(vec![src.to_owned()]));
        }

        let Some(src_idx) = self.edges().node_row_idx(src) else {
            return Ok(None); // src isolated — cannot reach anything
        };
        let Some(dst_idx) = self.edges().node_row_idx(dst) else {
            return Ok(None); // dst isolated — unreachable
        };

        Ok(
            dijkstra_single(self, src_idx, dst_idx, config)?.map(|idxs| {
                idxs.iter()
                    .filter_map(|&i| self.edge_node_id_by_idx(i))
                    .map(|s| s.to_owned())
                    .collect()
            }),
        )
    }

    /// Returns all shortest paths from `src` to `dst`.
    ///
    /// Each element is a `Vec<String>` of node `_id` values from `src` to `dst`,
    /// inclusive.  Returns `[]` when no path exists.
    ///
    /// # Errors
    ///
    /// Same conditions as [`shortest_path`](Self::shortest_path).
    ///
    /// # Complexity
    ///
    /// O((N + E) log N + P) where P is the total size of all returned paths.
    pub fn all_shortest_paths(
        &self,
        src: &str,
        dst: &str,
        config: &ShortestPathConfig,
    ) -> Result<Vec<Vec<String>>> {
        validate_endpoints(self, src, dst)?;
        validate_weight_col(self.edges(), config)?;

        if src == dst {
            return Ok(vec![vec![src.to_owned()]]);
        }

        let Some(src_idx) = self.edges().node_row_idx(src) else {
            return Ok(Vec::new());
        };
        let Some(dst_idx) = self.edges().node_row_idx(dst) else {
            return Ok(Vec::new());
        };

        Ok(dijkstra_all(self, src_idx, dst_idx, config)?
            .into_iter()
            .map(|idxs| {
                idxs.iter()
                    .filter_map(|&i| self.edge_node_id_by_idx(i))
                    .map(|s| s.to_owned())
                    .collect()
            })
            .collect())
    }

    /// Returns the single shortest path from `src` to `dst` using A* search.
    ///
    /// `heuristic` estimates the remaining cost between the current node `_id`
    /// and the destination `_id`. Passing `None` falls back to
    /// [`shortest_path`](Self::shortest_path), so callers can use one entry
    /// point for both Dijkstra and A*.
    ///
    /// The heuristic must return finite, non-negative estimates. Lynxes
    /// does not attempt to prove admissibility; correctness relative to
    /// Dijkstra is guaranteed only when the heuristic is admissible.
    #[allow(clippy::type_complexity)]
    pub fn astar_shortest_path(
        &self,
        src: &str,
        dst: &str,
        config: &ShortestPathConfig,
        heuristic: Option<&dyn Fn(&str, &str) -> f64>,
    ) -> Result<Option<Vec<String>>> {
        let Some(heuristic) = heuristic else {
            return self.shortest_path(src, dst, config);
        };

        validate_endpoints(self, src, dst)?;
        validate_weight_col(self.edges(), config)?;

        if src == dst {
            return Ok(Some(vec![src.to_owned()]));
        }

        let Some(src_idx) = self.edges().node_row_idx(src) else {
            return Ok(None);
        };
        let Some(dst_idx) = self.edges().node_row_idx(dst) else {
            return Ok(None);
        };

        Ok(
            astar_single(self, src_idx, dst_idx, config, heuristic)?.map(|idxs| {
                idxs.iter()
                    .filter_map(|&i| self.edge_node_id_by_idx(i))
                    .map(|s| s.to_owned())
                    .collect()
            }),
        )
    }

    /// Returns up to `k` shortest simple paths from `src` to `dst` using Yen's
    /// algorithm.
    ///
    /// Paths are ordered by total path cost; ties are broken deterministically
    /// by the compact-node sequence. `max_hops` limits the maximum number of
    /// edges in each returned path.
    pub fn k_shortest_paths(
        &self,
        src: &str,
        dst: &str,
        k: usize,
        max_hops: Option<usize>,
        config: &ShortestPathConfig,
    ) -> Result<Vec<Vec<String>>> {
        if k == 0 {
            return Ok(Vec::new());
        }

        validate_endpoints(self, src, dst)?;
        validate_weight_col(self.edges(), config)?;

        if src == dst {
            return Ok(vec![vec![src.to_owned()]]);
        }

        let Some(src_idx) = self.edges().node_row_idx(src) else {
            return Ok(Vec::new());
        };
        let Some(dst_idx) = self.edges().node_row_idx(dst) else {
            return Ok(Vec::new());
        };

        let empty_nodes = HashSet::new();
        let empty_edges = HashSet::new();
        let Some(first_path) = shortest_path_candidate(
            self,
            src_idx,
            dst_idx,
            config,
            &empty_nodes,
            &empty_edges,
            max_hops,
        )?
        else {
            return Ok(Vec::new());
        };

        let mut accepted = vec![first_path];
        let mut accepted_keys: HashSet<Vec<u32>> = HashSet::new();
        accepted_keys.insert(accepted[0].nodes.clone());
        let mut candidates: Vec<PathCandidate> = Vec::new();
        let mut candidate_keys: HashSet<Vec<u32>> = HashSet::new();

        while accepted.len() < k {
            let prev_path = accepted.last().expect("accepted is non-empty").clone();

            for spur_idx in 0..prev_path.nodes.len().saturating_sub(1) {
                let root_nodes = &prev_path.nodes[..=spur_idx];
                let root_edges = &prev_path.edge_rows[..spur_idx];
                let spur_node = root_nodes[spur_idx];

                let mut banned_edges: HashSet<u32> = HashSet::new();
                for path in &accepted {
                    if path.nodes.len() > spur_idx && path.nodes[..=spur_idx] == *root_nodes {
                        banned_edges.insert(path.edge_rows[spur_idx]);
                    }
                }

                let banned_nodes: HashSet<u32> = root_nodes[..spur_idx].iter().copied().collect();

                let remaining_hops = max_hops.map(|limit| limit.saturating_sub(root_edges.len()));
                let Some(spur_path) = shortest_path_candidate(
                    self,
                    spur_node,
                    dst_idx,
                    config,
                    &banned_nodes,
                    &banned_edges,
                    remaining_hops,
                )?
                else {
                    continue;
                };

                let mut total_nodes = root_nodes[..spur_idx].to_vec();
                total_nodes.extend_from_slice(&spur_path.nodes);

                if accepted_keys.contains(&total_nodes) || candidate_keys.contains(&total_nodes) {
                    continue;
                }

                let mut total_edges = root_edges.to_vec();
                total_edges.extend_from_slice(&spur_path.edge_rows);
                let root_cost = path_cost(self.edges(), root_edges, config)?;
                let total_cost = root_cost + spur_path.cost;

                candidate_keys.insert(total_nodes.clone());
                candidates.push(PathCandidate {
                    nodes: total_nodes,
                    edge_rows: total_edges,
                    cost: total_cost,
                });
            }

            if candidates.is_empty() {
                break;
            }

            let next_idx = select_best_candidate(&candidates);
            let next = candidates.swap_remove(next_idx);
            candidate_keys.remove(&next.nodes);
            accepted_keys.insert(next.nodes.clone());
            accepted.push(next);
        }

        Ok(accepted
            .into_iter()
            .map(|path| {
                path.nodes
                    .iter()
                    .filter_map(|&idx| self.edge_node_id_by_idx(idx))
                    .map(str::to_owned)
                    .collect()
            })
            .collect())
    }
}

// ── Priority-queue state ──────────────────────────────────────────────────────

/// Threshold for treating two accumulated costs as equal.
const COST_EPS: f64 = 1e-10;

#[derive(Debug, Clone, PartialEq)]
struct State {
    cost: f64,
    node_idx: u32,
}

impl Eq for State {}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for State {
    /// Min-heap by cost; equal costs are broken by lower `node_idx` first.
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse `cost` comparison for min-heap.
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.node_idx.cmp(&self.node_idx))
    }
}

#[derive(Debug, Clone)]
struct HopState {
    cost: f64,
    node_idx: u32,
    hops: usize,
}

impl Eq for HopState {}

impl PartialEq for HopState {
    fn eq(&self, other: &Self) -> bool {
        self.node_idx == other.node_idx
            && self.hops == other.hops
            && (self.cost - other.cost).abs() < COST_EPS
    }
}

impl PartialOrd for HopState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HopState {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.hops.cmp(&self.hops))
            .then_with(|| other.node_idx.cmp(&self.node_idx))
    }
}

#[derive(Debug, Clone)]
struct PathCandidate {
    nodes: Vec<u32>,
    edge_rows: Vec<u32>,
    cost: f64,
}

// ── Internal kernels ──────────────────────────────────────────────────────────

/// Single-path Dijkstra with early termination.
///
/// Returns the compact-index path `[src_idx, …, dst_idx]`, or `None` if
/// `dst_idx` is not reachable from `src_idx`.
fn dijkstra_single(
    graph: &GraphFrame,
    src_idx: u32,
    dst_idx: u32,
    config: &ShortestPathConfig,
) -> Result<Option<Vec<u32>>> {
    let n = graph.edges().node_count();
    let mut dist = vec![f64::INFINITY; n];
    let mut prev: Vec<Option<u32>> = vec![None; n];

    dist[src_idx as usize] = 0.0;
    let mut heap = BinaryHeap::new();
    heap.push(State {
        cost: 0.0,
        node_idx: src_idx,
    });

    while let Some(State { cost, node_idx }) = heap.pop() {
        if node_idx == dst_idx {
            return Ok(Some(reconstruct_path(&prev, src_idx, dst_idx)));
        }
        // Stale entry — skip.
        if cost > dist[node_idx as usize] + COST_EPS {
            continue;
        }

        for (nb, eid) in neighbor_pairs(graph.edges(), node_idx, config) {
            let w = edge_weight(graph.edges(), eid, config)?;
            let nc = cost + w;
            let old = dist[nb as usize];

            if nc < old - COST_EPS {
                dist[nb as usize] = nc;
                prev[nb as usize] = Some(node_idx);
                heap.push(State {
                    cost: nc,
                    node_idx: nb,
                });
            } else if (nc - old).abs() < COST_EPS {
                // Equal cost: prefer lower predecessor index (spec tie-break).
                if prev[nb as usize].is_some_and(|p| node_idx < p) {
                    prev[nb as usize] = Some(node_idx);
                }
            }
        }
    }

    Ok(None)
}

/// Two-pass all-paths Dijkstra.
///
/// Pass 1 — compute optimal distances from `src_idx`.
/// Pass 2 — walk every outgoing edge; record u as a predecessor of v when
///           `dist[u] + w(u,v) ≈ dist[v]`.
/// Pass 3 — enumerate all paths from `dst_idx` back to `src_idx` via
///           predecessors, then reverse each.
fn dijkstra_all(
    graph: &GraphFrame,
    src_idx: u32,
    dst_idx: u32,
    config: &ShortestPathConfig,
) -> Result<Vec<Vec<u32>>> {
    let n = graph.edges().node_count();

    // Pass 1.
    let dist = dijkstra_distances(graph, src_idx, config)?;
    if dist[dst_idx as usize].is_infinite() {
        return Ok(Vec::new());
    }

    // Pass 2.
    let mut all_preds: Vec<Vec<u32>> = vec![Vec::new(); n];
    for node_idx in 0..n as u32 {
        let d = dist[node_idx as usize];
        if d.is_infinite() {
            continue;
        }
        for (nb, eid) in neighbor_pairs(graph.edges(), node_idx, config) {
            let w = edge_weight(graph.edges(), eid, config)?;
            if (d + w - dist[nb as usize]).abs() < COST_EPS
                && !all_preds[nb as usize].contains(&node_idx)
            {
                all_preds[nb as usize].push(node_idx);
            }
        }
    }

    // Pass 3.
    Ok(enumerate_all_paths(&all_preds, src_idx, dst_idx))
}

/// Dijkstra pass that returns only the distance vector from `src_idx`.
fn dijkstra_distances(
    graph: &GraphFrame,
    src_idx: u32,
    config: &ShortestPathConfig,
) -> Result<Vec<f64>> {
    let n = graph.edges().node_count();
    let mut dist = vec![f64::INFINITY; n];
    dist[src_idx as usize] = 0.0;

    let mut heap = BinaryHeap::new();
    heap.push(State {
        cost: 0.0,
        node_idx: src_idx,
    });

    while let Some(State { cost, node_idx }) = heap.pop() {
        if cost > dist[node_idx as usize] + COST_EPS {
            continue;
        }
        for (nb, eid) in neighbor_pairs(graph.edges(), node_idx, config) {
            let w = edge_weight(graph.edges(), eid, config)?;
            let nc = cost + w;
            if nc < dist[nb as usize] - COST_EPS {
                dist[nb as usize] = nc;
                heap.push(State {
                    cost: nc,
                    node_idx: nb,
                });
            }
        }
    }

    Ok(dist)
}

/// A* single-path search with optional weighted edges.
fn astar_single(
    graph: &GraphFrame,
    src_idx: u32,
    dst_idx: u32,
    config: &ShortestPathConfig,
    heuristic: &dyn Fn(&str, &str) -> f64,
) -> Result<Option<Vec<u32>>> {
    let n = graph.edges().node_count();
    let mut g_score = vec![f64::INFINITY; n];
    let mut prev: Vec<Option<u32>> = vec![None; n];

    let dst_id = graph
        .edge_node_id_by_idx(dst_idx)
        .ok_or_else(|| GFError::InvalidConfig {
            message: format!("missing node id for destination compact index {}", dst_idx),
        })?;

    g_score[src_idx as usize] = 0.0;
    let mut heap = BinaryHeap::new();
    heap.push(State {
        cost: heuristic_cost(graph, src_idx, dst_id, heuristic)?,
        node_idx: src_idx,
    });

    while let Some(State { cost, node_idx }) = heap.pop() {
        let expected =
            g_score[node_idx as usize] + heuristic_cost(graph, node_idx, dst_id, heuristic)?;
        if cost > expected + COST_EPS {
            continue;
        }

        if node_idx == dst_idx {
            return Ok(Some(reconstruct_path(&prev, src_idx, dst_idx)));
        }

        for (nb, eid) in neighbor_pairs(graph.edges(), node_idx, config) {
            let tentative_g = g_score[node_idx as usize] + edge_weight(graph.edges(), eid, config)?;
            let old_g = g_score[nb as usize];

            if tentative_g < old_g - COST_EPS {
                g_score[nb as usize] = tentative_g;
                prev[nb as usize] = Some(node_idx);
                heap.push(State {
                    cost: tentative_g + heuristic_cost(graph, nb, dst_id, heuristic)?,
                    node_idx: nb,
                });
            } else if (tentative_g - old_g).abs() < COST_EPS
                && prev[nb as usize].is_some_and(|p| node_idx < p)
            {
                prev[nb as usize] = Some(node_idx);
            }
        }
    }

    Ok(None)
}

fn shortest_path_candidate(
    graph: &GraphFrame,
    src_idx: u32,
    dst_idx: u32,
    config: &ShortestPathConfig,
    banned_nodes: &HashSet<u32>,
    banned_edges: &HashSet<u32>,
    max_hops: Option<usize>,
) -> Result<Option<PathCandidate>> {
    if banned_nodes.contains(&src_idx) || banned_nodes.contains(&dst_idx) {
        return Ok(None);
    }

    let n = graph.edges().node_count();
    let hop_limit = max_hops.unwrap_or_else(|| n.saturating_sub(1));
    let mut dist = vec![f64::INFINITY; (hop_limit + 1) * n];
    let mut prev_node: Vec<Option<u32>> = vec![None; (hop_limit + 1) * n];
    let mut prev_hops: Vec<Option<usize>> = vec![None; (hop_limit + 1) * n];
    let mut prev_edge: Vec<Option<u32>> = vec![None; (hop_limit + 1) * n];
    let mut heap = BinaryHeap::new();

    dist[src_idx as usize] = 0.0;
    heap.push(HopState {
        cost: 0.0,
        node_idx: src_idx,
        hops: 0,
    });

    while let Some(HopState {
        cost,
        node_idx,
        hops,
    }) = heap.pop()
    {
        let slot = hop_slot(n, node_idx, hops);
        if cost > dist[slot] + COST_EPS {
            continue;
        }

        if node_idx == dst_idx {
            return Ok(Some(reconstruct_path_candidate(
                n, src_idx, dst_idx, hops, cost, &prev_node, &prev_hops, &prev_edge,
            )));
        }

        if hops >= hop_limit {
            continue;
        }

        for (nb, eid) in neighbor_pairs(graph.edges(), node_idx, config) {
            if banned_edges.contains(&eid) || banned_nodes.contains(&nb) {
                continue;
            }

            let next_hops = hops + 1;
            let next_slot = hop_slot(n, nb, next_hops);
            let next_cost = cost + edge_weight(graph.edges(), eid, config)?;

            if next_cost < dist[next_slot] - COST_EPS {
                dist[next_slot] = next_cost;
                prev_node[next_slot] = Some(node_idx);
                prev_hops[next_slot] = Some(hops);
                prev_edge[next_slot] = Some(eid);
                heap.push(HopState {
                    cost: next_cost,
                    node_idx: nb,
                    hops: next_hops,
                });
            } else if (next_cost - dist[next_slot]).abs() < COST_EPS
                && prev_node[next_slot].is_some_and(|p| node_idx < p)
            {
                prev_node[next_slot] = Some(node_idx);
                prev_hops[next_slot] = Some(hops);
                prev_edge[next_slot] = Some(eid);
            }
        }
    }

    Ok(None)
}

// ── Path reconstruction ───────────────────────────────────────────────────────

fn reconstruct_path(prev: &[Option<u32>], src_idx: u32, dst_idx: u32) -> Vec<u32> {
    let mut path = Vec::new();
    let mut cur = dst_idx;
    loop {
        path.push(cur);
        if cur == src_idx {
            break;
        }
        match prev[cur as usize] {
            Some(p) => cur = p,
            None => break, // unreachable in valid Dijkstra result
        }
    }
    path.reverse();
    path
}

#[allow(clippy::too_many_arguments)]
fn reconstruct_path_candidate(
    node_count: usize,
    src_idx: u32,
    dst_idx: u32,
    dst_hops: usize,
    cost: f64,
    prev_node: &[Option<u32>],
    prev_hops: &[Option<usize>],
    prev_edge: &[Option<u32>],
) -> PathCandidate {
    let mut nodes = Vec::new();
    let mut edge_rows = Vec::new();
    let mut cur_node = dst_idx;
    let mut cur_hops = dst_hops;

    loop {
        nodes.push(cur_node);
        if cur_node == src_idx && cur_hops == 0 {
            break;
        }

        let slot = hop_slot(node_count, cur_node, cur_hops);
        edge_rows.push(prev_edge[slot].expect("path state must have predecessor edge"));
        let next_node = prev_node[slot].expect("path state must have predecessor node");
        let next_hops = prev_hops[slot].expect("path state must have predecessor hop");
        cur_node = next_node;
        cur_hops = next_hops;
    }

    nodes.reverse();
    edge_rows.reverse();

    PathCandidate {
        nodes,
        edge_rows,
        cost,
    }
}

fn enumerate_all_paths(all_preds: &[Vec<u32>], src_idx: u32, dst_idx: u32) -> Vec<Vec<u32>> {
    let mut paths = Vec::new();
    let mut stack = vec![dst_idx];
    enumerate_rec(all_preds, src_idx, &mut stack, &mut paths);
    paths
}

fn enumerate_rec(
    all_preds: &[Vec<u32>],
    src_idx: u32,
    stack: &mut Vec<u32>,
    paths: &mut Vec<Vec<u32>>,
) {
    let cur = *stack.last().unwrap();
    if cur == src_idx {
        let mut path = stack.clone();
        path.reverse();
        paths.push(path);
        return;
    }
    for &pred in &all_preds[cur as usize] {
        stack.push(pred);
        enumerate_rec(all_preds, src_idx, stack, paths);
        stack.pop();
    }
}

// ── Edge helpers ─────────────────────────────────────────────────────────────

/// Returns `(neighbor_compact_idx, edge_row)` pairs reachable from `node_idx`
/// after applying direction and edge-type filters.
fn neighbor_pairs(
    edges: &EdgeFrame,
    node_idx: u32,
    config: &ShortestPathConfig,
) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    match config.direction {
        Direction::Out => {
            append_type_filtered(
                edges.out_neighbors(node_idx),
                edges.out_edge_ids(node_idx),
                &config.edge_type,
                edges,
                &mut out,
            );
        }
        Direction::In => {
            append_type_filtered(
                edges.in_neighbors(node_idx),
                edges.in_edge_ids(node_idx),
                &config.edge_type,
                edges,
                &mut out,
            );
        }
        Direction::Both | Direction::None => {
            append_type_filtered(
                edges.out_neighbors(node_idx),
                edges.out_edge_ids(node_idx),
                &config.edge_type,
                edges,
                &mut out,
            );
            append_type_filtered(
                edges.in_neighbors(node_idx),
                edges.in_edge_ids(node_idx),
                &config.edge_type,
                edges,
                &mut out,
            );
        }
    }
    out
}

fn append_type_filtered(
    neighbors: &[u32],
    edge_ids: &[u32],
    spec: &EdgeTypeSpec,
    edges: &EdgeFrame,
    out: &mut Vec<(u32, u32)>,
) {
    for (&nb, &eid) in neighbors.iter().zip(edge_ids.iter()) {
        if matches_type(edges, eid, spec) {
            out.push((nb, eid));
        }
    }
}

fn matches_type(edges: &EdgeFrame, edge_row: u32, spec: &EdgeTypeSpec) -> bool {
    match spec {
        EdgeTypeSpec::Any => true,
        EdgeTypeSpec::Single(t) => edges.edge_type_at(edge_row) == t.as_str(),
        EdgeTypeSpec::Multiple(ts) => {
            let et = edges.edge_type_at(edge_row);
            ts.iter().any(|t| t.as_str() == et)
        }
    }
}

/// Reads the weight of the edge at `edge_row`.
///
/// Returns `1.0` if `config.weight_col` is `None`.
fn edge_weight(edges: &EdgeFrame, edge_row: u32, config: &ShortestPathConfig) -> Result<f64> {
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

    let w: f64 = if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
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

    if w.is_nan() || w.is_infinite() {
        return Err(GFError::TypeMismatch {
            message: format!(
                "weight column '{}' contains non-finite value at edge row {}",
                col_name, row
            ),
        });
    }
    if w < 0.0 {
        return Err(GFError::NegativeWeight {
            column: col_name.to_owned(),
        });
    }

    Ok(w)
}

// ── Validation helpers ────────────────────────────────────────────────────────

fn validate_endpoints(graph: &GraphFrame, src: &str, dst: &str) -> Result<()> {
    if graph.nodes().row_index(src).is_none() {
        return Err(GFError::NodeNotFound { id: src.to_owned() });
    }
    if graph.nodes().row_index(dst).is_none() {
        return Err(GFError::NodeNotFound { id: dst.to_owned() });
    }
    Ok(())
}

fn validate_weight_col(edges: &EdgeFrame, config: &ShortestPathConfig) -> Result<()> {
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

fn heuristic_cost(
    graph: &GraphFrame,
    node_idx: u32,
    dst_id: &str,
    heuristic: &dyn Fn(&str, &str) -> f64,
) -> Result<f64> {
    let node_id = graph
        .edge_node_id_by_idx(node_idx)
        .ok_or_else(|| GFError::InvalidConfig {
            message: format!("missing node id for compact index {}", node_idx),
        })?;
    let cost = heuristic(node_id, dst_id);

    if !cost.is_finite() || cost < 0.0 {
        return Err(GFError::InvalidConfig {
            message: format!(
                "heuristic must return a finite, non-negative value for ({}, {})",
                node_id, dst_id
            ),
        });
    }

    Ok(cost)
}

fn hop_slot(node_count: usize, node_idx: u32, hops: usize) -> usize {
    hops * node_count + node_idx as usize
}

fn path_cost(edges: &EdgeFrame, edge_rows: &[u32], config: &ShortestPathConfig) -> Result<f64> {
    edge_rows.iter().try_fold(0.0, |acc, &edge_row| {
        edge_weight(edges, edge_row, config).map(|weight| acc + weight)
    })
}

fn select_best_candidate(candidates: &[PathCandidate]) -> usize {
    candidates
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| compare_candidates(left, right))
        .map(|(idx, _)| idx)
        .expect("candidates is non-empty")
}

fn compare_candidates(left: &PathCandidate, right: &PathCandidate) -> Ordering {
    left.cost
        .partial_cmp(&right.cost)
        .unwrap_or(Ordering::Equal)
        .then_with(|| left.nodes.cmp(&right.nodes))
}
