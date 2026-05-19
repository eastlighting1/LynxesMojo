//! Graph partitioning and distributed BFS across partitions.
//!
//! # Overview
//!
//! Large graphs that don't fit on one machine (or that benefit from parallel
//! processing across multiple nodes) are split into _shards_ — each a
//! self-contained `GraphFrame` with its own CSR index.  Edges whose endpoints
//! fall in **different** shards are captured in a shared `boundary_edges`
//! `EdgeFrame`.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │  GraphFrame (full)                                           │
//! │                                                              │
//! │   nodes: A, B, C, D, E, F                                   │
//! │   edges: A→B, A→C, B→D, C→E, D→F, E→F                      │
//! └──────────────────────────┬───────────────────────────────────┘
//!                            │ partition(n=2, Hash)
//!      ┌─────────────────────┴──────────────────────┐
//!      ▼                                            ▼
//! ┌──────────┐  boundary_edges  ┌──────────────────────┐
//! │ Shard 0  │  A→C, B→D        │ Shard 1              │
//! │ A, B, D  │◄────────────────►│ C, E, F              │
//! │ A→B      │                  │ C→E, D→F, E→F        │
//! └──────────┘                  └──────────────────────┘
//! ```
//!
//! # Distributed BFS
//!
//! `PartitionedGraph::distributed_expand` performs BFS across shards by
//! following boundary edges between rounds, accumulating visited nodes and
//! edges from all shards + the boundary EdgeFrame.

use std::collections::{HashMap, HashSet};

use arrow_array::{Array, StringArray};

use crate::{
    Direction, EdgeFrame, EdgeTypeSpec, GFError, GraphFrame, NodeFrame, Result, COL_EDGE_DST,
    COL_EDGE_SRC,
};

// ── Public types ──────────────────────────────────────────────────────────────

/// How to assign each node to a shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PartitionMethod {
    /// `hash(node_id) % n_shards` — deterministic, works well for random IDs.
    #[default]
    Hash,
    /// Lexicographic sort then divide into equal bands — good for ordered IDs.
    Range,
    /// All nodes sharing the same first `_label` go to the same shard.
    /// Nodes with no label fall back to hash partitioning.
    Label,
}

/// Statistics describing how balanced a partition is.
#[derive(Debug, Clone)]
pub struct PartitionStats {
    pub n_shards: usize,
    pub nodes_per_shard: Vec<usize>,
    pub edges_per_shard: Vec<usize>,
    pub boundary_edge_count: usize,
    /// `max_shard_nodes / avg_shard_nodes`.  1.0 = perfect balance.
    pub imbalance_ratio: f64,
}

/// A partitioned view of a `GraphFrame`.
///
/// Each shard is a fully self-contained `GraphFrame` (its own CSR index).
/// Edges crossing shard boundaries are stored in `boundary_edges` so that
/// distributed BFS can follow them without reading the full graph.
#[derive(Debug, Clone)]
pub struct PartitionedGraph {
    /// One `GraphFrame` per shard.
    pub shards: Vec<GraphFrame>,
    /// Edges whose `_src` and `_dst` belong to different shards.
    pub boundary_edges: EdgeFrame,
    pub n_shards: usize,
    /// `node_id → shard index` lookup built once at partition time.
    node_to_shard: HashMap<String, usize>,
}

impl PartitionedGraph {
    /// Which shard owns `node_id`?  O(1).
    pub fn shard_of(&self, node_id: &str) -> Option<usize> {
        self.node_to_shard.get(node_id).copied()
    }

    /// Partition statistics.
    pub fn stats(&self) -> PartitionStats {
        let nodes_per_shard: Vec<usize> = self.shards.iter().map(|s| s.node_count()).collect();
        let edges_per_shard: Vec<usize> = self.shards.iter().map(|s| s.edge_count()).collect();
        let boundary_edge_count = self.boundary_edges.len();
        let total: usize = nodes_per_shard.iter().sum();
        let avg = if self.n_shards == 0 {
            0.0
        } else {
            total as f64 / self.n_shards as f64
        };
        let max = nodes_per_shard.iter().copied().max().unwrap_or(0) as f64;
        let imbalance_ratio = if avg == 0.0 { 1.0 } else { max / avg };
        PartitionStats {
            n_shards: self.n_shards,
            nodes_per_shard,
            edges_per_shard,
            boundary_edge_count,
            imbalance_ratio,
        }
    }

    /// Merge all shards + boundary edges back into a single `GraphFrame`.
    ///
    /// Useful for validation: `partition → merge` should preserve total node
    /// and edge counts.
    pub fn merge(&self) -> Result<GraphFrame> {
        if self.shards.is_empty() {
            return Err(GFError::InvalidConfig {
                message: "cannot merge zero shards".to_owned(),
            });
        }
        // Collect all shard NodeFrames
        let all_node_refs: Vec<&NodeFrame> = self.shards.iter().map(|s| s.nodes()).collect();
        let merged_nodes = NodeFrame::concat(&all_node_refs)?;

        // Collect all intra-shard EdgeFrames + boundary_edges
        let mut all_edge_refs: Vec<&EdgeFrame> = self.shards.iter().map(|s| s.edges()).collect();
        all_edge_refs.push(&self.boundary_edges);
        let merged_edges = EdgeFrame::concat(&all_edge_refs)?;

        GraphFrame::new(merged_nodes, merged_edges)
    }

    /// Distributed BFS expand.
    ///
    /// Expands `seed_ids` by `hops` hops, following both intra-shard CSR
    /// edges and cross-shard `boundary_edges`.  Returns the merged
    /// (`NodeFrame`, `EdgeFrame`) of all discovered nodes and edges.
    pub fn distributed_expand(
        &self,
        seed_ids: &[&str],
        edge_type: &EdgeTypeSpec,
        hops: u32,
        direction: Direction,
    ) -> Result<(NodeFrame, EdgeFrame)> {
        let mut frontier: HashSet<String> = seed_ids.iter().map(|&s| s.to_owned()).collect();
        let mut visited: HashSet<String> = frontier.clone();
        let mut visited_edge_rows: Vec<HashSet<usize>> =
            (0..self.n_shards).map(|_| HashSet::new()).collect();
        let mut visited_boundary: HashSet<usize> = HashSet::new();

        for _ in 0..hops {
            let mut next_frontier: HashSet<String> = HashSet::new();

            // 1. Intra-shard expansion
            for (shard_idx, shard) in self.shards.iter().enumerate() {
                // Which frontier nodes live in this shard?
                let shard_seeds: Vec<&str> = frontier
                    .iter()
                    .filter(|id| self.shard_of(id) == Some(shard_idx))
                    .map(String::as_str)
                    .collect();
                if shard_seeds.is_empty() {
                    continue;
                }
                // Direct 1-hop expansion (avoids LazyGraphFrame dependency)
                let new_ids = direct_expand(shard, &shard_seeds, edge_type, direction);
                for id in new_ids {
                    if visited.insert(id.clone()) {
                        next_frontier.insert(id);
                    }
                }
                // Record which edge rows of this shard we've traversed
                // (collect all edges whose src is in our seed set)
                collect_reachable_edge_rows(
                    shard,
                    &shard_seeds,
                    edge_type,
                    direction,
                    &mut visited_edge_rows[shard_idx],
                );
            }

            // 2. Cross-shard expansion via boundary_edges
            let (new_cross_nodes, new_boundary_rows) = expand_via_boundary(
                &self.boundary_edges,
                &frontier,
                edge_type,
                direction,
                &visited,
            )?;
            for id in new_cross_nodes {
                if visited.insert(id.clone()) {
                    next_frontier.insert(id);
                }
            }
            visited_boundary.extend(new_boundary_rows);

            if next_frontier.is_empty() {
                break;
            }
            frontier = next_frontier;
        }

        // ── Build output ─────────────────────────────────────────────────────
        let visited_set: HashSet<&str> = visited.iter().map(String::as_str).collect();

        // Collect per-shard subgraphs of visited nodes
        let shard_subgraphs: Vec<GraphFrame> = self
            .shards
            .iter()
            .map(|shard| {
                let shard_ids: Vec<&str> = visited_set
                    .iter()
                    .copied()
                    .filter(|&id| {
                        self.shard_of(id).is_some() && { shard.nodes().row_index(id).is_some() }
                    })
                    .collect();
                if shard_ids.is_empty() {
                    // Return empty graph with same schema
                    shard.subgraph(&[])
                } else {
                    shard.subgraph(&shard_ids)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let node_frame_refs: Vec<&NodeFrame> = shard_subgraphs.iter().map(|g| g.nodes()).collect();
        let merged_nodes = NodeFrame::concat(&node_frame_refs)?;

        // Collect edges from shard subgraphs + filtered boundary
        let edge_frame_refs: Vec<&EdgeFrame> = shard_subgraphs.iter().map(|g| g.edges()).collect();
        let boundary_mask: Vec<bool> = (0..self.boundary_edges.len())
            .map(|i| visited_boundary.contains(&i))
            .collect();
        let boundary_sub = filter_edge_frame_by_rows(&self.boundary_edges, &boundary_mask)?;
        let mut all_ef_refs: Vec<&EdgeFrame> = edge_frame_refs;
        all_ef_refs.push(&boundary_sub);
        let merged_edges = EdgeFrame::concat(&all_ef_refs)?;

        Ok((merged_nodes, merged_edges))
    }
}

// ── GraphPartitioner ──────────────────────────────────────────────────────────

/// Splits a `GraphFrame` into N balanced shards.
///
/// # Example
///
/// ```ignore
/// use lynxes_core::{Direction, EdgeTypeSpec, GraphPartitionMethod, GraphPartitioner};
///
/// let pg = GraphPartitioner::by_hash(&graph, 4)?;
/// println!("{:?}", pg.stats());
/// let (nodes, edges) = pg.distributed_expand(&["alice"], &EdgeTypeSpec::Any, 2, Direction::Out)?;
/// ```
pub struct GraphPartitioner;

impl GraphPartitioner {
    /// Partition by `hash(node_id) % n_shards`.
    pub fn by_hash(graph: &GraphFrame, n_shards: usize) -> Result<PartitionedGraph> {
        Self::partition(graph, n_shards, PartitionMethod::Hash)
    }

    /// Partition by lexicographic range — each shard gets a consecutive band
    /// of node IDs when sorted alphabetically.
    pub fn by_range(graph: &GraphFrame, n_shards: usize) -> Result<PartitionedGraph> {
        Self::partition(graph, n_shards, PartitionMethod::Range)
    }

    /// Assign nodes to shards by their first label.
    /// All nodes with the same label go to the same shard (label-affinity).
    pub fn by_label(graph: &GraphFrame, n_shards: usize) -> Result<PartitionedGraph> {
        Self::partition(graph, n_shards, PartitionMethod::Label)
    }

    /// Core partitioning logic.
    pub fn partition(
        graph: &GraphFrame,
        n_shards: usize,
        method: PartitionMethod,
    ) -> Result<PartitionedGraph> {
        if n_shards == 0 {
            return Err(GFError::InvalidConfig {
                message: "n_shards must be >= 1".to_owned(),
            });
        }

        let node_batch = graph.nodes().to_record_batch();
        let id_col = node_batch
            .column_by_name("_id")
            .ok_or_else(|| GFError::ColumnNotFound {
                column: "_id".to_owned(),
            })?;
        let ids: &StringArray = id_col
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| GFError::TypeMismatch {
                message: "_id must be Utf8".to_owned(),
            })?;

        // Build node_to_shard map
        let assignments = assign_nodes(graph, ids, n_shards, method)?;
        let node_to_shard: HashMap<String, usize> = ids
            .iter()
            .zip(assignments.iter())
            .filter_map(|(id, &shard)| Some((id?.to_owned(), shard)))
            .collect();

        // Build per-shard NodeFrames via boolean masks
        let mut shard_node_masks: Vec<Vec<bool>> =
            (0..n_shards).map(|_| vec![false; ids.len()]).collect();
        for (row, &shard) in assignments.iter().enumerate() {
            shard_node_masks[shard][row] = true;
        }

        // Assign edges to shards (src and dst in same shard) or boundary
        let edge_batch = graph.edges().to_record_batch();
        let src_col = edge_batch
            .column_by_name(COL_EDGE_SRC)
            .ok_or_else(|| GFError::ColumnNotFound {
                column: COL_EDGE_SRC.to_owned(),
            })?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| GFError::TypeMismatch {
                message: "_src must be Utf8".to_owned(),
            })?;
        let dst_col = edge_batch
            .column_by_name(COL_EDGE_DST)
            .ok_or_else(|| GFError::ColumnNotFound {
                column: COL_EDGE_DST.to_owned(),
            })?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| GFError::TypeMismatch {
                message: "_dst must be Utf8".to_owned(),
            })?;

        let edge_count = edge_batch.num_rows();
        let mut shard_edge_masks: Vec<Vec<bool>> =
            (0..n_shards).map(|_| vec![false; edge_count]).collect();
        let mut boundary_mask = vec![false; edge_count];

        for row in 0..edge_count {
            let src = src_col.value(row);
            let dst = dst_col.value(row);
            let src_shard = node_to_shard.get(src).copied();
            let dst_shard = node_to_shard.get(dst).copied();
            match (src_shard, dst_shard) {
                (Some(s), Some(d)) if s == d => shard_edge_masks[s][row] = true,
                _ => boundary_mask[row] = true,
            }
        }

        // Build shards
        let mut shards = Vec::with_capacity(n_shards);
        for shard_idx in 0..n_shards {
            let node_mask = bool_vec_to_boolean_array(&shard_node_masks[shard_idx]);
            let edge_mask = bool_vec_to_boolean_array(&shard_edge_masks[shard_idx]);
            let shard_nodes = graph.nodes().filter(&node_mask)?;
            let shard_edges = graph.edges().filter(&edge_mask)?;
            shards.push(GraphFrame::new(shard_nodes, shard_edges)?);
        }

        // Build boundary EdgeFrame
        let boundary_bool = bool_vec_to_boolean_array(&boundary_mask);
        let boundary_edges = graph.edges().filter(&boundary_bool)?;

        Ok(PartitionedGraph {
            shards,
            boundary_edges,
            n_shards,
            node_to_shard,
        })
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Compute shard index for each node row.
fn assign_nodes(
    graph: &GraphFrame,
    ids: &StringArray,
    n_shards: usize,
    method: PartitionMethod,
) -> Result<Vec<usize>> {
    match method {
        PartitionMethod::Hash => Ok(ids
            .iter()
            .map(|id| hash_shard(id.unwrap_or(""), n_shards))
            .collect()),
        PartitionMethod::Range => {
            // Sort IDs lexicographically, assign bands
            let mut indexed: Vec<(usize, &str)> = ids
                .iter()
                .enumerate()
                .map(|(i, id)| (i, id.unwrap_or("")))
                .collect();
            indexed.sort_by_key(|&(_, id)| id);
            let mut result = vec![0usize; ids.len()];
            let band = (ids.len() + n_shards - 1) / n_shards.max(1);
            for (rank, (row, _)) in indexed.iter().enumerate() {
                result[*row] = (rank / band.max(1)).min(n_shards - 1);
            }
            Ok(result)
        }
        PartitionMethod::Label => {
            // Group by first label; unknown labels → hash fallback
            let node_batch = graph.nodes().to_record_batch();
            let label_col = node_batch.column_by_name("_label");
            if label_col.is_none() {
                // No label column — fall back to hash
                return assign_nodes(graph, ids, n_shards, PartitionMethod::Hash);
            }
            let label_col = label_col.unwrap();
            use arrow_array::ListArray;
            let list = label_col
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| GFError::TypeMismatch {
                    message: "_label must be List<Utf8>".to_owned(),
                })?;

            // Build label→shard map lazily
            let mut label_map: HashMap<String, usize> = HashMap::new();
            let mut next_shard = 0usize;
            let mut result = Vec::with_capacity(ids.len());
            for row in 0..ids.len() {
                let shard = if list.is_null(row) || list.value(row).is_empty() {
                    hash_shard(ids.value(row), n_shards)
                } else {
                    let first_label = list
                        .value(row)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .and_then(|a| {
                            if a.is_empty() {
                                None
                            } else {
                                Some(a.value(0).to_owned())
                            }
                        })
                        .unwrap_or_default();
                    *label_map.entry(first_label).or_insert_with(|| {
                        let s = next_shard % n_shards;
                        next_shard += 1;
                        s
                    })
                };
                result.push(shard);
            }
            Ok(result)
        }
    }
}

fn hash_shard(id: &str, n_shards: usize) -> usize {
    use std::hash::{Hash, Hasher};
    let mut hasher = ahash::AHasher::default();
    id.hash(&mut hasher);
    (hasher.finish() as usize) % n_shards
}

fn bool_vec_to_boolean_array(mask: &[bool]) -> arrow_array::BooleanArray {
    arrow_array::BooleanArray::from(mask.to_vec())
}

/// Build a filter `Expr` that matches nodes whose `_id` is in `ids`.
/// One-hop neighbor expansion via direct edge-column iteration.
///
/// Returns the IDs of nodes reachable from `seed_ids` in one hop following
/// `edge_type` and `direction`, without going through `LazyGraphFrame`.
fn direct_expand(
    shard: &GraphFrame,
    seed_ids: &[&str],
    edge_type: &EdgeTypeSpec,
    direction: Direction,
) -> Vec<String> {
    let seed_set: HashSet<&str> = seed_ids.iter().copied().collect();
    let edge_batch = shard.edges().to_record_batch();
    let src_col = match edge_batch.column_by_name(COL_EDGE_SRC) {
        Some(c) => c.clone(),
        None => return Vec::new(),
    };
    let dst_col = match edge_batch.column_by_name(COL_EDGE_DST) {
        Some(c) => c.clone(),
        None => return Vec::new(),
    };
    let src_arr = match src_col.as_any().downcast_ref::<StringArray>() {
        Some(a) => a,
        None => return Vec::new(),
    };
    let dst_arr = match dst_col.as_any().downcast_ref::<StringArray>() {
        Some(a) => a,
        None => return Vec::new(),
    };

    let mut found = Vec::new();
    for row in 0..edge_batch.num_rows() {
        if !edge_type_matches(shard.edges(), row, edge_type) {
            continue;
        }
        let src = src_arr.value(row);
        let dst = dst_arr.value(row);
        match direction {
            Direction::Out => {
                if seed_set.contains(src) {
                    found.push(dst.to_owned());
                }
            }
            Direction::In => {
                if seed_set.contains(dst) {
                    found.push(src.to_owned());
                }
            }
            Direction::Both | Direction::None => {
                if seed_set.contains(src) {
                    found.push(dst.to_owned());
                }
                if seed_set.contains(dst) {
                    found.push(src.to_owned());
                }
            }
        }
    }
    found
}

/// Collect edge row indices in `shard` that are reachable from `seed_ids`.
fn collect_reachable_edge_rows(
    shard: &GraphFrame,
    seed_ids: &[&str],
    edge_type: &EdgeTypeSpec,
    direction: Direction,
    visited_rows: &mut HashSet<usize>,
) {
    let edge_batch = shard.edges().to_record_batch();
    let src_arc = match edge_batch.column_by_name(COL_EDGE_SRC) {
        Some(c) => c.clone(),
        None => return,
    };
    let dst_arc = match edge_batch.column_by_name(COL_EDGE_DST) {
        Some(c) => c.clone(),
        None => return,
    };
    let src_arr = match src_arc.as_any().downcast_ref::<StringArray>() {
        Some(a) => a,
        None => return,
    };
    let dst_arr = match dst_arc.as_any().downcast_ref::<StringArray>() {
        Some(a) => a,
        None => return,
    };

    let seed_set: HashSet<&str> = seed_ids.iter().copied().collect();

    for row in 0..edge_batch.num_rows() {
        let src = src_arr.value(row);
        let dst = dst_arr.value(row);
        let matches = match direction {
            Direction::Out => seed_set.contains(src),
            Direction::In => seed_set.contains(dst),
            Direction::Both | Direction::None => seed_set.contains(src) || seed_set.contains(dst),
        };
        if matches && edge_type_matches(shard.edges(), row, edge_type) {
            visited_rows.insert(row);
        }
    }
}

fn edge_type_matches(edges: &EdgeFrame, row: usize, spec: &EdgeTypeSpec) -> bool {
    use crate::COL_EDGE_TYPE;
    match spec {
        EdgeTypeSpec::Any => true,
        EdgeTypeSpec::Single(t) => edges
            .to_record_batch()
            .column_by_name(COL_EDGE_TYPE)
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .map(|a| a.value(row) == t.as_str())
            .unwrap_or(false),
        EdgeTypeSpec::Multiple(ts) => edges
            .to_record_batch()
            .column_by_name(COL_EDGE_TYPE)
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .map(|a| ts.iter().any(|t| t == a.value(row)))
            .unwrap_or(false),
    }
}

/// Scan `boundary_edges` for edges whose source (or dest on `In`) is in
/// `frontier`, returning new node IDs and the edge row indices that were hit.
fn expand_via_boundary(
    boundary: &EdgeFrame,
    frontier: &HashSet<String>,
    edge_type: &EdgeTypeSpec,
    direction: Direction,
    visited: &HashSet<String>,
) -> Result<(Vec<String>, Vec<usize>)> {
    let batch = boundary.to_record_batch();
    let src_arc = match batch.column_by_name(COL_EDGE_SRC) {
        Some(c) => c.clone(),
        None => return Ok((Vec::new(), Vec::new())),
    };
    let dst_arc = match batch.column_by_name(COL_EDGE_DST) {
        Some(c) => c.clone(),
        None => return Ok((Vec::new(), Vec::new())),
    };
    let src_arr = match src_arc.as_any().downcast_ref::<StringArray>() {
        Some(a) => a,
        None => return Ok((Vec::new(), Vec::new())),
    };
    let dst_arr = match dst_arc.as_any().downcast_ref::<StringArray>() {
        Some(a) => a,
        None => return Ok((Vec::new(), Vec::new())),
    };

    let mut new_nodes = Vec::new();
    let mut new_rows = Vec::new();

    for row in 0..batch.num_rows() {
        let src = src_arr.value(row);
        let dst = dst_arr.value(row);
        if !edge_type_matches(boundary, row, edge_type) {
            continue;
        }
        match direction {
            Direction::Out => {
                if frontier.contains(src) && !visited.contains(dst) {
                    new_nodes.push(dst.to_owned());
                    new_rows.push(row);
                }
            }
            Direction::In => {
                if frontier.contains(dst) && !visited.contains(src) {
                    new_nodes.push(src.to_owned());
                    new_rows.push(row);
                }
            }
            Direction::Both | Direction::None => {
                if frontier.contains(src) && !visited.contains(dst) {
                    new_nodes.push(dst.to_owned());
                    new_rows.push(row);
                }
                if frontier.contains(dst) && !visited.contains(src) {
                    new_nodes.push(src.to_owned());
                    new_rows.push(row);
                }
            }
        }
    }
    Ok((new_nodes, new_rows))
}

/// Build a new `EdgeFrame` containing only the rows where `mask[row] == true`.
fn filter_edge_frame_by_rows(edges: &EdgeFrame, mask: &[bool]) -> Result<EdgeFrame> {
    let bool_arr = arrow_array::BooleanArray::from(mask.to_vec());
    edges.filter(&bool_arr)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        builder::{ListBuilder, StringBuilder},
        ArrayRef, Int8Array, RecordBatch, StringArray,
    };
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};

    use super::*;
    use crate::{Direction, EdgeTypeSpec};

    fn bridge_graph() -> GraphFrame {
        let node_schema = Arc::new(ArrowSchema::new(vec![
            Field::new(crate::COL_NODE_ID, DataType::Utf8, false),
            Field::new(
                crate::COL_NODE_LABEL,
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                false,
            ),
        ]));
        let mut labels = ListBuilder::new(StringBuilder::new());
        for _ in 0..6 {
            labels.values().append_value("Node");
            labels.append(true);
        }
        let nodes = NodeFrame::from_record_batch(
            RecordBatch::try_new(
                node_schema,
                vec![
                    Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e", "f"])) as ArrayRef,
                    Arc::new(labels.finish()) as ArrayRef,
                ],
            )
            .unwrap(),
        )
        .unwrap();

        let edge_schema = Arc::new(ArrowSchema::new(vec![
            Field::new(crate::COL_EDGE_SRC, DataType::Utf8, false),
            Field::new(crate::COL_EDGE_DST, DataType::Utf8, false),
            Field::new(crate::COL_EDGE_TYPE, DataType::Utf8, false),
            Field::new(crate::COL_EDGE_DIRECTION, DataType::Int8, false),
        ]));
        let edges = EdgeFrame::from_record_batch(
            RecordBatch::try_new(
                edge_schema,
                vec![
                    Arc::new(StringArray::from(vec!["a", "a", "b", "c", "d", "e"])) as ArrayRef,
                    Arc::new(StringArray::from(vec!["b", "c", "d", "e", "f", "f"])) as ArrayRef,
                    Arc::new(StringArray::from(vec!["KNOWS"; 6])) as ArrayRef,
                    Arc::new(Int8Array::from(vec![Direction::Out.as_i8(); 6])) as ArrayRef,
                ],
            )
            .unwrap(),
        )
        .unwrap();

        GraphFrame::new(nodes, edges).unwrap()
    }

    #[test]
    fn test_partition_node_count_preserved() {
        let g = bridge_graph();
        let pg = GraphPartitioner::by_hash(&g, 2).unwrap();
        let total: usize = pg.shards.iter().map(|s| s.node_count()).sum();
        assert_eq!(
            total,
            g.node_count(),
            "all nodes must appear in exactly one shard"
        );
    }

    #[test]
    fn test_partition_edge_count_total() {
        let g = bridge_graph();
        let pg = GraphPartitioner::by_hash(&g, 2).unwrap();
        let intra: usize = pg.shards.iter().map(|s| s.edge_count()).sum();
        let boundary = pg.boundary_edges.len();
        assert_eq!(
            intra + boundary,
            g.edge_count(),
            "intra + boundary must equal original"
        );
    }

    #[test]
    fn test_partition_merge_roundtrip() {
        let g = bridge_graph();
        let pg = GraphPartitioner::by_hash(&g, 3).unwrap();
        let merged = pg.merge().unwrap();
        assert_eq!(merged.node_count(), g.node_count());
        assert_eq!(merged.edge_count(), g.edge_count());
    }

    #[test]
    fn test_range_partition_covers_all_nodes() {
        let g = bridge_graph();
        let pg = GraphPartitioner::by_range(&g, 2).unwrap();
        let total: usize = pg.shards.iter().map(|s| s.node_count()).sum();
        assert_eq!(total, g.node_count());
    }

    #[test]
    fn test_label_partition_covers_all_nodes() {
        let g = bridge_graph();
        let pg = GraphPartitioner::by_label(&g, 2).unwrap();
        let total: usize = pg.shards.iter().map(|s| s.node_count()).sum();
        assert_eq!(total, g.node_count());
    }

    #[test]
    fn test_hash_partition_balance() {
        let g = bridge_graph();
        let pg = GraphPartitioner::by_hash(&g, 2).unwrap();
        let stats = pg.stats();
        // With 6 nodes and 2 shards, imbalance ratio should be ≤ 2.0
        assert!(
            stats.imbalance_ratio <= 2.0,
            "imbalance too high: {}",
            stats.imbalance_ratio
        );
    }

    #[test]
    fn test_single_shard_equals_original() {
        let g = bridge_graph();
        let pg = GraphPartitioner::by_hash(&g, 1).unwrap();
        assert_eq!(pg.shards.len(), 1);
        assert_eq!(pg.shards[0].node_count(), g.node_count());
        assert_eq!(pg.boundary_edges.len(), 0);
    }

    #[test]
    fn test_distributed_expand_reachability() {
        let g = bridge_graph();

        // Distributed 2-hop from "a" across 2 shards
        let pg = GraphPartitioner::by_hash(&g, 2).unwrap();
        let (dist_nodes, _edges) = pg
            .distributed_expand(&["a"], &EdgeTypeSpec::Any, 2, Direction::Out)
            .unwrap();
        let dist_ids: HashSet<String> = dist_nodes
            .id_column()
            .iter()
            .flatten()
            .map(str::to_owned)
            .collect();

        let expected_ids: HashSet<String> = ["a", "b", "c", "d", "e"]
            .into_iter()
            .map(str::to_owned)
            .collect();
        for id in &expected_ids {
            assert!(
                dist_ids.contains(id),
                "missing node {id} in distributed result"
            );
        }
        assert!(!dist_ids.contains("f"));
    }

    #[test]
    fn test_partition_stats_fields() {
        let g = bridge_graph();
        let pg = GraphPartitioner::by_hash(&g, 3).unwrap();
        let stats = pg.stats();
        assert_eq!(stats.n_shards, 3);
        assert_eq!(stats.nodes_per_shard.len(), 3);
        assert_eq!(stats.edges_per_shard.len(), 3);
        assert!(stats.imbalance_ratio >= 1.0);
    }
}
