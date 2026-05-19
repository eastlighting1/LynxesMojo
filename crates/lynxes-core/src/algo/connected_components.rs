//! Connected-components algorithms (ALG-004).
//!
//! Provides [`GraphFrame::connected_components`] and
//! [`GraphFrame::largest_connected_component`].

use std::collections::VecDeque;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, UInt32Array};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};

use crate::{frame::graph_frame::GraphFrame, NodeFrame, Result, COL_NODE_ID, COL_NODE_LABEL};

// ── GraphFrame public API ─────────────────────────────────────────────────────

impl GraphFrame {
    /// Assigns a component ID to every node using undirected BFS.
    ///
    /// Returns a [`NodeFrame`] with exactly three columns:
    ///
    /// | Column         | Type         | Notes                       |
    /// |----------------|--------------|-----------------------------|
    /// | `_id`          | `Utf8`       | same order as input         |
    /// | `_label`       | `List<Utf8>` | preserved from input        |
    /// | `component_id` | `UInt32`     | starts at 0, discovery order |
    ///
    /// # Semantics
    ///
    /// Connectivity is **undirected**: outgoing and incoming edges both contribute
    /// adjacency, so `a→b` and `b→a` are treated identically.
    ///
    /// Component IDs are assigned in NodeFrame row order: the component
    /// containing row 0 receives ID 0, the next newly-seen component receives
    /// ID 1, and so on.
    ///
    /// # Complexity
    ///
    /// O(N + E).
    pub fn connected_components(&self) -> Result<NodeFrame> {
        let n = self.nodes().len();
        if n == 0 {
            return build_empty_cc_output(self);
        }
        let (component_ids, _num_components) = compute_components(self);
        build_cc_output(self, &component_ids)
    }

    /// Returns the induced [`GraphFrame`] for the connected component with the
    /// most nodes.
    ///
    /// When multiple components share the maximum node count, the component
    /// with the **lowest** `component_id` (i.e. the one whose first member
    /// appears earliest in NodeFrame row order) is returned.
    ///
    /// Returns an empty `GraphFrame` if the graph has no nodes.
    ///
    /// # Complexity
    ///
    /// O(N + E) for component discovery, plus subgraph materialisation.
    pub fn largest_connected_component(&self) -> Result<GraphFrame> {
        let n = self.nodes().len();
        if n == 0 {
            return self.subgraph(&[]);
        }

        let (component_ids, num_components) = compute_components(self);

        if num_components == 0 {
            return self.subgraph(&[]);
        }

        // Count nodes per component.
        let mut sizes = vec![0usize; num_components as usize];
        for &cid in &component_ids {
            sizes[cid as usize] += 1;
        }

        // Largest by size; equal sizes break on lowest component_id.
        // In max_by: for equal size, `bi.cmp(ai)` returns Greater when bi > ai,
        // meaning the element with the *lower* index wins (a is preferred).
        let largest_cid = sizes
            .iter()
            .enumerate()
            .max_by(|(ai, as_), (bi, bs)| as_.cmp(bs).then_with(|| bi.cmp(ai)))
            .map(|(cid, _)| cid as u32)
            .unwrap(); // safe: num_components > 0

        let id_col = self.nodes().id_column();
        let node_ids: Vec<&str> = component_ids
            .iter()
            .enumerate()
            .filter(|(_, &cid)| cid == largest_cid)
            .map(|(row, _)| id_col.value(row))
            .collect();

        self.subgraph(&node_ids)
    }
}

// ── Core BFS kernel ───────────────────────────────────────────────────────────

/// Assigns component IDs to every node via undirected BFS.
///
/// Returns `(component_ids, num_components)` where `component_ids` is indexed
/// by NodeFrame row.
///
/// Component IDs start at 0 and are assigned in NodeFrame row order.
fn compute_components(graph: &GraphFrame) -> (Vec<u32>, u32) {
    let n = graph.nodes().len();
    let id_col = graph.nodes().id_column();

    // ── Index tables ─────────────────────────────────────────────────────────

    // NodeFrame row → EdgeFrame compact index (None for isolated nodes).
    let nf_to_eidx: Vec<Option<u32>> = (0..n)
        .map(|row| graph.edges().node_row_idx(id_col.value(row)))
        .collect();

    // EdgeFrame compact index → NodeFrame row (usize::MAX = unmapped).
    let ec = graph.edges().node_count();
    let mut eidx_to_nf = vec![usize::MAX; ec];
    for (nf_row, maybe_eidx) in nf_to_eidx.iter().enumerate() {
        if let Some(&eidx) = maybe_eidx.as_ref() {
            eidx_to_nf[eidx as usize] = nf_row;
        }
    }

    // ── BFS ──────────────────────────────────────────────────────────────────

    let mut component_ids: Vec<Option<u32>> = vec![None; n];
    let mut next_cid = 0u32;
    let mut queue: VecDeque<usize> = VecDeque::new();

    for start in 0..n {
        if component_ids[start].is_some() {
            continue;
        }

        let cid = next_cid;
        next_cid += 1;
        component_ids[start] = Some(cid);
        queue.push_back(start);

        while let Some(nf_row) = queue.pop_front() {
            let edge_idx = match nf_to_eidx[nf_row] {
                Some(idx) => idx,
                None => continue, // isolated node — no adjacency to expand
            };

            // Undirected traversal: follow both outgoing and incoming edges.
            for &nb_eidx in graph.edges().out_neighbors(edge_idx) {
                let nb_nf = eidx_to_nf[nb_eidx as usize];
                if nb_nf != usize::MAX && component_ids[nb_nf].is_none() {
                    component_ids[nb_nf] = Some(cid);
                    queue.push_back(nb_nf);
                }
            }
            for &nb_eidx in graph.edges().in_neighbors(edge_idx) {
                let nb_nf = eidx_to_nf[nb_eidx as usize];
                if nb_nf != usize::MAX && component_ids[nb_nf].is_none() {
                    component_ids[nb_nf] = Some(cid);
                    queue.push_back(nb_nf);
                }
            }
        }
    }

    let ids: Vec<u32> = component_ids.into_iter().map(|c| c.unwrap_or(0)).collect();
    (ids, next_cid)
}

// ── Output builders ───────────────────────────────────────────────────────────

fn build_cc_output(graph: &GraphFrame, component_ids: &[u32]) -> Result<NodeFrame> {
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

    let cid_col = Arc::new(UInt32Array::from(component_ids.to_vec())) as ArrayRef;

    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        label_field,
        Field::new("component_id", DataType::UInt32, false),
    ]));

    let batch = RecordBatch::try_new(schema, vec![id_col, label_col, cid_col])
        .map_err(std::io::Error::other)?;

    NodeFrame::from_record_batch(batch)
}

fn build_empty_cc_output(graph: &GraphFrame) -> Result<NodeFrame> {
    let label_field = graph
        .nodes()
        .schema()
        .field_with_name(COL_NODE_LABEL)
        .expect("NodeFrame schema has _label field")
        .clone();

    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        label_field,
        Field::new("component_id", DataType::UInt32, false),
    ]));

    NodeFrame::from_record_batch(RecordBatch::new_empty(schema))
}
