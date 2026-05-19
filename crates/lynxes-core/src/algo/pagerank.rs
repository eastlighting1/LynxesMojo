//! PageRank algorithm (ALG-003).
//!
//! Provides [`GraphFrame::pagerank`].

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
    RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};

use crate::{
    frame::graph_frame::GraphFrame, EdgeFrame, GFError, NodeFrame, Result, COL_EDGE_DST,
    COL_EDGE_SRC, COL_NODE_ID, COL_NODE_LABEL,
};

// ── Public configuration ─────────────────────────────────────────────────────

/// Configuration for PageRank computation.
pub struct PageRankConfig {
    /// Damping factor.  Must be in the open interval `(0.0, 1.0)`.
    ///
    /// Typical value: `0.85`.
    pub damping: f64,

    /// Maximum number of power-iteration rounds.  Must be `> 0`.
    pub max_iter: usize,

    /// Convergence threshold.  Iteration stops early when the L1 norm of the
    /// rank-vector change falls below this value.  Must be `> 0`.
    pub epsilon: f64,

    /// Optional numeric edge column used as edge weight.
    ///
    /// `None` means each outgoing edge carries uniform weight `1`.
    /// With a weight column, each node distributes its mass proportionally to
    /// outgoing edge weights (normalised by the total outgoing weight).
    /// A node whose total outgoing weight is zero is treated as a dangling node.
    pub weight_col: Option<String>,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping: 0.85,
            max_iter: 100,
            epsilon: 1e-6,
            weight_col: None,
        }
    }
}

// ── GraphFrame public API ─────────────────────────────────────────────────────

impl GraphFrame {
    /// Computes PageRank over the node set.
    ///
    /// Returns a [`NodeFrame`] with exactly three columns:
    ///
    /// | Column     | Type      | Notes                       |
    /// |------------|-----------|-----------------------------|
    /// | `_id`      | `Utf8`    | node identity, same order   |
    /// | `_label`   | `List<Utf8>` | preserved from input     |
    /// | `pagerank` | `Float64` | converged score             |
    ///
    /// Row order matches the input [`NodeFrame`].
    ///
    /// # Algorithm
    ///
    /// Power iteration with dangling-node redistribution:
    ///
    /// ```text
    /// PR(v) = (1−d)/N + d × (dangling_sum/N + Σ_{u→v} PR(u) × w(u,v) / out_weight(u))
    /// ```
    ///
    /// Iteration stops when the L1 norm of the rank-vector change drops below
    /// `config.epsilon`, or after `config.max_iter` rounds.
    ///
    /// # Errors
    ///
    /// - [`GFError::InvalidConfig`] — `damping ∉ (0,1)`, `max_iter == 0`, or
    ///   `epsilon ≤ 0`.
    /// - [`GFError::ColumnNotFound`] — `weight_col` absent from `edges`.
    /// - [`GFError::TypeMismatch`] — `weight_col` is non-numeric or contains
    ///   null / NaN / infinite values.
    /// - [`GFError::NegativeWeight`] — a weight value is negative.
    ///
    /// # Complexity
    ///
    /// O(iter × (N + E)).
    pub fn pagerank(&self, config: &PageRankConfig) -> Result<NodeFrame> {
        validate_pr_config(config)?;
        validate_pr_weight_col(self.edges(), config)?;

        let n = self.nodes().len();
        if n == 0 {
            return build_empty_output(self);
        }

        // Pre-compute in-adjacency: in_edges[v_nf] = [(u_nf, raw_weight), ...]
        // and the total outgoing raw weight per NodeFrame row.
        let (in_edges, out_weight) = build_in_adjacency(self, config)?;

        // Uniform initial distribution.
        let mut ranks = vec![1.0 / n as f64; n];

        for _ in 0..config.max_iter {
            // Collect dangling-node mass (nodes with no usable outgoing weight).
            let dangling_sum: f64 = (0..n)
                .filter(|&i| out_weight[i] == 0.0)
                .map(|i| ranks[i])
                .sum();

            // Base teleportation + dangling redistribution, uniform over all nodes.
            let base = (1.0 - config.damping) / n as f64 + config.damping * dangling_sum / n as f64;

            let mut new_ranks = vec![base; n];

            // Push each source node's normalised contribution to its destinations.
            for v in 0..n {
                for &(u, w) in &in_edges[v] {
                    let ow = out_weight[u];
                    if ow > 0.0 {
                        new_ranks[v] += config.damping * ranks[u] * w / ow;
                    }
                }
            }

            // L1 convergence check.
            let l1: f64 = new_ranks
                .iter()
                .zip(ranks.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();

            ranks = new_ranks;

            if l1 < config.epsilon {
                break;
            }
        }

        build_output(self, ranks)
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Builds the in-adjacency list and out-weight sums used by the iteration loop.
///
/// Returns `(in_edges, out_weight)` where both are indexed by NodeFrame row.
///
/// `in_edges[v] = [(u, raw_weight), ...]` — raw (unnormalised) weights of
/// edges pointing at v.
/// `out_weight[u]` — sum of raw outgoing weights from u (0.0 for dangling).
#[allow(clippy::type_complexity)]
fn build_in_adjacency(
    graph: &GraphFrame,
    config: &PageRankConfig,
) -> Result<(Vec<Vec<(usize, f64)>>, Vec<f64>)> {
    let n = graph.nodes().len();
    let mut in_edges: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut out_weight = vec![0.0f64; n];

    let edges = graph.edges();
    let eb = edges.to_record_batch();

    let src_col = eb
        .column_by_name(COL_EDGE_SRC)
        .expect("validated EdgeFrame has _src")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("_src is Utf8");

    let dst_col = eb
        .column_by_name(COL_EDGE_DST)
        .expect("validated EdgeFrame has _dst")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("_dst is Utf8");

    for edge_row in 0..edges.len() {
        let src_id = src_col.value(edge_row);
        let dst_id = dst_col.value(edge_row);

        // Edges to/from nodes absent from NodeFrame are silently ignored
        // (cannot happen in a validated GraphFrame, but guarded for safety).
        let Some(src_nf) = graph.nodes().row_index(src_id).map(|r| r as usize) else {
            continue;
        };
        let Some(dst_nf) = graph.nodes().row_index(dst_id).map(|r| r as usize) else {
            continue;
        };

        let w = pr_edge_weight(edges, edge_row as u32, config)?;

        in_edges[dst_nf].push((src_nf, w));
        out_weight[src_nf] += w;
    }

    Ok((in_edges, out_weight))
}

/// Reads the weight for a single edge row.
///
/// Returns `1.0` when `config.weight_col` is `None`.
fn pr_edge_weight(edges: &EdgeFrame, edge_row: u32, config: &PageRankConfig) -> Result<f64> {
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
                "non-finite weight in column '{}' at edge row {}",
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

/// Builds the output NodeFrame with columns `_id`, `_label`, `pagerank`.
fn build_output(graph: &GraphFrame, ranks: Vec<f64>) -> Result<NodeFrame> {
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

    let pr_col = Arc::new(Float64Array::from(ranks)) as ArrayRef;

    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        label_field,
        Field::new("pagerank", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(schema, vec![id_col, label_col, pr_col])
        .map_err(std::io::Error::other)?;

    NodeFrame::from_record_batch(batch)
}

/// Returns a valid empty NodeFrame with the correct three-column schema.
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
        Field::new("pagerank", DataType::Float64, false),
    ]));

    NodeFrame::from_record_batch(RecordBatch::new_empty(schema))
}

// ── Config / weight-column validation ────────────────────────────────────────

fn validate_pr_config(config: &PageRankConfig) -> Result<()> {
    if config.damping <= 0.0 || config.damping >= 1.0 {
        return Err(GFError::InvalidConfig {
            message: format!("damping must be in (0.0, 1.0), got {}", config.damping),
        });
    }
    if config.max_iter == 0 {
        return Err(GFError::InvalidConfig {
            message: "max_iter must be > 0".to_owned(),
        });
    }
    if config.epsilon <= 0.0 {
        return Err(GFError::InvalidConfig {
            message: format!("epsilon must be > 0.0, got {}", config.epsilon),
        });
    }
    Ok(())
}

fn validate_pr_weight_col(edges: &EdgeFrame, config: &PageRankConfig) -> Result<()> {
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
