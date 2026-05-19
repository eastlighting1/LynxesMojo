//! BFS multi-source traversal kernel (ALG-001).
//!
//! This module provides the canonical BFS traversal kernel used by
//! `Expand`, `k_hop_subgraph`, and `has_path` wrappers.

use arrow_array::{
    Array, BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
    ListArray, StringArray,
};
use hashbrown::HashSet;

use crate::{
    frame::graph_frame::GraphFrame, BinaryOp, Direction, EdgeFrame, EdgeTypeSpec, Expr, GFError,
    NodeFrame, Result, ScalarValue,
};

// ── Public API ──────────────────────────────────────────────────────────────

/// Configuration for a BFS traversal.
///
/// # Defaults
///
/// Constructed via [`BfsConfig::new`]:
/// - `direction`: [`Direction::Out`]
/// - `edge_type`: [`EdgeTypeSpec::Any`]
/// - `pre_filter`: `None`
pub struct BfsConfig<'a> {
    /// Maximum number of hops to expand from the seed set.
    ///
    /// `0` returns only the root nodes without any edge expansion.
    pub hops: u32,

    /// Which direction to follow edges.
    pub direction: Direction,

    /// Edge type restriction for traversal expansion.
    pub edge_type: EdgeTypeSpec,

    /// Optional node predicate.  A candidate node that evaluates to `false`
    /// is not added to the frontier and is excluded from the result.
    ///
    /// Root nodes are **not** filtered by this predicate.
    pub pre_filter: Option<&'a Expr>,
}

impl<'a> BfsConfig<'a> {
    /// Creates a `BfsConfig` with `hops` depth and all-default traversal options.
    pub fn new(hops: u32) -> Self {
        Self {
            hops,
            direction: Direction::Out,
            edge_type: EdgeTypeSpec::Any,
            pre_filter: None,
        }
    }
}

/// Multi-source BFS traversal kernel.
///
/// Starting from all nodes in `roots`, expands up to `config.hops` hops
/// following edges that match `config.edge_type` in `config.direction`.
/// Before a candidate node enters the next frontier, it is tested against
/// `config.pre_filter` (if provided); failing nodes are excluded.
///
/// # Returns
///
/// A pair `(NodeFrame, EdgeFrame)`:
/// - `NodeFrame` contains all visited nodes (roots plus reachable nodes that
///   pass `pre_filter`).
/// - `EdgeFrame` contains all traversed edges (edges whose destination was
///   admitted into the frontier).
///
/// # Errors
///
/// - [`GFError::NodeNotFound`] if any root ID is absent from `graph.nodes()`.
///
/// # Edge cases
///
/// - Empty `roots` → empty valid `(NodeFrame, EdgeFrame)`.
/// - `hops == 0` → result contains only root nodes; edge frame is empty.
/// - Root IDs that exist in `NodeFrame` but have no edges in `EdgeFrame`
///   (isolated nodes) are included in the node result without expansion.
///
/// # Complexity
///
/// O(N + E) over the traversed connected region.
pub fn bfs(
    graph: &GraphFrame,
    roots: &[&str],
    config: &BfsConfig,
) -> Result<(NodeFrame, EdgeFrame)> {
    // ── Seed phase ──────────────────────────────────────────────────────────

    // EdgeFrame-local compact node indices of visited nodes.
    let mut visited: HashSet<u32> = HashSet::new();
    // NodeFrame row indices included in the result.
    let mut result_node_rows: HashSet<u32> = HashSet::new();
    // Edge row indices included in the result.
    let mut result_edge_rows: HashSet<u32> = HashSet::new();

    let mut frontier: Vec<u32> = Vec::new();

    for &root in roots {
        // Validate existence in NodeFrame.
        let node_row = graph
            .nodes()
            .row_index(root)
            .ok_or_else(|| GFError::NodeNotFound {
                id: root.to_owned(),
            })?;

        result_node_rows.insert(node_row);

        // Seed EdgeFrame-local index (may be absent for isolated nodes).
        if let Some(edge_idx) = graph.edges().node_row_idx(root) {
            if visited.insert(edge_idx) {
                frontier.push(edge_idx);
            }
        }
    }

    // ── Expansion phase ─────────────────────────────────────────────────────

    for _ in 0..config.hops {
        if frontier.is_empty() {
            break;
        }
        let current = std::mem::take(&mut frontier);

        for node_idx in current {
            expand_node(
                graph,
                node_idx,
                &config.direction,
                &config.edge_type,
                config.pre_filter,
                &mut visited,
                &mut result_node_rows,
                &mut result_edge_rows,
                &mut frontier,
            );
        }
    }

    // ── Result materialisation ───────────────────────────────────────────────

    // Build result NodeFrame by filtering on row index membership.
    let node_mask: BooleanArray = (0..graph.nodes().len() as u32)
        .map(|r| Some(result_node_rows.contains(&r)))
        .collect();
    let result_nodes = graph.nodes().filter(&node_mask)?;

    // Build result EdgeFrame by filtering on edge row membership.
    let mut edge_mask_values = vec![false; graph.edges().len()];
    for &row in &result_edge_rows {
        edge_mask_values[row as usize] = true;
    }
    let result_edges = graph
        .edges()
        .filter(&BooleanArray::from(edge_mask_values))?;

    Ok((result_nodes, result_edges))
}

// ── Private helpers ──────────────────────────────────────────────────────────

/// Expands `node_idx` in the EdgeFrame and admits neighbours into `frontier`.
#[allow(clippy::too_many_arguments)]
fn expand_node(
    graph: &GraphFrame,
    node_idx: u32,
    direction: &Direction,
    edge_type: &EdgeTypeSpec,
    pre_filter: Option<&Expr>,
    visited: &mut HashSet<u32>,
    result_node_rows: &mut HashSet<u32>,
    result_edge_rows: &mut HashSet<u32>,
    frontier: &mut Vec<u32>,
) {
    let edges = graph.edges();

    let pairs: Vec<(u32, u32)> = match direction {
        Direction::Out => collect_pairs(
            edges.out_neighbors(node_idx),
            edges.out_edge_ids(node_idx),
            edge_type,
            edges,
        ),
        Direction::In => collect_pairs(
            edges.in_neighbors(node_idx),
            edges.in_edge_ids(node_idx),
            edge_type,
            edges,
        ),
        Direction::Both | Direction::None => {
            let mut p = collect_pairs(
                edges.out_neighbors(node_idx),
                edges.out_edge_ids(node_idx),
                edge_type,
                edges,
            );
            p.extend(collect_pairs(
                edges.in_neighbors(node_idx),
                edges.in_edge_ids(node_idx),
                edge_type,
                edges,
            ));
            p
        }
    };

    for (neighbor_idx, edge_row) in pairs {
        // If destination already admitted, record the edge and move on.
        if visited.contains(&neighbor_idx) {
            result_edge_rows.insert(edge_row);
            continue;
        }

        // Map EdgeFrame compact index → string ID → NodeFrame row.
        let node_id = match graph.edge_node_id_by_idx(neighbor_idx) {
            Some(id) => id,
            None => continue,
        };
        let node_row = match graph.node_row_by_id(node_id) {
            Some(r) => r,
            None => continue,
        };

        // Apply pre_filter (evaluated against NodeFrame row).
        if let Some(filter_expr) = pre_filter {
            if !eval_bool(filter_expr, graph.nodes(), node_row as usize) {
                continue; // node excluded — edge not recorded
            }
        }

        // Destination admitted: record node, edge, and enqueue.
        visited.insert(neighbor_idx);
        result_node_rows.insert(node_row);
        result_edge_rows.insert(edge_row);
        frontier.push(neighbor_idx);
    }
}

/// Zips `neighbors` and `edge_ids` slices, retaining only pairs whose edge
/// matches `edge_type`.
fn collect_pairs(
    neighbors: &[u32],
    edge_ids: &[u32],
    edge_type: &EdgeTypeSpec,
    edges: &EdgeFrame,
) -> Vec<(u32, u32)> {
    neighbors
        .iter()
        .zip(edge_ids.iter())
        .filter(|(_, &eid)| matches_edge_type(edges, eid, edge_type))
        .map(|(&nb, &eid)| (nb, eid))
        .collect()
}

/// Returns `true` if the edge at `edge_row` satisfies `spec`.
fn matches_edge_type(edges: &EdgeFrame, edge_row: u32, spec: &EdgeTypeSpec) -> bool {
    match spec {
        EdgeTypeSpec::Any => true,
        EdgeTypeSpec::Single(t) => edges.edge_type_at(edge_row) == t.as_str(),
        EdgeTypeSpec::Multiple(ts) => {
            let et = edges.edge_type_at(edge_row);
            ts.iter().any(|t| t.as_str() == et)
        }
    }
}

// ── Minimal Expr evaluator ───────────────────────────────────────────────────
//
// Supports common predicates used in BFS pre_filter:
//   - Literal bool
//   - Col op Literal (Eq, NotEq, Gt, GtEq, Lt, LtEq) on string / numeric columns
//   - ListContains for _label columns
//   - And, Or, Not combinators
//
// Unsupported variants conservatively return `false`.

/// A dynamically-typed value produced by expression evaluation.
#[derive(Debug, PartialEq, PartialOrd)]
enum DynValue<'a> {
    Null,
    Str(&'a str),
    Int(i64),
    Float(f64),
    Bool(bool),
}

/// Evaluates `expr` to a boolean for the node at `row` in `nodes`.
///
/// Unsupported expression shapes return `false` (conservative: the node is
/// excluded from the frontier).
fn eval_bool(expr: &Expr, nodes: &NodeFrame, row: usize) -> bool {
    match expr {
        Expr::Literal {
            value: ScalarValue::Bool(b),
        } => *b,
        Expr::And { left, right } => eval_bool(left, nodes, row) && eval_bool(right, nodes, row),
        Expr::Or { left, right } => eval_bool(left, nodes, row) || eval_bool(right, nodes, row),
        Expr::Not { expr } => !eval_bool(expr, nodes, row),
        Expr::BinaryOp { left, op, right } => eval_cmp(left, op, right, nodes, row),
        Expr::ListContains { expr, item } => eval_list_contains(expr, item, nodes, row),
        _ => false,
    }
}

/// Evaluates a comparison between two sub-expressions at `row`.
fn eval_cmp(left: &Expr, op: &BinaryOp, right: &Expr, nodes: &NodeFrame, row: usize) -> bool {
    let lv = eval_value(left, nodes, row);
    let rv = eval_value(right, nodes, row);

    match (lv, rv) {
        (DynValue::Null, _) | (_, DynValue::Null) => false,
        (DynValue::Str(l), DynValue::Str(r)) => cmp_ord(l, op, r),
        (DynValue::Int(l), DynValue::Int(r)) => cmp_ord(l, op, r),
        (DynValue::Float(l), DynValue::Float(r)) => cmp_ord(l, op, r),
        (DynValue::Int(l), DynValue::Float(r)) => cmp_ord(l as f64, op, r),
        (DynValue::Float(l), DynValue::Int(r)) => cmp_ord(l, op, r as f64),
        (DynValue::Bool(l), DynValue::Bool(r)) => match op {
            BinaryOp::Eq => l == r,
            BinaryOp::NotEq => l != r,
            _ => false,
        },
        _ => false,
    }
}

/// Applies a comparison operator to two `PartialOrd` values.
fn cmp_ord<T: PartialOrd + PartialEq>(l: T, op: &BinaryOp, r: T) -> bool {
    match op {
        BinaryOp::Eq => l == r,
        BinaryOp::NotEq => l != r,
        BinaryOp::Gt => l > r,
        BinaryOp::GtEq => l >= r,
        BinaryOp::Lt => l < r,
        BinaryOp::LtEq => l <= r,
        _ => false, // arithmetic ops not meaningful in boolean context
    }
}

/// Evaluates `list_expr` as a list column and checks whether `item_expr`
/// (a string literal) is present among its elements at `row`.
fn eval_list_contains(list_expr: &Expr, item_expr: &Expr, nodes: &NodeFrame, row: usize) -> bool {
    let Expr::Col { name } = list_expr else {
        return false;
    };
    let item_val = eval_value(item_expr, nodes, row);
    let DynValue::Str(target) = item_val else {
        return false;
    };

    let Some(col) = nodes.column(name) else {
        return false;
    };
    let Some(list_arr) = col.as_any().downcast_ref::<ListArray>() else {
        return false;
    };

    let values = list_arr.value(row);
    let Some(str_arr) = values.as_any().downcast_ref::<StringArray>() else {
        return false;
    };

    str_arr.iter().any(|v| v == Some(target))
}

/// Reads a scalar value from `expr` at `row` in `nodes`.
///
/// Handles `Expr::Literal` and `Expr::Col` (Utf8, integer, float, boolean
/// Arrow arrays).  Everything else returns [`DynValue::Null`].
fn eval_value<'a>(expr: &'a Expr, nodes: &'a NodeFrame, row: usize) -> DynValue<'a> {
    match expr {
        Expr::Literal { value } => match value {
            ScalarValue::Null => DynValue::Null,
            ScalarValue::String(s) => DynValue::Str(s.as_str()),
            ScalarValue::Int(i) => DynValue::Int(*i),
            ScalarValue::Float(f) => DynValue::Float(*f),
            ScalarValue::Bool(b) => DynValue::Bool(*b),
            ScalarValue::List(_) => DynValue::Null,
        },
        Expr::Col { name } => {
            let Some(col) = nodes.column(name) else {
                return DynValue::Null;
            };
            let col = col.as_ref();

            if col.is_null(row) {
                return DynValue::Null;
            }

            // Utf8
            if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                return DynValue::Str(arr.value(row));
            }
            // Integer variants (widening to i64)
            if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
                return DynValue::Int(arr.value(row));
            }
            if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
                return DynValue::Int(arr.value(row) as i64);
            }
            if let Some(arr) = col.as_any().downcast_ref::<Int16Array>() {
                return DynValue::Int(arr.value(row) as i64);
            }
            if let Some(arr) = col.as_any().downcast_ref::<Int8Array>() {
                return DynValue::Int(arr.value(row) as i64);
            }
            // Float variants (widening to f64)
            if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
                return DynValue::Float(arr.value(row));
            }
            if let Some(arr) = col.as_any().downcast_ref::<Float32Array>() {
                return DynValue::Float(arr.value(row) as f64);
            }
            // Boolean
            if let Some(arr) = col.as_any().downcast_ref::<arrow_array::BooleanArray>() {
                return DynValue::Bool(arr.value(row));
            }

            DynValue::Null
        }
        _ => DynValue::Null,
    }
}
