#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::sync::Arc;

use arrow_array::BooleanArray;
use hashbrown::HashSet;

use lynxes_core::{
    Direction, EdgeFrame, GraphFrame, NodeFrame, Result, COL_EDGE_DIRECTION, COL_EDGE_DST,
    COL_EDGE_SRC,
};
use lynxes_plan::{EdgeTypeSpec, Expr, LogicalPlan, PatternStep};

use super::{
    candidate_passes_pre_filter, execute, extract_node_frontier, int8_array, matches_edge_type,
    string_array, ExecutionValue,
};
pub(crate) fn execute_limit_aware(
    input: &LogicalPlan,
    source_graph: Arc<GraphFrame>,
    n: usize,
) -> Result<ExecutionValue> {
    match input {
        LogicalPlan::Expand {
            input: inner,
            edge_type,
            hops,
            direction,
            pre_filter,
        } => {
            let frontier_val = execute(inner, source_graph.clone())?;
            let frontier = extract_node_frontier(frontier_val, "LimitAware Expand")?;
            Ok(ExecutionValue::Graph(expand_graph(
                source_graph.as_ref(),
                &frontier,
                edge_type,
                *hops as usize,
                *direction,
                pre_filter.as_ref(),
                Some(n),
            )?))
        }
        LogicalPlan::Traverse {
            input: inner,
            pattern,
        } => {
            let frontier_val = execute(inner, source_graph.clone())?;
            let frontier = extract_node_frontier(frontier_val, "LimitAware Traverse")?;
            Ok(ExecutionValue::Graph(traverse_graph(
                source_graph.as_ref(),
                &frontier,
                pattern,
                Some(n),
            )?))
        }
        // Hint not directly above an expansion node ??fall through without limit.
        _ => execute(input, source_graph),
    }
}

/// Executes `input` and, when it is a Sort node, performs a partial top-K sort
/// rather than a full sort, yielding O(E log n) instead of O(E log E).
pub(crate) fn expand_graph_raw(
    graph: &GraphFrame,
    frontier_ids: &[String],
    edge_type: &EdgeTypeSpec,
    hops: usize,
    direction: Direction,
    pre_filter: Option<&Expr>,
    stop_at: Option<usize>,
) -> Result<(HashSet<String>, HashSet<usize>)> {
    let edge_node_ids = build_edge_node_ids(graph.edges())?;
    let mut visited: HashSet<String> = frontier_ids.iter().cloned().collect();
    let mut current: HashSet<String> = visited.clone();
    // HashSet deduplicates retained edge rows naturally (same edge via two paths).
    let mut retained_rows: HashSet<usize> = HashSet::new();

    // Pre-extract the _direction column for O(1) per-edge direction checks.
    let edge_batch = graph.edges().to_record_batch();
    let dir_col = int8_array(edge_batch, COL_EDGE_DIRECTION)?;

    'expand: for _ in 0..hops {
        let mut next: HashSet<String> = HashSet::new();

        for frontier_id in &current {
            // Resolve EdgeFrame-local index; nodes absent from edge set have no edges.
            let Some(local_idx) = graph.edges().node_row_idx(frontier_id) else {
                continue;
            };

            match direction {
                Direction::Out => {
                    // Follow edges where this node is the _src.
                    let n_locals = graph.edges().out_neighbors(local_idx);
                    let e_rows = graph.edges().out_edge_ids(local_idx);
                    for (&n_local, &e_row) in n_locals.iter().zip(e_rows) {
                        let edge_dir = Direction::try_from(dir_col.value(e_row as usize))?;
                        // Semantic Out traversal: only Out and Both edges.
                        if !matches!(edge_dir, Direction::Out | Direction::Both) {
                            continue;
                        }
                        if !matches_edge_type(graph.edges().edge_type_at(e_row), edge_type) {
                            continue;
                        }
                        let Some(candidate) =
                            edge_node_ids.get(n_local as usize).map(String::as_str)
                        else {
                            continue;
                        };
                        if candidate_passes_pre_filter(graph.nodes(), candidate, pre_filter)? {
                            let is_new = visited.insert(candidate.to_owned());
                            if is_new {
                                next.insert(candidate.to_owned());
                            }
                            retained_rows.insert(e_row as usize);
                            if stop_at.is_some_and(|lim| visited.len() >= lim) {
                                break 'expand;
                            }
                        }
                    }
                }
                Direction::In => {
                    // Follow edges where this node is the _dst (reverse CSR).
                    let n_locals = graph.edges().in_neighbors(local_idx);
                    let e_rows = graph.edges().in_edge_ids(local_idx);
                    for (&n_local, &e_row) in n_locals.iter().zip(e_rows) {
                        let edge_dir = Direction::try_from(dir_col.value(e_row as usize))?;
                        // Semantic In traversal: Out, Both, and In edges.
                        if !matches!(edge_dir, Direction::Out | Direction::Both | Direction::In) {
                            continue;
                        }
                        if !matches_edge_type(graph.edges().edge_type_at(e_row), edge_type) {
                            continue;
                        }
                        let Some(candidate) =
                            edge_node_ids.get(n_local as usize).map(String::as_str)
                        else {
                            continue;
                        };
                        if candidate_passes_pre_filter(graph.nodes(), candidate, pre_filter)? {
                            let is_new = visited.insert(candidate.to_owned());
                            if is_new {
                                next.insert(candidate.to_owned());
                            }
                            retained_rows.insert(e_row as usize);
                            if stop_at.is_some_and(|lim| visited.len() >= lim) {
                                break 'expand;
                            }
                        }
                    }
                }
                Direction::Both | Direction::None => {
                    // Follow all edges regardless of semantic direction.
                    for (&n_local, &e_row) in graph
                        .edges()
                        .out_neighbors(local_idx)
                        .iter()
                        .zip(graph.edges().out_edge_ids(local_idx))
                    {
                        if !matches_edge_type(graph.edges().edge_type_at(e_row), edge_type) {
                            continue;
                        }
                        let Some(candidate) =
                            edge_node_ids.get(n_local as usize).map(String::as_str)
                        else {
                            continue;
                        };
                        if candidate_passes_pre_filter(graph.nodes(), candidate, pre_filter)? {
                            let is_new = visited.insert(candidate.to_owned());
                            if is_new {
                                next.insert(candidate.to_owned());
                            }
                            retained_rows.insert(e_row as usize);
                            if stop_at.is_some_and(|lim| visited.len() >= lim) {
                                break 'expand;
                            }
                        }
                    }
                    for (&n_local, &e_row) in graph
                        .edges()
                        .in_neighbors(local_idx)
                        .iter()
                        .zip(graph.edges().in_edge_ids(local_idx))
                    {
                        if !matches_edge_type(graph.edges().edge_type_at(e_row), edge_type) {
                            continue;
                        }
                        let Some(candidate) =
                            edge_node_ids.get(n_local as usize).map(String::as_str)
                        else {
                            continue;
                        };
                        if candidate_passes_pre_filter(graph.nodes(), candidate, pre_filter)? {
                            let is_new = visited.insert(candidate.to_owned());
                            if is_new {
                                next.insert(candidate.to_owned());
                            }
                            retained_rows.insert(e_row as usize);
                            if stop_at.is_some_and(|lim| visited.len() >= lim) {
                                break 'expand;
                            }
                        }
                    }
                }
            }
        }

        if next.is_empty() {
            break;
        }
        current = next;
    }

    Ok((visited, retained_rows))
}

/// Materialises the final `GraphFrame` from the raw BFS outputs.
///
/// Shared by the serial path and the parallel merge step.
pub(crate) fn build_expand_result(
    graph: &GraphFrame,
    visited: HashSet<String>,
    retained_rows: HashSet<usize>,
) -> Result<GraphFrame> {
    let retained_ids: Vec<&str> = visited.iter().map(String::as_str).collect();
    let nodes = graph.subgraph(&retained_ids)?.nodes().clone();

    let mask: BooleanArray = (0..graph.edges().len())
        .map(|row| Some(retained_rows.contains(&row)))
        .collect();
    let edges = graph.edges().filter(&mask)?;

    GraphFrame::new(nodes, edges)
}

/// Serial CSR multi-hop expansion.  Thin wrapper around `expand_graph_raw` +
/// `build_expand_result`.
pub(crate) fn expand_graph(
    graph: &GraphFrame,
    frontier: &NodeFrame,
    edge_type: &EdgeTypeSpec,
    hops: usize,
    direction: Direction,
    pre_filter: Option<&Expr>,
    stop_at: Option<usize>,
) -> Result<GraphFrame> {
    let frontier_ids: Vec<String> = frontier
        .id_column()
        .iter()
        .flatten()
        .map(str::to_owned)
        .collect();
    let (visited, retained_rows) = expand_graph_raw(
        graph,
        &frontier_ids,
        edge_type,
        hops,
        direction,
        pre_filter,
        stop_at,
    )?;
    build_expand_result(graph, visited, retained_rows)
}

/// Parallel frontier-partitioned `Expand` execution.
///
/// The frontier is split into `rayon::current_num_threads()` contiguous chunks.
/// Each chunk runs `expand_graph_raw` independently on a Rayon thread; the
/// per-shard `(visited, retained_rows)` sets are then unioned into a single
/// result.
///
/// Correctness: a node reachable from the full frontier within `hops` hops is
/// reachable from at least one shard's subset of the frontier ??it appears in
/// at least one shard's visited set ??it appears in the union.
///
/// Trade-off: nodes reachable from multiple shards are expanded redundantly.
/// This is acceptable when the frontier is large relative to the reachable set,
/// which is the case the `PartitionParallel` optimizer targets.
pub(crate) fn execute_partition_parallel(
    input: &LogicalPlan,
    source_graph: Arc<GraphFrame>,
) -> Result<ExecutionValue> {
    match input {
        LogicalPlan::Expand {
            input: inner,
            edge_type,
            hops,
            direction,
            pre_filter,
        } => {
            let frontier_val = execute(inner, source_graph.clone())?;
            let frontier = extract_node_frontier(frontier_val, "PartitionParallel Expand")?;

            let all_ids: Vec<String> = frontier
                .id_column()
                .iter()
                .flatten()
                .map(str::to_owned)
                .collect();

            #[cfg(not(target_arch = "wasm32"))]
            let n_threads = rayon::current_num_threads();
            #[cfg(target_arch = "wasm32")]
            let n_threads = 1usize;

            // Below this threshold the Rayon overhead outweighs the gain; fall
            // back to the serial path.
            if all_ids.len() < 2 * n_threads {
                return Ok(ExecutionValue::Graph(expand_graph(
                    source_graph.as_ref(),
                    &frontier,
                    edge_type,
                    *hops as usize,
                    *direction,
                    pre_filter.as_ref(),
                    None,
                )?));
            }

            let chunk_size = all_ids.len().div_ceil(n_threads);
            let graph_ref = source_graph.as_ref();
            let hops_u = *hops as usize;

            // Parallel shard execution (Rayon on native, serial on wasm).
            #[cfg(not(target_arch = "wasm32"))]
            let partial: Vec<Result<(HashSet<String>, HashSet<usize>)>> = all_ids
                .par_chunks(chunk_size)
                .map(|chunk| {
                    expand_graph_raw(
                        graph_ref,
                        chunk,
                        edge_type,
                        hops_u,
                        *direction,
                        pre_filter.as_ref(),
                        None,
                    )
                })
                .collect();
            #[cfg(target_arch = "wasm32")]
            let partial: Vec<Result<(HashSet<String>, HashSet<usize>)>> = all_ids
                .chunks(chunk_size)
                .map(|chunk| {
                    expand_graph_raw(
                        graph_ref,
                        chunk,
                        edge_type,
                        hops_u,
                        *direction,
                        pre_filter.as_ref(),
                        None,
                    )
                })
                .collect();

            // Sequentially merge partial results (the union is the correct answer).
            let mut visited: HashSet<String> = HashSet::new();
            let mut retained_rows: HashSet<usize> = HashSet::new();
            for result in partial {
                let (v, r) = result?;
                visited.extend(v);
                retained_rows.extend(r);
            }

            Ok(ExecutionValue::Graph(build_expand_result(
                source_graph.as_ref(),
                visited,
                retained_rows,
            )?))
        }
        // PatternRoots: PatternMatch executor is not yet implemented; fall through.
        _ => execute(input, source_graph),
    }
}

pub(crate) fn traverse_graph(
    graph: &GraphFrame,
    start: &NodeFrame,
    pattern: &[PatternStep],
    stop_at: Option<usize>,
) -> Result<GraphFrame> {
    let mut frontier: HashSet<String> = start
        .id_column()
        .iter()
        .flatten()
        .map(str::to_owned)
        .collect();
    let mut visited = frontier.clone();
    let mut retained_rows: HashSet<usize> = HashSet::new();

    for step in pattern {
        let (next, rows) = expand_frontier_csr(graph, &frontier, &step.edge_type, step.direction)?;
        if next.is_empty() {
            break;
        }
        retained_rows.extend(rows);
        visited.extend(next.iter().cloned());
        frontier = next;
        if stop_at.is_some_and(|lim| visited.len() >= lim) {
            break;
        }
    }

    let retained_ids: Vec<&str> = visited.iter().map(String::as_str).collect();
    let nodes = graph.subgraph(&retained_ids)?.nodes().clone();
    let mask: BooleanArray = (0..graph.edges().len())
        .map(|row| Some(retained_rows.contains(&row)))
        .collect();
    let edges = graph.edges().filter(&mask)?;

    GraphFrame::new(nodes, edges)
}

/// CSR-based single-hop frontier expansion used by `traverse_graph`.
///
/// Returns the set of newly discovered neighbour IDs and the set of edge row
/// indices that were traversed.  No visited-dedup is performed here; callers
/// accumulate `visited` themselves across steps.
pub(crate) fn expand_frontier_csr(
    graph: &GraphFrame,
    frontier: &HashSet<String>,
    edge_type: &EdgeTypeSpec,
    direction: Direction,
) -> Result<(HashSet<String>, HashSet<usize>)> {
    let edge_node_ids = build_edge_node_ids(graph.edges())?;
    let mut next: HashSet<String> = HashSet::new();
    let mut rows: HashSet<usize> = HashSet::new();

    for frontier_id in frontier {
        let Some(local_idx) = graph.edges().node_row_idx(frontier_id) else {
            continue;
        };

        match direction {
            Direction::Out => {
                for (&n_local, &e_row) in graph
                    .edges()
                    .out_neighbors(local_idx)
                    .iter()
                    .zip(graph.edges().out_edge_ids(local_idx))
                {
                    if !matches_edge_type(graph.edges().edge_type_at(e_row), edge_type) {
                        continue;
                    }
                    if let Some(candidate) = edge_node_ids.get(n_local as usize) {
                        next.insert(candidate.to_owned());
                        rows.insert(e_row as usize);
                    }
                }
            }
            Direction::In => {
                for (&n_local, &e_row) in graph
                    .edges()
                    .in_neighbors(local_idx)
                    .iter()
                    .zip(graph.edges().in_edge_ids(local_idx))
                {
                    if !matches_edge_type(graph.edges().edge_type_at(e_row), edge_type) {
                        continue;
                    }
                    if let Some(candidate) = edge_node_ids.get(n_local as usize) {
                        next.insert(candidate.to_owned());
                        rows.insert(e_row as usize);
                    }
                }
            }
            Direction::Both | Direction::None => {
                for (&n_local, &e_row) in graph
                    .edges()
                    .out_neighbors(local_idx)
                    .iter()
                    .zip(graph.edges().out_edge_ids(local_idx))
                {
                    if !matches_edge_type(graph.edges().edge_type_at(e_row), edge_type) {
                        continue;
                    }
                    if let Some(candidate) = edge_node_ids.get(n_local as usize) {
                        next.insert(candidate.to_owned());
                        rows.insert(e_row as usize);
                    }
                }
                for (&n_local, &e_row) in graph
                    .edges()
                    .in_neighbors(local_idx)
                    .iter()
                    .zip(graph.edges().in_edge_ids(local_idx))
                {
                    if !matches_edge_type(graph.edges().edge_type_at(e_row), edge_type) {
                        continue;
                    }
                    if let Some(candidate) = edge_node_ids.get(n_local as usize) {
                        next.insert(candidate.to_owned());
                        rows.insert(e_row as usize);
                    }
                }
            }
        }
    }

    Ok((next, rows))
}

pub(crate) fn build_edge_node_ids(edges: &EdgeFrame) -> Result<Vec<String>> {
    let batch = edges.to_record_batch();
    let src_col = string_array(batch, COL_EDGE_SRC)?;
    let dst_col = string_array(batch, COL_EDGE_DST)?;
    let mut node_ids = vec![String::new(); edges.node_count()];

    for row in 0..edges.len() {
        for id in [src_col.value(row), dst_col.value(row)] {
            if let Some(idx) = edges.node_row_idx(id) {
                if node_ids[idx as usize].is_empty() {
                    node_ids[idx as usize] = id.to_owned();
                }
            }
        }
    }

    Ok(node_ids)
}
