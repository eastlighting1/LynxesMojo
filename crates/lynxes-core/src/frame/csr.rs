#[cfg(not(target_arch = "wasm32"))]
use rayon::slice::ParallelSliceMut;

/// Compressed Sparse Row (CSR) adjacency index.
///
/// Stores the outgoing edges of every node in sorted order so that all lookups
/// are a single bounds-checked slice operation.
///
/// # Layout
///
/// ```text
/// Example: nodes 0??1,2], 1??2], 2??0,2]
///
///   offsets:   [0, 2, 3, 5]          (length = node_count + 1)
///   neighbors: [1, 2, 2, 0, 2]       (length = edge_count)
///   edge_ids:  [e0, e1, e2, e3, e4]  (length = edge_count; EdgeFrame row indices)
///
///   node 0 ??neighbors[0..2] = [1, 2]
///   node 1 ??neighbors[2..3] = [2]
///   node 2 ??neighbors[3..5] = [0, 2]
/// ```
///
/// Within each node's range the entries are sorted by `(neighbor, edge_id)`,
/// making multi-edge graphs deterministic and enabling future binary search.
#[derive(Debug, Clone)]
pub struct CsrIndex {
    /// Length `node_count + 1`.  `offsets[i]..offsets[i+1]` is the slice range
    /// for node `i` in both `neighbors` and `edge_ids`.
    offsets: Vec<u32>,

    /// Destination node row indices, length `edge_count`.
    neighbors: Vec<u32>,

    /// EdgeFrame row indices parallel to `neighbors`, length `edge_count`.
    edge_ids: Vec<u32>,
}

impl CsrIndex {
    // ?? Construction (FRM-005) ???????????????????????????????????????????????

    /// Builds a CSR index from parallel source and destination row-index slices.
    ///
    /// - `src_rows[i]` and `dst_rows[i]` are the source and destination **node row
    ///   indices** (not `_id` strings) for edge `i`.
    /// - `node_count` must cover every index that appears in `src_rows`; callers
    ///   are responsible for ensuring this (EdgeFrame validates it at construction).
    /// - Within each node's range entries are sorted by `(dst_row, edge_id)`.
    ///
    /// # Panics
    /// Panics if `src_rows.len() != dst_rows.len()`.
    ///
    /// # Complexity
    /// O(E log E) time, O(N + E) space.
    pub fn build(src_rows: &[u32], dst_rows: &[u32], node_count: usize) -> Self {
        assert_eq!(
            src_rows.len(),
            dst_rows.len(),
            "src_rows and dst_rows must have equal length"
        );

        let edge_count = src_rows.len();

        // Pack edges as (src, dst, original_edge_id) and sort so that all edges
        // leaving the same source node are contiguous, then by (dst, edge_id).
        let mut edges: Vec<(u32, u32, u32)> = src_rows
            .iter()
            .zip(dst_rows.iter())
            .enumerate()
            .map(|(i, (&src, &dst))| (src, dst, i as u32))
            .collect();
        edges.sort_unstable(); // sorts lexicographically: (src, dst, edge_id)

        Self::build_from_sorted_edges(edges, node_count, edge_count)
    }

    /// Builds a reverse CSR index by swapping the source and destination roles.
    ///
    /// This is the canonical helper for `EdgeFrame` inbound adjacency:
    /// given forward edges `src -> dst`, the reverse CSR stores `dst -> src`.
    ///
    /// # Complexity
    /// O(E log E) time for the sort step, but the sorting work is parallelized
    /// with Rayon to reduce large reverse-index build latency.
    pub fn build_reverse(src_rows: &[u32], dst_rows: &[u32], node_count: usize) -> Self {
        assert_eq!(
            src_rows.len(),
            dst_rows.len(),
            "src_rows and dst_rows must have equal length"
        );

        let edge_count = src_rows.len();
        let mut reverse_edges: Vec<(u32, u32, u32)> = src_rows
            .iter()
            .zip(dst_rows.iter())
            .enumerate()
            .map(|(i, (&src, &dst))| (dst, src, i as u32))
            .collect();
        #[cfg(not(target_arch = "wasm32"))]
        reverse_edges.par_sort_unstable();
        #[cfg(target_arch = "wasm32")]
        reverse_edges.sort_unstable();

        Self::build_from_sorted_edges(reverse_edges, node_count, edge_count)
    }

    fn build_from_sorted_edges(
        edges: Vec<(u32, u32, u32)>,
        node_count: usize,
        edge_count: usize,
    ) -> Self {
        // Degree-count pass: tally how many edges leave each node.
        let mut offsets = vec![0u32; node_count + 1];
        for &(src, _, _) in &edges {
            offsets[src as usize + 1] += 1;
        }

        // Prefix-sum pass: convert degree counts to start positions.
        for i in 1..=node_count {
            offsets[i] += offsets[i - 1];
        }

        // Fill neighbors and edge_ids in the sorted order established above.
        let mut neighbors = vec![0u32; edge_count];
        let mut edge_ids = vec![0u32; edge_count];
        for (pos, &(_, dst, eid)) in edges.iter().enumerate() {
            neighbors[pos] = dst;
            edge_ids[pos] = eid;
        }

        Self {
            offsets,
            neighbors,
            edge_ids,
        }
    }

    // ?? Properties ??????????????????????????????????????????????????????????

    /// Number of nodes this index covers.
    ///
    /// Nodes with no outgoing edges are fully represented (degree 0).
    pub fn node_count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Total number of edges stored in this index.
    #[allow(dead_code)] // used by EdgeFrame stats and future algorithms
    pub fn edge_count(&self) -> usize {
        self.neighbors.len()
    }

    // ?? Lookups (FRM-006) ????????????????????????????????????????????????????

    /// Returns the destination node row indices for all edges leaving `node_idx`.
    ///
    /// Returns an empty slice if `node_idx >= node_count` (no panic).
    ///
    /// # Complexity
    /// O(1) ??two offset reads and one slice.
    pub fn neighbors(&self, node_idx: u32) -> &[u32] {
        match self.range(node_idx) {
            Some((start, end)) => &self.neighbors[start..end],
            None => &[],
        }
    }

    /// Returns the EdgeFrame row indices for all edges leaving `node_idx`.
    ///
    /// Parallel to [`neighbors`](Self::neighbors): `edge_ids(i)[k]` is the row
    /// index of the edge that reaches `neighbors(i)[k]`.
    ///
    /// Returns an empty slice if `node_idx >= node_count` (no panic).
    ///
    /// # Complexity
    /// O(1) ??two offset reads and one slice.
    pub fn edge_ids(&self, node_idx: u32) -> &[u32] {
        match self.range(node_idx) {
            Some((start, end)) => &self.edge_ids[start..end],
            None => &[],
        }
    }

    /// Returns the out-degree of `node_idx` (number of outgoing edges).
    ///
    /// Returns `0` if `node_idx >= node_count` (no panic).
    ///
    /// # Complexity
    /// O(1).
    pub fn degree(&self, node_idx: u32) -> usize {
        let idx = node_idx as usize;
        if idx >= self.node_count() {
            return 0;
        }
        (self.offsets[idx + 1] - self.offsets[idx]) as usize
    }

    pub(crate) fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    pub(crate) fn raw_edge_ids(&self) -> &[u32] {
        &self.edge_ids
    }

    // ?? Private helpers ??????????????????????????????????????????????????????

    /// Returns `Some((start, end))` ??the `[start, end)` range in `neighbors` /
    /// `edge_ids` for `node_idx`, or `None` if the index is out of bounds.
    #[inline]
    fn range(&self, node_idx: u32) -> Option<(usize, usize)> {
        let idx = node_idx as usize;
        if idx >= self.node_count() {
            return None;
        }
        Some((self.offsets[idx] as usize, self.offsets[idx + 1] as usize))
    }
}
