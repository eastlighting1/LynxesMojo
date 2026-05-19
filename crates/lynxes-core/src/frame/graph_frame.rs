use std::{collections::VecDeque, sync::Arc};

use arrow_array::{Array, ArrayRef, BooleanArray, Int64Array, ListArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use hashbrown::{HashMap, HashSet};

use super::mutable_graph_frame::MutableGraphFrame;
use crate::mojo_runtime::{self, StructuralDegreeInputs};
use crate::{EdgeFrame, GFError, NodeFrame, Result, COL_EDGE_DST, COL_EDGE_SRC, COL_NODE_LABEL};

#[cfg(not(target_arch = "wasm32"))]
type MutableParts = (
    NodeFrame,
    EdgeFrame,
    Option<crate::Schema>,
    HashMap<String, u32>,
    Vec<String>,
    HashMap<String, u32>,
    Vec<String>,
);

/// In-memory graph composed of a `NodeFrame` and an `EdgeFrame`.
///
/// `GraphFrame` validates that every edge endpoint exists in `nodes` (unless
/// constructed with [`new_unchecked`](Self::new_unchecked)) and caches the
/// node-id to row-index mapping for future graph algorithms.
#[derive(Debug)]
pub struct GraphFrame {
    nodes: NodeFrame,
    edges: EdgeFrame,
    schema: Option<crate::Schema>,
    node_id_to_idx: HashMap<String, u32>,
    node_idx_to_id: Vec<String>,
    edge_node_id_to_idx: HashMap<String, u32>,
    edge_node_idx_to_id: Vec<String>,
}

impl Clone for GraphFrame {
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
            schema: self.schema.clone(),
            node_id_to_idx: self.node_id_to_idx.clone(),
            node_idx_to_id: self.node_idx_to_id.clone(),
            edge_node_id_to_idx: self.edge_node_id_to_idx.clone(),
            edge_node_idx_to_id: self.edge_node_idx_to_id.clone(),
        }
    }
}

impl GraphFrame {
    /// Constructs a `GraphFrame` after checking that every edge endpoint exists
    /// in the node set.
    ///
    /// # Errors
    /// Returns `GFError::DanglingEdge` if any `_src` or `_dst` value is absent
    /// from `nodes`.
    pub fn new(nodes: NodeFrame, edges: EdgeFrame) -> Result<Self> {
        let graph = Self::new_unchecked(nodes, edges);
        graph.validate_edge_endpoints()?;
        Ok(graph)
    }

    pub fn new_with_schema(
        nodes: NodeFrame,
        edges: EdgeFrame,
        schema: Option<crate::Schema>,
        validate_schema: bool,
    ) -> Result<Self> {
        let mut graph = Self::new_unchecked(nodes, edges);
        graph.schema = schema;
        graph.validate_edge_endpoints()?;
        if validate_schema {
            if let Some(schema) = graph.schema.as_ref() {
                let errors = schema.validate_graph(&graph);
                if !errors.is_empty() {
                    return Err(GFError::schema_validation(errors));
                }
            }
        }
        Ok(graph)
    }

    /// Constructs a `GraphFrame` without validating edge endpoints.
    ///
    /// Intended for internal use when the caller already knows the frames are
    /// consistent, or when the validation is performed separately.
    pub fn new_unchecked(nodes: NodeFrame, edges: EdgeFrame) -> Self {
        let id_column = nodes.id_column();
        let mut node_id_to_idx = HashMap::with_capacity(id_column.len());
        let mut node_idx_to_id = Vec::with_capacity(id_column.len());

        for (row_idx, id) in id_column.iter().enumerate() {
            let id = id.expect("validated _id column is non-null");
            node_id_to_idx.insert(id.to_owned(), row_idx as u32);
            node_idx_to_id.push(id.to_owned());
        }

        let (edge_node_id_to_idx, edge_node_idx_to_id) = build_edge_node_mapping(&edges);

        Self {
            nodes,
            edges,
            schema: None,
            node_id_to_idx,
            node_idx_to_id,
            edge_node_id_to_idx,
            edge_node_idx_to_id,
        }
    }

    pub fn nodes(&self) -> &NodeFrame {
        &self.nodes
    }

    pub fn edges(&self) -> &EdgeFrame {
        &self.edges
    }

    pub fn schema(&self) -> Option<&crate::Schema> {
        self.schema.as_ref()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Materializes the graph topology as a COO `(src, dst)` pair.
    ///
    /// The returned coordinates use the `EdgeFrame` local compact node index
    /// space, not the `NodeFrame` row index. This matches the CSR adjacency
    /// owned by `EdgeFrame` and is the index space that later GNN-oriented
    /// utilities consume.
    ///
    /// Nodes that exist in `NodeFrame` but never appear as an edge endpoint do
    /// not appear in the returned coordinate arrays.
    pub fn to_coo(&self) -> (Int64Array, Int64Array) {
        let edge_count = self.edges.len();
        let mut src = Vec::with_capacity(edge_count);
        let mut dst = Vec::with_capacity(edge_count);

        for src_idx in 0..self.edges.node_count() as u32 {
            for &dst_idx in self.edges.out_neighbors(src_idx) {
                src.push(src_idx as i64);
                dst.push(dst_idx as i64);
            }
        }

        (Int64Array::from(src), Int64Array::from(dst))
    }

    /// Converts the eager immutable graph into a mutable wrapper without
    /// cloning the node frame or edge frame payloads.
    pub fn into_mutable(self) -> MutableGraphFrame {
        MutableGraphFrame::from_graph_frame(self)
    }

    pub fn density(&self) -> f64 {
        let n = self.node_count();
        if n < 2 {
            return 0.0;
        }
        self.edge_count() as f64 / (n * (n - 1)) as f64
    }

    /// Computes Mojo-backed structural degree features for each node.
    ///
    /// This is intentionally a Mojo-only feature: the Rust side prepares stable
    /// compact graph buffers and materializes the Arrow result, while the degree
    /// scan itself is performed by the bundled Mojo shared library.
    pub fn structural_features(&self, edge_type: Option<&str>) -> Result<NodeFrame> {
        for column in ["out_degree", "in_degree", "total_degree"] {
            if self.nodes.column(column).is_some() {
                return Err(GFError::SchemaMismatch {
                    message: format!(
                        "structural_features would overwrite existing node column '{column}'"
                    ),
                });
            }
        }

        let node_to_edge_idx: Vec<i64> = self
            .nodes
            .id_column()
            .iter()
            .map(|id| {
                let id = id.expect("validated _id column is non-null");
                self.edge_node_id_to_idx
                    .get(id)
                    .map(|idx| *idx as i64)
                    .unwrap_or(-1)
            })
            .collect();
        let edge_allowed = self.edge_type_mask(edge_type);
        let out_csr = self.edges.out_csr();
        let in_csr = self.edges.in_csr_ref();

        let (out_degree, in_degree, total_degree) =
            mojo_runtime::compute_structural_degrees(StructuralDegreeInputs {
                node_to_edge_idx: &node_to_edge_idx,
                out_offsets: out_csr.offsets(),
                out_edge_ids: out_csr.raw_edge_ids(),
                in_offsets: in_csr.offsets(),
                in_edge_ids: in_csr.raw_edge_ids(),
                edge_allowed: &edge_allowed,
            })?;

        self.with_structural_feature_columns(out_degree, in_degree, total_degree)
    }

    /// Returns the outgoing neighbor IDs of `id`.
    ///
    /// # Errors
    /// Returns `GFError::NodeNotFound` if `id` is absent from the node set.
    pub fn out_neighbors(&self, id: &str) -> Result<Vec<&str>> {
        let edge_idx = match self.edge_node_idx(id)? {
            Some(idx) => idx,
            None => return Ok(Vec::new()),
        };
        Ok(self
            .edges
            .out_neighbors(edge_idx)
            .iter()
            .filter_map(|&idx| self.edge_node_id(idx))
            .collect())
    }

    /// Returns the incoming neighbor IDs of `id`.
    ///
    /// # Errors
    /// Returns `GFError::NodeNotFound` if `id` is absent from the node set.
    pub fn in_neighbors(&self, id: &str) -> Result<Vec<&str>> {
        let edge_idx = match self.edge_node_idx(id)? {
            Some(idx) => idx,
            None => return Ok(Vec::new()),
        };
        Ok(self
            .edges
            .in_neighbors(edge_idx)
            .iter()
            .filter_map(|&idx| self.edge_node_id(idx))
            .collect())
    }

    /// Returns the neighbor IDs of `id` according to `direction`.
    ///
    /// `Direction::Both` and `Direction::None` both return the deduplicated union
    /// of inbound and outbound neighbors.
    pub fn neighbors(&self, id: &str, direction: crate::Direction) -> Result<Vec<&str>> {
        match direction {
            crate::Direction::Out => self.out_neighbors(id),
            crate::Direction::In => self.in_neighbors(id),
            crate::Direction::Both | crate::Direction::None => {
                let edge_idx = match self.edge_node_idx(id)? {
                    Some(idx) => idx,
                    None => return Ok(Vec::new()),
                };
                Ok(self.collect_union_neighbors(edge_idx))
            }
        }
    }

    /// Returns the out-degree of `id`.
    ///
    /// # Errors
    /// Returns `GFError::NodeNotFound` if `id` is absent from the node set.
    pub fn out_degree(&self, id: &str) -> Result<usize> {
        Ok(match self.edge_node_idx(id)? {
            Some(idx) => self.edges.out_degree(idx),
            None => 0,
        })
    }

    /// Returns the in-degree of `id`.
    ///
    /// # Errors
    /// Returns `GFError::NodeNotFound` if `id` is absent from the node set.
    pub fn in_degree(&self, id: &str) -> Result<usize> {
        Ok(match self.edge_node_idx(id)? {
            Some(idx) => self.edges.in_degree(idx),
            None => 0,
        })
    }

    /// Returns the subgraph induced by `node_ids`.
    ///
    /// Unknown IDs are silently ignored. An edge is retained only when both of
    /// its endpoints are present in the resulting node set.
    pub fn subgraph(&self, node_ids: &[&str]) -> Result<Self> {
        let included_ids: HashSet<&str> = node_ids.iter().copied().collect();

        let node_mask: BooleanArray = self
            .nodes
            .id_column()
            .iter()
            .map(|id| Some(included_ids.contains(id.expect("validated _id column is non-null"))))
            .collect();
        let sub_nodes = self.nodes.filter(&node_mask)?;

        let retained_ids = collect_node_ids(&sub_nodes);
        let edge_mask = edge_endpoint_mask(&self.edges, &retained_ids)?;
        let sub_edges = self.edges.filter(&edge_mask)?;

        Self::new(sub_nodes, sub_edges)
    }

    /// Returns the subgraph induced by nodes whose `_label` list contains
    /// `label`.
    pub fn subgraph_by_label(&self, label: &str) -> Result<Self> {
        let label_column = self
            .nodes
            .column(COL_NODE_LABEL)
            .expect("_label must exist in a validated NodeFrame")
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("_label must be List<Utf8>");

        let node_mask: BooleanArray = (0..self.nodes.len())
            .map(|row| Some(node_has_label(label_column, row, label)))
            .collect();
        let sub_nodes = self.nodes.filter(&node_mask)?;

        let retained_ids = collect_node_ids(&sub_nodes);
        let edge_mask = edge_endpoint_mask(&self.edges, &retained_ids)?;
        let sub_edges = self.edges.filter(&edge_mask)?;

        Self::new(sub_nodes, sub_edges)
    }

    /// Returns the subgraph induced by edges whose `_type` equals `edge_type`.
    ///
    /// The resulting node set contains only the endpoints referenced by the
    /// retained edges.
    pub fn subgraph_by_edge_type(&self, edge_type: &str) -> Result<Self> {
        let sub_edges = self.edges.filter_by_type(edge_type)?;
        let retained_ids = collect_edge_endpoint_ids(&sub_edges)?;

        let node_mask: BooleanArray = self
            .nodes
            .id_column()
            .iter()
            .map(|id| Some(retained_ids.contains(id.expect("validated _id column is non-null"))))
            .collect();
        let sub_nodes = self.nodes.filter(&node_mask)?;

        Self::new(sub_nodes, sub_edges)
    }

    /// Returns the `k`-hop subgraph around `root` using BFS over both inbound
    /// and outbound adjacency.
    ///
    /// `k = 0` returns a single-node graph containing only `root`.
    ///
    /// # Errors
    /// Returns `GFError::NodeNotFound` if `root` is absent from the node set.
    pub fn k_hop_subgraph(&self, root: &str, k: usize) -> Result<Self> {
        if !self.node_id_to_idx.contains_key(root) {
            return Err(GFError::NodeNotFound {
                id: root.to_owned(),
            });
        }

        let Some(root_edge_idx) = self.edge_node_id_to_idx.get(root).copied() else {
            return self.subgraph(&[root]);
        };
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        visited.insert(root_edge_idx);
        queue.push_back((root_edge_idx, 0usize));

        while let Some((node_idx, depth)) = queue.pop_front() {
            if depth == k {
                continue;
            }

            for &next in self.edges.out_neighbors(node_idx) {
                if visited.insert(next) {
                    queue.push_back((next, depth + 1));
                }
            }
            for &next in self.edges.in_neighbors(node_idx) {
                if visited.insert(next) {
                    queue.push_back((next, depth + 1));
                }
            }
        }

        let retained: Vec<&str> = visited
            .iter()
            .filter_map(|&idx| self.edge_node_id(idx))
            .collect();
        self.subgraph(&retained)
    }

    fn validate_edge_endpoints(&self) -> Result<()> {
        let src_col = edge_string_column(self.edges.to_record_batch(), COL_EDGE_SRC)?;
        let dst_col = edge_string_column(self.edges.to_record_batch(), COL_EDGE_DST)?;

        for row in 0..self.edges.len() {
            let src = src_col.value(row);
            let dst = dst_col.value(row);

            if !self.node_id_to_idx.contains_key(src) || !self.node_id_to_idx.contains_key(dst) {
                return Err(GFError::DanglingEdge {
                    src: src.to_owned(),
                    dst: dst.to_owned(),
                });
            }
        }

        Ok(())
    }

    #[allow(dead_code)]
    fn node_id(&self, idx: u32) -> Option<&str> {
        self.node_idx_to_id.get(idx as usize).map(|s| s.as_str())
    }

    /// Maps an EdgeFrame-local compact node index back to its string node ID.
    ///
    /// Used by the algorithm layer to reconstruct node IDs from BFS traversal results.
    pub(crate) fn edge_node_id_by_idx(&self, idx: u32) -> Option<&str> {
        self.edge_node_idx_to_id
            .get(idx as usize)
            .map(|s| s.as_str())
    }

    /// Maps a string node ID to its NodeFrame row index.
    ///
    /// Returns `None` if `id` is absent from the node set.
    pub(crate) fn node_row_by_id(&self, id: &str) -> Option<u32> {
        self.node_id_to_idx.get(id).copied()
    }

    fn edge_node_id(&self, idx: u32) -> Option<&str> {
        self.edge_node_idx_to_id
            .get(idx as usize)
            .map(|s| s.as_str())
    }

    fn edge_node_idx(&self, id: &str) -> Result<Option<u32>> {
        if !self.node_id_to_idx.contains_key(id) {
            return Err(GFError::NodeNotFound { id: id.to_owned() });
        }

        Ok(self.edge_node_id_to_idx.get(id).copied())
    }

    fn collect_union_neighbors(&self, edge_idx: u32) -> Vec<&str> {
        let mut seen = HashSet::new();
        let mut neighbors = Vec::new();

        for &idx in self.edges.out_neighbors(edge_idx) {
            if seen.insert(idx) {
                if let Some(id) = self.edge_node_id(idx) {
                    neighbors.push(id);
                }
            }
        }
        for &idx in self.edges.in_neighbors(edge_idx) {
            if seen.insert(idx) {
                if let Some(id) = self.edge_node_id(idx) {
                    neighbors.push(id);
                }
            }
        }

        neighbors
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn into_mutable_parts(self) -> MutableParts {
        (
            self.nodes,
            self.edges,
            self.schema,
            self.node_id_to_idx,
            self.node_idx_to_id,
            self.edge_node_id_to_idx,
            self.edge_node_idx_to_id,
        )
    }

    fn edge_type_mask(&self, edge_type: Option<&str>) -> Vec<u8> {
        let mut mask = vec![0u8; self.edges.len()];
        match edge_type {
            Some(edge_type) => {
                for &row in self.edges.edge_rows_by_type(edge_type) {
                    mask[row as usize] = 1;
                }
            }
            None => mask.fill(1),
        }
        mask
    }

    fn with_structural_feature_columns(
        &self,
        out_degree: Vec<i64>,
        in_degree: Vec<i64>,
        total_degree: Vec<i64>,
    ) -> Result<NodeFrame> {
        let len = self.nodes.len();
        for values in [&out_degree, &in_degree, &total_degree] {
            if values.len() != len {
                return Err(GFError::LengthMismatch {
                    expected: len,
                    actual: values.len(),
                });
            }
        }

        let mut fields: Vec<Field> = self
            .nodes
            .schema()
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .collect();
        fields.push(Field::new("out_degree", DataType::Int64, false));
        fields.push(Field::new("in_degree", DataType::Int64, false));
        fields.push(Field::new("total_degree", DataType::Int64, false));

        let mut columns: Vec<ArrayRef> = self.nodes.to_record_batch().columns().to_vec();
        columns.push(Arc::new(Int64Array::from(out_degree)) as ArrayRef);
        columns.push(Arc::new(Int64Array::from(in_degree)) as ArrayRef);
        columns.push(Arc::new(Int64Array::from(total_degree)) as ArrayRef);

        let batch = RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), columns)
            .map_err(|error| GFError::IoError(std::io::Error::other(error)))?;
        NodeFrame::from_record_batch(batch)
    }
}

fn build_edge_node_mapping(edges: &EdgeFrame) -> (HashMap<String, u32>, Vec<String>) {
    let src_col = edge_string_column(edges.to_record_batch(), COL_EDGE_SRC)
        .expect("validated EdgeFrame must contain Utf8 _src");
    let dst_col = edge_string_column(edges.to_record_batch(), COL_EDGE_DST)
        .expect("validated EdgeFrame must contain Utf8 _dst");

    let mut edge_node_id_to_idx = HashMap::with_capacity(edges.node_count());
    let mut edge_node_idx_to_id = vec![String::new(); edges.node_count()];

    for row in 0..edges.len() {
        for id in [src_col.value(row), dst_col.value(row)] {
            if let Some(idx) = edges.node_row_idx(id) {
                if !edge_node_id_to_idx.contains_key(id) {
                    edge_node_id_to_idx.insert(id.to_owned(), idx);
                    edge_node_idx_to_id[idx as usize] = id.to_owned();
                }
            }
        }
    }

    (edge_node_id_to_idx, edge_node_idx_to_id)
}

fn collect_node_ids(nodes: &NodeFrame) -> HashSet<String> {
    nodes
        .id_column()
        .iter()
        .map(|id| id.expect("validated _id column is non-null").to_owned())
        .collect()
}

fn collect_edge_endpoint_ids(edges: &EdgeFrame) -> Result<HashSet<String>> {
    let src_col = edge_string_column(edges.to_record_batch(), COL_EDGE_SRC)?;
    let dst_col = edge_string_column(edges.to_record_batch(), COL_EDGE_DST)?;

    let mut ids = HashSet::with_capacity(edges.len().saturating_mul(2));
    for row in 0..edges.len() {
        ids.insert(src_col.value(row).to_owned());
        ids.insert(dst_col.value(row).to_owned());
    }

    Ok(ids)
}

fn edge_endpoint_mask(edges: &EdgeFrame, retained_ids: &HashSet<String>) -> Result<BooleanArray> {
    let src_col = edge_string_column(edges.to_record_batch(), COL_EDGE_SRC)?;
    let dst_col = edge_string_column(edges.to_record_batch(), COL_EDGE_DST)?;

    Ok((0..edges.len())
        .map(|row| {
            Some(
                retained_ids.contains(src_col.value(row))
                    && retained_ids.contains(dst_col.value(row)),
            )
        })
        .collect())
}

fn edge_string_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    batch
        .column_by_name(name)
        .ok_or_else(|| GFError::MissingReservedColumn {
            column: name.to_owned(),
        })?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| GFError::ReservedColumnType {
            column: name.to_owned(),
            expected: "Utf8".to_owned(),
            actual: "non-Utf8 array".to_owned(),
        })
}

fn node_has_label(label_column: &ListArray, row: usize, label: &str) -> bool {
    let labels = label_column.value(row);
    let labels = labels
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("_label list values must be Utf8");

    labels.iter().any(|value| value == Some(label))
}
