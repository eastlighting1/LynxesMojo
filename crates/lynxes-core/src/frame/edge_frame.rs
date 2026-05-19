use std::sync::{Arc, OnceLock};

use arrow_array::{
    new_null_array, Array, ArrayRef, BooleanArray, Int8Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use hashbrown::HashMap;

use super::csr::CsrIndex;
use super::{graph_frame::GraphFrame, node_frame::NodeFrame};
use crate::{
    Direction, GFError, Result, COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE,
    EDGE_RESERVED_COLUMNS,
};

/// Columnar edge storage with an immediately-built outgoing CSR adjacency index.
///
/// # Schema
///
/// Reserved columns are always present in this order:
/// `[_src, _dst, _type, _direction, ...user columns]`
///
/// | Column | Type | Nullable |
/// |--------|------|----------|
/// | `_src` | `Utf8` | No ??source node `_id` |
/// | `_dst` | `Utf8` | No ??destination node `_id` |
/// | `_type` | `Utf8` | No ??edge type label |
/// | `_direction` | `Int8` | No ??`Direction` encoding: 0=Out, 1=In, 2=Both, 3=None |
///
/// # Node Indexing
///
/// `EdgeFrame` is self-contained: it builds a compact local node index from the
/// unique IDs seen in `_src` and `_dst`.  This index is **not** the same as the
/// `NodeFrame` row index.  `GraphFrame::new` builds the cross-mapping when the
/// two frames are combined.
#[derive(Debug)]
pub struct EdgeFrame {
    data: RecordBatch,

    /// Outgoing adjacency index: source local-node-idx ??destination local-node-idx + edge row.
    /// Built immediately during `from_record_batch`.
    out_csr: Arc<CsrIndex>,

    /// Incoming adjacency index: destination local-node-idx ??source local-node-idx + edge row.
    /// Built lazily on the first call to `in_neighbors` (FRM-009).
    #[allow(dead_code)] // populated and read in FRM-009
    in_csr: OnceLock<CsrIndex>,

    /// Edge type ??sorted list of edge row indices.  O(1) filter-by-type lookup.
    type_index: HashMap<String, Vec<u32>>,

    /// Unique node IDs seen in `_src` / `_dst` ??compact local node index.
    ///
    /// Assigned in first-appearance order while scanning rows
    /// (checking `_src` before `_dst` within each row).
    node_index: HashMap<String, u32>,
}

impl Clone for EdgeFrame {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            out_csr: Arc::clone(&self.out_csr),
            in_csr: OnceLock::new(),
            type_index: self.type_index.clone(),
            node_index: self.node_index.clone(),
        }
    }
}

impl EdgeFrame {
    // ?? Construction (FRM-007) ???????????????????????????????????????????????

    /// Returns an empty `EdgeFrame` with the given schema and no edges.
    ///
    /// The schema must satisfy the EdgeFrame invariants (reserved columns present
    /// with correct types). No validation is performed here ??callers must ensure
    /// the schema is valid.
    pub fn empty(schema: &ArrowSchema) -> Self {
        Self {
            data: RecordBatch::new_empty(Arc::new(schema.clone())),
            out_csr: Arc::new(CsrIndex::build(&[], &[], 0)),
            in_csr: OnceLock::new(),
            type_index: HashMap::new(),
            node_index: HashMap::new(),
        }
    }

    /// Constructs an `EdgeFrame` from a `RecordBatch`.
    ///
    /// Validates the schema, then immediately builds:
    /// - `node_index` ??compact local node index from unique `_src`/`_dst` values
    /// - `out_csr` ??outgoing adjacency index using the local node index
    /// - `type_index` ??edge type ??row index mapping
    ///
    /// # Errors
    ///
    /// - `GFError::MissingReservedColumn` ??`_src`, `_dst`, `_type`, or `_direction` absent
    /// - `GFError::ReservedColumnType` ??wrong type or null values in a reserved column
    /// - `GFError::InvalidDirection` ??a `_direction` value is not in {0, 1, 2, 3}
    /// - `GFError::ReservedColumnName` ??a user column name begins with `_`
    ///
    /// # Complexity
    /// O(E log E) ??dominated by CSR sort.
    pub fn from_record_batch(batch: RecordBatch) -> Result<Self> {
        validate_edge_schema(&batch)?;
        Ok(Self::from_valid_batch(batch))
    }

    // ?? Properties ??????????????????????????????????????????????????????????

    /// Number of edges (rows) in this frame.
    pub fn len(&self) -> usize {
        self.data.num_rows()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of unique node IDs seen across `_src` and `_dst`.
    pub fn node_count(&self) -> usize {
        self.node_index.len()
    }

    pub fn schema(&self) -> &ArrowSchema {
        self.data.schema_ref().as_ref()
    }

    pub fn column_names(&self) -> Vec<&str> {
        self.data
            .schema_ref()
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect()
    }

    pub fn column(&self, name: &str) -> Option<&ArrayRef> {
        self.data
            .schema()
            .index_of(name)
            .ok()
            .map(|idx| self.data.column(idx))
    }

    /// Returns the EdgeFrame-local integer index for the given node `_id`.
    ///
    /// Returns `None` if `id` does not appear as a `_src` or `_dst` value in
    /// this frame.
    ///
    /// This index is used by `out_csr` / `in_csr` internally.  It is **not**
    /// the same as the `NodeFrame` row index ??`GraphFrame` builds that mapping.
    pub fn node_row_idx(&self, id: &str) -> Option<u32> {
        self.node_index.get(id).copied()
    }

    /// Rehydrate this edge frame into a validated [`GraphFrame`] using `nodes`.
    ///
    /// This is sugar over [`GraphFrame::new`] that keeps the edge-owned API
    /// discoverable when callers already hold an `EdgeFrame`.
    pub fn with_nodes(&self, nodes: NodeFrame) -> Result<GraphFrame> {
        GraphFrame::new(nodes, self.clone())
    }

    pub fn to_record_batch(&self) -> &RecordBatch {
        &self.data
    }

    // ?? CSR access (used by FRM-008, FRM-009, algorithms) ???????????????????

    /// Returns destination local-node-indices for all edges leaving `node_idx`.
    ///
    /// `node_idx` is the **EdgeFrame-local** index (from [`node_row_idx`]).
    /// Returns an empty slice for out-of-bounds indices.
    pub fn out_neighbors(&self, node_idx: u32) -> &[u32] {
        self.out_csr.neighbors(node_idx)
    }

    /// Returns edge row indices for all edges leaving `node_idx`.
    ///
    /// Parallel to [`out_neighbors`]: `out_edge_ids(i)[k]` is the row index of
    /// the edge reaching `out_neighbors(i)[k]`.
    pub fn out_edge_ids(&self, node_idx: u32) -> &[u32] {
        self.out_csr.edge_ids(node_idx)
    }

    /// Out-degree of `node_idx`.  Returns 0 for out-of-bounds indices.
    pub fn out_degree(&self, node_idx: u32) -> usize {
        self.out_csr.degree(node_idx)
    }

    pub(crate) fn out_csr(&self) -> &CsrIndex {
        self.out_csr.as_ref()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn out_csr_arc(&self) -> Arc<CsrIndex> {
        Arc::clone(&self.out_csr)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn with_out_csr(&self, out_csr: Arc<CsrIndex>) -> Self {
        Self {
            data: self.data.clone(),
            out_csr,
            in_csr: OnceLock::new(),
            type_index: self.type_index.clone(),
            node_index: self.node_index.clone(),
        }
    }

    /// Returns source local-node-indices for all edges entering `node_idx`.
    ///
    /// The reverse CSR is built lazily on first access and then cached for all
    /// subsequent inbound lookups.
    pub fn in_neighbors(&self, node_idx: u32) -> &[u32] {
        self.in_csr().neighbors(node_idx)
    }

    /// Returns edge row indices for all edges entering `node_idx`.
    ///
    /// Parallel to [`in_neighbors`]: `in_edge_ids(i)[k]` is the row index of
    /// the edge whose source is `in_neighbors(i)[k]` and whose destination is `i`.
    pub fn in_edge_ids(&self, node_idx: u32) -> &[u32] {
        self.in_csr().edge_ids(node_idx)
    }

    /// In-degree of `node_idx`. Returns 0 for out-of-bounds indices.
    pub fn in_degree(&self, node_idx: u32) -> usize {
        self.in_csr().degree(node_idx)
    }

    pub(crate) fn in_csr_ref(&self) -> &CsrIndex {
        self.in_csr()
    }

    // ?? Type index access ????????????????????????????????????????????????????

    /// Returns the row indices of all edges with the given `_type` value.
    ///
    /// Returns an empty slice if `edge_type` is not present in this frame.
    pub fn edge_rows_by_type(&self, edge_type: &str) -> &[u32] {
        self.type_index
            .get(edge_type)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Returns all distinct edge type labels present in this frame.
    pub fn edge_types(&self) -> Vec<&str> {
        self.type_index.keys().map(|s| s.as_str()).collect()
    }

    /// Returns the `_type` string value of the edge at `edge_row`.
    ///
    /// # Panics
    /// Panics if `edge_row >= self.len()` ??callers must stay within bounds.
    ///
    /// # Safety
    /// The `_type` column is guaranteed non-null and Utf8 by EdgeFrame invariants.
    pub fn edge_type_at(&self, edge_row: u32) -> &str {
        let idx = self
            .data
            .schema_ref()
            .index_of(COL_EDGE_TYPE)
            .expect("validated EdgeFrame always has _type");
        // SAFETY: _type is validated as non-null Utf8 during from_record_batch.
        self.data
            .column(idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("_type is Utf8")
            .value(edge_row as usize)
    }

    // ?? Filter / select (FRM-008) ????????????????????????????????????????????

    /// Retains only the rows where `mask` is `true`.
    ///
    /// Null values in `mask` are treated as `false` (the row is dropped).
    /// Rebuilds `node_index`, `out_csr`, and `type_index` from the filtered rows.
    /// Returns a new `EdgeFrame`; `self` is not modified.
    ///
    /// # Errors
    /// `GFError::LengthMismatch` if `mask.len() != self.len()`.
    pub fn filter(&self, mask: &BooleanArray) -> Result<Self> {
        if mask.len() != self.len() {
            return Err(GFError::LengthMismatch {
                expected: self.len(),
                actual: mask.len(),
            });
        }

        let filtered_columns: Vec<ArrayRef> = self
            .data
            .columns()
            .iter()
            .map(|col| arrow::compute::filter(col.as_ref(), mask))
            .collect::<std::result::Result<_, _>>()
            .map_err(std::io::Error::other)?;

        let filtered_batch = RecordBatch::try_new(self.data.schema_ref().clone(), filtered_columns)
            .map_err(std::io::Error::other)?;

        Ok(Self::from_valid_batch(filtered_batch))
    }

    /// Returns a new `EdgeFrame` containing only edges whose `_type` equals `edge_type`.
    ///
    /// If `edge_type` is not present, returns an empty frame (no error).
    /// Uses `type_index` for O(matching_edges) mask construction.
    pub fn filter_by_type(&self, edge_type: &str) -> Result<Self> {
        let row_indices = match self.type_index.get(edge_type) {
            Some(rows) => rows,
            None => return Ok(Self::empty(self.schema())),
        };

        let mut mask_values = vec![false; self.len()];
        for &idx in row_indices {
            mask_values[idx as usize] = true;
        }
        self.filter(&BooleanArray::from(mask_values))
    }

    /// Returns a new `EdgeFrame` containing only edges whose `_type` is in `types`.
    ///
    /// If none of the given types are present, returns an empty frame (no error).
    /// Duplicate type strings in `types` are silently ignored.
    pub fn filter_by_types(&self, types: &[&str]) -> Result<Self> {
        let mut mask_values = vec![false; self.len()];
        for &edge_type in types {
            if let Some(rows) = self.type_index.get(edge_type) {
                for &idx in rows {
                    mask_values[idx as usize] = true;
                }
            }
        }
        self.filter(&BooleanArray::from(mask_values))
    }

    /// Returns a new `EdgeFrame` containing only the requested columns,
    /// plus the four reserved columns `_src`, `_dst`, `_type`, `_direction`
    /// (always present and always first).
    ///
    /// Output column order: `[_src, _dst, _type, _direction, ...requested user columns
    /// in request order]`. Reserved column names in `columns` are silently deduplicated.
    ///
    /// Since `select` does not change the row set, `node_index`, `out_csr`, and
    /// `type_index` are cloned directly ??no index rebuild needed.
    ///
    /// # Errors
    /// `GFError::ColumnNotFound` if any requested column is absent.
    pub fn select(&self, columns: &[&str]) -> Result<Self> {
        let schema = self.data.schema_ref();

        for &col in columns {
            if schema.index_of(col).is_err() {
                return Err(GFError::ColumnNotFound {
                    column: col.to_owned(),
                });
            }
        }

        let mut final_names: Vec<&str> = EDGE_RESERVED_COLUMNS.to_vec();
        for &col in columns {
            if !EDGE_RESERVED_COLUMNS.contains(&col) && !final_names.contains(&col) {
                final_names.push(col);
            }
        }

        let new_fields: Vec<Field> = final_names
            .iter()
            .map(|name| schema.field_with_name(name).unwrap().clone())
            .collect();
        let new_columns: Vec<ArrayRef> = final_names
            .iter()
            .map(|name| self.data.column(schema.index_of(name).unwrap()).clone())
            .collect();

        let new_batch = RecordBatch::try_new(Arc::new(ArrowSchema::new(new_fields)), new_columns)
            .map_err(std::io::Error::other)?;

        // Row set is unchanged ??clone indexes directly.
        Ok(Self {
            data: new_batch,
            out_csr: Arc::clone(&self.out_csr),
            in_csr: OnceLock::new(), // OnceLock is not Clone; let in_csr reinitialize on demand
            type_index: self.type_index.clone(),
            node_index: self.node_index.clone(),
        })
    }

    // ?? Set operations (FRM-010) ??????????????????????????????????????????????

    /// Concatenates one or more `EdgeFrame`s into a new frame.
    ///
    /// - Column order: reserved columns first (`_src`, `_dst`, `_type`, `_direction`),
    ///   then user columns in first-appearance order across `frames`.
    /// - If the same column appears with type `T` in one frame and `Optional(T)` in
    ///   another, the result column is `Optional(T)`.
    /// - If a column is absent from some frames it is filled with nulls and the
    ///   result column is promoted to nullable.
    /// - If the same column appears with incompatible types in different frames,
    ///   returns `GFError::TypeMismatch`.
    ///
    /// Rebuilds `node_index`, `out_csr`, and `type_index` from the concatenated rows.
    ///
    /// # Errors
    /// `GFError::InvalidConfig` if `frames` is empty.
    pub fn concat(frames: &[&EdgeFrame]) -> Result<Self> {
        if frames.is_empty() {
            return Err(GFError::InvalidConfig {
                message: "concat requires at least one frame".to_owned(),
            });
        }
        if frames.len() == 1 {
            return Ok(Self {
                data: frames[0].data.clone(),
                out_csr: Arc::clone(&frames[0].out_csr),
                in_csr: OnceLock::new(),
                type_index: frames[0].type_index.clone(),
                node_index: frames[0].node_index.clone(),
            });
        }

        let merged_fields = concat_merge_fields(frames)?;
        let merged_schema = Arc::new(ArrowSchema::new(merged_fields.clone()));

        let merged_columns: Vec<ArrayRef> = merged_fields
            .iter()
            .map(|field| concat_build_column(field, frames))
            .collect::<Result<_>>()?;

        let batch =
            RecordBatch::try_new(merged_schema, merged_columns).map_err(std::io::Error::other)?;

        Ok(Self::from_valid_batch(batch))
    }

    // ?? Private helpers ??????????????????????????????????????????????????????

    /// Constructs an `EdgeFrame` from a batch that is already known to have a
    /// valid schema (skips type/name validation, rebuilds all indexes from scratch).
    ///
    /// # Panics
    /// Panics if `_src`, `_dst`, or `_type` columns are absent or not `Utf8`.
    fn from_valid_batch(data: RecordBatch) -> Self {
        let src_col = data
            .column_by_name(COL_EDGE_SRC)
            .expect("_src must exist in a validated batch")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("_src must be Utf8");
        let dst_col = data
            .column_by_name(COL_EDGE_DST)
            .expect("_dst must exist in a validated batch")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("_dst must be Utf8");

        let mut node_index: HashMap<String, u32> = HashMap::new();
        let mut src_rows: Vec<u32> = Vec::with_capacity(data.num_rows());
        let mut dst_rows: Vec<u32> = Vec::with_capacity(data.num_rows());
        let mut next_idx: u32 = 0;

        for row in 0..data.num_rows() {
            let src_id = src_col.value(row);
            let dst_id = dst_col.value(row);

            let src_idx = *node_index.entry(src_id.to_owned()).or_insert_with(|| {
                let idx = next_idx;
                next_idx += 1;
                idx
            });
            let dst_idx = *node_index.entry(dst_id.to_owned()).or_insert_with(|| {
                let idx = next_idx;
                next_idx += 1;
                idx
            });

            src_rows.push(src_idx);
            dst_rows.push(dst_idx);
        }

        let out_csr = Arc::new(CsrIndex::build(&src_rows, &dst_rows, node_index.len()));

        let type_col = data
            .column_by_name(COL_EDGE_TYPE)
            .expect("_type must exist in a validated batch")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("_type must be Utf8");
        let mut type_index: HashMap<String, Vec<u32>> = HashMap::new();
        for (row, edge_type) in type_col.iter().enumerate() {
            let edge_type = edge_type.expect("_type is non-null in a validated batch");
            type_index
                .entry(edge_type.to_owned())
                .or_default()
                .push(row as u32);
        }

        Self {
            data,
            out_csr,
            in_csr: OnceLock::new(),
            type_index,
            node_index,
        }
    }

    fn in_csr(&self) -> &CsrIndex {
        self.in_csr.get_or_init(|| {
            let src_col = self
                .data
                .column_by_name(COL_EDGE_SRC)
                .expect("_src must exist in a validated batch")
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("_src must be Utf8");
            let dst_col = self
                .data
                .column_by_name(COL_EDGE_DST)
                .expect("_dst must exist in a validated batch")
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("_dst must be Utf8");

            let mut src_rows = Vec::with_capacity(self.data.num_rows());
            let mut dst_rows = Vec::with_capacity(self.data.num_rows());

            for row in 0..self.data.num_rows() {
                let src_id = src_col.value(row);
                let dst_id = dst_col.value(row);
                let src_idx = self
                    .node_index
                    .get(src_id)
                    .copied()
                    .expect("validated node_index must contain _src ids");
                let dst_idx = self
                    .node_index
                    .get(dst_id)
                    .copied()
                    .expect("validated node_index must contain _dst ids");

                src_rows.push(src_idx);
                dst_rows.push(dst_idx);
            }

            CsrIndex::build_reverse(&src_rows, &dst_rows, self.node_index.len())
        })
    }
}

// ?? Private validation helpers ???????????????????????????????????????????????

fn validate_edge_schema(batch: &RecordBatch) -> Result<()> {
    validate_reserved_columns_present(batch)?;
    validate_reserved_column_types(batch)?;
    validate_reserved_column_values(batch)?;
    validate_user_column_names(batch)?;
    Ok(())
}

/// Computes the merged schema fields for `EdgeFrame::concat`.
///
/// Column order: `EDGE_RESERVED_COLUMNS` first, then user columns in first-appearance
/// order across `frames`. Returns `GFError::TypeMismatch` if the same column name
/// appears with incompatible `DataType`s in different frames.
fn concat_merge_fields(frames: &[&EdgeFrame]) -> Result<Vec<Field>> {
    let mut all_names: Vec<String> = EDGE_RESERVED_COLUMNS
        .iter()
        .map(|s| s.to_string())
        .collect();
    for frame in frames {
        for field in frame.data.schema_ref().fields() {
            let name = field.name().as_str();
            if !EDGE_RESERVED_COLUMNS.contains(&name) && !all_names.iter().any(|n| n == name) {
                all_names.push(name.to_owned());
            }
        }
    }

    let mut merged_fields = Vec::with_capacity(all_names.len());
    for col_name in &all_names {
        let mut base_field: Option<Field> = None;
        let mut present_in_all = true;

        for frame in frames {
            match frame.data.schema_ref().field_with_name(col_name) {
                Ok(field) => {
                    base_field = Some(match base_field {
                        None => field.clone(),
                        Some(prev) => {
                            if prev.data_type() != field.data_type() {
                                return Err(GFError::TypeMismatch {
                                    message: format!(
                                        "column '{}': type {:?} is incompatible with {:?}",
                                        col_name,
                                        field.data_type(),
                                        prev.data_type(),
                                    ),
                                });
                            }
                            let nullable = prev.is_nullable() || field.is_nullable();
                            Field::new(col_name.as_str(), prev.data_type().clone(), nullable)
                        }
                    });
                }
                Err(_) => {
                    present_in_all = false;
                }
            }
        }

        let mut field =
            base_field.expect("col_name was added because at least one frame contains it");
        if !present_in_all {
            field = Field::new(field.name(), field.data_type().clone(), true);
        }
        merged_fields.push(field);
    }

    Ok(merged_fields)
}

/// Concatenates one column across all frames, substituting an all-null array for
/// any frame that lacks the column.
fn concat_build_column(field: &Field, frames: &[&EdgeFrame]) -> Result<ArrayRef> {
    let owned: Vec<ArrayRef> = frames
        .iter()
        .map(|frame| match frame.data.column_by_name(field.name()) {
            Some(col) => col.clone(),
            None => new_null_array(field.data_type(), frame.len()),
        })
        .collect();

    let refs: Vec<&dyn Array> = owned.iter().map(|a| a.as_ref()).collect();
    arrow::compute::concat(&refs).map_err(|e| std::io::Error::other(e).into())
}

fn validate_reserved_columns_present(batch: &RecordBatch) -> Result<()> {
    for col in [
        COL_EDGE_SRC,
        COL_EDGE_DST,
        COL_EDGE_TYPE,
        COL_EDGE_DIRECTION,
    ] {
        if batch.schema().column_with_name(col).is_none() {
            return Err(GFError::MissingReservedColumn {
                column: col.to_owned(),
            });
        }
    }
    Ok(())
}

fn validate_reserved_column_types(batch: &RecordBatch) -> Result<()> {
    let schema = batch.schema_ref();

    for col in [COL_EDGE_SRC, COL_EDGE_DST, COL_EDGE_TYPE] {
        let field = schema
            .field_with_name(col)
            .map_err(|_| GFError::MissingReservedColumn {
                column: col.to_owned(),
            })?;
        if field.data_type() != &DataType::Utf8 {
            return Err(GFError::ReservedColumnType {
                column: col.to_owned(),
                expected: "Utf8".to_owned(),
                actual: format!("{:?}", field.data_type()),
            });
        }
    }

    let dir_field =
        schema
            .field_with_name(COL_EDGE_DIRECTION)
            .map_err(|_| GFError::MissingReservedColumn {
                column: COL_EDGE_DIRECTION.to_owned(),
            })?;
    if dir_field.data_type() != &DataType::Int8 {
        return Err(GFError::ReservedColumnType {
            column: COL_EDGE_DIRECTION.to_owned(),
            expected: "Int8".to_owned(),
            actual: format!("{:?}", dir_field.data_type()),
        });
    }

    Ok(())
}

fn validate_reserved_column_values(batch: &RecordBatch) -> Result<()> {
    // _src, _dst, _type must be non-null Utf8.
    for col_name in [COL_EDGE_SRC, COL_EDGE_DST, COL_EDGE_TYPE] {
        let col = batch
            .column_by_name(col_name)
            .ok_or_else(|| GFError::MissingReservedColumn {
                column: col_name.to_owned(),
            })?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| GFError::ReservedColumnType {
                column: col_name.to_owned(),
                expected: "Utf8".to_owned(),
                actual: "non-Utf8".to_owned(),
            })?;
        if col.null_count() > 0 {
            return Err(GFError::ReservedColumnType {
                column: col_name.to_owned(),
                expected: "non-null Utf8".to_owned(),
                actual: "Utf8 with nulls".to_owned(),
            });
        }
    }

    // _direction must be non-null Int8 with values in {0, 1, 2, 3}.
    let dir_col = batch
        .column_by_name(COL_EDGE_DIRECTION)
        .ok_or_else(|| GFError::MissingReservedColumn {
            column: COL_EDGE_DIRECTION.to_owned(),
        })?
        .as_any()
        .downcast_ref::<Int8Array>()
        .ok_or_else(|| GFError::ReservedColumnType {
            column: COL_EDGE_DIRECTION.to_owned(),
            expected: "Int8".to_owned(),
            actual: "non-Int8".to_owned(),
        })?;
    if dir_col.null_count() > 0 {
        return Err(GFError::ReservedColumnType {
            column: COL_EDGE_DIRECTION.to_owned(),
            expected: "non-null Int8".to_owned(),
            actual: "Int8 with nulls".to_owned(),
        });
    }
    for i in 0..dir_col.len() {
        Direction::try_from(dir_col.value(i))?;
    }

    Ok(())
}

fn validate_user_column_names(batch: &RecordBatch) -> Result<()> {
    for field in batch.schema().fields() {
        let name = field.name().as_str();
        if name.starts_with('_') && !EDGE_RESERVED_COLUMNS.contains(&name) {
            return Err(GFError::ReservedColumnName {
                column: name.to_owned(),
            });
        }
    }
    Ok(())
}
