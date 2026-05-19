use std::sync::Arc;

use arrow_array::{
    new_null_array, Array, ArrayRef, BooleanArray, ListArray, RecordBatch, StringArray, UInt32Array,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use hashbrown::HashMap;

use super::{edge_frame::EdgeFrame, graph_frame::GraphFrame};
use crate::{GFError, Result, COL_NODE_ID, COL_NODE_LABEL, NODE_RESERVED_COLUMNS};

/// Columnar node storage with O(1) user-id to row-index lookup.
#[derive(Debug)]
pub struct NodeFrame {
    data: RecordBatch,
    id_index: HashMap<String, u32>,
}

impl Clone for NodeFrame {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            id_index: self.id_index.clone(),
        }
    }
}

impl NodeFrame {
    pub fn empty(schema: &ArrowSchema) -> Self {
        Self {
            data: RecordBatch::new_empty(Arc::new(schema.clone())),
            id_index: HashMap::new(),
        }
    }

    pub fn from_record_batch(batch: RecordBatch) -> Result<Self> {
        validate_node_schema(&batch)?;
        let id_column = id_string_array(&batch)?;
        let mut id_index = HashMap::with_capacity(id_column.len());

        for (row_idx, id) in id_column.iter().enumerate() {
            let id = id.expect("validated non-null _id column");
            if id_index.insert(id.to_owned(), row_idx as u32).is_some() {
                return Err(GFError::DuplicateNodeId { id: id.to_owned() });
            }
        }

        Ok(Self {
            data: batch,
            id_index,
        })
    }

    pub fn len(&self) -> usize {
        self.data.num_rows()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn schema(&self) -> &ArrowSchema {
        self.data.schema_ref().as_ref()
    }

    pub fn column_names(&self) -> Vec<&str> {
        self.data
            .schema_ref()
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect()
    }

    pub fn column(&self, name: &str) -> Option<&ArrayRef> {
        self.data
            .schema()
            .index_of(name)
            .ok()
            .map(|idx| self.data.column(idx))
    }

    pub fn id_column(&self) -> &StringArray {
        self.data
            .column_by_name(COL_NODE_ID)
            .expect("validated _id column exists")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("validated _id column has Utf8 type")
    }

    /// Returns the internal row index for the given `_id` value.
    ///
    /// Prefer [`row`] for data access. This method is exposed for advanced callers
    /// that need direct row-level addressing (e.g. custom algorithms).
    ///
    /// **Stability:** unstable ??row indices are an implementation detail and may be
    /// re-addressed if the storage layout changes.
    pub fn row_index(&self, id: &str) -> Option<u32> {
        self.id_index.get(id).copied()
    }

    /// Returns a single-row `RecordBatch` slice for the node with the given `_id`.
    ///
    /// Returns `None` if `id` is not present in this frame.
    /// The returned batch shares the underlying Arrow buffers with `self` (zero-copy).
    ///
    /// # Example
    /// ```ignore
    /// if let Some(row) = frame.row("alice") {
    ///     // row has the same schema as frame, with exactly one row
    /// }
    /// ```
    pub fn row(&self, id: &str) -> Option<RecordBatch> {
        let row_idx = *self.id_index.get(id)? as usize;
        Some(self.data.slice(row_idx, 1))
    }

    /// Gathers the rows at `row_ids` into a new `RecordBatch`.
    ///
    /// The output preserves the schema and row order implied by `row_ids`.
    /// Repeated indices are allowed and duplicate the corresponding rows.
    ///
    /// An empty `row_ids` slice returns an empty batch with the same schema.
    pub fn gather_rows(&self, row_ids: &[u32]) -> Result<RecordBatch> {
        if row_ids.is_empty() {
            return Ok(RecordBatch::new_empty(self.data.schema_ref().clone()));
        }

        if let Some(&row_id) = row_ids
            .iter()
            .find(|&&row_id| row_id as usize >= self.len())
        {
            return Err(GFError::InvalidConfig {
                message: format!(
                    "gather_rows row id {} is out of bounds for frame of length {}",
                    row_id,
                    self.len()
                ),
            });
        }

        let indices = UInt32Array::from(row_ids.to_vec());
        let gathered_columns: Vec<ArrayRef> = self
            .data
            .columns()
            .iter()
            .map(|col| arrow::compute::take(col.as_ref(), &indices, None))
            .collect::<std::result::Result<_, _>>()
            .map_err(|err| GFError::InvalidConfig {
                message: format!("gather_rows failed for row ids {:?}: {}", row_ids, err),
            })?;

        RecordBatch::try_new(self.data.schema_ref().clone(), gathered_columns)
            .map_err(std::io::Error::other)
            .map_err(Into::into)
    }

    /// Retains only the rows where `mask` is `true`.
    ///
    /// - Null values in `mask` are treated as `false` (the row is dropped).
    /// - Returns a new `NodeFrame`; `self` is not modified.
    /// - `id_index` is rebuilt from the filtered result.
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

        // Filter every column with the same mask, then reassemble.
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

    /// Returns a new `NodeFrame` containing only the requested columns,
    /// plus the reserved columns `_id` and `_label` (always present).
    ///
    /// Output column order: `[_id, _label, ...requested user columns in request order]`.
    /// If a requested column name matches a reserved column it is silently deduplicated.
    ///
    /// # Errors
    /// `GFError::ColumnNotFound` if any requested column is absent.
    pub fn select(&self, columns: &[&str]) -> Result<Self> {
        let schema = self.data.schema_ref();

        // Validate every requested column exists before doing any work.
        for &col in columns {
            if schema.index_of(col).is_err() {
                return Err(GFError::ColumnNotFound {
                    column: col.to_owned(),
                });
            }
        }

        // Build final ordered list: reserved first, then user columns (no duplicates).
        let mut final_names: Vec<&str> = NODE_RESERVED_COLUMNS.to_vec();
        for &col in columns {
            if !NODE_RESERVED_COLUMNS.contains(&col) && !final_names.contains(&col) {
                final_names.push(col);
            }
        }

        // Project schema and columns in the same order.
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

        // Row set is unchanged ??clone id_index directly (O(N) but unavoidable for HashMap).
        Ok(Self {
            data: new_batch,
            id_index: self.id_index.clone(),
        })
    }

    /// Returns a new `NodeFrame` covering `self[offset .. offset + length]`.
    ///
    /// The underlying Arrow buffers are shared (zero-copy). `id_index` is rebuilt
    /// from the sliced rows, which is O(length).
    ///
    /// # Panics
    /// Panics if `offset + length > self.len()` ??same contract as `RecordBatch::slice`.
    pub fn slice(&self, offset: usize, length: usize) -> Self {
        Self::from_valid_batch(self.data.slice(offset, length))
    }

    /// Rehydrate this node frame into a validated [`GraphFrame`] using `edges`.
    ///
    /// This is sugar over [`GraphFrame::new`] that keeps the node-owned API
    /// discoverable when callers already hold a `NodeFrame`.
    pub fn with_edges(&self, edges: EdgeFrame) -> Result<GraphFrame> {
        GraphFrame::new(self.clone(), edges)
    }

    pub fn to_record_batch(&self) -> &RecordBatch {
        &self.data
    }

    // ?? Set operations (FRM-004) ?????????????????????????????????????????????

    /// Concatenates one or more `NodeFrame`s into a new frame.
    ///
    /// - Column order: reserved columns first (`_id`, `_label`), then user columns
    ///   in first-appearance order across `frames`.
    /// - If the same column appears with type `T` in one frame and `Optional(T)` in
    ///   another, the result column is `Optional(T)`.
    /// - If a column is absent from some frames it is filled with nulls and the
    ///   result column is promoted to nullable.
    /// - If the same column appears with incompatible types in different frames,
    ///   returns `GFError::TypeMismatch`.
    /// - If any `_id` value appears more than once across the combined rows,
    ///   returns `GFError::DuplicateNodeId`.
    ///
    /// # Errors
    /// `GFError::InvalidConfig` if `frames` is empty.
    pub fn concat(frames: &[&NodeFrame]) -> Result<Self> {
        if frames.is_empty() {
            return Err(GFError::InvalidConfig {
                message: "concat requires at least one frame".to_owned(),
            });
        }
        if frames.len() == 1 {
            // Fast path: nothing to merge ??clone the single frame directly.
            return Ok(Self {
                data: frames[0].data.clone(),
                id_index: frames[0].id_index.clone(),
            });
        }

        // Compute the merged schema, then build each column.
        let merged_fields = concat_merge_fields(frames)?;
        let merged_schema = Arc::new(ArrowSchema::new(merged_fields.clone()));

        let merged_columns: Vec<ArrayRef> = merged_fields
            .iter()
            .map(|field| concat_build_column(field, frames))
            .collect::<Result<_>>()?;

        let batch =
            RecordBatch::try_new(merged_schema, merged_columns).map_err(std::io::Error::other)?;

        // Build id_index while checking for cross-frame duplicate _id values.
        let id_col = batch
            .column_by_name(COL_NODE_ID)
            .expect("_id always present after merge")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("_id is Utf8");

        let mut id_index = HashMap::with_capacity(id_col.len());
        for (row_idx, id) in id_col.iter().enumerate() {
            let id = id.expect("_id is non-null in all input frames");
            if id_index.insert(id.to_owned(), row_idx as u32).is_some() {
                return Err(GFError::DuplicateNodeId { id: id.to_owned() });
            }
        }

        Ok(Self {
            data: batch,
            id_index,
        })
    }

    /// Returns a new `NodeFrame` containing only the rows whose `_id` values are
    /// present in **both** `self` and `other`.
    ///
    /// Row order and column values are taken from `self`.
    /// The two frames must have identical schemas (same columns, same types, same
    /// order). Returns `GFError::SchemaMismatch` if they differ.
    pub fn intersect(&self, other: &NodeFrame) -> Result<Self> {
        if self.schema() != other.schema() {
            return Err(GFError::SchemaMismatch {
                message: format!(
                    "intersect requires identical schemas; \
                     left columns: {:?}, right columns: {:?}",
                    self.column_names(),
                    other.column_names(),
                ),
            });
        }
        let mask: BooleanArray = self
            .id_column()
            .iter()
            .map(|id| Some(other.id_index.contains_key(id.expect("_id non-null"))))
            .collect();
        self.filter(&mask)
    }

    /// Returns a new `NodeFrame` containing only the rows whose `_id` values are
    /// present in `self` but **not** in `other`.
    ///
    /// Row order and column values are taken from `self`.
    /// The two frames must have identical schemas (same columns, same types, same
    /// order). Returns `GFError::SchemaMismatch` if they differ.
    pub fn difference(&self, other: &NodeFrame) -> Result<Self> {
        if self.schema() != other.schema() {
            return Err(GFError::SchemaMismatch {
                message: format!(
                    "difference requires identical schemas; \
                     left columns: {:?}, right columns: {:?}",
                    self.column_names(),
                    other.column_names(),
                ),
            });
        }
        let mask: BooleanArray = self
            .id_column()
            .iter()
            .map(|id| Some(!other.id_index.contains_key(id.expect("_id non-null"))))
            .collect();
        self.filter(&mask)
    }

    // ?? Private helpers ??????????????????????????????????????????????????????

    /// Constructs a `NodeFrame` from a batch that is already known to have a
    /// valid schema (skips type/name validation, rebuilds id_index only).
    ///
    /// # Panics
    /// Panics if `_id` column is absent or contains nulls ??callers must ensure
    /// the batch originates from a previously validated `NodeFrame`.
    fn from_valid_batch(data: RecordBatch) -> Self {
        let id_col = data
            .column_by_name(COL_NODE_ID)
            .expect("_id column must exist in a validated batch")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("_id column must be Utf8");

        let mut id_index = HashMap::with_capacity(id_col.len());
        for (row_idx, id) in id_col.iter().enumerate() {
            let id = id.expect("_id values are non-null in a validated batch");
            id_index.insert(id.to_owned(), row_idx as u32);
        }

        Self { data, id_index }
    }
}

fn validate_node_schema(batch: &RecordBatch) -> Result<()> {
    validate_reserved_columns_present(batch)?;
    validate_reserved_column_types(batch)?;
    validate_reserved_column_values(batch)?;
    validate_user_column_names(batch)?;
    Ok(())
}

fn validate_reserved_columns_present(batch: &RecordBatch) -> Result<()> {
    for column in [COL_NODE_ID, COL_NODE_LABEL] {
        if batch.schema().column_with_name(column).is_none() {
            return Err(GFError::MissingReservedColumn {
                column: column.to_owned(),
            });
        }
    }

    Ok(())
}

fn validate_reserved_column_types(batch: &RecordBatch) -> Result<()> {
    let schema = batch.schema_ref();

    let id_field =
        schema
            .field_with_name(COL_NODE_ID)
            .map_err(|_| GFError::MissingReservedColumn {
                column: COL_NODE_ID.to_owned(),
            })?;
    if id_field.data_type() != &DataType::Utf8 {
        return Err(GFError::ReservedColumnType {
            column: COL_NODE_ID.to_owned(),
            expected: "Utf8".to_owned(),
            actual: format!("{:?}", id_field.data_type()),
        });
    }

    let label_field =
        schema
            .field_with_name(COL_NODE_LABEL)
            .map_err(|_| GFError::MissingReservedColumn {
                column: COL_NODE_LABEL.to_owned(),
            })?;
    match label_field.data_type() {
        DataType::List(field) if field.data_type() == &DataType::Utf8 => {}
        actual => {
            return Err(GFError::ReservedColumnType {
                column: COL_NODE_LABEL.to_owned(),
                expected: "List<Utf8>".to_owned(),
                actual: format!("{:?}", actual),
            });
        }
    }

    Ok(())
}

fn validate_reserved_column_values(batch: &RecordBatch) -> Result<()> {
    let id_column = id_string_array(batch)?;
    if id_column.null_count() > 0 {
        return Err(GFError::ReservedColumnType {
            column: COL_NODE_ID.to_owned(),
            expected: "non-null Utf8".to_owned(),
            actual: "Utf8 with nulls".to_owned(),
        });
    }

    let label_column = label_list_array(batch)?;
    if label_column.null_count() > 0 {
        return Err(GFError::ReservedColumnType {
            column: COL_NODE_LABEL.to_owned(),
            expected: "non-null List<Utf8>".to_owned(),
            actual: "List<Utf8> with nulls".to_owned(),
        });
    }

    let label_values = label_column
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| GFError::ReservedColumnType {
            column: COL_NODE_LABEL.to_owned(),
            expected: "List<Utf8>".to_owned(),
            actual: format!("{:?}", label_column.values().data_type()),
        })?;
    if label_values.null_count() > 0 {
        return Err(GFError::ReservedColumnType {
            column: COL_NODE_LABEL.to_owned(),
            expected: "List<Utf8> with non-null child values".to_owned(),
            actual: "List<Utf8> with null child values".to_owned(),
        });
    }

    Ok(())
}

fn validate_user_column_names(batch: &RecordBatch) -> Result<()> {
    for field in batch.schema().fields() {
        let name = field.name();
        if name.starts_with('_') && name != COL_NODE_ID && name != COL_NODE_LABEL {
            return Err(GFError::ReservedColumnName {
                column: name.to_owned(),
            });
        }
    }

    Ok(())
}

/// Computes the merged schema fields for `NodeFrame::concat`.
///
/// Column order: `NODE_RESERVED_COLUMNS` first, then user columns in first-appearance
/// order across `frames`.  Returns `GFError::TypeMismatch` if the same column name
/// appears with incompatible `DataType`s in different frames.
fn concat_merge_fields(frames: &[&NodeFrame]) -> Result<Vec<Field>> {
    // Collect column names in stable order.
    let mut all_names: Vec<String> = NODE_RESERVED_COLUMNS
        .iter()
        .map(|s| s.to_string())
        .collect();
    for frame in frames {
        for field in frame.data.schema_ref().fields() {
            let name = field.name().as_str();
            if !NODE_RESERVED_COLUMNS.contains(&name) && !all_names.iter().any(|n| n == name) {
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
                            // Propagate nullability: if either side is nullable, result is too.
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

        // Column absent in at least one frame ??must be nullable in the result.
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
fn concat_build_column(field: &Field, frames: &[&NodeFrame]) -> Result<ArrayRef> {
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

fn id_string_array(batch: &RecordBatch) -> Result<&StringArray> {
    batch
        .column_by_name(COL_NODE_ID)
        .ok_or_else(|| GFError::MissingReservedColumn {
            column: COL_NODE_ID.to_owned(),
        })?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| GFError::ReservedColumnType {
            column: COL_NODE_ID.to_owned(),
            expected: "Utf8".to_owned(),
            actual: "non-Utf8 array".to_owned(),
        })
}

fn label_list_array(batch: &RecordBatch) -> Result<&ListArray> {
    batch
        .column_by_name(COL_NODE_LABEL)
        .ok_or_else(|| GFError::MissingReservedColumn {
            column: COL_NODE_LABEL.to_owned(),
        })?
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| GFError::ReservedColumnType {
            column: COL_NODE_LABEL.to_owned(),
            expected: "List<Utf8>".to_owned(),
            actual: "non-List array".to_owned(),
        })
}
