use std::{fs::File, path::Path, sync::Arc};

use arrow::{
    array::{
        new_empty_array, Array, ArrayRef, ListBuilder, RecordBatch, StringArray, StringBuilder,
    },
    csv::reader::{Format, ReaderBuilder},
    datatypes::{DataType, Field, Schema as ArrowSchema},
};
use lynxes_core::{GFError, NodeFrame, Result, COL_NODE_ID, COL_NODE_LABEL};

const DEFAULT_BATCH_SIZE: usize = 65_536;

#[derive(Debug, Clone)]
pub struct CsvNodeReadOptions {
    pub label: Option<String>,
    pub id_col: Option<String>,
    pub id_prefix: Option<String>,
    pub columns: Option<Vec<String>>,
    pub schema_overrides: Vec<(String, DataType)>,
    pub infer_schema_rows: Option<usize>,
    pub batch_size: usize,
    pub has_header: bool,
    pub delimiter: u8,
}

impl Default for CsvNodeReadOptions {
    fn default() -> Self {
        Self {
            label: None,
            id_col: None,
            id_prefix: None,
            columns: None,
            schema_overrides: Vec::new(),
            infer_schema_rows: None,
            batch_size: DEFAULT_BATCH_SIZE,
            has_header: true,
            delimiter: b',',
        }
    }
}

pub fn read_csv_nodes(path: impl AsRef<Path>, options: &CsvNodeReadOptions) -> Result<NodeFrame> {
    let path = path.as_ref();
    let format = Format::default()
        .with_header(options.has_header)
        .with_delimiter(options.delimiter);

    let mut infer_file = File::open(path)?;
    let (source_schema, _) = format
        .infer_schema(&mut infer_file, options.infer_schema_rows)
        .map_err(|err| GFError::InvalidConfig {
            message: format!("failed to infer CSV schema for {}: {err}", path.display()),
        })?;
    let source_schema = apply_schema_overrides(source_schema, &options.schema_overrides)?;
    let projection = projection_indices(&source_schema, options)?;
    let read_schema = projected_schema(&source_schema, projection.as_deref());

    let read_file = File::open(path)?;
    let mut reader_builder = ReaderBuilder::new(Arc::new(source_schema.clone()))
        .with_format(format)
        .with_batch_size(options.batch_size.max(1));
    if let Some(projection) = projection {
        reader_builder = reader_builder.with_projection(projection);
    }
    let reader = reader_builder
        .build(read_file)
        .map_err(|err| GFError::InvalidConfig {
            message: format!("failed to open CSV reader for {}: {err}", path.display()),
        })?;

    let batches = reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|err| GFError::InvalidConfig {
            message: format!("failed to read CSV rows from {}: {err}", path.display()),
        })?;
    let source_columns = concat_source_columns(&read_schema, &batches)?;
    let rows = source_columns
        .first()
        .map(|array| array.len())
        .unwrap_or_else(|| batches.iter().map(RecordBatch::num_rows).sum());

    let source_names: Vec<&str> = read_schema
        .fields()
        .iter()
        .map(|field| field.name().as_str())
        .collect();

    let id_array = build_id_array(&read_schema, &source_columns, rows, options)?;
    let label_array = build_label_array(&read_schema, &source_columns, rows, options)?;

    let mut fields = vec![
        Field::new(COL_NODE_ID, DataType::Utf8, false),
        Field::new(
            COL_NODE_LABEL,
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            false,
        ),
    ];
    let mut columns = vec![id_array, label_array];

    append_user_columns(
        &mut fields,
        &mut columns,
        &read_schema,
        &source_columns,
        &source_names,
        options.columns.as_deref(),
    )?;

    let batch = RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), columns)
        .map_err(std::io::Error::other)?;
    NodeFrame::from_record_batch(batch)
}

fn apply_schema_overrides(
    schema: ArrowSchema,
    overrides: &[(String, DataType)],
) -> Result<ArrowSchema> {
    if overrides.is_empty() {
        return Ok(schema);
    }

    let mut fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect();
    for (name, dtype) in overrides {
        let idx = schema.index_of(name).map_err(|_| GFError::ColumnNotFound {
            column: name.clone(),
        })?;
        fields[idx] = Field::new(name, dtype.clone(), true);
    }

    Ok(ArrowSchema::new(fields))
}

fn projection_indices(
    schema: &ArrowSchema,
    options: &CsvNodeReadOptions,
) -> Result<Option<Vec<usize>>> {
    let Some(requested_columns) = options.columns.as_deref() else {
        return Ok(None);
    };

    let mut names = Vec::new();
    if let Some(id_col) = options.id_col.as_deref() {
        names.push(id_col);
    } else if schema.index_of(COL_NODE_ID).is_ok() {
        names.push(COL_NODE_ID);
    }
    if options.label.is_none() {
        names.push(COL_NODE_LABEL);
    }
    for name in requested_columns {
        names.push(name);
    }

    let mut projection = Vec::new();
    for name in names {
        let idx = schema.index_of(name).map_err(|_| GFError::ColumnNotFound {
            column: name.to_owned(),
        })?;
        if !projection.contains(&idx) {
            projection.push(idx);
        }
    }

    Ok(Some(projection))
}

fn projected_schema(schema: &ArrowSchema, projection: Option<&[usize]>) -> ArrowSchema {
    let Some(projection) = projection else {
        return schema.clone();
    };

    let fields: Vec<Field> = projection
        .iter()
        .map(|idx| schema.field(*idx).clone())
        .collect();
    ArrowSchema::new(fields)
}

fn append_user_columns(
    fields: &mut Vec<Field>,
    columns: &mut Vec<ArrayRef>,
    schema: &ArrowSchema,
    source_columns: &[ArrayRef],
    source_names: &[&str],
    requested_columns: Option<&[String]>,
) -> Result<()> {
    if let Some(requested_columns) = requested_columns {
        let mut emitted = Vec::new();
        for name in requested_columns {
            let name = name.as_str();
            if name == COL_NODE_ID || name == COL_NODE_LABEL || emitted.contains(&name) {
                continue;
            }
            if name.starts_with('_') {
                return Err(GFError::ReservedColumnName {
                    column: name.to_owned(),
                });
            }
            let idx = schema.index_of(name).map_err(|_| GFError::ColumnNotFound {
                column: name.to_owned(),
            })?;
            fields.push(schema.field(idx).clone());
            columns.push(source_columns[idx].clone());
            emitted.push(name);
        }
        return Ok(());
    }

    for (idx, name) in source_names.iter().enumerate() {
        if *name == COL_NODE_ID || *name == COL_NODE_LABEL {
            continue;
        }
        if name.starts_with('_') {
            return Err(GFError::ReservedColumnName {
                column: (*name).to_owned(),
            });
        }
        fields.push(schema.field(idx).clone());
        columns.push(source_columns[idx].clone());
    }

    Ok(())
}

fn concat_source_columns(schema: &ArrowSchema, batches: &[RecordBatch]) -> Result<Vec<ArrayRef>> {
    let mut columns = Vec::with_capacity(schema.fields().len());

    for col_idx in 0..schema.fields().len() {
        if batches.is_empty() {
            columns.push(new_empty_array(schema.field(col_idx).data_type()));
            continue;
        }

        let arrays: Vec<ArrayRef> = batches
            .iter()
            .map(|batch| batch.column(col_idx).clone())
            .collect();
        let refs: Vec<&dyn Array> = arrays.iter().map(|array| array.as_ref()).collect();
        let array = arrow::compute::concat(&refs).map_err(std::io::Error::other)?;
        columns.push(array);
    }

    Ok(columns)
}

fn build_id_array(
    schema: &ArrowSchema,
    source_columns: &[ArrayRef],
    rows: usize,
    options: &CsvNodeReadOptions,
) -> Result<ArrayRef> {
    if let Some(id_col) = options.id_col.as_deref() {
        let idx = schema
            .index_of(id_col)
            .map_err(|_| GFError::ColumnNotFound {
                column: id_col.to_owned(),
            })?;
        return cast_non_null_utf8(&source_columns[idx], id_col);
    }

    if let Ok(idx) = schema.index_of(COL_NODE_ID) {
        return cast_non_null_utf8(&source_columns[idx], COL_NODE_ID);
    }

    let prefix = options.id_prefix.as_deref().unwrap_or("row");
    let values = (0..rows).map(|idx| format!("{prefix}_{idx}"));
    Ok(Arc::new(StringArray::from_iter_values(values)))
}

fn build_label_array(
    schema: &ArrowSchema,
    source_columns: &[ArrayRef],
    rows: usize,
    options: &CsvNodeReadOptions,
) -> Result<ArrayRef> {
    if let Some(label) = options.label.as_deref() {
        return Ok(Arc::new(repeated_label_array(label, rows)));
    }

    let idx = schema
        .index_of(COL_NODE_LABEL)
        .map_err(|_| GFError::MissingReservedColumn {
            column: COL_NODE_LABEL.to_owned(),
        })?;
    let labels = cast_non_null_utf8(&source_columns[idx], COL_NODE_LABEL)?;
    let labels = labels
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("cast_non_null_utf8 returns StringArray");
    Ok(Arc::new(singleton_label_array(labels)))
}

fn cast_non_null_utf8(array: &ArrayRef, name: &str) -> Result<ArrayRef> {
    if array.null_count() > 0 {
        return Err(GFError::ReservedColumnType {
            column: name.to_owned(),
            expected: "non-null Utf8".to_owned(),
            actual: format!("{:?} with nulls", array.data_type()),
        });
    }

    arrow::compute::cast(array.as_ref(), &DataType::Utf8).map_err(|_| GFError::InvalidCast {
        from: format!("{:?}", array.data_type()),
        to: format!("Utf8 ({name})"),
    })
}

fn repeated_label_array(label: &str, rows: usize) -> arrow::array::ListArray {
    let mut builder = ListBuilder::new(StringBuilder::new());
    for _ in 0..rows {
        builder.values().append_value(label);
        builder.append(true);
    }
    builder.finish()
}

fn singleton_label_array(labels: &StringArray) -> arrow::array::ListArray {
    let mut builder = ListBuilder::new(StringBuilder::new());
    for label in labels.iter() {
        builder
            .values()
            .append_value(label.expect("validated non-null _label"));
        builder.append(true);
    }
    builder.finish()
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    fn temp_csv(name: &str, contents: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "lynxes_csv_test_{}_{}.csv",
            name,
            std::process::id()
        ));
        let mut file = File::create(&path).unwrap();
        file.write_all(contents.as_bytes()).unwrap();
        path
    }

    #[test]
    fn synthetic_id_and_label() {
        let path = temp_csv("synthetic", "title,year\nMoon,2009\nArrival,2016\n");
        let frame = read_csv_nodes(
            &path,
            &CsvNodeReadOptions {
                label: Some("RawMovie".to_owned()),
                id_prefix: Some("raw_movie".to_owned()),
                ..Default::default()
            },
        )
        .unwrap();

        let ids: Vec<_> = frame.id_column().iter().map(|id| id.unwrap()).collect();
        assert_eq!(ids, ["raw_movie_0", "raw_movie_1"]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn id_col_is_cast_to_reserved_id() {
        let path = temp_csv("id_col", "id,title\n10,Alien\n11,Aliens\n");
        let frame = read_csv_nodes(
            &path,
            &CsvNodeReadOptions {
                label: Some("RawMovie".to_owned()),
                id_col: Some("id".to_owned()),
                ..Default::default()
            },
        )
        .unwrap();

        let ids: Vec<_> = frame.id_column().iter().map(|id| id.unwrap()).collect();
        assert_eq!(ids, ["10", "11"]);
        assert!(frame.column("id").is_some());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn projection_pushdown_keeps_id_dependency() {
        let path = temp_csv(
            "projection",
            "id,title,cast,votes\n10,Alien,\"[{\"\"name\"\":\"\"Sigourney Weaver\"\"}]\",100\n11,Aliens,\"[]\",200\n",
        );
        let frame = read_csv_nodes(
            &path,
            &CsvNodeReadOptions {
                label: Some("RawMovie".to_owned()),
                id_col: Some("id".to_owned()),
                columns: Some(vec!["title".to_owned()]),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(frame.column_names(), ["_id", "_label", "title"]);
        let ids: Vec<_> = frame.id_column().iter().map(|id| id.unwrap()).collect();
        assert_eq!(ids, ["10", "11"]);
        assert!(frame.column("cast").is_none());
        assert!(frame.column("votes").is_none());
        assert!(frame.column("id").is_none());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn schema_override_can_read_string_view_payload() {
        let path = temp_csv(
            "string_view",
            "id,title,cast\n10,Alien,\"[{\"\"id\"\":1,\"\"name\"\":\"\"Ripley\"\"}]\"\n11,Aliens,\"[]\"\n",
        );
        let frame = read_csv_nodes(
            &path,
            &CsvNodeReadOptions {
                label: Some("RawMovie".to_owned()),
                id_col: Some("id".to_owned()),
                columns: Some(vec!["cast".to_owned()]),
                schema_overrides: vec![("cast".to_owned(), DataType::Utf8View)],
                ..Default::default()
            },
        )
        .unwrap();

        let cast = frame.column("cast").unwrap();
        assert_eq!(cast.data_type(), &DataType::Utf8View);
        assert_eq!(frame.column_names(), ["_id", "_label", "cast"]);
        std::fs::remove_file(path).unwrap();
    }
}
