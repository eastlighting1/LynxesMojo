use std::fs::File;
use std::path::Path;

use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::arrow::ProjectionMask;
use parquet::file::properties::WriterProperties;

use lynxes_core::{
    EdgeFrame, GFError, GraphFrame, NodeFrame, Result, EDGE_RESERVED_COLUMNS, NODE_RESERVED_COLUMNS,
};

#[derive(Debug, Clone, Default)]
pub struct ParquetReadOptions {
    pub node_columns: Option<Vec<String>>,
    pub edge_columns: Option<Vec<String>>,
}

pub fn write_parquet_graph<P1, P2>(graph: &GraphFrame, node_path: P1, edge_path: P2) -> Result<()>
where
    P1: AsRef<Path>,
    P2: AsRef<Path>,
{
    write_record_batch_parquet(graph.nodes().to_record_batch(), node_path)?;
    write_record_batch_parquet(graph.edges().to_record_batch(), edge_path)?;
    Ok(())
}

pub fn read_parquet_graph<P1, P2>(node_path: P1, edge_path: P2) -> Result<GraphFrame>
where
    P1: AsRef<Path>,
    P2: AsRef<Path>,
{
    read_parquet_graph_with_options(node_path, edge_path, &ParquetReadOptions::default())
}

pub fn read_parquet_graph_with_options<P1, P2>(
    node_path: P1,
    edge_path: P2,
    options: &ParquetReadOptions,
) -> Result<GraphFrame>
where
    P1: AsRef<Path>,
    P2: AsRef<Path>,
{
    let nodes = read_node_frame_parquet(node_path, options.node_columns.as_deref())?;
    let edges = read_edge_frame_parquet(edge_path, options.edge_columns.as_deref())?;
    GraphFrame::new(nodes, edges)
}

fn write_record_batch_parquet<P>(batch: &arrow_array::RecordBatch, path: P) -> Result<()>
where
    P: AsRef<Path>,
{
    let file = File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props)).map_err(io_other)?;
    writer.write(batch).map_err(io_other)?;
    writer.close().map_err(io_other)?;
    Ok(())
}

fn read_node_frame_parquet<P>(path: P, columns: Option<&[String]>) -> Result<NodeFrame>
where
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let mut builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(io_other)?;
    if let Some(columns) = columns {
        let projection = projection_mask(
            builder.schema().fields(),
            builder.parquet_schema(),
            columns,
            &NODE_RESERVED_COLUMNS,
        )?;
        builder = builder.with_projection(projection);
    }

    let reader = builder.build().map_err(io_other)?;
    let mut frames = Vec::new();
    for batch in reader {
        frames.push(NodeFrame::from_record_batch(batch.map_err(io_other)?)?);
    }
    concat_node_frames(&frames)
}

fn read_edge_frame_parquet<P>(path: P, columns: Option<&[String]>) -> Result<EdgeFrame>
where
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let mut builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(io_other)?;
    if let Some(columns) = columns {
        let projection = projection_mask(
            builder.schema().fields(),
            builder.parquet_schema(),
            columns,
            &EDGE_RESERVED_COLUMNS,
        )?;
        builder = builder.with_projection(projection);
    }

    let reader = builder.build().map_err(io_other)?;
    let mut frames = Vec::new();
    for batch in reader {
        frames.push(EdgeFrame::from_record_batch(batch.map_err(io_other)?)?);
    }
    concat_edge_frames(&frames)
}

fn projection_mask(
    fields: &arrow_schema::Fields,
    parquet_schema: &parquet::schema::types::SchemaDescriptor,
    columns: &[String],
    reserved: &[&str],
) -> Result<ProjectionMask> {
    let mut requested: Vec<String> = reserved.iter().map(|name| (*name).to_owned()).collect();
    for column in columns {
        if !requested.iter().any(|existing| existing == column) {
            requested.push(column.clone());
        }
    }

    let mut indices = Vec::with_capacity(requested.len());
    for column in &requested {
        let index = fields
            .iter()
            .position(|field| field.name().as_str() == column.as_str())
            .ok_or_else(|| GFError::ColumnNotFound {
                column: column.clone(),
            })?;
        indices.push(index);
    }
    Ok(ProjectionMask::leaves(parquet_schema, indices))
}

fn concat_node_frames(frames: &[NodeFrame]) -> Result<NodeFrame> {
    if frames.is_empty() {
        return Err(GFError::ParseError {
            message: "parquet node file produced no record batches".to_owned(),
        });
    }
    let refs: Vec<&NodeFrame> = frames.iter().collect();
    NodeFrame::concat(&refs)
}

fn concat_edge_frames(frames: &[EdgeFrame]) -> Result<EdgeFrame> {
    if frames.is_empty() {
        return Err(GFError::ParseError {
            message: "parquet edge file produced no record batches".to_owned(),
        });
    }
    let refs: Vec<&EdgeFrame> = frames.iter().collect();
    EdgeFrame::concat(&refs)
}

fn io_other(error: impl std::error::Error + Send + Sync + 'static) -> GFError {
    GFError::IoError(std::io::Error::other(error))
}
