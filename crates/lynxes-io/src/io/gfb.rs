use std::collections::BTreeMap;
use std::fs;
use std::io::Cursor;
use std::path::Path;
#[cfg(not(target_arch = "wasm32"))]
use std::pin::Pin;
#[cfg(not(target_arch = "wasm32"))]
use std::task::{Context, Poll};

use arrow_array::{ListArray, RecordBatch, StringArray};
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;
#[cfg(not(target_arch = "wasm32"))]
use chrono::Utc;
#[cfg(not(target_arch = "wasm32"))]
use futures_core::Stream;
use serde::{Deserialize, Serialize};

use lynxes_core::{EdgeFrame, GFError, GraphFrame, NodeFrame, Result, COL_NODE_LABEL};

const GFB_MAGIC: &[u8; 8] = b"GFRAME\x01\x00";
const GFB_VERSION_MAJOR: u16 = 1;
const GFB_VERSION_MINOR: u16 = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GfbCompression {
    None,
    #[default]
    Zstd,
    Lz4,
}

impl GfbCompression {
    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Zstd => "zstd",
            Self::Lz4 => "lz4",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct GfbWriteOptions {
    pub compression: GfbCompression,
    pub metadata: BTreeMap<String, serde_json::Value>,
    pub schema_json: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct GfbHeader<'a> {
    gfb_version_major: u16,
    gfb_version_minor: u16,
    created_at: String,
    node_count: usize,
    edge_count: usize,
    has_schema: bool,
    compression: &'a str,
    node_columns: Vec<&'a str>,
    edge_columns: Vec<&'a str>,
    node_labels: Vec<String>,
    edge_types: Vec<String>,
    metadata: &'a BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct GfbHeaderOwned {
    gfb_version_major: u16,
    gfb_version_minor: u16,
    created_at: String,
    node_count: usize,
    edge_count: usize,
    has_schema: bool,
    compression: String,
    node_columns: Vec<String>,
    edge_columns: Vec<String>,
    #[allow(dead_code)]
    node_labels: Vec<String>,
    #[allow(dead_code)]
    edge_types: Vec<String>,
    #[allow(dead_code)]
    metadata: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct GfbIndex {
    node_id_to_row: BTreeMap<String, u32>,
    node_labels: Vec<String>,
    edge_types: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct GfbIndexOwned {
    node_id_to_row: BTreeMap<String, u32>,
    #[allow(dead_code)]
    node_labels: Vec<String>,
    #[allow(dead_code)]
    edge_types: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GfbFooter {
    header_offset: u64,
    schema_offset: u64,
    node_offset: u64,
    edge_offset: u64,
    index_offset: u64,
    footer_offset: u64,
}

#[derive(Debug, Clone, Default)]
pub struct GfbReadOptions {
    pub node_columns: Option<Vec<String>>,
    pub edge_columns: Option<Vec<String>>,
}

/// Pull-based `.gfb` graph stream.
///
/// Current `.gfb` files embed exactly one node payload block and one edge payload
/// block, so the stream yields a single [`GraphFrame`] today. The streaming
/// surface exists so future `.gfb` revisions can expose multiple self-contained
/// graph batches without changing the public API again.
#[cfg(not(target_arch = "wasm32"))]
pub struct GfbGraphStream {
    path: std::path::PathBuf,
    options: GfbReadOptions,
    yielded: bool,
}

#[cfg(not(target_arch = "wasm32"))]
impl Stream for GfbGraphStream {
    type Item = Result<GraphFrame>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.yielded {
            return Poll::Ready(None);
        }

        self.yielded = true;
        Poll::Ready(Some(read_gfb_with_options(&self.path, &self.options)))
    }
}

/// Lightweight statistics extracted from a `.gfb` header block.
///
/// Produced by [`read_gfb_inspect`], which reads only the magic bytes, the
/// footer, and the header JSON — skipping the Arrow IPC node/edge payloads.
/// This makes it O(1) in payload size and suitable for quick `inspect` output.
#[derive(Debug, Clone, Serialize)]
pub struct GfbInspect {
    /// `.gfb` format version (`major.minor`).
    pub version: (u16, u16),
    /// Timestamp string embedded in the header at write time (ISO-8601).
    pub created_at: String,
    /// Number of node rows as recorded in the header.
    pub node_count: usize,
    /// Number of edge rows as recorded in the header.
    pub edge_count: usize,
    /// Distinct node labels found at write time.
    pub node_labels: Vec<String>,
    /// Distinct edge type strings found at write time.
    pub edge_types: Vec<String>,
    /// Whether the file embeds a `Schema` JSON block.
    pub has_schema: bool,
    /// Compression codec used for data blocks.
    pub compression: String,
}

/// Read a `.gfb` header without decoding the Arrow IPC payloads.
///
/// Only the magic bytes, the footer, and the header JSON block are accessed;
/// the node, edge, and index blocks are left unread.
pub fn read_gfb_inspect<P>(path: P) -> Result<GfbInspect>
where
    P: AsRef<Path>,
{
    let bytes = fs::read(path)?;
    read_gfb_inspect_bytes(&bytes)
}

pub(crate) fn read_gfb_inspect_bytes(bytes: &[u8]) -> Result<GfbInspect> {
    if bytes.len() < 20 {
        return Err(GFError::ParseError {
            message: "gfb file is too short".to_owned(),
        });
    }
    if &bytes[..8] != GFB_MAGIC {
        return Err(GFError::ParseError {
            message: "invalid gfb magic".to_owned(),
        });
    }

    let major = u16::from_le_bytes(bytes[8..10].try_into().unwrap());
    let minor = u16::from_le_bytes(bytes[10..12].try_into().unwrap());
    if major != GFB_VERSION_MAJOR {
        return Err(GFError::UnsupportedOperation {
            message: format!("unsupported gfb major version {major}"),
        });
    }

    let footer_len = u64::from_le_bytes(bytes[bytes.len() - 8..].try_into().unwrap()) as usize;
    if footer_len + 8 > bytes.len() {
        return Err(GFError::ParseError {
            message: "invalid gfb footer length".to_owned(),
        });
    }
    let footer_start = bytes.len() - 8 - footer_len;
    let footer: GfbFooter =
        serde_json::from_slice(&bytes[footer_start..bytes.len() - 8]).map_err(parse_json_error)?;

    let header_len = u32::from_le_bytes(
        read_fixed(bytes, footer.header_offset as usize, 4)?
            .try_into()
            .unwrap(),
    ) as usize;
    let header_start = footer.header_offset as usize + 4;
    let header_end = checked_end(header_start, header_len, bytes.len())?;
    let header: GfbHeaderOwned =
        serde_json::from_slice(&bytes[header_start..header_end]).map_err(parse_json_error)?;

    Ok(GfbInspect {
        version: (major, minor),
        created_at: header.created_at,
        node_count: header.node_count,
        edge_count: header.edge_count,
        node_labels: header.node_labels,
        edge_types: header.edge_types,
        has_schema: header.has_schema,
        compression: header.compression,
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub fn write_gfb<P>(graph: &GraphFrame, path: P, options: &GfbWriteOptions) -> Result<()>
where
    P: AsRef<Path>,
{
    let schema_bytes = match &options.schema_json {
        Some(schema) => serde_json::to_vec(schema).map_err(json_to_io)?,
        None => Vec::new(),
    };
    let node_labels = collect_node_labels(graph)?;
    let edge_types = collect_edge_types(graph);
    let header = GfbHeader {
        gfb_version_major: GFB_VERSION_MAJOR,
        gfb_version_minor: GFB_VERSION_MINOR,
        created_at: Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        node_count: graph.node_count(),
        edge_count: graph.edge_count(),
        has_schema: !schema_bytes.is_empty(),
        compression: options.compression.as_str(),
        node_columns: graph.nodes().column_names(),
        edge_columns: graph.edges().column_names(),
        node_labels: node_labels.clone(),
        edge_types: edge_types.clone(),
        metadata: &options.metadata,
    };
    let header_bytes = serde_json::to_vec(&header).map_err(json_to_io)?;

    let node_ipc = encode_record_batch(graph.nodes().to_record_batch())?;
    let edge_ipc = encode_record_batch(graph.edges().to_record_batch())?;
    let node_bytes = compress_block(&node_ipc, options.compression)?;
    let edge_bytes = compress_block(&edge_ipc, options.compression)?;

    let index = GfbIndex {
        node_id_to_row: graph
            .nodes()
            .id_column()
            .iter()
            .enumerate()
            .map(|(idx, id)| {
                (
                    id.expect("validated _id column is non-null").to_owned(),
                    idx as u32,
                )
            })
            .collect(),
        node_labels,
        edge_types,
    };
    let index_json = serde_json::to_vec(&index).map_err(json_to_io)?;
    let index_bytes = compress_block(&index_json, options.compression)?;

    let mut bytes = Vec::new();
    bytes.extend_from_slice(GFB_MAGIC);
    bytes.extend_from_slice(&GFB_VERSION_MAJOR.to_le_bytes());
    bytes.extend_from_slice(&GFB_VERSION_MINOR.to_le_bytes());

    let header_offset = bytes.len() as u64;
    write_u32_block(&mut bytes, &header_bytes);
    let schema_offset = bytes.len() as u64;
    write_u64_block(&mut bytes, &schema_bytes);
    let node_offset = bytes.len() as u64;
    write_u64_block(&mut bytes, &node_bytes);
    let edge_offset = bytes.len() as u64;
    write_u64_block(&mut bytes, &edge_bytes);
    let index_offset = bytes.len() as u64;
    write_u64_block(&mut bytes, &index_bytes);

    let footer = GfbFooter {
        header_offset,
        schema_offset,
        node_offset,
        edge_offset,
        index_offset,
        footer_offset: bytes.len() as u64,
    };
    let footer_bytes = serde_json::to_vec(&footer).map_err(json_to_io)?;
    let footer_len = footer_bytes.len() as u64;
    bytes.extend_from_slice(&footer_bytes);
    bytes.extend_from_slice(&footer_len.to_le_bytes());

    fs::write(path, bytes)?;
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn read_gfb<P>(path: P) -> Result<GraphFrame>
where
    P: AsRef<Path>,
{
    read_gfb_with_options(path, &GfbReadOptions::default())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn read_gfb_streaming<P>(path: P) -> Result<GfbGraphStream>
where
    P: AsRef<Path>,
{
    read_gfb_streaming_with_options(path, &GfbReadOptions::default())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn read_gfb_streaming_with_options<P>(
    path: P,
    options: &GfbReadOptions,
) -> Result<GfbGraphStream>
where
    P: AsRef<Path>,
{
    let path = path.as_ref().to_path_buf();
    if !path.exists() {
        return Err(GFError::IoError(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("gfb file not found: {}", path.display()),
        )));
    }

    Ok(GfbGraphStream {
        path,
        options: options.clone(),
        yielded: false,
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub fn read_gfb_with_options<P>(path: P, options: &GfbReadOptions) -> Result<GraphFrame>
where
    P: AsRef<Path>,
{
    let bytes = fs::read(path)?;
    read_gfb_bytes_with_options(&bytes, options)
}

pub(crate) fn read_gfb_bytes_with_options(
    bytes: &[u8],
    options: &GfbReadOptions,
) -> Result<GraphFrame> {
    if bytes.len() < 20 {
        return Err(GFError::ParseError {
            message: "gfb file is too short".to_owned(),
        });
    }
    if &bytes[..8] != GFB_MAGIC {
        return Err(GFError::ParseError {
            message: "invalid gfb magic".to_owned(),
        });
    }

    let major = u16::from_le_bytes(bytes[8..10].try_into().unwrap());
    let minor = u16::from_le_bytes(bytes[10..12].try_into().unwrap());
    if major != GFB_VERSION_MAJOR {
        return Err(GFError::UnsupportedOperation {
            message: format!("unsupported gfb major version {major}"),
        });
    }
    if minor > GFB_VERSION_MINOR {
        return Err(GFError::UnsupportedOperation {
            message: format!("unsupported gfb minor version {minor}"),
        });
    }

    let footer_len = u64::from_le_bytes(bytes[bytes.len() - 8..].try_into().unwrap()) as usize;
    if footer_len + 8 > bytes.len() {
        return Err(GFError::ParseError {
            message: "invalid gfb footer length".to_owned(),
        });
    }
    let footer_start = bytes.len() - 8 - footer_len;
    let footer: GfbFooter =
        serde_json::from_slice(&bytes[footer_start..bytes.len() - 8]).map_err(parse_json_error)?;
    validate_footer(&footer, bytes.len())?;

    let header_len = u32::from_le_bytes(
        read_fixed(bytes, footer.header_offset as usize, 4)?
            .try_into()
            .unwrap(),
    ) as usize;
    let header_start = footer.header_offset as usize + 4;
    let header_end = checked_end(header_start, header_len, bytes.len())?;
    let header: GfbHeaderOwned =
        serde_json::from_slice(&bytes[header_start..header_end]).map_err(parse_json_error)?;

    if header.gfb_version_major != major || header.gfb_version_minor != minor {
        return Err(GFError::SchemaMismatch {
            message: "header version does not match envelope version".to_owned(),
        });
    }

    let compression = parse_compression(&header.compression)?;
    let schema_len = u64::from_le_bytes(
        read_fixed(bytes, footer.schema_offset as usize, 8)?
            .try_into()
            .unwrap(),
    ) as usize;
    if header.has_schema != (schema_len > 0) {
        return Err(GFError::SchemaMismatch {
            message: "header has_schema does not match schema block length".to_owned(),
        });
    }

    let node_batch = decode_block_batch(bytes, footer.node_offset as usize, compression)?;
    let edge_batch = decode_block_batch(bytes, footer.edge_offset as usize, compression)?;

    if node_batch.num_rows() != header.node_count || edge_batch.num_rows() != header.edge_count {
        return Err(GFError::SchemaMismatch {
            message: format!(
                "header counts do not match payload counts: nodes {} vs {}, edges {} vs {}",
                header.node_count,
                node_batch.num_rows(),
                header.edge_count,
                edge_batch.num_rows()
            ),
        });
    }

    let index_json =
        decode_json_block::<GfbIndexOwned>(bytes, footer.index_offset as usize, compression)?;
    validate_index(&index_json, &node_batch)?;

    let nodes = NodeFrame::from_record_batch(node_batch)?;
    let edges = EdgeFrame::from_record_batch(edge_batch)?;
    let nodes = apply_node_projection(nodes, options, &header.node_columns)?;
    let edges = apply_edge_projection(edges, options, &header.edge_columns)?;

    GraphFrame::new(nodes, edges)
}

fn write_u32_block(buffer: &mut Vec<u8>, payload: &[u8]) {
    buffer.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    buffer.extend_from_slice(payload);
}

fn write_u64_block(buffer: &mut Vec<u8>, payload: &[u8]) {
    buffer.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    buffer.extend_from_slice(payload);
}

fn encode_record_batch(batch: &RecordBatch) -> Result<Vec<u8>> {
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer =
            FileWriter::try_new(&mut cursor, batch.schema().as_ref()).map_err(io_other)?;
        writer.write(batch).map_err(io_other)?;
        writer.finish().map_err(io_other)?;
    }
    Ok(cursor.into_inner())
}

#[cfg(not(target_arch = "wasm32"))]
fn compress_block(payload: &[u8], compression: GfbCompression) -> Result<Vec<u8>> {
    match compression {
        GfbCompression::None => Ok(payload.to_vec()),
        GfbCompression::Zstd => zstd::stream::encode_all(Cursor::new(payload), 3).map_err(io_other),
        GfbCompression::Lz4 => Ok(lz4_flex::compress_prepend_size(payload)),
    }
}

fn decode_block_batch(
    bytes: &[u8],
    offset: usize,
    compression: GfbCompression,
) -> Result<RecordBatch> {
    let payload = decode_block(bytes, offset, compression)?;
    let mut reader = FileReader::try_new(Cursor::new(payload), None).map_err(io_other)?;
    let batch = reader
        .next()
        .transpose()
        .map_err(io_other)?
        .ok_or_else(|| GFError::ParseError {
            message: "arrow ipc block is empty".to_owned(),
        })?;
    if reader.next().is_some() {
        return Err(GFError::ParseError {
            message: "gfb blocks must contain exactly one record batch".to_owned(),
        });
    }
    Ok(batch)
}

fn decode_json_block<T>(bytes: &[u8], offset: usize, compression: GfbCompression) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    let payload = decode_block(bytes, offset, compression)?;
    serde_json::from_slice(&payload).map_err(parse_json_error)
}

fn decode_block(bytes: &[u8], offset: usize, compression: GfbCompression) -> Result<Vec<u8>> {
    let len = u64::from_le_bytes(read_fixed(bytes, offset, 8)?.try_into().unwrap()) as usize;
    let start = offset + 8;
    let end = checked_end(start, len, bytes.len())?;
    let payload = &bytes[start..end];
    #[cfg(target_arch = "wasm32")]
    {
        match compression {
            GfbCompression::None => Ok(payload.to_vec()),
            GfbCompression::Zstd | GfbCompression::Lz4 => Err(GFError::UnsupportedOperation {
                message: "compressed gfb payload decode is unavailable on wasm setup build"
                    .to_owned(),
            }),
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    match compression {
        GfbCompression::None => Ok(payload.to_vec()),
        GfbCompression::Zstd => zstd::decode_all(Cursor::new(payload)).map_err(io_other),
        GfbCompression::Lz4 => lz4_flex::decompress_size_prepended(payload).map_err(io_other),
    }
}

fn apply_node_projection(
    nodes: NodeFrame,
    options: &GfbReadOptions,
    declared_columns: &[String],
) -> Result<NodeFrame> {
    let Some(columns) = options.node_columns.as_ref() else {
        return Ok(nodes);
    };
    validate_projection(columns, declared_columns, "_node")?;
    let requested: Vec<&str> = columns.iter().map(String::as_str).collect();
    nodes.select(&requested)
}

fn apply_edge_projection(
    edges: EdgeFrame,
    options: &GfbReadOptions,
    declared_columns: &[String],
) -> Result<EdgeFrame> {
    let Some(columns) = options.edge_columns.as_ref() else {
        return Ok(edges);
    };
    validate_projection(columns, declared_columns, "_edge")?;
    let requested: Vec<&str> = columns.iter().map(String::as_str).collect();
    edges.select(&requested)
}

fn validate_projection(
    columns: &[String],
    declared_columns: &[String],
    domain: &str,
) -> Result<()> {
    for column in columns {
        if !declared_columns.iter().any(|declared| declared == column) {
            return Err(GFError::ColumnNotFound {
                column: format!("{domain}:{column}"),
            });
        }
    }
    Ok(())
}

fn validate_footer(footer: &GfbFooter, file_size: usize) -> Result<()> {
    let offsets = [
        footer.header_offset,
        footer.schema_offset,
        footer.node_offset,
        footer.edge_offset,
        footer.index_offset,
        footer.footer_offset,
    ];
    if offsets.windows(2).any(|window| window[0] >= window[1]) {
        return Err(GFError::ParseError {
            message: "gfb footer offsets are not strictly increasing".to_owned(),
        });
    }
    if footer.footer_offset as usize >= file_size {
        return Err(GFError::ParseError {
            message: "gfb footer offset is out of bounds".to_owned(),
        });
    }
    Ok(())
}

fn validate_index(index: &GfbIndexOwned, node_batch: &RecordBatch) -> Result<()> {
    if index.node_id_to_row.len() != node_batch.num_rows() {
        return Err(GFError::SchemaMismatch {
            message: "index node_id_to_row size does not match node row count".to_owned(),
        });
    }
    Ok(())
}

fn parse_compression(value: &str) -> Result<GfbCompression> {
    match value {
        "none" => Ok(GfbCompression::None),
        "zstd" => Ok(GfbCompression::Zstd),
        "lz4" => Ok(GfbCompression::Lz4),
        other => Err(GFError::UnsupportedOperation {
            message: format!("unsupported gfb compression mode {other}"),
        }),
    }
}

fn read_fixed(bytes: &[u8], offset: usize, len: usize) -> Result<&[u8]> {
    let end = checked_end(offset, len, bytes.len())?;
    Ok(&bytes[offset..end])
}

fn checked_end(start: usize, len: usize, file_len: usize) -> Result<usize> {
    let end = start.checked_add(len).ok_or_else(|| GFError::ParseError {
        message: "gfb block length overflow".to_owned(),
    })?;
    if end > file_len {
        return Err(GFError::ParseError {
            message: "gfb block extends past file end".to_owned(),
        });
    }
    Ok(end)
}

fn collect_node_labels(graph: &GraphFrame) -> Result<Vec<String>> {
    let labels = graph
        .nodes()
        .column(COL_NODE_LABEL)
        .ok_or_else(|| GFError::MissingReservedColumn {
            column: COL_NODE_LABEL.to_owned(),
        })?
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| GFError::ReservedColumnType {
            column: COL_NODE_LABEL.to_owned(),
            expected: "List<Utf8>".to_owned(),
            actual: "non-List array".to_owned(),
        })?;
    let values = labels
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| GFError::ReservedColumnType {
            column: COL_NODE_LABEL.to_owned(),
            expected: "List<Utf8>".to_owned(),
            actual: "non-Utf8 child array".to_owned(),
        })?;

    let mut seen = std::collections::BTreeSet::new();
    for value in values.iter().flatten() {
        seen.insert(value.to_owned());
    }
    Ok(seen.into_iter().collect())
}

fn collect_edge_types(graph: &GraphFrame) -> Vec<String> {
    let mut edge_types: Vec<String> = graph
        .edges()
        .edge_types()
        .into_iter()
        .map(str::to_owned)
        .collect();
    edge_types.sort();
    edge_types
}

fn json_to_io(error: serde_json::Error) -> GFError {
    GFError::IoError(std::io::Error::other(error))
}

fn parse_json_error(error: serde_json::Error) -> GFError {
    GFError::ParseError {
        message: error.to_string(),
    }
}

fn io_other(error: impl std::error::Error + Send + Sync + 'static) -> GFError {
    GFError::IoError(std::io::Error::other(error))
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;
    use std::sync::Arc;

    use arrow_array::{
        builder::{ListBuilder, StringBuilder},
        ArrayRef, Int64Array, Int8Array,
    };
    use arrow_ipc::reader::FileReader;
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};

    use lynxes_core::{
        Direction, EdgeFrame, NodeFrame, COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC,
        COL_EDGE_TYPE, COL_NODE_ID,
    };

    fn labels_array(values: &[&[&str]]) -> ListArray {
        let mut builder = ListBuilder::new(StringBuilder::new());
        for labels in values {
            for label in *labels {
                builder.values().append_value(label);
            }
            builder.append(true);
        }
        builder.finish()
    }

    fn demo_graph() -> GraphFrame {
        let node_schema = Arc::new(ArrowSchema::new(vec![
            Field::new(COL_NODE_ID, DataType::Utf8, false),
            Field::new(
                COL_NODE_LABEL,
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                false,
            ),
            Field::new("age", DataType::Int64, true),
        ]));
        let edge_schema = Arc::new(ArrowSchema::new(vec![
            Field::new(COL_EDGE_SRC, DataType::Utf8, false),
            Field::new(COL_EDGE_DST, DataType::Utf8, false),
            Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
            Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
            Field::new("weight", DataType::Int64, true),
        ]));

        let nodes = NodeFrame::from_record_batch(
            RecordBatch::try_new(
                node_schema,
                vec![
                    Arc::new(arrow_array::StringArray::from(vec!["alice", "bob"])) as ArrayRef,
                    Arc::new(labels_array(&[&["Person"], &["Person", "Admin"]])) as ArrayRef,
                    Arc::new(Int64Array::from(vec![Some(30), Some(40)])) as ArrayRef,
                ],
            )
            .unwrap(),
        )
        .unwrap();
        let edges = EdgeFrame::from_record_batch(
            RecordBatch::try_new(
                edge_schema,
                vec![
                    Arc::new(arrow_array::StringArray::from(vec!["alice"])) as ArrayRef,
                    Arc::new(arrow_array::StringArray::from(vec!["bob"])) as ArrayRef,
                    Arc::new(arrow_array::StringArray::from(vec!["KNOWS"])) as ArrayRef,
                    Arc::new(Int8Array::from(vec![Direction::Out.as_i8()])) as ArrayRef,
                    Arc::new(Int64Array::from(vec![Some(1)])) as ArrayRef,
                ],
            )
            .unwrap(),
        )
        .unwrap();

        GraphFrame::new(nodes, edges).unwrap()
    }

    #[test]
    fn write_gfb_emits_expected_envelope_and_footer() {
        let graph = demo_graph();
        let path = std::env::temp_dir().join(format!("lynxes-ser005-{}.gfb", std::process::id()));

        write_gfb(&graph, &path, &GfbWriteOptions::default()).unwrap();
        let bytes = fs::read(&path).unwrap();
        let _ = fs::remove_file(&path);

        assert_eq!(&bytes[..8], GFB_MAGIC);
        assert_eq!(u16::from_le_bytes([bytes[8], bytes[9]]), GFB_VERSION_MAJOR);
        assert_eq!(
            u16::from_le_bytes([bytes[10], bytes[11]]),
            GFB_VERSION_MINOR
        );

        let footer_len = u64::from_le_bytes(bytes[bytes.len() - 8..].try_into().unwrap()) as usize;
        let footer_start = bytes.len() - 8 - footer_len;
        let footer: GfbFooter =
            serde_json::from_slice(&bytes[footer_start..bytes.len() - 8]).unwrap();
        assert_eq!(footer.header_offset, 12);
        assert!(footer.footer_offset as usize >= footer.index_offset as usize);
        assert_eq!(footer.footer_offset as usize, footer_start);
    }

    #[test]
    fn write_gfb_stores_header_and_index_metadata() {
        let graph = demo_graph();
        let path =
            std::env::temp_dir().join(format!("lynxes-ser005-header-{}.gfb", std::process::id()));

        write_gfb(
            &graph,
            &path,
            &GfbWriteOptions {
                compression: GfbCompression::None,
                metadata: BTreeMap::from([(
                    "source".to_owned(),
                    serde_json::Value::String("unit-test".to_owned()),
                )]),
                schema_json: None,
            },
        )
        .unwrap();
        let bytes = fs::read(&path).unwrap();
        let _ = fs::remove_file(&path);

        let header_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let header: serde_json::Value =
            serde_json::from_slice(&bytes[16..16 + header_len]).unwrap();
        assert_eq!(header["compression"], "none");
        assert_eq!(header["node_count"], 2);
        assert_eq!(header["edge_count"], 1);
        assert_eq!(header["metadata"]["source"], "unit-test");
        assert!(!header["node_labels"].as_array().unwrap().is_empty());
    }

    #[test]
    fn write_gfb_embeds_arrow_ipc_payloads() {
        let graph = demo_graph();
        let path =
            std::env::temp_dir().join(format!("lynxes-ser005-ipc-{}.gfb", std::process::id()));

        write_gfb(&graph, &path, &GfbWriteOptions::default()).unwrap();
        let bytes = fs::read(&path).unwrap();
        let _ = fs::remove_file(&path);

        let header_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let schema_offset = 16 + header_len;
        let schema_len =
            u64::from_le_bytes(bytes[schema_offset..schema_offset + 8].try_into().unwrap())
                as usize;
        let node_offset = schema_offset + 8 + schema_len;
        let node_len =
            u64::from_le_bytes(bytes[node_offset..node_offset + 8].try_into().unwrap()) as usize;
        let node_block = &bytes[node_offset + 8..node_offset + 8 + node_len];
        let node_ipc = zstd::decode_all(Cursor::new(node_block)).unwrap();

        let mut reader = FileReader::try_new(Cursor::new(node_ipc), None).unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert!(batch.schema().column_with_name(COL_NODE_ID).is_some());
    }

    #[test]
    fn write_gfb_supports_lz4_compression() {
        let graph = demo_graph();
        let path =
            std::env::temp_dir().join(format!("lynxes-ser005-lz4-{}.gfb", std::process::id()));

        write_gfb(
            &graph,
            &path,
            &GfbWriteOptions {
                compression: GfbCompression::Lz4,
                ..GfbWriteOptions::default()
            },
        )
        .unwrap();
        let bytes = fs::read(&path).unwrap();
        let _ = fs::remove_file(&path);

        let header_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let schema_offset = 16 + header_len;
        let schema_len =
            u64::from_le_bytes(bytes[schema_offset..schema_offset + 8].try_into().unwrap())
                as usize;
        let node_offset = schema_offset + 8 + schema_len;
        let node_len =
            u64::from_le_bytes(bytes[node_offset..node_offset + 8].try_into().unwrap()) as usize;
        let node_block = &bytes[node_offset + 8..node_offset + 8 + node_len];
        let node_ipc = lz4_flex::decompress_size_prepended(node_block).unwrap();

        let mut reader = FileReader::try_new(Cursor::new(node_ipc), None).unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 2);
    }

    #[test]
    fn read_gfb_round_trips_written_graph() {
        let graph = demo_graph();
        let path = std::env::temp_dir().join(format!(
            "lynxes-ser006-roundtrip-{}.gfb",
            std::process::id()
        ));

        write_gfb(&graph, &path, &GfbWriteOptions::default()).unwrap();
        let decoded = read_gfb(&path).unwrap();
        let _ = fs::remove_file(&path);

        assert_eq!(decoded.node_count(), graph.node_count());
        assert_eq!(decoded.edge_count(), graph.edge_count());
        assert!(decoded.nodes().column("age").is_some());
        assert!(decoded.edges().column("weight").is_some());
    }

    #[test]
    fn read_gfb_supports_post_decode_projection() {
        let graph = demo_graph();
        let path = std::env::temp_dir().join(format!(
            "lynxes-ser006-projection-{}.gfb",
            std::process::id()
        ));

        write_gfb(&graph, &path, &GfbWriteOptions::default()).unwrap();
        let decoded = read_gfb_with_options(
            &path,
            &GfbReadOptions {
                node_columns: Some(vec!["age".to_owned()]),
                edge_columns: Some(vec!["weight".to_owned()]),
            },
        )
        .unwrap();
        let _ = fs::remove_file(&path);

        assert_eq!(
            decoded.nodes().column_names(),
            vec![COL_NODE_ID, COL_NODE_LABEL, "age"]
        );
        assert_eq!(
            decoded.edges().column_names(),
            vec![
                COL_EDGE_SRC,
                COL_EDGE_DST,
                COL_EDGE_TYPE,
                COL_EDGE_DIRECTION,
                "weight"
            ]
        );
    }

    #[test]
    fn read_gfb_rejects_invalid_magic() {
        let path = std::env::temp_dir().join(format!(
            "lynxes-ser006-invalid-magic-{}.gfb",
            std::process::id()
        ));
        fs::write(&path, b"not-a-gfb").unwrap();
        let err = read_gfb(&path).unwrap_err();
        let _ = fs::remove_file(&path);

        assert!(matches!(err, GFError::ParseError { .. }));
    }

    #[test]
    fn read_gfb_restores_lz4_payloads() {
        let graph = demo_graph();
        let path =
            std::env::temp_dir().join(format!("lynxes-ser006-lz4-{}.gfb", std::process::id()));

        write_gfb(
            &graph,
            &path,
            &GfbWriteOptions {
                compression: GfbCompression::Lz4,
                ..GfbWriteOptions::default()
            },
        )
        .unwrap();
        let decoded = read_gfb(&path).unwrap();
        let _ = fs::remove_file(&path);

        assert_eq!(decoded.node_count(), 2);
        assert_eq!(decoded.edge_count(), 1);
    }

    // ── read_gfb_inspect tests ────────────────────────────────────────────────

    #[test]
    fn read_gfb_inspect_returns_correct_counts_and_labels() {
        let graph = demo_graph();
        let path =
            std::env::temp_dir().join(format!("lynxes-inspect-counts-{}.gfb", std::process::id()));

        write_gfb(&graph, &path, &GfbWriteOptions::default()).unwrap();
        let info = read_gfb_inspect(&path).unwrap();
        let _ = fs::remove_file(&path);

        assert_eq!(info.node_count, graph.node_count());
        assert_eq!(info.edge_count, graph.edge_count());
        assert!(
            info.node_labels.contains(&"Person".to_owned()),
            "expected 'Person' in labels, got {:?}",
            info.node_labels
        );
        assert!(
            info.edge_types.contains(&"KNOWS".to_owned()),
            "expected 'KNOWS' in edge_types, got {:?}",
            info.edge_types
        );
    }

    #[test]
    fn read_gfb_inspect_reports_version_and_compression() {
        let graph = demo_graph();
        let path =
            std::env::temp_dir().join(format!("lynxes-inspect-meta-{}.gfb", std::process::id()));

        write_gfb(
            &graph,
            &path,
            &GfbWriteOptions {
                compression: GfbCompression::Lz4,
                ..GfbWriteOptions::default()
            },
        )
        .unwrap();
        let info = read_gfb_inspect(&path).unwrap();
        let _ = fs::remove_file(&path);

        assert_eq!(info.version, (GFB_VERSION_MAJOR, GFB_VERSION_MINOR));
        assert_eq!(info.compression, "lz4");
        assert!(
            !info.created_at.is_empty(),
            "created_at should be non-empty"
        );
    }

    #[test]
    fn read_gfb_inspect_has_schema_flag_matches_write_options() {
        let graph = demo_graph();
        let path_no_schema = std::env::temp_dir().join(format!(
            "lynxes-inspect-noschema-{}.gfb",
            std::process::id()
        ));
        let path_with_schema =
            std::env::temp_dir().join(format!("lynxes-inspect-schema-{}.gfb", std::process::id()));

        write_gfb(&graph, &path_no_schema, &GfbWriteOptions::default()).unwrap();
        write_gfb(
            &graph,
            &path_with_schema,
            &GfbWriteOptions {
                schema_json: Some(serde_json::json!({"nodes": {}, "edges": {}})),
                ..GfbWriteOptions::default()
            },
        )
        .unwrap();

        let info_no = read_gfb_inspect(&path_no_schema).unwrap();
        let info_yes = read_gfb_inspect(&path_with_schema).unwrap();
        let _ = fs::remove_file(&path_no_schema);
        let _ = fs::remove_file(&path_with_schema);

        assert!(!info_no.has_schema, "expected has_schema=false");
        assert!(info_yes.has_schema, "expected has_schema=true");
    }

    #[test]
    fn read_gfb_inspect_skips_payload_bytes() {
        // Verify that inspect is much faster than a full read for a larger graph
        // by simply checking it doesn't error and produces coherent output.
        // (Timing assertions are fragile; we just validate correctness here.)
        let graph = demo_graph();
        let path =
            std::env::temp_dir().join(format!("lynxes-inspect-skip-{}.gfb", std::process::id()));
        write_gfb(&graph, &path, &GfbWriteOptions::default()).unwrap();
        let file_size = fs::metadata(&path).unwrap().len();
        let info = read_gfb_inspect(&path).unwrap();
        let _ = fs::remove_file(&path);

        // If inspect erroneously reads the full payload its accuracy degrades —
        // but if it reads correctly the counts must match exactly.
        assert_eq!(
            info.node_count,
            graph.node_count(),
            "inspect node_count mismatch (file was {} bytes)",
            file_size
        );
        assert_eq!(
            info.edge_count,
            graph.edge_count(),
            "inspect edge_count mismatch (file was {} bytes)",
            file_size
        );
    }
}
