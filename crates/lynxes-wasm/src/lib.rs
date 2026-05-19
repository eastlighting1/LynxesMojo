use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

const GFB_MAGIC: &[u8; 8] = b"GFRAME\x01\x00";

#[derive(Serialize)]
struct GfbInfo {
    version: [u16; 2],
    node_count: usize,
    edge_count: usize,
    compression: String,
    has_schema: bool,
}

#[derive(Deserialize)]
struct GfbFooter {
    header_offset: u64,
    // other offsets not needed for header inspection
    #[allow(dead_code)]
    schema_offset: u64,
    #[allow(dead_code)]
    node_offset: u64,
    #[allow(dead_code)]
    edge_offset: u64,
    #[allow(dead_code)]
    index_offset: u64,
    #[allow(dead_code)]
    footer_offset: u64,
}

#[derive(Deserialize)]
struct GfbHeaderSlim {
    node_count: usize,
    edge_count: usize,
    has_schema: bool,
    compression: String,
}

/// Returns the Lynxes engine version string.
#[wasm_bindgen]
pub fn wasm_version() -> String {
    lynxes_core::version().to_owned()
}

/// Inspects a `.gfb` file from raw bytes and returns a JS object with header metadata.
///
/// Returned object shape:
/// ```json
/// { "version": [1, 0], "node_count": 4, "edge_count": 5, "compression": "zstd", "has_schema": false }
/// ```
#[wasm_bindgen]
pub fn inspect_gfb_bytes(bytes: &[u8]) -> Result<JsValue, JsError> {
    let info = inspect_impl(bytes).map_err(|e| JsError::new(&e))?;
    serde_wasm_bindgen::to_value(&info).map_err(|e| JsError::new(&e.to_string()))
}

fn inspect_impl(bytes: &[u8]) -> Result<GfbInfo, String> {
    if bytes.len() < 20 {
        return Err("gfb file is too short".to_owned());
    }
    if &bytes[..8] != GFB_MAGIC {
        return Err("not a valid .gfb file (bad magic bytes)".to_owned());
    }

    let major = u16::from_le_bytes(bytes[8..10].try_into().unwrap());
    let minor = u16::from_le_bytes(bytes[10..12].try_into().unwrap());

    // Footer length is stored in the last 8 bytes.
    let footer_len = u64::from_le_bytes(bytes[bytes.len() - 8..].try_into().unwrap()) as usize;
    if footer_len + 8 > bytes.len() {
        return Err("invalid gfb footer length".to_owned());
    }
    let footer_start = bytes.len() - 8 - footer_len;
    let footer: GfbFooter = serde_json::from_slice(&bytes[footer_start..bytes.len() - 8])
        .map_err(|e| format!("footer parse error: {e}"))?;

    // Header: 4-byte length prefix then JSON.
    let h_off = footer.header_offset as usize;
    if h_off + 4 > bytes.len() {
        return Err("header offset out of range".to_owned());
    }
    let header_len = u32::from_le_bytes(bytes[h_off..h_off + 4].try_into().unwrap()) as usize;
    let header_start = h_off + 4;
    let header_end = header_start + header_len;
    if header_end > bytes.len() {
        return Err("header block out of range".to_owned());
    }
    let header: GfbHeaderSlim = serde_json::from_slice(&bytes[header_start..header_end])
        .map_err(|e| format!("header parse error: {e}"))?;

    Ok(GfbInfo {
        version: [major, minor],
        node_count: header.node_count,
        edge_count: header.edge_count,
        compression: header.compression,
        has_schema: header.has_schema,
    })
}
