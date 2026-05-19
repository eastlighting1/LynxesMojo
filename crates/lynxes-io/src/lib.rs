pub mod io;

pub use io::{
    parse_gf, read_gfb_inspect, write_gf, GfbCompression, GfbInspect, GfbReadOptions,
    GfbWriteOptions, ParsedEdgeDecl, ParsedGfDocument, ParsedNodeDecl,
};
#[cfg(not(target_arch = "wasm32"))]
pub use io::{
    read_csv_nodes, read_gfb, read_gfb_streaming, read_gfb_streaming_with_options,
    read_gfb_with_options, read_parquet_graph, read_parquet_graph_with_options, write_gfb,
    write_parquet_graph, CsvNodeReadOptions, GfbGraphStream, ParquetReadOptions,
};
