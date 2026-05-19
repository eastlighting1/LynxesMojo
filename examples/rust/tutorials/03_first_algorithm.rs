use std::fs;

use lynxes::{parse_gf, PageRankConfig, ShortestPathConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // These examples assume you are running from a repository checkout and can
    // read the shared example graphs under examples/data.
    let simple = fs::read_to_string("examples/data/example_simple.gf")?;
    let simple_graph = parse_gf(&simple)?.to_graph_frame()?;

    // Eager algorithms execute immediately and return engine-native outputs that
    // can be inspected without switching to a different abstraction first.
    let path = simple_graph.shortest_path("alice", "charlie", &ShortestPathConfig::default())?;
    println!("shortest path: {:?}", path);

    let ranks = simple_graph.pagerank(&PageRankConfig::default())?;
    println!("pagerank rows: {}", ranks.len());
    Ok(())
}
