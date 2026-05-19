use std::fs;

use lynxes::parse_gf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // These examples assume you are running from a repository checkout and can
    // read the shared example graphs under examples/data.
    let source = fs::read_to_string("examples/data/example_simple.gf")?;
    let graph = parse_gf(&source)?.to_graph_frame()?;

    // GraphFrame keeps graph structure and Arrow-backed payloads together, so
    // basic inspection starts from one engine-native value.
    println!("nodes: {}", graph.node_count());
    println!("edges: {}", graph.edge_count());
    Ok(())
}
