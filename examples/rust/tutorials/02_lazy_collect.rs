use std::fs;

use lynxes::{parse_gf, BinaryOp, Direction, EdgeTypeSpec, Expr, ScalarValue};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // These examples assume you are running from a repository checkout and can
    // read the shared example graphs under examples/data.
    let source = fs::read_to_string("examples/data/example_simple.gf")?;
    let graph = parse_gf(&source)?.to_graph_frame()?;

    let result = graph
        .lazy()
        .filter_nodes(Expr::BinaryOp {
            left: Box::new(Expr::Col {
                name: "_id".to_owned(),
            }),
            op: BinaryOp::Eq,
            right: Box::new(Expr::Literal {
                value: ScalarValue::String("alice".to_owned()),
            }),
        })
        // The lazy API builds a logical query first; collect() is the point where
        // Lynxes executes the traversal and materializes the subgraph.
        .expand(EdgeTypeSpec::Single("KNOWS".to_owned()), 2, Direction::Out)
        .collect()?;

    println!("expanded nodes: {}", result.node_count());
    println!("expanded edges: {}", result.edge_count());
    Ok(())
}
