use std::{
    fs,
    path::{Path, PathBuf},
};

use lynxes_core::{
    BinaryOp, Direction, EdgeFrame, EdgeTypeSpec, Expr, GFError, GraphFrame, NodeFrame, Result,
    ScalarValue, COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
};
use lynxes_io::{parse_gf, read_gfb};

use lynxes_lazy::LazyGraphFrame;

use crate::connector::{Connector, ConnectorFuture, ExpandResult};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GFConnectorFormat {
    Gf,
    Gfb,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GFConnector {
    path: PathBuf,
    format: GFConnectorFormat,
}

impl GFConnector {
    pub fn new<P>(path: P) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        let path = path.as_ref().to_path_buf();
        let format = infer_format(&path)?;
        Ok(Self { path, format })
    }

    pub fn from_gf<P>(path: P) -> Self
    where
        P: AsRef<Path>,
    {
        Self {
            path: path.as_ref().to_path_buf(),
            format: GFConnectorFormat::Gf,
        }
    }

    pub fn from_gfb<P>(path: P) -> Self
    where
        P: AsRef<Path>,
    {
        Self {
            path: path.as_ref().to_path_buf(),
            format: GFConnectorFormat::Gfb,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn format(&self) -> &GFConnectorFormat {
        &self.format
    }

    fn read_graph(&self) -> Result<GraphFrame> {
        match self.format {
            GFConnectorFormat::Gf => {
                let source = fs::read_to_string(&self.path)?;
                parse_gf(&source)?.to_graph_frame()
            }
            GFConnectorFormat::Gfb => read_gfb(&self.path),
        }
    }
}

impl Connector for GFConnector {
    fn cache_source_key(&self) -> Option<String> {
        Some(format!("gf://{}", self.path.display()))
    }

    fn load_nodes<'a>(
        &'a self,
        labels: Option<&'a [&'a str]>,
        columns: Option<&'a [&'a str]>,
        predicate: Option<&'a Expr>,
        batch_size: usize,
    ) -> ConnectorFuture<'a, NodeFrame> {
        Box::pin(async move {
            validate_batch_size(batch_size)?;
            let graph = self.read_graph()?;
            let mut lazy = LazyGraphFrame::from_graph(&graph);

            if let Some(label_predicate) = labels_to_predicate(labels) {
                lazy = lazy.filter_nodes(label_predicate);
            }
            if let Some(predicate) = predicate {
                lazy = lazy.filter_nodes(predicate.clone());
            }
            if let Some(columns) = columns {
                let columns: Vec<String> =
                    columns.iter().map(|column| (*column).to_owned()).collect();
                lazy = lazy.select_nodes(columns);
            }

            lazy.collect_nodes()
        })
    }

    fn load_edges<'a>(
        &'a self,
        edge_types: Option<&'a [&'a str]>,
        columns: Option<&'a [&'a str]>,
        predicate: Option<&'a Expr>,
        batch_size: usize,
    ) -> ConnectorFuture<'a, EdgeFrame> {
        Box::pin(async move {
            validate_batch_size(batch_size)?;
            let graph = self.read_graph()?;
            let mut lazy = LazyGraphFrame::from_graph(&graph);

            if let Some(edge_predicate) = edge_types_to_predicate(edge_types) {
                lazy = lazy.filter_edges(edge_predicate);
            }
            if let Some(predicate) = predicate {
                lazy = lazy.filter_edges(predicate.clone());
            }
            if let Some(columns) = columns {
                let columns: Vec<String> =
                    columns.iter().map(|column| (*column).to_owned()).collect();
                lazy = lazy.select_edges(columns);
            }

            lazy.collect_edges()
        })
    }

    fn expand<'a>(
        &'a self,
        node_ids: &'a [&'a str],
        edge_type: &'a EdgeTypeSpec,
        hops: u32,
        direction: Direction,
        node_predicate: Option<&'a Expr>,
    ) -> ConnectorFuture<'a, ExpandResult> {
        Box::pin(async move {
            validate_hops(hops)?;
            let graph = self.read_graph()?;
            expand_graph(&graph, node_ids, edge_type, hops, direction, node_predicate)
        })
    }

    fn write<'a>(&'a self, graph: &'a GraphFrame) -> ConnectorFuture<'a, ()> {
        Box::pin(async move {
            match self.format {
                GFConnectorFormat::Gfb => {
                    lynxes_io::write_gfb(graph, &self.path, &lynxes_io::GfbWriteOptions::default())
                }
                GFConnectorFormat::Gf => Err(GFError::UnsupportedOperation {
                    message: "GFConnector write() currently supports .gfb targets only".to_owned(),
                }),
            }
        })
    }
}

fn infer_format(path: &Path) -> Result<GFConnectorFormat> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("gf") => Ok(GFConnectorFormat::Gf),
        Some("gfb") => Ok(GFConnectorFormat::Gfb),
        other => Err(GFError::InvalidConfig {
            message: format!(
                "unsupported GFConnector path extension {:?}; expected .gf or .gfb",
                other
            ),
        }),
    }
}

fn validate_batch_size(batch_size: usize) -> Result<()> {
    if batch_size == 0 {
        return Err(GFError::InvalidConfig {
            message: "batch_size must be greater than zero".to_owned(),
        });
    }
    Ok(())
}

fn validate_hops(hops: u32) -> Result<()> {
    if hops == 0 {
        return Err(GFError::InvalidConfig {
            message: "hops must be greater than zero".to_owned(),
        });
    }
    Ok(())
}

fn labels_to_predicate(labels: Option<&[&str]>) -> Option<Expr> {
    let labels = labels?;
    if labels.is_empty() {
        return Some(Expr::Literal {
            value: ScalarValue::Bool(false),
        });
    }

    labels
        .iter()
        .map(|label| Expr::ListContains {
            expr: Box::new(Expr::Col {
                name: COL_NODE_LABEL.to_owned(),
            }),
            item: Box::new(Expr::Literal {
                value: ScalarValue::String((*label).to_owned()),
            }),
        })
        .reduce(|left, right| Expr::Or {
            left: Box::new(left),
            right: Box::new(right),
        })
}

fn edge_types_to_predicate(edge_types: Option<&[&str]>) -> Option<Expr> {
    let edge_types = edge_types?;
    if edge_types.is_empty() {
        return Some(Expr::Literal {
            value: ScalarValue::Bool(false),
        });
    }

    edge_types
        .iter()
        .map(|edge_type| Expr::BinaryOp {
            left: Box::new(Expr::Col {
                name: COL_EDGE_TYPE.to_owned(),
            }),
            op: BinaryOp::Eq,
            right: Box::new(Expr::Literal {
                value: ScalarValue::String((*edge_type).to_owned()),
            }),
        })
        .reduce(|left, right| Expr::Or {
            left: Box::new(left),
            right: Box::new(right),
        })
}

fn node_ids_to_predicate(node_ids: &[&str]) -> Option<Expr> {
    if node_ids.is_empty() {
        return None;
    }

    node_ids
        .iter()
        .map(|node_id| Expr::BinaryOp {
            left: Box::new(Expr::Col {
                name: COL_NODE_ID.to_owned(),
            }),
            op: BinaryOp::Eq,
            right: Box::new(Expr::Literal {
                value: ScalarValue::String((*node_id).to_owned()),
            }),
        })
        .reduce(|left, right| Expr::Or {
            left: Box::new(left),
            right: Box::new(right),
        })
}

fn expand_graph(
    graph: &GraphFrame,
    node_ids: &[&str],
    edge_type: &EdgeTypeSpec,
    hops: u32,
    direction: Direction,
    node_predicate: Option<&Expr>,
) -> Result<ExpandResult> {
    if node_ids.is_empty() {
        let graph = graph.subgraph(&[])?;
        return Ok((graph.nodes().clone(), graph.edges().clone()));
    }

    let seed_predicate = node_ids_to_predicate(node_ids).ok_or_else(|| GFError::InvalidConfig {
        message: "expand requires at least one seed node id".to_owned(),
    })?;

    let expanded = LazyGraphFrame::from_graph(graph)
        .filter_nodes(seed_predicate)
        .expand(edge_type.clone(), hops, direction)
        .collect()?;

    let expanded = if let Some(predicate) = node_predicate {
        let filtered = LazyGraphFrame::from_graph(&expanded)
            .filter_nodes(predicate.clone())
            .collect_nodes()?;
        let retained_ids: Vec<&str> = filtered.id_column().iter().flatten().collect();
        expanded.subgraph(&retained_ids)?
    } else {
        expanded
    };

    Ok((expanded.nodes().clone(), expanded.edges().clone()))
}
