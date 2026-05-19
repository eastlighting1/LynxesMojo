use std::{future::Future, pin::Pin};

use crate::{Direction, EdgeFrame, EdgeTypeSpec, Expr, GFError, GraphFrame, NodeFrame, Result};

pub type ConnectorFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;

pub type ExpandResult = (NodeFrame, EdgeFrame);

/// Canonical connector contract for loading and persisting graph-shaped data.
///
/// This is intentionally object-safe so a logical `Scan` can hold a backend
/// source behind `Arc<dyn Connector>`.
pub trait Connector: Send + Sync + std::fmt::Debug {
    fn cache_source_key(&self) -> Option<String> {
        None
    }

    fn load_nodes<'a>(
        &'a self,
        labels: Option<&'a [&'a str]>,
        columns: Option<&'a [&'a str]>,
        predicate: Option<&'a Expr>,
        batch_size: usize,
    ) -> ConnectorFuture<'a, NodeFrame> {
        let _ = (labels, columns, predicate);
        Box::pin(async move {
            validate_batch_size(batch_size)?;
            Err(unsupported_operation("load_nodes"))
        })
    }

    fn load_edges<'a>(
        &'a self,
        edge_types: Option<&'a [&'a str]>,
        columns: Option<&'a [&'a str]>,
        predicate: Option<&'a Expr>,
        batch_size: usize,
    ) -> ConnectorFuture<'a, EdgeFrame> {
        let _ = (edge_types, columns, predicate);
        Box::pin(async move {
            validate_batch_size(batch_size)?;
            Err(unsupported_operation("load_edges"))
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
        let _ = (node_ids, edge_type, direction, node_predicate);
        Box::pin(async move {
            validate_hops(hops)?;
            Err(unsupported_operation("expand"))
        })
    }

    fn write<'a>(&'a self, graph: &'a GraphFrame) -> ConnectorFuture<'a, ()> {
        let _ = graph;
        Box::pin(async move { Err(unsupported_operation("write")) })
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

fn unsupported_operation(method: &str) -> GFError {
    GFError::UnsupportedOperation {
        message: format!("connector method {method} is not implemented"),
    }
}
