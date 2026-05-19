//! Lynxes — graph-native analytics engine built on Apache Arrow.
//!
//! This is the stable public umbrella crate.
//! All user-facing types and functions are re-exported from here.
//! Internal implementation crates (`lynxes-core`, `lynxes-plan`, etc.)
//! are not part of the stable API and may change without notice.

pub use lynxes_core::*;
pub use lynxes_io::*;
pub use lynxes_lazy::LazyGraphFrame;

#[cfg(not(target_arch = "wasm32"))]
pub use lynxes_connect::{
    AqlBindVars, AqlQuery, AqlValue, ArangoBackend, ArangoConfig, ArangoConnector, CypherParams,
    CypherQuery, CypherValue, FlightAuth, FlightConfig, FlightConnector, FlightGraphService,
    FlightServerConfig, FlightTlsConfig, Neo4jBackend, Neo4jConfig, Neo4jConnector, SparqlBackend,
    SparqlConfig, SparqlConnector, SparqlParams, SparqlQuery, SparqlValue,
};
pub use lynxes_connect::{GFConnector, GFConnectorFormat};
