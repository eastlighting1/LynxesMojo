// Wasm-safe connectors (file I/O only, no network)
pub use self::gf_connector::{GFConnector, GFConnectorFormat};
pub use self::r#trait::{Connector, ConnectorFuture, ExpandResult};

mod gf_connector;
mod r#trait;

// Native-only connectors (require tokio, tonic, reqwest)
#[cfg(not(target_arch = "wasm32"))]
pub use self::arangodb_connector::{
    AqlBindVars, AqlQuery, AqlValue, ArangoBackend, ArangoConfig, ArangoConnector,
};
#[cfg(not(target_arch = "wasm32"))]
pub use self::flight_connector::{
    FlightAuth, FlightConfig, FlightConnector, FlightGraphService, FlightServerConfig,
    FlightTlsConfig,
};
#[cfg(not(target_arch = "wasm32"))]
pub use self::neo4j_connector::{
    CypherParams, CypherQuery, CypherValue, Neo4jBackend, Neo4jConfig, Neo4jConnector,
};
#[cfg(not(target_arch = "wasm32"))]
pub use self::sparql_connector::{
    SparqlBackend, SparqlConfig, SparqlConnector, SparqlParams, SparqlQuery, SparqlValue,
};

#[cfg(not(target_arch = "wasm32"))]
mod arangodb_connector;
#[cfg(not(target_arch = "wasm32"))]
mod flight_connector;
#[cfg(not(target_arch = "wasm32"))]
mod neo4j_connector;
#[cfg(not(target_arch = "wasm32"))]
mod sparql_connector;
