#![allow(clippy::result_large_err)]

pub mod connector;

#[cfg(not(target_arch = "wasm32"))]
pub use connector::{
    AqlBindVars, AqlQuery, AqlValue, ArangoBackend, ArangoConfig, ArangoConnector, CypherParams,
    CypherQuery, CypherValue, FlightAuth, FlightConfig, FlightConnector, FlightGraphService,
    FlightServerConfig, FlightTlsConfig, Neo4jBackend, Neo4jConfig, Neo4jConnector, SparqlBackend,
    SparqlConfig, SparqlConnector, SparqlParams, SparqlQuery, SparqlValue,
};
pub use connector::{Connector, ConnectorFuture, ExpandResult, GFConnector, GFConnectorFormat};
