use std::{
    collections::HashMap,
    future::Future,
    net::SocketAddr,
    path::{Path, PathBuf},
    pin::Pin,
    sync::{Arc, RwLock},
    time::Duration,
};

use arrow_array::RecordBatch;
use arrow_flight::{
    encode::FlightDataEncoderBuilder,
    error::FlightError,
    flight_descriptor::DescriptorType,
    flight_service_server::{FlightService, FlightServiceServer},
    utils::flight_data_to_batches,
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightEndpoint, FlightInfo,
    HandshakeRequest, HandshakeResponse, PollInfo, PutResult, SchemaAsIpc, SchemaResult, Ticket,
};
use arrow_ipc::writer::IpcWriteOptions;
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use futures::{stream, Stream, StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{
    transport::{Certificate, ClientTlsConfig, Endpoint, Identity, Server, ServerTlsConfig},
    Request, Response, Status,
};

use lynxes_core::{
    BinaryOp, Direction, EdgeFrame, EdgeTypeSpec, Expr, GFError, GraphFrame, NodeFrame, Result,
    ScalarValue, COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
};

use lynxes_lazy::LazyGraphFrame;

use crate::connector::{Connector, ConnectorFuture, ExpandResult};

type FlightDataStream = Pin<Box<dyn Stream<Item = std::result::Result<FlightData, Status>> + Send>>;
type PutResultStream = Pin<Box<dyn Stream<Item = std::result::Result<PutResult, Status>> + Send>>;
type ActionResultStream =
    Pin<Box<dyn Stream<Item = std::result::Result<arrow_flight::Result, Status>> + Send>>;
type FlightInfoStream = Pin<Box<dyn Stream<Item = std::result::Result<FlightInfo, Status>> + Send>>;
type ActionTypeStream = Pin<Box<dyn Stream<Item = std::result::Result<ActionType, Status>> + Send>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlightTlsConfig {
    pub domain_name: String,
    pub ca_cert_path: Option<PathBuf>,
    pub client_cert_path: Option<PathBuf>,
    pub client_key_path: Option<PathBuf>,
    pub server_cert_path: Option<PathBuf>,
    pub server_key_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlightAuth {
    Bearer { token: String },
    Basic { username: String, password: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlightConfig {
    pub endpoint: String,
    pub graph_name: String,
    pub tls: Option<FlightTlsConfig>,
    pub auth: Option<FlightAuth>,
    pub timeout: Duration,
}

impl FlightConfig {
    pub fn new(endpoint: impl Into<String>, graph_name: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            graph_name: graph_name.into(),
            tls: None,
            auth: None,
            timeout: Duration::from_secs(30),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct FlightServerConfig {
    pub tls: Option<FlightTlsConfig>,
    pub auth: Option<FlightAuth>,
    pub public_location: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FlightConnector {
    config: FlightConfig,
}

impl FlightConnector {
    pub fn new(config: FlightConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &FlightConfig {
        &self.config
    }

    async fn connect_client(&self) -> Result<arrow_flight::FlightClient> {
        let mut endpoint = Endpoint::from_shared(self.config.endpoint.clone()).map_err(|err| {
            GFError::ConnectorError {
                message: format!("invalid Flight endpoint {}: {err}", self.config.endpoint),
            }
        })?;
        endpoint = endpoint.timeout(self.config.timeout);

        if let Some(tls) = &self.config.tls {
            let mut tls_config = ClientTlsConfig::new().domain_name(tls.domain_name.clone());
            if let Some(path) = &tls.ca_cert_path {
                tls_config = tls_config.ca_certificate(load_certificate(path)?);
            }
            if let (Some(cert_path), Some(key_path)) = (&tls.client_cert_path, &tls.client_key_path)
            {
                tls_config = tls_config.identity(load_identity(cert_path, key_path)?);
            }
            endpoint = endpoint
                .tls_config(tls_config)
                .map_err(|err| GFError::ConnectorError {
                    message: format!("invalid Flight client TLS config: {err}"),
                })?;
        }

        let channel = endpoint
            .connect()
            .await
            .map_err(|err| GFError::ConnectorError {
                message: format!(
                    "failed to connect to Flight endpoint {}: {err}",
                    self.config.endpoint
                ),
            })?;
        let mut client = arrow_flight::FlightClient::new(channel);
        if let Some(auth) = &self.config.auth {
            client
                .add_header("authorization", &auth_header_value(auth))
                .map_err(|err| GFError::ConnectorError {
                    message: format!("failed to attach Flight auth header: {err}"),
                })?;
        }
        Ok(client)
    }

    async fn fetch_nodes(
        &self,
        labels: Option<Vec<String>>,
        columns: Option<Vec<String>>,
        predicate: Option<Expr>,
    ) -> Result<NodeFrame> {
        let request = FlightTicketRequest::LoadNodes {
            graph: self.config.graph_name.clone(),
            labels,
            columns,
            predicate,
        };
        let batch = self.fetch_single_batch(request).await?;
        NodeFrame::from_record_batch(batch)
    }

    async fn fetch_edges(
        &self,
        edge_types: Option<Vec<String>>,
        columns: Option<Vec<String>>,
        predicate: Option<Expr>,
    ) -> Result<EdgeFrame> {
        let request = FlightTicketRequest::LoadEdges {
            graph: self.config.graph_name.clone(),
            edge_types,
            columns,
            predicate,
        };
        let batch = self.fetch_single_batch(request).await?;
        EdgeFrame::from_record_batch(batch)
    }

    async fn fetch_expand(
        &self,
        dataset: FlightDataset,
        node_ids: Vec<String>,
        edge_type: EdgeTypeSpec,
        hops: u32,
        direction: Direction,
        node_predicate: Option<Expr>,
    ) -> Result<RecordBatch> {
        let request = match dataset {
            FlightDataset::Nodes => FlightTicketRequest::ExpandNodes {
                graph: self.config.graph_name.clone(),
                node_ids,
                edge_type: FlightEdgeTypeSpec::from(&edge_type),
                hops,
                direction: FlightDirection::from(direction),
                node_predicate,
            },
            FlightDataset::Edges => FlightTicketRequest::ExpandEdges {
                graph: self.config.graph_name.clone(),
                node_ids,
                edge_type: FlightEdgeTypeSpec::from(&edge_type),
                hops,
                direction: FlightDirection::from(direction),
                node_predicate,
            },
        };
        self.fetch_single_batch(request).await
    }

    async fn fetch_single_batch(&self, request: FlightTicketRequest) -> Result<RecordBatch> {
        let ticket =
            Ticket::new(
                serde_json::to_vec(&request).map_err(|err| GFError::ConnectorError {
                    message: format!("failed to encode Flight ticket request: {err}"),
                })?,
            );
        let mut client = self.connect_client().await?;
        let batches: Vec<RecordBatch> = client
            .do_get(ticket)
            .await
            .map_err(flight_client_error)?
            .try_collect()
            .await
            .map_err(flight_client_error)?;

        match batches.len() {
            0 => Err(GFError::ConnectorError {
                message: "Flight do_get returned no record batches".to_owned(),
            }),
            1 => Ok(batches.into_iter().next().unwrap()),
            _ => Err(GFError::ConnectorError {
                message: "Flight do_get returned multiple batches for a single-frame request"
                    .to_owned(),
            }),
        }
    }

    async fn put_batch(&self, dataset: FlightDataset, batch: RecordBatch) -> Result<()> {
        let descriptor = FlightDescriptor::new_path(vec![
            "lynxes".to_owned(),
            self.config.graph_name.clone(),
            dataset.as_str().to_owned(),
        ]);
        let schema = batch.schema();
        let input = stream::iter(vec![Ok::<RecordBatch, FlightError>(batch)]);
        let flight_data = FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .with_flight_descriptor(Some(descriptor))
            .build(input);

        let mut client = self.connect_client().await?;
        let mut response = client
            .do_put(flight_data)
            .await
            .map_err(flight_client_error)?;
        while let Some(_ack) = response.try_next().await.map_err(flight_client_error)? {}
        Ok(())
    }
}

impl Connector for FlightConnector {
    fn cache_source_key(&self) -> Option<String> {
        Some(format!(
            "flight://{}#{}",
            self.config.endpoint, self.config.graph_name
        ))
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
            self.fetch_nodes(
                labels.map(|labels| labels.iter().map(|value| (*value).to_owned()).collect()),
                columns.map(|columns| columns.iter().map(|value| (*value).to_owned()).collect()),
                predicate.cloned(),
            )
            .await
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
            self.fetch_edges(
                edge_types.map(|values| values.iter().map(|value| (*value).to_owned()).collect()),
                columns.map(|columns| columns.iter().map(|value| (*value).to_owned()).collect()),
                predicate.cloned(),
            )
            .await
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
            let node_ids: Vec<String> = node_ids.iter().map(|value| (*value).to_owned()).collect();
            let nodes = NodeFrame::from_record_batch(
                self.fetch_expand(
                    FlightDataset::Nodes,
                    node_ids.clone(),
                    edge_type.clone(),
                    hops,
                    direction,
                    node_predicate.cloned(),
                )
                .await?,
            )?;
            let edges = EdgeFrame::from_record_batch(
                self.fetch_expand(
                    FlightDataset::Edges,
                    node_ids,
                    edge_type.clone(),
                    hops,
                    direction,
                    node_predicate.cloned(),
                )
                .await?,
            )?;
            Ok((nodes, edges))
        })
    }

    fn write<'a>(&'a self, graph: &'a GraphFrame) -> ConnectorFuture<'a, ()> {
        Box::pin(async move {
            self.put_batch(
                FlightDataset::Nodes,
                graph.nodes().to_record_batch().clone(),
            )
            .await?;
            self.put_batch(
                FlightDataset::Edges,
                graph.edges().to_record_batch().clone(),
            )
            .await?;
            Ok(())
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct FlightGraphService {
    graphs: Arc<RwLock<HashMap<String, GraphFrame>>>,
    staged_uploads: Arc<RwLock<HashMap<String, StagedUpload>>>,
    config: FlightServerConfig,
}

impl FlightGraphService {
    pub fn new() -> Self {
        Self::with_config(FlightServerConfig::default())
    }

    pub fn with_config(config: FlightServerConfig) -> Self {
        Self {
            graphs: Arc::new(RwLock::new(HashMap::new())),
            staged_uploads: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    pub fn insert_graph(&self, name: impl Into<String>, graph: GraphFrame) -> Option<GraphFrame> {
        self.graphs.write().unwrap().insert(name.into(), graph)
    }

    pub fn graph(&self, name: &str) -> Option<GraphFrame> {
        self.graphs.read().unwrap().get(name).cloned()
    }

    pub fn graph_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.graphs.read().unwrap().keys().cloned().collect();
        names.sort();
        names
    }

    pub async fn serve(self, addr: SocketAddr) -> Result<()> {
        let mut builder = Server::builder();
        if let Some(tls) = &self.config.tls {
            builder = builder.tls_config(server_tls_config(tls)?).map_err(|err| {
                GFError::ConnectorError {
                    message: format!("invalid Flight server TLS config: {err}"),
                }
            })?;
        }

        builder
            .add_service(FlightServiceServer::new(self))
            .serve(addr)
            .await
            .map_err(|err| GFError::ConnectorError {
                message: format!("Flight server error: {err}"),
            })
    }

    pub async fn serve_with_shutdown<F>(self, addr: SocketAddr, shutdown: F) -> Result<()>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let mut builder = Server::builder();
        if let Some(tls) = &self.config.tls {
            builder = builder.tls_config(server_tls_config(tls)?).map_err(|err| {
                GFError::ConnectorError {
                    message: format!("invalid Flight server TLS config: {err}"),
                }
            })?;
        }

        builder
            .add_service(FlightServiceServer::new(self))
            .serve_with_shutdown(addr, shutdown)
            .await
            .map_err(|err| GFError::ConnectorError {
                message: format!("Flight server error: {err}"),
            })
    }

    pub async fn serve_listener<F>(self, listener: TcpListener, shutdown: F) -> Result<()>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let mut builder = Server::builder();
        if let Some(tls) = &self.config.tls {
            builder = builder.tls_config(server_tls_config(tls)?).map_err(|err| {
                GFError::ConnectorError {
                    message: format!("invalid Flight server TLS config: {err}"),
                }
            })?;
        }

        builder
            .add_service(FlightServiceServer::new(self))
            .serve_with_incoming_shutdown(TcpListenerStream::new(listener), shutdown)
            .await
            .map_err(|err| GFError::ConnectorError {
                message: format!("Flight server error: {err}"),
            })
    }

    fn require_authorized<T>(&self, request: &Request<T>) -> std::result::Result<(), Status> {
        let Some(expected) = self.config.auth.as_ref() else {
            return Ok(());
        };

        let actual = request
            .metadata()
            .get("authorization")
            .and_then(|value| value.to_str().ok())
            .ok_or_else(|| Status::unauthenticated("missing authorization header"))?;
        if actual != auth_header_value(expected) {
            return Err(Status::unauthenticated("invalid authorization header"));
        }
        Ok(())
    }

    fn graph_or_not_found(&self, graph: &str) -> std::result::Result<GraphFrame, Status> {
        self.graph(graph)
            .ok_or_else(|| Status::not_found(format!("graph not found: {graph}")))
    }
}

#[tonic::async_trait]
impl FlightService for FlightGraphService {
    type HandshakeStream = Pin<
        Box<dyn Stream<Item = std::result::Result<HandshakeResponse, Status>> + Send + 'static>,
    >;
    type ListFlightsStream = FlightInfoStream;
    type DoGetStream = FlightDataStream;
    type DoPutStream = PutResultStream;
    type DoExchangeStream = FlightDataStream;
    type DoActionStream = ActionResultStream;
    type ListActionsStream = ActionTypeStream;

    async fn handshake(
        &self,
        request: Request<tonic::Streaming<HandshakeRequest>>,
    ) -> std::result::Result<Response<Self::HandshakeStream>, Status> {
        self.require_authorized(&request)?;
        let mut input = request.into_inner();
        let first = input.message().await?.unwrap_or(HandshakeRequest {
            protocol_version: 0,
            payload: Default::default(),
        });
        let response = HandshakeResponse {
            protocol_version: first.protocol_version,
            payload: first.payload,
        };
        Ok(Response::new(Box::pin(stream::iter(vec![Ok(response)]))))
    }

    async fn list_flights(
        &self,
        request: Request<Criteria>,
    ) -> std::result::Result<Response<Self::ListFlightsStream>, Status> {
        self.require_authorized(&request)?;
        let infos: Vec<_> = self
            .graph_names()
            .into_iter()
            .flat_map(|graph| {
                [
                    self.flight_info(&graph, FlightDataset::Nodes),
                    self.flight_info(&graph, FlightDataset::Edges),
                ]
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Response::new(Box::pin(stream::iter(
            infos.into_iter().map(Ok),
        ))))
    }

    async fn get_flight_info(
        &self,
        request: Request<FlightDescriptor>,
    ) -> std::result::Result<Response<FlightInfo>, Status> {
        self.require_authorized(&request)?;
        let (graph, dataset) = parse_descriptor(request.get_ref())?;
        Ok(Response::new(self.flight_info(&graph, dataset)?))
    }

    async fn poll_flight_info(
        &self,
        request: Request<FlightDescriptor>,
    ) -> std::result::Result<Response<PollInfo>, Status> {
        self.require_authorized(&request)?;
        let info = self
            .get_flight_info(Request::new(request.into_inner()))
            .await?
            .into_inner();
        Ok(Response::new(PollInfo::new().with_info(info)))
    }

    async fn get_schema(
        &self,
        request: Request<FlightDescriptor>,
    ) -> std::result::Result<Response<SchemaResult>, Status> {
        self.require_authorized(&request)?;
        let (graph_name, dataset) = parse_descriptor(request.get_ref())?;
        let graph = self.graph_or_not_found(&graph_name)?;
        let schema = match dataset {
            FlightDataset::Nodes => (*graph.nodes().schema()).clone(),
            FlightDataset::Edges => (*graph.edges().schema()).clone(),
        };
        let schema = SchemaAsIpc::new(&schema, &IpcWriteOptions::default())
            .try_into()
            .map_err(arrow_status)?;
        Ok(Response::new(schema))
    }

    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> std::result::Result<Response<Self::DoGetStream>, Status> {
        self.require_authorized(&request)?;
        let ticket = parse_ticket(request.get_ref())?;
        let batch = match ticket {
            FlightTicketRequest::LoadNodes {
                graph,
                labels,
                columns,
                predicate,
            } => load_nodes_from_graph(
                &self.graph_or_not_found(&graph)?,
                labels.as_deref(),
                columns.as_deref(),
                predicate.as_ref(),
            )
            .map_err(gf_status)?
            .to_record_batch()
            .clone(),
            FlightTicketRequest::LoadEdges {
                graph,
                edge_types,
                columns,
                predicate,
            } => load_edges_from_graph(
                &self.graph_or_not_found(&graph)?,
                edge_types.as_deref(),
                columns.as_deref(),
                predicate.as_ref(),
            )
            .map_err(gf_status)?
            .to_record_batch()
            .clone(),
            FlightTicketRequest::ExpandNodes {
                graph,
                node_ids,
                edge_type,
                hops,
                direction,
                node_predicate,
            } => expand_graph(
                &self.graph_or_not_found(&graph)?,
                &node_ids.iter().map(String::as_str).collect::<Vec<_>>(),
                &EdgeTypeSpec::from(edge_type),
                hops,
                Direction::from(direction),
                node_predicate.as_ref(),
            )
            .map_err(gf_status)?
            .0
            .to_record_batch()
            .clone(),
            FlightTicketRequest::ExpandEdges {
                graph,
                node_ids,
                edge_type,
                hops,
                direction,
                node_predicate,
            } => expand_graph(
                &self.graph_or_not_found(&graph)?,
                &node_ids.iter().map(String::as_str).collect::<Vec<_>>(),
                &EdgeTypeSpec::from(edge_type),
                hops,
                Direction::from(direction),
                node_predicate.as_ref(),
            )
            .map_err(gf_status)?
            .1
            .to_record_batch()
            .clone(),
        };

        let stream = FlightDataEncoderBuilder::new()
            .with_schema(batch.schema())
            .build(stream::iter(vec![Ok::<RecordBatch, FlightError>(batch)]))
            .map(|item| item.map_err(flight_status));
        Ok(Response::new(Box::pin(stream)))
    }

    async fn do_put(
        &self,
        request: Request<tonic::Streaming<FlightData>>,
    ) -> std::result::Result<Response<Self::DoPutStream>, Status> {
        self.require_authorized(&request)?;
        let mut input = request.into_inner();
        let mut messages = Vec::new();
        while let Some(message) = input.message().await? {
            messages.push(message);
        }
        if messages.is_empty() {
            return Err(Status::invalid_argument(
                "do_put requires at least one FlightData message",
            ));
        }

        let descriptor = messages[0]
            .flight_descriptor
            .clone()
            .ok_or_else(|| Status::invalid_argument("Flight do_put missing descriptor"))?;
        let (graph_name, dataset) = parse_descriptor(&descriptor)?;
        let batches = flight_data_to_batches(&messages).map_err(arrow_status)?;
        let batch = batches.into_iter().next().ok_or_else(|| {
            Status::invalid_argument("Flight do_put did not contain a record batch payload")
        })?;

        let mut staged = self.staged_uploads.write().unwrap();
        let entry = staged.entry(graph_name.clone()).or_default();
        match dataset {
            FlightDataset::Nodes => {
                entry.nodes = Some(NodeFrame::from_record_batch(batch).map_err(gf_status)?)
            }
            FlightDataset::Edges => {
                entry.edges = Some(EdgeFrame::from_record_batch(batch).map_err(gf_status)?)
            }
        }
        if let (Some(nodes), Some(edges)) = (&entry.nodes, &entry.edges) {
            let graph = GraphFrame::new(nodes.clone(), edges.clone()).map_err(gf_status)?;
            self.graphs
                .write()
                .unwrap()
                .insert(graph_name.clone(), graph);
            staged.remove(&graph_name);
        }

        let ack = PutResult {
            app_metadata: format!("stored {} for {}", dataset.as_str(), graph_name)
                .into_bytes()
                .into(),
        };
        Ok(Response::new(Box::pin(stream::iter(vec![Ok(ack)]))))
    }

    async fn do_exchange(
        &self,
        request: Request<tonic::Streaming<FlightData>>,
    ) -> std::result::Result<Response<Self::DoExchangeStream>, Status> {
        self.require_authorized(&request)?;
        Err(Status::unimplemented(
            "Flight do_exchange is not implemented",
        ))
    }

    async fn do_action(
        &self,
        request: Request<Action>,
    ) -> std::result::Result<Response<Self::DoActionStream>, Status> {
        self.require_authorized(&request)?;
        let action = request.into_inner();
        match action.r#type.as_str() {
            "delete_graph" => {
                let graph_name = std::str::from_utf8(&action.body)
                    .map_err(|_| Status::invalid_argument("delete_graph body must be utf8"))?;
                self.graphs.write().unwrap().remove(graph_name);
                Ok(Response::new(Box::pin(stream::iter(vec![Ok(
                    arrow_flight::Result::new(format!("deleted:{graph_name}").into_bytes()),
                )]))))
            }
            _ => Err(Status::unimplemented(format!(
                "unsupported Flight action {}",
                action.r#type
            ))),
        }
    }

    async fn list_actions(
        &self,
        request: Request<Empty>,
    ) -> std::result::Result<Response<Self::ListActionsStream>, Status> {
        self.require_authorized(&request)?;
        let actions = vec![Ok(ActionType {
            r#type: "delete_graph".to_owned(),
            description: "Delete a stored graph by name".to_owned(),
        })];
        Ok(Response::new(Box::pin(stream::iter(actions))))
    }
}

impl FlightGraphService {
    fn flight_info(
        &self,
        graph_name: &str,
        dataset: FlightDataset,
    ) -> std::result::Result<FlightInfo, Status> {
        let graph = self.graph_or_not_found(graph_name)?;
        let batch = match dataset {
            FlightDataset::Nodes => graph.nodes().to_record_batch(),
            FlightDataset::Edges => graph.edges().to_record_batch(),
        };
        let descriptor = FlightDescriptor::new_path(vec![
            "lynxes".to_owned(),
            graph_name.to_owned(),
            dataset.as_str().to_owned(),
        ]);
        let ticket_request = match dataset {
            FlightDataset::Nodes => FlightTicketRequest::LoadNodes {
                graph: graph_name.to_owned(),
                labels: None,
                columns: None,
                predicate: None,
            },
            FlightDataset::Edges => FlightTicketRequest::LoadEdges {
                graph: graph_name.to_owned(),
                edge_types: None,
                columns: None,
                predicate: None,
            },
        };
        let ticket =
            Ticket::new(serde_json::to_vec(&ticket_request).map_err(|err| {
                Status::internal(format!("failed to encode Flight ticket: {err}"))
            })?);

        let mut endpoint = FlightEndpoint::new().with_ticket(ticket);
        if let Some(location) = &self.config.public_location {
            endpoint = endpoint.with_location(location.clone());
        }

        FlightInfo::new()
            .try_with_schema(batch.schema().as_ref())
            .map_err(arrow_status)
            .map(|info| {
                info.with_descriptor(descriptor)
                    .with_endpoint(endpoint)
                    .with_total_records(batch.num_rows() as i64)
                    .with_total_bytes(-1)
            })
    }
}

#[derive(Debug, Clone, Default)]
struct StagedUpload {
    nodes: Option<NodeFrame>,
    edges: Option<EdgeFrame>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FlightDataset {
    Nodes,
    Edges,
}

impl FlightDataset {
    fn as_str(self) -> &'static str {
        match self {
            Self::Nodes => "nodes",
            Self::Edges => "edges",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FlightDirection {
    Out,
    In,
    Both,
    None,
}

impl From<Direction> for FlightDirection {
    fn from(value: Direction) -> Self {
        match value {
            Direction::Out => Self::Out,
            Direction::In => Self::In,
            Direction::Both => Self::Both,
            Direction::None => Self::None,
        }
    }
}

impl From<FlightDirection> for Direction {
    fn from(value: FlightDirection) -> Self {
        match value {
            FlightDirection::Out => Self::Out,
            FlightDirection::In => Self::In,
            FlightDirection::Both => Self::Both,
            FlightDirection::None => Self::None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum FlightEdgeTypeSpec {
    Single { value: String },
    Multiple { values: Vec<String> },
    Any,
}

impl From<&EdgeTypeSpec> for FlightEdgeTypeSpec {
    fn from(value: &EdgeTypeSpec) -> Self {
        match value {
            EdgeTypeSpec::Single(value) => Self::Single {
                value: value.clone(),
            },
            EdgeTypeSpec::Multiple(values) => Self::Multiple {
                values: values.clone(),
            },
            EdgeTypeSpec::Any => Self::Any,
        }
    }
}

impl From<FlightEdgeTypeSpec> for EdgeTypeSpec {
    fn from(value: FlightEdgeTypeSpec) -> Self {
        match value {
            FlightEdgeTypeSpec::Single { value } => Self::Single(value),
            FlightEdgeTypeSpec::Multiple { values } => Self::Multiple(values),
            FlightEdgeTypeSpec::Any => Self::Any,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum FlightTicketRequest {
    LoadNodes {
        graph: String,
        labels: Option<Vec<String>>,
        columns: Option<Vec<String>>,
        predicate: Option<Expr>,
    },
    LoadEdges {
        graph: String,
        edge_types: Option<Vec<String>>,
        columns: Option<Vec<String>>,
        predicate: Option<Expr>,
    },
    ExpandNodes {
        graph: String,
        node_ids: Vec<String>,
        edge_type: FlightEdgeTypeSpec,
        hops: u32,
        direction: FlightDirection,
        node_predicate: Option<Expr>,
    },
    ExpandEdges {
        graph: String,
        node_ids: Vec<String>,
        edge_type: FlightEdgeTypeSpec,
        hops: u32,
        direction: FlightDirection,
        node_predicate: Option<Expr>,
    },
}

fn parse_ticket(ticket: &Ticket) -> std::result::Result<FlightTicketRequest, Status> {
    serde_json::from_slice(&ticket.ticket)
        .map_err(|err| Status::invalid_argument(format!("invalid Flight ticket payload: {err}")))
}

fn parse_descriptor(
    descriptor: &FlightDescriptor,
) -> std::result::Result<(String, FlightDataset), Status> {
    if descriptor.r#type() != DescriptorType::Path {
        return Err(Status::invalid_argument(
            "Flight descriptor must use path form for lynxes service",
        ));
    }
    if descriptor.path.len() != 3 || descriptor.path[0] != "lynxes" {
        return Err(Status::invalid_argument(format!(
            "invalid Flight descriptor path {:?}",
            descriptor.path
        )));
    }
    let dataset = match descriptor.path[2].as_str() {
        "nodes" => FlightDataset::Nodes,
        "edges" => FlightDataset::Edges,
        other => {
            return Err(Status::invalid_argument(format!(
                "unsupported Flight descriptor dataset {other}"
            )))
        }
    };
    Ok((descriptor.path[1].clone(), dataset))
}

fn load_nodes_from_graph(
    graph: &GraphFrame,
    labels: Option<&[String]>,
    columns: Option<&[String]>,
    predicate: Option<&Expr>,
) -> Result<NodeFrame> {
    let mut lazy = LazyGraphFrame::from_graph(graph);
    if let Some(label_predicate) = labels_to_predicate(labels) {
        lazy = lazy.filter_nodes(label_predicate);
    }
    if let Some(predicate) = predicate {
        lazy = lazy.filter_nodes(predicate.clone());
    }
    if let Some(columns) = columns {
        lazy = lazy.select_nodes(columns.to_vec());
    }
    lazy.collect_nodes()
}

fn load_edges_from_graph(
    graph: &GraphFrame,
    edge_types: Option<&[String]>,
    columns: Option<&[String]>,
    predicate: Option<&Expr>,
) -> Result<EdgeFrame> {
    let mut lazy = LazyGraphFrame::from_graph(graph);
    if let Some(edge_predicate) = edge_types_to_predicate(edge_types) {
        lazy = lazy.filter_edges(edge_predicate);
    }
    if let Some(predicate) = predicate {
        lazy = lazy.filter_edges(predicate.clone());
    }
    if let Some(columns) = columns {
        lazy = lazy.select_edges(columns.to_vec());
    }
    lazy.collect_edges()
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

fn labels_to_predicate(labels: Option<&[String]>) -> Option<Expr> {
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
                value: ScalarValue::String(label.clone()),
            }),
        })
        .reduce(|left, right| Expr::Or {
            left: Box::new(left),
            right: Box::new(right),
        })
}

fn edge_types_to_predicate(edge_types: Option<&[String]>) -> Option<Expr> {
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
                value: ScalarValue::String(edge_type.clone()),
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

fn auth_header_value(auth: &FlightAuth) -> String {
    match auth {
        FlightAuth::Bearer { token } => format!("Bearer {token}"),
        FlightAuth::Basic { username, password } => {
            let payload = BASE64_STANDARD.encode(format!("{username}:{password}"));
            format!("Basic {payload}")
        }
    }
}

fn load_certificate(path: &Path) -> Result<Certificate> {
    let pem = std::fs::read(path)?;
    Ok(Certificate::from_pem(pem))
}

fn load_identity(cert_path: &Path, key_path: &Path) -> Result<Identity> {
    let cert = std::fs::read(cert_path)?;
    let key = std::fs::read(key_path)?;
    Ok(Identity::from_pem(cert, key))
}

fn server_tls_config(config: &FlightTlsConfig) -> Result<ServerTlsConfig> {
    let cert_path = config
        .server_cert_path
        .as_ref()
        .ok_or_else(|| GFError::InvalidConfig {
            message: "Flight server TLS requires server_cert_path".to_owned(),
        })?;
    let key_path = config
        .server_key_path
        .as_ref()
        .ok_or_else(|| GFError::InvalidConfig {
            message: "Flight server TLS requires server_key_path".to_owned(),
        })?;
    let identity = load_identity(cert_path, key_path)?;
    let mut tls = ServerTlsConfig::new().identity(identity);
    if let Some(ca_path) = &config.ca_cert_path {
        tls = tls.client_ca_root(load_certificate(ca_path)?);
    }
    Ok(tls)
}

fn flight_status(err: FlightError) -> Status {
    Status::internal(err.to_string())
}

fn flight_client_error(err: FlightError) -> GFError {
    GFError::ConnectorError {
        message: err.to_string(),
    }
}

fn arrow_status(err: impl std::fmt::Display) -> Status {
    Status::internal(err.to_string())
}

fn gf_status(err: GFError) -> Status {
    Status::internal(err.to_string())
}
