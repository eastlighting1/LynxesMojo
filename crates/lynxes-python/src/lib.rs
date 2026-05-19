#![allow(clippy::useless_conversion, clippy::wrong_self_convention)]

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, HashMap},
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::{
    array::{
        make_array, Array, ArrayData, ArrayRef, BooleanArray, BooleanBuilder, Float32Array,
        Float64Array, Float64Builder, Int16Array, Int32Array, Int64Array, Int64Builder, Int8Array,
        LargeStringArray, ListArray, ListBuilder, StringArray, StringBuilder, StringViewArray,
        UInt32Array, UInt64Array,
    },
    datatypes::{DataType, Field, Fields, Schema},
    pyarrow::{PyArrowType, ToPyArrow},
    record_batch::RecordBatch,
};
use lynxes_connect::{
    ArangoConfig, ArangoConnector, Connector as BackendConnector, Neo4jConfig, Neo4jConnector,
    SparqlConfig, SparqlConnector,
};
use lynxes_core::{
    configure_mojo_runtime, AttrStatsSummary, BetweennessConfig, Connector, Direction,
    DisplayOptions, DisplaySlice, DisplayView, EdgeFrame, EdgeTypeSpec, Expr, GFError,
    GlimpseSummary, GraphFrame, GraphInfo, GraphPartitionMethod, GraphPartitioner,
    MutableGraphFrame, NodeFrame, PageRankConfig, PartitionedGraph, SampledSubgraph,
    SamplingConfig, SchemaSummary, ShortestPathConfig, StructureStats, COL_EDGE_DIRECTION,
    COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE, COL_NODE_LABEL, EDGE_RESERVED_COLUMNS,
    NODE_RESERVED_COLUMNS,
};
use lynxes_io::{
    parse_gf, read_csv_nodes, read_gfb, read_parquet_graph, write_gf as core_write_gf, write_gfb,
    write_parquet_graph, CsvNodeReadOptions,
};
use lynxes_lazy::LazyGraphFrame;
use lynxes_plan::{
    AggExpr, BinaryOp, Pattern, PatternNodeConstraint, PatternStep, PatternStepConstraint,
    ScalarValue, StringOp, UnaryOp,
};
use pyo3::{
    basic::CompareOp,
    exceptions::{
        PyImportError, PyIndexError, PyKeyError, PyNotImplementedError, PyOSError, PyRuntimeError,
        PyTypeError, PyValueError,
    },
    prelude::*,
    types::{PyAny, PyBytes, PyDict, PyList, PyTuple, PyType},
    wrap_pyfunction,
};
#[pyclass(name = "NodeFrame", module = "lynxes")]
#[derive(Clone)]
struct PyNodeFrame {
    inner: Arc<NodeFrame>,
}

#[pyclass(name = "EdgeFrame", module = "lynxes")]
#[derive(Clone)]
struct PyEdgeFrame {
    inner: Arc<EdgeFrame>,
    node_ids: Arc<Vec<String>>,
}

#[pyclass(name = "GraphFrame", module = "lynxes")]
#[derive(Clone)]
struct PyGraphFrame {
    inner: Arc<GraphFrame>,
}

#[pyclass(name = "MutableGraphFrame", module = "lynxes")]
struct PyMutableGraphFrame {
    inner: Option<MutableGraphFrame>,
}

#[pyclass(name = "LazyGraphFrame", module = "lynxes")]
#[derive(Clone)]
struct PyLazyGraphFrame {
    inner: LazyGraphFrame,
}

#[pyclass(name = "Expr", module = "lynxes")]
#[derive(Clone)]
struct PyExpr {
    inner: Expr,
}

#[pyclass(name = "AggExpr", module = "lynxes")]
#[derive(Clone)]
struct PyAggExpr {
    inner: AggExpr,
}

#[pyclass(name = "PartitionedGraph", module = "lynxes")]
#[derive(Clone)]
struct PyPartitionedGraph {
    inner: PartitionedGraph,
}

impl PyPartitionedGraph {
    fn new(inner: PartitionedGraph) -> Self {
        Self { inner }
    }
}

/// Namespace returned by `expr.str` — provides string predicate builders.
#[pyclass(name = "StringExprNamespace", module = "lynxes")]
#[derive(Clone)]
struct PyStrExprNamespace {
    inner: Expr,
}

#[pyclass(name = "PatternNode", module = "lynxes")]
#[derive(Clone)]
struct PyPatternNode {
    alias: String,
    label: Option<String>,
    props: Vec<String>,
}

#[pyclass(name = "PatternEdge", module = "lynxes")]
#[derive(Clone)]
struct PyPatternEdge {
    alias: Option<String>,
    edge_type: Option<String>,
    optional: bool,
    min_hops: u32,
    max_hops: Option<u32>,
}

#[pyclass(name = "SampledSubgraph", module = "lynxes")]
#[derive(Clone)]
struct PySampledSubgraph {
    inner: SampledSubgraph,
}

impl PyNodeFrame {
    fn new(inner: NodeFrame) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }

    fn from_arc(inner: Arc<NodeFrame>) -> Self {
        Self { inner }
    }

    fn to_arrow_impl(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.inner
            .to_record_batch()
            .clone()
            .to_pyarrow(py)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    fn render_display_view(
        &self,
        view: FramePreviewView,
        rows: usize,
        sort_by: Option<String>,
        descending: bool,
        width: Option<usize>,
    ) -> PyResult<String> {
        render_frame_preview(
            &format!(
                "NodeFrame(rows={}, columns={}, order={})",
                self.inner.len(),
                self.inner.schema().fields().len(),
                frame_order_name(sort_by.is_some(), descending)
            ),
            self.inner.to_record_batch(),
            view,
            rows.max(1),
            sort_by,
            descending,
            width,
        )
    }
}

impl PyEdgeFrame {
    fn new(inner: EdgeFrame) -> Self {
        let node_ids = Arc::new(build_edge_node_ids(&inner));
        Self {
            inner: Arc::new(inner),
            node_ids,
        }
    }

    fn from_arc(inner: Arc<EdgeFrame>) -> Self {
        let node_ids = Arc::new(build_edge_node_ids(inner.as_ref()));
        Self { inner, node_ids }
    }

    fn to_arrow_impl(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.inner
            .to_record_batch()
            .clone()
            .to_pyarrow(py)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    fn render_display_view(
        &self,
        view: FramePreviewView,
        rows: usize,
        sort_by: Option<String>,
        descending: bool,
        width: Option<usize>,
    ) -> PyResult<String> {
        render_frame_preview(
            &format!(
                "EdgeFrame(rows={}, columns={}, nodes={}, order={})",
                self.inner.len(),
                self.inner.schema().fields().len(),
                self.inner.node_count(),
                frame_order_name(sort_by.is_some(), descending)
            ),
            self.inner.to_record_batch(),
            view,
            rows.max(1),
            sort_by,
            descending,
            width,
        )
    }
}

impl PyGraphFrame {
    fn new(inner: GraphFrame) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}

impl PyLazyGraphFrame {
    fn new(inner: LazyGraphFrame) -> Self {
        Self { inner }
    }
}

impl PyMutableGraphFrame {
    fn new(inner: MutableGraphFrame) -> Self {
        Self { inner: Some(inner) }
    }

    fn inner_mut(&mut self) -> PyResult<&mut MutableGraphFrame> {
        self.inner.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err(
                "MutableGraphFrame has already been frozen and can no longer be used",
            )
        })
    }
}

impl PySampledSubgraph {
    fn new(inner: SampledSubgraph) -> Self {
        Self { inner }
    }
}

#[derive(Debug, Clone)]
struct BackendConnectorAdapter {
    inner: Arc<dyn BackendConnector>,
}

impl Connector for BackendConnectorAdapter {
    fn cache_source_key(&self) -> Option<String> {
        self.inner.cache_source_key()
    }

    fn load_nodes<'a>(
        &'a self,
        labels: Option<&'a [&'a str]>,
        columns: Option<&'a [&'a str]>,
        predicate: Option<&'a Expr>,
        batch_size: usize,
    ) -> lynxes_core::ConnectorFuture<'a, NodeFrame> {
        self.inner
            .load_nodes(labels, columns, predicate, batch_size)
    }

    fn load_edges<'a>(
        &'a self,
        edge_types: Option<&'a [&'a str]>,
        columns: Option<&'a [&'a str]>,
        predicate: Option<&'a Expr>,
        batch_size: usize,
    ) -> lynxes_core::ConnectorFuture<'a, EdgeFrame> {
        self.inner
            .load_edges(edge_types, columns, predicate, batch_size)
    }

    fn expand<'a>(
        &'a self,
        node_ids: &'a [&'a str],
        edge_type: &'a EdgeTypeSpec,
        hops: u32,
        direction: Direction,
        node_predicate: Option<&'a Expr>,
    ) -> lynxes_core::ConnectorFuture<'a, lynxes_core::ExpandResult> {
        self.inner
            .expand(node_ids, edge_type, hops, direction, node_predicate)
    }

    fn write<'a>(&'a self, graph: &'a GraphFrame) -> lynxes_core::ConnectorFuture<'a, ()> {
        self.inner.write(graph)
    }
}

impl PyExpr {
    fn new(inner: Expr) -> Self {
        Self { inner }
    }

    fn binary(&self, other: &Bound<'_, PyAny>, op: BinaryOp) -> PyResult<Self> {
        Ok(Self::new(Expr::BinaryOp {
            left: Box::new(self.inner.clone()),
            op,
            right: Box::new(expr_from_py_operand(other)?),
        }))
    }
}

impl PyAggExpr {
    fn new(inner: AggExpr) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyNodeFrame {
    #[classmethod]
    fn from_dict(_cls: &Bound<'_, PyType>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let batch = record_batch_from_py_mapping(data, FrameKind::Node)?;
        let frame = NodeFrame::from_record_batch(batch).map_err(gf_error_to_py_err)?;
        Ok(Self::new(frame))
    }

    #[classmethod]
    fn from_arrow(_cls: &Bound<'_, PyType>, input: &Bound<'_, PyAny>) -> PyResult<Self> {
        let batch = record_batch_from_pyarrow_input(input)?;
        let frame = NodeFrame::from_record_batch(batch).map_err(gf_error_to_py_err)?;
        Ok(Self::new(frame))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn node_count(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn column_names(&self) -> Vec<String> {
        self.inner
            .column_names()
            .into_iter()
            .map(str::to_owned)
            .collect()
    }

    fn ids(&self) -> Vec<String> {
        let ids = self.inner.id_column();
        (0..ids.len())
            .map(|idx| ids.value(idx).to_owned())
            .collect()
    }

    fn column_values(&self, name: &str, py: Python<'_>) -> PyResult<PyObject> {
        let column = self
            .inner
            .to_record_batch()
            .column_by_name(name)
            .ok_or_else(|| PyKeyError::new_err(format!("column not found: {name}")))?;
        let py_array = column
            .to_data()
            .to_pyarrow(py)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        let values = py_array.bind(py).call_method0("to_pylist")?;
        Ok(values.unbind())
    }

    fn to_rows(&self, py: Python<'_>) -> PyResult<PyObject> {
        record_batch_to_py_rows(self.inner.to_record_batch(), py)
    }

    fn to_pylist(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.to_rows(py)
    }

    fn filter(&self, mask: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mask = extract_boolean_mask(mask)?;
        let frame = self.inner.filter(&mask).map_err(gf_error_to_py_err)?;
        Ok(Self::new(frame))
    }

    fn select(&self, columns: Vec<String>) -> PyResult<Self> {
        let columns_ref: Vec<&str> = columns.iter().map(String::as_str).collect();
        let frame = self
            .inner
            .select(&columns_ref)
            .map_err(gf_error_to_py_err)?;
        Ok(Self::new(frame))
    }

    #[pyo3(signature = (*, include=None, exclude_reserved=true, numeric_only=true))]
    fn feature_columns(
        &self,
        include: Option<Vec<String>>,
        exclude_reserved: bool,
        numeric_only: bool,
    ) -> PyResult<Vec<String>> {
        node_feature_columns(self.inner.as_ref(), include, exclude_reserved, numeric_only)
    }

    fn take(&self, indices: &Bound<'_, PyAny>) -> PyResult<Self> {
        let row_ids = extract_row_indices(indices, self.inner.len())?;
        let batch = self
            .inner
            .gather_rows(&row_ids)
            .map_err(gf_error_to_py_err)?;
        let frame = NodeFrame::from_record_batch(batch).map_err(gf_error_to_py_err)?;
        Ok(Self::new(frame))
    }

    #[pyo3(signature = (columns=None, indices=None, dtype=None, contiguous=true))]
    fn to_numpy(
        &self,
        py: Python<'_>,
        columns: Option<Vec<String>>,
        indices: Option<&Bound<'_, PyAny>>,
        dtype: Option<&Bound<'_, PyAny>>,
        contiguous: bool,
    ) -> PyResult<PyObject> {
        let columns = resolve_feature_columns_for_export(self.inner.as_ref(), columns)?;
        let batch = selected_node_batch_to_pyarrow(self.inner.as_ref(), indices, py)?;
        let selected = select_pyarrow_columns(&batch.bind(py), &columns)?;
        let numpy = py.import_bound("numpy").map_err(|_| {
            PyImportError::new_err(
                "NodeFrame.to_numpy requires numpy; install it with `pip install numpy`",
            )
        })?;
        let rows = selected.bind(py).getattr("num_rows")?.extract::<usize>()?;

        let matrix = if columns.is_empty() {
            let shape = (rows, 0usize);
            let kwargs = PyDict::new_bound(py);
            if let Some(dtype) = dtype {
                kwargs.set_item("dtype", dtype)?;
            }
            numpy.call_method("empty", (shape,), Some(&kwargs))?
        } else {
            let arrays = PyList::empty_bound(py);
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("zero_copy_only", false)?;
            for column in &columns {
                let arrow_array = selected.bind(py).call_method1("column", (column,))?;
                let numpy_array = arrow_array.call_method("to_numpy", (), Some(&kwargs))?;
                arrays.append(numpy_array)?;
            }
            let mut matrix = numpy.call_method1("column_stack", (arrays,))?;
            if let Some(dtype) = dtype {
                let kwargs = PyDict::new_bound(py);
                kwargs.set_item("copy", false)?;
                matrix = matrix.call_method("astype", (dtype,), Some(&kwargs))?;
            }
            matrix
        };

        let matrix = if contiguous {
            numpy.call_method1("ascontiguousarray", (matrix,))?
        } else {
            matrix
        };
        Ok(matrix.unbind())
    }

    #[pyo3(signature = (columns=None, indices=None, dtype=None, device=None, contiguous=true, out=None))]
    fn to_tensor(
        &self,
        py: Python<'_>,
        columns: Option<Vec<String>>,
        indices: Option<&Bound<'_, PyAny>>,
        dtype: Option<&Bound<'_, PyAny>>,
        device: Option<&Bound<'_, PyAny>>,
        contiguous: bool,
        out: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        if let Some(out_tensor) = out {
            let columns = resolve_feature_columns_for_export(self.inner.as_ref(), columns)?;
            let batch = selected_node_batch_to_pyarrow(self.inner.as_ref(), indices, py)?;
            let selected = select_pyarrow_columns(&batch.bind(py), &columns)?;

            let _numpy = py.import_bound("numpy").map_err(|_| {
                PyImportError::new_err("NodeFrame.to_tensor(out=...) requires numpy")
            })?;

            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("zero_copy_only", false)?;

            let out_np = out_tensor.call_method0("numpy")?;
            for (i, column) in columns.iter().enumerate() {
                let arrow_array = selected.bind(py).call_method1("column", (column,))?;
                let numpy_array = arrow_array.call_method("to_numpy", (), Some(&kwargs))?;

                let builtins = py.import_bound("builtins")?;
                let slice_class = builtins.getattr("slice")?;
                let slice_all = slice_class.call1((py.None(), py.None(), py.None()))?;

                let index_tuple =
                    PyTuple::new_bound(py, [slice_all.into_any(), i.into_py(py).into_bound(py)]);
                out_np.set_item(index_tuple, numpy_array)?;
            }
            return Ok(out_tensor.clone().unbind());
        }

        let numpy_array = self.to_numpy(py, columns, indices, None, true)?;
        let torch = py.import_bound("torch").map_err(|_| {
            PyImportError::new_err(
                "NodeFrame.to_tensor requires PyTorch; install it with `pip install torch`",
            )
        })?;

        let kwargs = PyDict::new_bound(py);
        if let Some(dtype) = dtype {
            if let Ok(name) = dtype.extract::<String>() {
                kwargs.set_item("dtype", torch.getattr(name.as_str())?)?;
            } else {
                kwargs.set_item("dtype", dtype)?;
            }
        } else {
            kwargs.set_item("dtype", torch.getattr("float32")?)?;
        }
        if let Some(device) = device {
            kwargs.set_item("device", device)?;
        }

        let tensor = torch.call_method("as_tensor", (numpy_array.bind(py),), Some(&kwargs))?;
        let tensor = if contiguous {
            tensor.call_method0("contiguous")?
        } else {
            tensor
        };
        Ok(tensor.unbind())
    }

    /// Concatenate multiple `NodeFrame`s into one (union of rows, schemas must be compatible).
    #[classmethod]
    fn concat(_cls: &Bound<'_, PyType>, frames: Vec<PyRef<'_, PyNodeFrame>>) -> PyResult<Self> {
        let inner_refs: Vec<&NodeFrame> = frames.iter().map(|f| f.inner.as_ref()).collect();
        let merged = NodeFrame::concat(&inner_refs).map_err(gf_error_to_py_err)?;
        Ok(Self::new(merged))
    }

    /// Return rows whose `_id` appears in *both* `self` and `other`.
    fn intersect(&self, other: PyRef<'_, PyNodeFrame>) -> PyResult<Self> {
        let result = self
            .inner
            .intersect(&other.inner)
            .map_err(gf_error_to_py_err)?;
        Ok(Self::new(result))
    }

    /// Return rows whose `_id` is in `self` but **not** in `other`.
    fn difference(&self, other: PyRef<'_, PyNodeFrame>) -> PyResult<Self> {
        let result = self
            .inner
            .difference(&other.inner)
            .map_err(gf_error_to_py_err)?;
        Ok(Self::new(result))
    }

    fn with_edges(&self, edges: PyRef<'_, PyEdgeFrame>) -> PyResult<PyGraphFrame> {
        let graph = self
            .inner
            .with_edges((*edges.inner).clone())
            .map_err(gf_error_to_py_err)?;
        Ok(PyGraphFrame::new(graph))
    }

    fn gather_rows(&self, row_ids: Vec<u32>, py: Python<'_>) -> PyResult<PyObject> {
        self.inner
            .gather_rows(&row_ids)
            .map_err(gf_error_to_py_err)?
            .to_pyarrow(py)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    fn __repr__(&self) -> String {
        match self.render_display_view(FramePreviewView::Table, 10, None, false, Some(100)) {
            Ok(rendered) => rendered,
            Err(err) => format!("NodeFrame(<display error: {err}>)"),
        }
    }

    #[pyo3(signature = (n=5, *, sort_by=None, descending=false, width=None))]
    fn head(
        &self,
        n: usize,
        sort_by: Option<String>,
        descending: bool,
        width: Option<usize>,
    ) -> PyResult<String> {
        self.render_display_view(FramePreviewView::Head, n, sort_by, descending, width)
    }

    #[pyo3(signature = (n=5, *, sort_by=None, descending=false, width=None))]
    fn tail(
        &self,
        n: usize,
        sort_by: Option<String>,
        descending: bool,
        width: Option<usize>,
    ) -> PyResult<String> {
        self.render_display_view(FramePreviewView::Tail, n, sort_by, descending, width)
    }

    fn info(&self) -> String {
        render_node_frame_info(self.inner.as_ref())
    }

    fn schema(&self) -> String {
        render_frame_schema(
            "NodeFrame",
            self.inner.to_record_batch(),
            &NODE_RESERVED_COLUMNS,
        )
    }

    #[pyo3(signature = (n=3, *, sort_by=None, descending=false, width=None))]
    fn glimpse(
        &self,
        n: usize,
        sort_by: Option<String>,
        descending: bool,
        width: Option<usize>,
    ) -> PyResult<String> {
        render_frame_glimpse(
            "NodeFrame",
            self.inner.to_record_batch(),
            n.max(1),
            sort_by,
            descending,
            width,
        )
    }

    #[pyo3(signature = (mode="all"))]
    fn describe(&self, mode: &str) -> PyResult<String> {
        describe_node_frame(self.inner.as_ref(), mode)
    }

    fn to_arrow(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.to_arrow_impl(py)
    }

    fn to_pyarrow(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.to_arrow_impl(py)
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let cls = py.get_type_bound::<PyNodeFrame>();
        let from_arrow = cls.getattr("from_arrow")?.unbind();
        let batch = self.to_arrow_impl(py)?;
        let args = PyTuple::new_bound(py, [batch]).unbind();
        Ok(PyTuple::new_bound(py, [from_arrow, args.into()])
            .unbind()
            .into())
    }
}

#[pymethods]
impl PyEdgeFrame {
    #[classmethod]
    fn from_dict(_cls: &Bound<'_, PyType>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let batch = record_batch_from_py_mapping(data, FrameKind::Edge)?;
        let frame = EdgeFrame::from_record_batch(batch).map_err(gf_error_to_py_err)?;
        Ok(Self::new(frame))
    }

    #[classmethod]
    fn from_arrow(_cls: &Bound<'_, PyType>, batch: PyArrowType<RecordBatch>) -> PyResult<Self> {
        let frame = EdgeFrame::from_record_batch(batch.0).map_err(gf_error_to_py_err)?;
        Ok(Self::new(frame))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn edge_count(&self) -> usize {
        self.inner.len()
    }

    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn column_names(&self) -> Vec<String> {
        self.inner
            .column_names()
            .into_iter()
            .map(str::to_owned)
            .collect()
    }

    fn edge_types(&self) -> Vec<String> {
        self.inner
            .edge_types()
            .into_iter()
            .map(str::to_owned)
            .collect()
    }

    fn column_values(&self, name: &str, py: Python<'_>) -> PyResult<PyObject> {
        let column = self
            .inner
            .to_record_batch()
            .column_by_name(name)
            .ok_or_else(|| PyKeyError::new_err(format!("column not found: {name}")))?;
        let py_array = column
            .to_data()
            .to_pyarrow(py)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        let values = py_array.bind(py).call_method0("to_pylist")?;
        Ok(values.unbind())
    }

    fn filter(&self, mask: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mask = extract_boolean_mask(mask)?;
        let frame = self.inner.filter(&mask).map_err(gf_error_to_py_err)?;
        Ok(Self::new(frame))
    }

    fn filter_by_type(&self, edge_type: &str) -> PyResult<Self> {
        let frame = self
            .inner
            .filter_by_type(edge_type)
            .map_err(gf_error_to_py_err)?;
        Ok(Self::new(frame))
    }

    fn filter_by_types(&self, edge_types: Vec<String>) -> PyResult<Self> {
        let edge_types_ref: Vec<&str> = edge_types.iter().map(String::as_str).collect();
        let frame = self
            .inner
            .filter_by_types(&edge_types_ref)
            .map_err(gf_error_to_py_err)?;
        Ok(Self::new(frame))
    }

    fn select(&self, columns: Vec<String>) -> PyResult<Self> {
        let columns_ref: Vec<&str> = columns.iter().map(String::as_str).collect();
        let frame = self
            .inner
            .select(&columns_ref)
            .map_err(gf_error_to_py_err)?;
        Ok(Self::new(frame))
    }

    fn out_neighbors(&self, node_id: &str) -> PyResult<Vec<String>> {
        let Some(node_idx) = self.inner.node_row_idx(node_id) else {
            return Err(PyKeyError::new_err(format!("node not found: {node_id}")));
        };

        self.inner
            .out_neighbors(node_idx)
            .iter()
            .map(|&idx| {
                self.node_ids.get(idx as usize).cloned().ok_or_else(|| {
                    PyRuntimeError::new_err(format!("invalid local node index: {idx}"))
                })
            })
            .collect()
    }

    fn in_neighbors(&self, node_id: &str) -> PyResult<Vec<String>> {
        let Some(node_idx) = self.inner.node_row_idx(node_id) else {
            return Err(PyKeyError::new_err(format!("node not found: {node_id}")));
        };

        self.inner
            .in_neighbors(node_idx)
            .iter()
            .map(|&idx| {
                self.node_ids.get(idx as usize).cloned().ok_or_else(|| {
                    PyRuntimeError::new_err(format!("invalid local node index: {idx}"))
                })
            })
            .collect()
    }

    #[pyo3(signature = (node_id, direction="out"))]
    fn neighbors(&self, node_id: &str, direction: &str) -> PyResult<Vec<String>> {
        let Some(node_idx) = self.inner.node_row_idx(node_id) else {
            return Err(PyKeyError::new_err(format!("node not found: {node_id}")));
        };
        let direction = python_to_direction(direction)?;

        match direction {
            Direction::Out => self.out_neighbors(node_id),
            Direction::In => self.in_neighbors(node_id),
            Direction::Both | Direction::None => {
                let mut seen = BTreeSet::new();
                let mut out = Vec::new();
                for &idx in self
                    .inner
                    .out_neighbors(node_idx)
                    .iter()
                    .chain(self.inner.in_neighbors(node_idx).iter())
                {
                    let id = self.node_ids.get(idx as usize).ok_or_else(|| {
                        PyRuntimeError::new_err(format!("invalid local node index: {idx}"))
                    })?;
                    if seen.insert(id.clone()) {
                        out.push(id.clone());
                    }
                }
                Ok(out)
            }
        }
    }

    fn out_degree(&self, node_id: &str) -> PyResult<usize> {
        let Some(node_idx) = self.inner.node_row_idx(node_id) else {
            return Err(PyKeyError::new_err(format!("node not found: {node_id}")));
        };
        Ok(self.inner.out_degree(node_idx))
    }

    fn in_degree(&self, node_id: &str) -> PyResult<usize> {
        let Some(node_idx) = self.inner.node_row_idx(node_id) else {
            return Err(PyKeyError::new_err(format!("node not found: {node_id}")));
        };
        Ok(self.inner.in_degree(node_idx))
    }

    fn with_nodes(&self, nodes: PyRef<'_, PyNodeFrame>) -> PyResult<PyGraphFrame> {
        let graph = self
            .inner
            .with_nodes((*nodes.inner).clone())
            .map_err(gf_error_to_py_err)?;
        Ok(PyGraphFrame::new(graph))
    }

    fn __repr__(&self) -> String {
        match self.render_display_view(FramePreviewView::Table, 10, None, false, Some(100)) {
            Ok(rendered) => rendered,
            Err(err) => format!("EdgeFrame(<display error: {err}>)"),
        }
    }

    #[pyo3(signature = (n=5, *, sort_by=None, descending=false, width=None))]
    fn head(
        &self,
        n: usize,
        sort_by: Option<String>,
        descending: bool,
        width: Option<usize>,
    ) -> PyResult<String> {
        self.render_display_view(FramePreviewView::Head, n, sort_by, descending, width)
    }

    #[pyo3(signature = (n=5, *, sort_by=None, descending=false, width=None))]
    fn tail(
        &self,
        n: usize,
        sort_by: Option<String>,
        descending: bool,
        width: Option<usize>,
    ) -> PyResult<String> {
        self.render_display_view(FramePreviewView::Tail, n, sort_by, descending, width)
    }

    fn info(&self) -> String {
        render_edge_frame_info(self.inner.as_ref())
    }

    fn schema(&self) -> String {
        render_frame_schema(
            "EdgeFrame",
            self.inner.to_record_batch(),
            &EDGE_RESERVED_COLUMNS,
        )
    }

    #[pyo3(signature = (n=3, *, sort_by=None, descending=false, width=None))]
    fn glimpse(
        &self,
        n: usize,
        sort_by: Option<String>,
        descending: bool,
        width: Option<usize>,
    ) -> PyResult<String> {
        render_frame_glimpse(
            "EdgeFrame",
            self.inner.to_record_batch(),
            n.max(1),
            sort_by,
            descending,
            width,
        )
    }

    #[pyo3(signature = (mode="all"))]
    fn describe(&self, mode: &str) -> PyResult<String> {
        describe_edge_frame(self.inner.as_ref(), mode)
    }

    fn to_arrow(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.to_arrow_impl(py)
    }

    fn to_pyarrow(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.to_arrow_impl(py)
    }
}

#[pymethods]
impl PyGraphFrame {
    #[classmethod]
    fn from_dicts(
        _cls: &Bound<'_, PyType>,
        nodes: &Bound<'_, PyAny>,
        edges: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        graph_from_py_mappings(nodes, edges)
    }

    #[classmethod]
    fn from_frames(
        _cls: &Bound<'_, PyType>,
        nodes: PyRef<'_, PyNodeFrame>,
        edges: PyRef<'_, PyEdgeFrame>,
    ) -> PyResult<Self> {
        let graph = GraphFrame::new((*nodes.inner).clone(), (*edges.inner).clone())
            .map_err(gf_error_to_py_err)?;
        Ok(Self::new(graph))
    }

    fn nodes(&self) -> PyNodeFrame {
        PyNodeFrame::from_arc(Arc::new(self.inner.nodes().clone()))
    }

    fn edges(&self) -> PyEdgeFrame {
        PyEdgeFrame::from_arc(Arc::new(self.inner.edges().clone()))
    }

    fn to_coo(&self, py: Python<'_>) -> PyResult<(PyObject, PyObject)> {
        let (src, dst) = self.inner.to_coo();
        let src: PyObject = src
            .to_data()
            .to_pyarrow(py)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        let dst: PyObject = dst
            .to_data()
            .to_pyarrow(py)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        Ok((src, dst))
    }

    #[pyo3(signature = (edge_type=None))]
    fn structural_features(&self, edge_type: Option<&str>) -> PyResult<PyNodeFrame> {
        let features = self
            .inner
            .structural_features(edge_type)
            .map_err(gf_error_to_py_err)?;
        Ok(PyNodeFrame::from_arc(Arc::new(features)))
    }

    #[pyo3(signature = (seed_nodes, *, hops=1, fan_out=None, direction="out", edge_type=None, replace=false))]
    fn sample_neighbors(
        &self,
        seed_nodes: Vec<String>,
        hops: usize,
        fan_out: Option<Vec<usize>>,
        direction: &str,
        edge_type: Option<&Bound<'_, PyAny>>,
        replace: bool,
    ) -> PyResult<PySampledSubgraph> {
        let direction = python_to_direction(direction)?;
        let edge_type = normalize_edge_type_spec(edge_type)?;
        let fan_out = fan_out.unwrap_or_else(|| vec![25; hops.max(1)]);
        let config = SamplingConfig {
            hops,
            fan_out,
            direction,
            edge_type,
            replace,
        };
        let seed_refs: Vec<&str> = seed_nodes.iter().map(String::as_str).collect();
        let sampled = self
            .inner
            .sample_neighbors(&seed_refs, &config)
            .map_err(gf_error_to_py_err)?;
        Ok(PySampledSubgraph::new(sampled))
    }

    #[pyo3(signature = (start_nodes, *, length=80, walks_per_node=10, direction="out", edge_type=None))]
    fn random_walk(
        &self,
        start_nodes: Vec<String>,
        length: usize,
        walks_per_node: usize,
        direction: &str,
        edge_type: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Vec<u32>>> {
        let direction = python_to_direction(direction)?;
        let edge_type = normalize_edge_type_spec(edge_type)?;
        let start_refs: Vec<&str> = start_nodes.iter().map(String::as_str).collect();
        self.inner
            .random_walk(&start_refs, length, walks_per_node, direction, &edge_type)
            .map_err(gf_error_to_py_err)
    }

    fn into_mutable(&self) -> PyMutableGraphFrame {
        PyMutableGraphFrame::new(self.inner.as_ref().clone().into_mutable())
    }

    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    fn density(&self) -> f64 {
        self.inner.density()
    }

    fn __repr__(&self) -> String {
        match self.inner.display_slice(DisplayOptions {
            view: DisplayView::Table,
            max_rows: 10,
            width: Some(100),
            sort_by: None,
            expand_attrs: false,
            attrs: Vec::new(),
        }) {
            Ok(slice) => render_python_display_slice(&slice),
            Err(err) => format!("GraphFrame(<display error: {err}>)"),
        }
    }

    #[pyo3(signature = (n=5, *, sort_by=None, expand_attrs=false, attrs=None, width=None))]
    fn head(
        &self,
        n: usize,
        sort_by: Option<String>,
        expand_attrs: bool,
        attrs: Option<Vec<String>>,
        width: Option<usize>,
    ) -> PyResult<String> {
        self.render_display_view(
            DisplayView::Head,
            n,
            sort_by,
            expand_attrs,
            attrs.unwrap_or_default(),
            width,
        )
    }

    #[pyo3(signature = (n=5, *, sort_by=None, expand_attrs=false, attrs=None, width=None))]
    fn tail(
        &self,
        n: usize,
        sort_by: Option<String>,
        expand_attrs: bool,
        attrs: Option<Vec<String>>,
        width: Option<usize>,
    ) -> PyResult<String> {
        self.render_display_view(
            DisplayView::Tail,
            n,
            sort_by,
            expand_attrs,
            attrs.unwrap_or_default(),
            width,
        )
    }

    fn info(&self) -> String {
        render_python_info(&self.inner.display_info())
    }

    fn schema(&self) -> String {
        render_python_schema(&self.inner.display_schema())
    }

    #[pyo3(signature = (n=3, *, sort_by=None, expand_attrs=true, attrs=None, width=None))]
    fn glimpse(
        &self,
        n: usize,
        sort_by: Option<String>,
        expand_attrs: bool,
        attrs: Option<Vec<String>>,
        width: Option<usize>,
    ) -> PyResult<String> {
        let glimpse = self
            .inner
            .display_glimpse(DisplayOptions {
                view: DisplayView::Head,
                max_rows: n.max(1),
                width,
                sort_by,
                expand_attrs,
                attrs: attrs.unwrap_or_default(),
            })
            .map_err(gf_error_to_py_err)?;
        Ok(render_python_glimpse(&glimpse))
    }

    #[pyo3(signature = (mode="all"))]
    fn describe(&self, mode: &str) -> PyResult<String> {
        describe_graph(self.inner.as_ref(), mode)
    }

    fn lazy(&self) -> PyLazyGraphFrame {
        PyLazyGraphFrame::new(LazyGraphFrame::from_graph(self.inner.as_ref()))
    }

    fn subgraph(&self, node_ids: Vec<String>) -> PyResult<Self> {
        let node_ids_ref: Vec<&str> = node_ids.iter().map(String::as_str).collect();
        let graph = self
            .inner
            .subgraph(&node_ids_ref)
            .map_err(gf_error_to_py_err)?;
        Ok(Self::new(graph))
    }

    fn subgraph_by_label(&self, label: &str) -> PyResult<Self> {
        let graph = self
            .inner
            .subgraph_by_label(label)
            .map_err(gf_error_to_py_err)?;
        Ok(Self::new(graph))
    }

    fn subgraph_by_edge_type(&self, edge_type: &str) -> PyResult<Self> {
        let graph = self
            .inner
            .subgraph_by_edge_type(edge_type)
            .map_err(gf_error_to_py_err)?;
        Ok(Self::new(graph))
    }

    fn k_hop_subgraph(&self, root: &str, k: usize) -> PyResult<Self> {
        let graph = self
            .inner
            .k_hop_subgraph(root, k)
            .map_err(gf_error_to_py_err)?;
        Ok(Self::new(graph))
    }

    fn out_neighbors(&self, node_id: &str) -> PyResult<Vec<String>> {
        self.inner
            .out_neighbors(node_id)
            .map(|values| values.into_iter().map(str::to_owned).collect())
            .map_err(gf_error_to_py_err)
    }

    fn in_neighbors(&self, node_id: &str) -> PyResult<Vec<String>> {
        self.inner
            .in_neighbors(node_id)
            .map(|values| values.into_iter().map(str::to_owned).collect())
            .map_err(gf_error_to_py_err)
    }

    #[pyo3(signature = (node_id, direction="out"))]
    fn neighbors(&self, node_id: &str, direction: &str) -> PyResult<Vec<String>> {
        let direction = python_to_direction(direction)?;
        self.inner
            .neighbors(node_id, direction)
            .map(|values| values.into_iter().map(str::to_owned).collect())
            .map_err(gf_error_to_py_err)
    }

    fn out_degree(&self, node_id: &str) -> PyResult<usize> {
        self.inner.out_degree(node_id).map_err(gf_error_to_py_err)
    }

    fn in_degree(&self, node_id: &str) -> PyResult<usize> {
        self.inner.in_degree(node_id).map_err(gf_error_to_py_err)
    }

    #[pyo3(signature = (*, damping=0.85, max_iter=100, epsilon=1e-6, weight_col=None))]
    fn pagerank(
        &self,
        damping: f64,
        max_iter: usize,
        epsilon: f64,
        weight_col: Option<String>,
    ) -> PyResult<PyNodeFrame> {
        let config = PageRankConfig {
            damping,
            max_iter,
            epsilon,
            weight_col,
        };
        let nodes = self.inner.pagerank(&config).map_err(gf_error_to_py_err)?;
        Ok(PyNodeFrame::new(nodes))
    }

    fn connected_components(&self) -> PyResult<PyNodeFrame> {
        let nodes = self
            .inner
            .connected_components()
            .map_err(gf_error_to_py_err)?;
        Ok(PyNodeFrame::new(nodes))
    }

    fn largest_connected_component(&self) -> PyResult<Self> {
        let graph = self
            .inner
            .largest_connected_component()
            .map_err(gf_error_to_py_err)?;
        Ok(Self::new(graph))
    }

    #[pyo3(signature = (src, dst, *, weight_col=None, edge_type=None, direction="out"))]
    fn shortest_path(
        &self,
        src: &str,
        dst: &str,
        weight_col: Option<String>,
        edge_type: Option<String>,
        direction: &str,
    ) -> PyResult<Option<Vec<String>>> {
        let config = ShortestPathConfig {
            weight_col,
            edge_type: shortest_path_edge_type(edge_type),
            direction: python_to_direction(direction)?,
        };
        self.inner
            .shortest_path(src, dst, &config)
            .map_err(gf_error_to_py_err)
    }

    #[pyo3(signature = (src, dst, *, weight_col=None, edge_type=None, direction="out"))]
    fn all_shortest_paths(
        &self,
        src: &str,
        dst: &str,
        weight_col: Option<String>,
        edge_type: Option<String>,
        direction: &str,
    ) -> PyResult<Vec<Vec<String>>> {
        let config = ShortestPathConfig {
            weight_col,
            edge_type: shortest_path_edge_type(edge_type),
            direction: python_to_direction(direction)?,
        };
        self.inner
            .all_shortest_paths(src, dst, &config)
            .map_err(gf_error_to_py_err)
    }

    #[pyo3(signature = (*, weight_col=None))]
    fn betweenness_centrality(&self, weight_col: Option<String>) -> PyResult<PyNodeFrame> {
        let nodes = self
            .inner
            .betweenness_centrality_with_config(&BetweennessConfig { weight_col })
            .map_err(gf_error_to_py_err)?;
        Ok(PyNodeFrame::new(nodes))
    }

    #[pyo3(signature = (direction="out"))]
    fn degree_centrality(&self, direction: &str) -> PyResult<PyNodeFrame> {
        let nodes = self
            .inner
            .degree_centrality(python_to_direction(direction)?)
            .map_err(gf_error_to_py_err)?;
        Ok(PyNodeFrame::new(nodes))
    }

    #[pyo3(signature = (*, algorithm="louvain", resolution=1.0, seed=None))]
    fn community_detection(
        &self,
        algorithm: &str,
        resolution: f64,
        seed: Option<u64>,
    ) -> PyResult<PyNodeFrame> {
        let algorithm = match algorithm {
            "louvain" => lynxes_core::CommunityAlgorithm::Louvain,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unsupported community_detection algorithm: {other}"
                )))
            }
        };

        let nodes = self
            .inner
            .community_detection(lynxes_core::CommunityConfig {
                algorithm,
                resolution,
                seed,
            })
            .map_err(gf_error_to_py_err)?;
        Ok(PyNodeFrame::new(nodes))
    }

    #[pyo3(signature = (src, dst, *, max_hops=None))]
    fn has_path(&self, src: &str, dst: &str, max_hops: Option<usize>) -> PyResult<bool> {
        self.inner
            .has_path(src, dst, max_hops)
            .map_err(gf_error_to_py_err)
    }

    fn write_gf(&self, path: &Bound<'_, PyAny>) -> PyResult<()> {
        let path = path_from_py_any(path)?;
        write_gf_impl(self.inner.as_ref(), &path)
    }

    fn write_gfb(&self, path: &Bound<'_, PyAny>) -> PyResult<()> {
        let path = path_from_py_any(path)?;
        write_gfb(
            self.inner.as_ref(),
            path,
            &lynxes_io::GfbWriteOptions::default(),
        )
        .map_err(gf_error_to_py_err)
    }

    fn write_parquet_graph(
        &self,
        nodes_path: &Bound<'_, PyAny>,
        edges_path: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let nodes_path = path_from_py_any(nodes_path)?;
        let edges_path = path_from_py_any(edges_path)?;
        write_parquet_graph(self.inner.as_ref(), nodes_path, edges_path).map_err(gf_error_to_py_err)
    }

    fn write_rdf(&self, path: &Bound<'_, PyAny>) -> PyResult<()> {
        let path = path_from_py_any(path)?;
        unsupported_write_impl("write_rdf", &path)
    }

    fn write_owl(&self, path: &Bound<'_, PyAny>) -> PyResult<()> {
        let path = path_from_py_any(path)?;
        unsupported_write_impl("write_owl", &path)
    }

    /// Partition this graph into `n_shards` balanced shards.
    ///
    /// ```python
    /// pg = graph.partition(4, strategy="hash")
    /// pg = graph.partition(4, strategy="range")
    /// pg = graph.partition(4, strategy="label")
    /// ```
    #[pyo3(signature = (n_shards, strategy = "hash"))]
    fn partition(&self, n_shards: usize, strategy: &str) -> PyResult<PyPartitionedGraph> {
        let method = match strategy {
            "range" => GraphPartitionMethod::Range,
            "label" => GraphPartitionMethod::Label,
            _ => GraphPartitionMethod::Hash,
        };
        let pg = GraphPartitioner::partition(self.inner.as_ref(), n_shards, method)
            .map_err(gf_error_to_py_err)?;
        Ok(PyPartitionedGraph::new(pg))
    }
}

impl PyGraphFrame {
    fn render_display_view(
        &self,
        view: DisplayView,
        rows: usize,
        sort_by: Option<String>,
        expand_attrs: bool,
        attrs: Vec<String>,
        width: Option<usize>,
    ) -> PyResult<String> {
        let slice = self
            .inner
            .display_slice(DisplayOptions {
                view,
                max_rows: rows.max(1),
                width,
                sort_by,
                expand_attrs,
                attrs,
            })
            .map_err(gf_error_to_py_err)?;
        Ok(render_python_display_slice(&slice))
    }
}

#[pymethods]
impl PyLazyGraphFrame {
    fn filter_nodes(&self, expr: PyRef<'_, PyExpr>) -> Self {
        Self::new(self.inner.clone().filter_nodes(expr.inner.clone()))
    }

    fn filter_edges(&self, expr: PyRef<'_, PyExpr>) -> Self {
        Self::new(self.inner.clone().filter_edges(expr.inner.clone()))
    }

    fn select_nodes(&self, columns: Vec<String>) -> Self {
        Self::new(self.inner.clone().select_nodes(columns))
    }

    fn select_edges(&self, columns: Vec<String>) -> Self {
        Self::new(self.inner.clone().select_edges(columns))
    }

    #[pyo3(signature = (edge_type=None, *, hops=1, direction="out"))]
    fn expand(
        &self,
        edge_type: Option<&Bound<'_, PyAny>>,
        hops: u32,
        direction: &str,
    ) -> PyResult<Self> {
        if hops == 0 {
            return Err(PyValueError::new_err("hops must be greater than zero"));
        }
        let direction = python_to_direction(direction)?;
        let edge_type = normalize_edge_type_spec(edge_type)?;
        Ok(Self::new(
            self.inner.clone().expand(edge_type, hops, direction),
        ))
    }

    fn aggregate_neighbors(&self, edge_type: &str, agg: PyRef<'_, PyAggExpr>) -> Self {
        Self::new(
            self.inner
                .clone()
                .aggregate_neighbors(edge_type.to_owned(), agg.inner.clone()),
        )
    }

    /// Declare a structural pattern to match.
    ///
    /// `steps` must be a list that alternates `PatternNode` / `PatternEdge` / `PatternNode` …
    /// with at least three items (one edge hop minimum).
    ///
    /// ```python
    /// lazy.match_pattern([
    ///     gf.node("a", "Person"),
    ///     gf.edge("KNOWS"),
    ///     gf.node("b", "Person"),
    /// ])
    /// ```
    #[pyo3(signature = (steps, where_=None))]
    fn match_pattern(
        &self,
        steps: &Bound<'_, PyAny>,
        where_: Option<PyRef<'_, PyExpr>>,
    ) -> PyResult<Self> {
        let pattern = pattern_from_py_steps(steps)?;
        let where_expr = where_.map(|e| e.inner.clone());
        Ok(Self::new(
            self.inner.clone().match_pattern(pattern, where_expr),
        ))
    }

    #[pyo3(signature = (by, descending=false))]
    fn sort(&self, by: &str, descending: bool) -> Self {
        Self::new(self.inner.clone().sort(by.to_owned(), descending))
    }

    fn limit(&self, n: usize) -> Self {
        Self::new(self.inner.clone().limit(n))
    }

    fn explain(&self) -> String {
        self.inner.explain()
    }

    fn collect(&self, py: Python<'_>) -> PyResult<PyObject> {
        match self.inner.clone().collect() {
            Ok(graph) => Py::new(py, PyGraphFrame::new(graph)).map(|obj| obj.into_py(py)),
            Err(err) => match self.inner.clone().collect_pattern_rows() {
                Ok(batch) => batch
                    .to_pyarrow(py)
                    .map_err(|arrow_err| PyRuntimeError::new_err(arrow_err.to_string())),
                Err(pattern_err) => {
                    if matches!(
                        pattern_err,
                        GFError::UnsupportedOperation { .. } | GFError::DomainMismatch { .. }
                    ) {
                        Err(gf_error_to_py_err(err))
                    } else {
                        Err(gf_error_to_py_err(pattern_err))
                    }
                }
            },
        }
    }

    fn collect_nodes(&self) -> PyResult<PyNodeFrame> {
        let nodes = self
            .inner
            .clone()
            .collect_nodes()
            .map_err(gf_error_to_py_err)?;
        Ok(PyNodeFrame::new(nodes))
    }

    fn collect_edges(&self) -> PyResult<PyEdgeFrame> {
        let edges = self
            .inner
            .clone()
            .collect_edges()
            .map_err(gf_error_to_py_err)?;
        Ok(PyEdgeFrame::new(edges))
    }
}

#[pymethods]
impl PySampledSubgraph {
    #[getter]
    fn node_indices(&self) -> Vec<u32> {
        self.inner.node_indices.clone()
    }

    #[getter]
    fn edge_src(&self) -> Vec<u32> {
        self.inner.edge_src.clone()
    }

    #[getter]
    fn edge_dst(&self) -> Vec<u32> {
        self.inner.edge_dst.clone()
    }

    #[getter]
    fn edge_row_ids(&self) -> Vec<u32> {
        self.inner.edge_row_ids.clone()
    }

    #[getter]
    fn node_row_ids(&self) -> Vec<u32> {
        self.inner.node_row_ids.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "SampledSubgraph(nodes={}, edges={})",
            self.inner.node_indices.len(),
            self.inner.edge_src.len()
        )
    }
}

#[pymethods]
impl PyMutableGraphFrame {
    fn add_node<'py>(
        mut slf: PyRefMut<'py, Self>,
        node: PyRef<'_, PyNodeFrame>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner_mut()?
            .add_node((*node.inner).clone())
            .map_err(gf_error_to_py_err)?;
        Ok(slf)
    }

    fn add_nodes_batch<'py>(
        mut slf: PyRefMut<'py, Self>,
        nodes: PyRef<'_, PyNodeFrame>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner_mut()?
            .add_nodes_batch((*nodes.inner).clone())
            .map_err(gf_error_to_py_err)?;
        Ok(slf)
    }

    #[pyo3(signature = (src, dst, *, edge_type=None, direction="out", attrs=None))]
    fn add_edge<'py>(
        mut slf: PyRefMut<'py, Self>,
        src: &str,
        dst: &str,
        edge_type: Option<&str>,
        direction: &str,
        attrs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let direction = python_to_direction(direction)?;
        let schema = {
            let inner = slf.inner_mut()?;
            inner.edge_schema()
        };
        let edge = edge_row_from_schema(schema.as_ref(), src, dst, edge_type, direction, attrs)?;

        slf.inner_mut()?
            .add_edge_row(edge)
            .map_err(gf_error_to_py_err)?;
        Ok(slf)
    }

    fn delete_node<'py>(mut slf: PyRefMut<'py, Self>, id: &str) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner_mut()?
            .delete_node(id)
            .map_err(gf_error_to_py_err)?;
        Ok(slf)
    }

    fn delete_edge<'py>(
        mut slf: PyRefMut<'py, Self>,
        edge_row: u32,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner_mut()?
            .delete_edge(edge_row)
            .map_err(gf_error_to_py_err)?;
        Ok(slf)
    }

    fn update_node<'py>(
        mut slf: PyRefMut<'py, Self>,
        old_id: &str,
        node: PyRef<'_, PyNodeFrame>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner_mut()?
            .update_node(old_id, (*node.inner).clone())
            .map_err(gf_error_to_py_err)?;
        Ok(slf)
    }

    fn update_edge<'py>(
        mut slf: PyRefMut<'py, Self>,
        edge_row: u32,
        src: &str,
        dst: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner_mut()?
            .update_edge(edge_row, src, dst)
            .map_err(gf_error_to_py_err)?;
        Ok(slf)
    }

    fn compact<'py>(mut slf: PyRefMut<'py, Self>) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner_mut()?.compact().map_err(gf_error_to_py_err)?;
        Ok(slf)
    }

    fn freeze(&mut self) -> PyResult<PyGraphFrame> {
        let inner = self.inner.take().ok_or_else(|| {
            PyRuntimeError::new_err(
                "MutableGraphFrame has already been frozen and can no longer be used",
            )
        })?;
        let graph = inner.freeze().map_err(gf_error_to_py_err)?;
        Ok(PyGraphFrame::new(graph))
    }

    fn __repr__(&self) -> String {
        if self.inner.is_some() {
            "MutableGraphFrame(active)".to_owned()
        } else {
            "MutableGraphFrame(frozen)".to_owned()
        }
    }
}

#[pymethods]
impl PyExpr {
    fn __repr__(&self) -> String {
        format!("Expr({:?})", self.inner)
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(PyTypeError::new_err(
            "symbolic expressions do not support truth-value testing; use &, |, and ~",
        ))
    }

    fn contains(&self, item: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self::new(Expr::ListContains {
            expr: Box::new(self.inner.clone()),
            item: Box::new(expr_from_py_operand(item)?),
        }))
    }

    fn cast(&self, dtype: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self::new(Expr::Cast {
            expr: Box::new(self.inner.clone()),
            dtype: extract_dtype(dtype)?,
        }))
    }

    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<Self> {
        let op = match op {
            CompareOp::Eq => BinaryOp::Eq,
            CompareOp::Ne => BinaryOp::NotEq,
            CompareOp::Gt => BinaryOp::Gt,
            CompareOp::Ge => BinaryOp::GtEq,
            CompareOp::Lt => BinaryOp::Lt,
            CompareOp::Le => BinaryOp::LtEq,
        };
        self.binary(other, op)
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(other, BinaryOp::Add)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(other, BinaryOp::Sub)
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(other, BinaryOp::Mul)
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(other, BinaryOp::Div)
    }

    fn __and__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self::new(Expr::And {
            left: Box::new(self.inner.clone()),
            right: Box::new(expr_from_py_operand(other)?),
        }))
    }

    fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self::new(Expr::Or {
            left: Box::new(self.inner.clone()),
            right: Box::new(expr_from_py_operand(other)?),
        }))
    }

    fn __invert__(&self) -> Self {
        Self::new(Expr::Not {
            expr: Box::new(self.inner.clone()),
        })
    }

    fn __neg__(&self) -> Self {
        Self::new(Expr::UnaryOp {
            op: UnaryOp::Neg,
            expr: Box::new(self.inner.clone()),
        })
    }

    /// Return a string-operation namespace for this expression.
    ///
    /// ```python
    /// gf.col("name").str.contains("Alice")
    /// gf.col("name").str.startswith("Al")
    /// gf.col("name").str.endswith("ce")
    /// ```
    #[getter]
    fn str(&self) -> PyStrExprNamespace {
        PyStrExprNamespace {
            inner: self.inner.clone(),
        }
    }
}

#[pymethods]
impl PyStrExprNamespace {
    fn __repr__(&self) -> String {
        format!("StringExprNamespace({:?})", self.inner)
    }

    /// `expr.str.contains(pat)` — true when the column value contains the substring.
    fn contains(&self, pat: &str) -> PyExpr {
        PyExpr::new(Expr::StringOp {
            op: StringOp::Contains,
            expr: Box::new(self.inner.clone()),
            pattern: Box::new(Expr::Literal {
                value: lynxes_core::ScalarValue::String(pat.to_owned()),
            }),
        })
    }

    /// `expr.str.startswith(pat)` — true when the column value starts with the prefix.
    fn startswith(&self, pat: &str) -> PyExpr {
        PyExpr::new(Expr::StringOp {
            op: StringOp::StartsWith,
            expr: Box::new(self.inner.clone()),
            pattern: Box::new(Expr::Literal {
                value: lynxes_core::ScalarValue::String(pat.to_owned()),
            }),
        })
    }

    /// `expr.str.endswith(pat)` — true when the column value ends with the suffix.
    fn endswith(&self, pat: &str) -> PyExpr {
        PyExpr::new(Expr::StringOp {
            op: StringOp::EndsWith,
            expr: Box::new(self.inner.clone()),
            pattern: Box::new(Expr::Literal {
                value: lynxes_core::ScalarValue::String(pat.to_owned()),
            }),
        })
    }
}

#[pymethods]
impl PyPartitionedGraph {
    fn __repr__(&self) -> String {
        format!(
            "PartitionedGraph(n_shards={}, boundary_edges={})",
            self.inner.n_shards,
            self.inner.boundary_edges.len()
        )
    }

    /// Number of shards.
    #[getter]
    fn n_shards(&self) -> usize {
        self.inner.n_shards
    }

    /// Number of boundary edges (cross-shard).
    #[getter]
    fn boundary_edge_count(&self) -> usize {
        self.inner.boundary_edges.len()
    }

    /// List of shard GraphFrames.
    fn shards(&self) -> Vec<PyGraphFrame> {
        self.inner
            .shards
            .iter()
            .map(|s| PyGraphFrame::new(s.clone()))
            .collect()
    }

    /// Merge all shards back into a single GraphFrame.
    fn merge(&self) -> PyResult<PyGraphFrame> {
        let g = self.inner.merge().map_err(gf_error_to_py_err)?;
        Ok(PyGraphFrame::new(g))
    }

    /// Partition statistics as a plain Python dict.
    /// Keys: `n_shards`, `nodes_per_shard`, `edges_per_shard`,
    /// `boundary_edge_count`, `imbalance_ratio`.
    fn stats<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyDict> {
        let s = self.inner.stats();
        let d = pyo3::types::PyDict::new_bound(py);
        d.set_item("n_shards", s.n_shards).unwrap();
        d.set_item("nodes_per_shard", s.nodes_per_shard).unwrap();
        d.set_item("edges_per_shard", s.edges_per_shard).unwrap();
        d.set_item("boundary_edge_count", s.boundary_edge_count)
            .unwrap();
        d.set_item("imbalance_ratio", s.imbalance_ratio).unwrap();
        d
    }

    /// Which shard owns `node_id`?  Returns `None` if not found.
    fn shard_of(&self, node_id: &str) -> Option<usize> {
        self.inner.shard_of(node_id)
    }

    /// Distributed BFS expand.
    ///
    /// ```python
    /// nodes, edges = pg.distributed_expand(["alice"], edge_type="KNOWS", hops=2, direction="out")
    /// ```
    #[pyo3(signature = (seed_ids, edge_type=None, hops=1, direction="out"))]
    fn distributed_expand(
        &self,
        seed_ids: Vec<String>,
        edge_type: Option<&str>,
        hops: u32,
        direction: &str,
    ) -> PyResult<(PyNodeFrame, PyEdgeFrame)> {
        let et = match edge_type {
            Some(edge_type) => EdgeTypeSpec::Single(edge_type.to_owned()),
            None => EdgeTypeSpec::Any,
        };
        let dir = python_to_direction(direction)?;
        let seed_refs: Vec<&str> = seed_ids.iter().map(String::as_str).collect();
        let (nf, ef) = self
            .inner
            .distributed_expand(&seed_refs, &et, hops, dir)
            .map_err(gf_error_to_py_err)?;
        let ef_node_ids: Vec<String> = nf.id_column().iter().flatten().map(str::to_owned).collect();
        Ok((
            PyNodeFrame::new(nf),
            PyEdgeFrame {
                inner: std::sync::Arc::new(ef),
                node_ids: std::sync::Arc::new(ef_node_ids),
            },
        ))
    }
}

#[pymethods]
impl PyAggExpr {
    fn __repr__(&self) -> String {
        format!("AggExpr({:?})", self.inner)
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(PyTypeError::new_err(
            "symbolic aggregations do not support truth-value testing",
        ))
    }

    /// Override the output column name produced by this aggregation.
    ///
    /// ```python
    /// gf.count().alias("follower_count")
    /// gf.sum(gf.col("weight")).alias("total_weight")
    /// ```
    fn alias(&self, name: &str) -> Self {
        Self::new(AggExpr::Alias {
            expr: Box::new(self.inner.clone()),
            name: name.to_owned(),
        })
    }
}

#[pymethods]
impl PyPatternNode {
    #[getter]
    fn alias(&self) -> String {
        self.alias.clone()
    }

    #[getter]
    fn label(&self) -> Option<String> {
        self.label.clone()
    }

    #[getter]
    fn props(&self) -> Vec<String> {
        self.props.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "PatternNode(alias={:?}, label={:?}, props={:?})",
            self.alias, self.label, self.props
        )
    }
}

#[pymethods]
impl PyPatternEdge {
    #[getter]
    fn edge_type(&self) -> Option<String> {
        self.edge_type.clone()
    }

    #[getter]
    fn optional(&self) -> bool {
        self.optional
    }

    #[getter]
    fn min_hops(&self) -> u32 {
        self.min_hops
    }

    #[getter]
    fn max_hops(&self) -> Option<u32> {
        self.max_hops
    }

    fn __repr__(&self) -> String {
        format!(
            "PatternEdge(edge_type={:?}, optional={}, min_hops={}, max_hops={:?})",
            self.edge_type, self.optional, self.min_hops, self.max_hops
        )
    }
}

#[pyfunction]
fn col(name: &str) -> PyExpr {
    PyExpr::new(expr_from_col_name(name))
}

#[pyfunction]
fn count() -> PyAggExpr {
    PyAggExpr::new(AggExpr::Count)
}

#[pyfunction]
fn sum(expr: &Bound<'_, PyAny>) -> PyResult<PyAggExpr> {
    Ok(PyAggExpr::new(AggExpr::Sum {
        expr: normalize_agg_expr_input(expr)?,
    }))
}

#[pyfunction]
fn mean(expr: &Bound<'_, PyAny>) -> PyResult<PyAggExpr> {
    Ok(PyAggExpr::new(AggExpr::Mean {
        expr: normalize_agg_expr_input(expr)?,
    }))
}

#[pyfunction]
fn list(expr: &Bound<'_, PyAny>) -> PyResult<PyAggExpr> {
    Ok(PyAggExpr::new(AggExpr::List {
        expr: normalize_agg_expr_input(expr)?,
    }))
}

#[pyfunction]
fn first(expr: &Bound<'_, PyAny>) -> PyResult<PyAggExpr> {
    Ok(PyAggExpr::new(AggExpr::First {
        expr: normalize_agg_expr_input(expr)?,
    }))
}

#[pyfunction]
fn last(expr: &Bound<'_, PyAny>) -> PyResult<PyAggExpr> {
    Ok(PyAggExpr::new(AggExpr::Last {
        expr: normalize_agg_expr_input(expr)?,
    }))
}

#[pyfunction]
#[pyo3(signature = (alias, label=None, props=None))]
fn node(alias: &str, label: Option<&str>, props: Option<Vec<String>>) -> PyPatternNode {
    PyPatternNode {
        alias: alias.to_owned(),
        label: label.map(str::to_owned),
        props: props.unwrap_or_default(),
    }
}

#[pyfunction]
#[pyo3(signature = (edge_type=None, *, alias=None, optional=false, min_hops=1, max_hops=None))]
fn edge(
    edge_type: Option<&str>,
    alias: Option<&str>,
    optional: bool,
    min_hops: u32,
    max_hops: Option<u32>,
) -> PyResult<PyPatternEdge> {
    if min_hops == 0 {
        return Err(PyValueError::new_err("min_hops must be greater than zero"));
    }
    if let Some(max_hops) = max_hops {
        if max_hops < min_hops {
            return Err(PyValueError::new_err(
                "max_hops must be greater than or equal to min_hops",
            ));
        }
    }

    Ok(PyPatternEdge {
        alias: alias.map(str::to_owned),
        edge_type: edge_type.map(str::to_owned),
        optional,
        min_hops,
        max_hops,
    })
}

#[pyfunction]
#[pyo3(signature = (*, nodes, edges))]
fn graph(nodes: &Bound<'_, PyAny>, edges: &Bound<'_, PyAny>) -> PyResult<PyGraphFrame> {
    graph_from_py_mappings(nodes, edges)
}

#[pyfunction]
#[pyo3(signature = (path, *, label=None, id_col=None, id_prefix=None, columns=None, schema_overrides=None, infer_schema_rows=None, batch_size=65536, has_header=true, delimiter=","))]
fn read_csv_native_py(
    path: &Bound<'_, PyAny>,
    label: Option<String>,
    id_col: Option<String>,
    id_prefix: Option<String>,
    columns: Option<Vec<String>>,
    schema_overrides: Option<&Bound<'_, PyAny>>,
    infer_schema_rows: Option<usize>,
    batch_size: usize,
    has_header: bool,
    delimiter: &str,
) -> PyResult<PyNodeFrame> {
    let path = path_from_py_any(path)?;
    let delimiter = csv_delimiter_byte(delimiter)?;
    let schema_overrides = csv_schema_overrides_from_py(schema_overrides)?;
    let frame = read_csv_nodes(
        path,
        &CsvNodeReadOptions {
            label,
            id_col,
            id_prefix,
            columns,
            schema_overrides,
            infer_schema_rows,
            batch_size,
            has_header,
            delimiter,
        },
    )
    .map_err(gf_error_to_py_err)?;
    Ok(PyNodeFrame::new(frame))
}

#[pyfunction]
fn read_gf(path: &Bound<'_, PyAny>) -> PyResult<PyGraphFrame> {
    let path = path_from_py_any(path)?;
    let source = fs::read_to_string(&path)
        .map_err(GFError::IoError)
        .map_err(gf_error_to_py_err)?;
    let graph = parse_gf(&source)
        .and_then(|document| document.to_graph_frame())
        .map_err(gf_error_to_py_err)?;
    Ok(PyGraphFrame::new(graph))
}

#[pyfunction]
fn read_gfb_py(path: &Bound<'_, PyAny>) -> PyResult<PyGraphFrame> {
    let path = path_from_py_any(path)?;
    let graph = read_gfb(path).map_err(gf_error_to_py_err)?;
    Ok(PyGraphFrame::new(graph))
}

#[pyfunction]
fn read_parquet_graph_py(
    nodes_path: &Bound<'_, PyAny>,
    edges_path: &Bound<'_, PyAny>,
) -> PyResult<PyGraphFrame> {
    let nodes_path = path_from_py_any(nodes_path)?;
    let edges_path = path_from_py_any(edges_path)?;
    let graph = read_parquet_graph(nodes_path, edges_path).map_err(gf_error_to_py_err)?;
    Ok(PyGraphFrame::new(graph))
}

#[pyfunction]
fn write_gf(graph: PyRef<'_, PyGraphFrame>, path: &Bound<'_, PyAny>) -> PyResult<()> {
    let path = path_from_py_any(path)?;
    write_gf_impl(graph.inner.as_ref(), &path)
}

#[pyfunction]
fn write_gfb_py(graph: PyRef<'_, PyGraphFrame>, path: &Bound<'_, PyAny>) -> PyResult<()> {
    let path = path_from_py_any(path)?;
    write_gfb(
        graph.inner.as_ref(),
        path,
        &lynxes_io::GfbWriteOptions::default(),
    )
    .map_err(gf_error_to_py_err)
}

#[pyfunction]
fn write_parquet_graph_py(
    graph: PyRef<'_, PyGraphFrame>,
    nodes_path: &Bound<'_, PyAny>,
    edges_path: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let nodes_path = path_from_py_any(nodes_path)?;
    let edges_path = path_from_py_any(edges_path)?;
    write_parquet_graph(graph.inner.as_ref(), nodes_path, edges_path).map_err(gf_error_to_py_err)
}

/// Create a lazy graph frame backed by a Neo4j database.
///
/// The frame is not executed until `.collect()` is called.
/// Requires the `neo4j` feature (currently uses an unsupported-backend stub
/// that raises a runtime error on `.collect()` unless a real backend is linked).
///
/// ```python
/// lazy = gf.read_neo4j("bolt://localhost:7687", "neo4j", "password")
/// result = lazy.filter_nodes(gf.col("age") > 30).collect()
/// ```
#[pyfunction]
#[pyo3(signature = (uri, user, password, database=None))]
fn read_neo4j(uri: &str, user: &str, password: &str, database: Option<&str>) -> PyLazyGraphFrame {
    let config = Neo4jConfig {
        uri: uri.to_owned(),
        user: user.to_owned(),
        password: password.to_owned(),
        database: database.map(str::to_owned),
    };
    let connector_impl: std::sync::Arc<dyn BackendConnector> =
        std::sync::Arc::new(Neo4jConnector::new(config));
    let connector: std::sync::Arc<dyn Connector> = std::sync::Arc::new(BackendConnectorAdapter {
        inner: connector_impl,
    });
    PyLazyGraphFrame {
        inner: LazyGraphFrame::from_connector(connector),
    }
}

/// Create a lazy graph frame backed by an ArangoDB graph.
///
/// ```python
/// lazy = gf.read_arangodb(
///     endpoint="http://localhost:8529",
///     database="mydb",
///     graph="social",
///     vertex_collection="persons",
///     edge_collection="knows",
///     username="root",
///     password="secret",
/// )
/// ```
#[pyfunction]
#[pyo3(signature = (endpoint, database, graph, vertex_collection, edge_collection, username="", password=""))]
fn read_arangodb(
    endpoint: &str,
    database: &str,
    graph: &str,
    vertex_collection: &str,
    edge_collection: &str,
    username: &str,
    password: &str,
) -> PyLazyGraphFrame {
    let config = ArangoConfig {
        endpoint: endpoint.to_owned(),
        database: database.to_owned(),
        graph: graph.to_owned(),
        vertex_collection: vertex_collection.to_owned(),
        edge_collection: edge_collection.to_owned(),
        username: username.to_owned(),
        password: password.to_owned(),
    };
    let connector_impl: std::sync::Arc<dyn BackendConnector> =
        std::sync::Arc::new(ArangoConnector::new(config));
    let connector: std::sync::Arc<dyn Connector> = std::sync::Arc::new(BackendConnectorAdapter {
        inner: connector_impl,
    });
    PyLazyGraphFrame {
        inner: LazyGraphFrame::from_connector(connector),
    }
}

/// Create a lazy graph frame backed by a SPARQL endpoint.
///
/// ```python
/// lazy = gf.read_sparql(
///     endpoint="https://dbpedia.org/sparql",
///     node_template="SELECT ?id ?label WHERE { ?id rdfs:label ?label }",
///     edge_template="SELECT ?src ?dst WHERE { ?src ?edge ?dst }",
/// )
/// ```
#[pyfunction]
#[pyo3(signature = (endpoint, node_template, edge_template, expand_template=None))]
fn read_sparql(
    endpoint: &str,
    node_template: &str,
    edge_template: &str,
    expand_template: Option<&str>,
) -> PyLazyGraphFrame {
    let config = SparqlConfig {
        endpoint: endpoint.to_owned(),
        node_template: node_template.to_owned(),
        edge_template: edge_template.to_owned(),
        expand_template: expand_template.map(str::to_owned),
    };
    let connector_impl: std::sync::Arc<dyn BackendConnector> =
        std::sync::Arc::new(SparqlConnector::new(config));
    let connector: std::sync::Arc<dyn Connector> = std::sync::Arc::new(BackendConnectorAdapter {
        inner: connector_impl,
    });
    PyLazyGraphFrame {
        inner: LazyGraphFrame::from_connector(connector),
    }
}

#[pyfunction]
fn write_rdf(graph: PyRef<'_, PyGraphFrame>, path: &Bound<'_, PyAny>) -> PyResult<()> {
    let _ = graph;
    let path = path_from_py_any(path)?;
    unsupported_write_impl("write_rdf", &path)
}

#[pyfunction]
fn write_owl(graph: PyRef<'_, PyGraphFrame>, path: &Bound<'_, PyAny>) -> PyResult<()> {
    let _ = graph;
    let path = path_from_py_any(path)?;
    unsupported_write_impl("write_owl", &path)
}

/// Partition a `GraphFrame` into `n_shards` shards.
///
/// Convenience top-level alias for `graph.partition(n_shards, strategy)`.
///
/// Parameters
/// ----------
/// graph : GraphFrame
/// n_shards : int
///     Number of partitions.
/// strategy : str, optional
///     ``"hash"`` (default), ``"range"``, or ``"label"``.
///
/// Returns
/// -------
/// PartitionedGraph
#[pyfunction]
#[pyo3(signature = (graph, n_shards, strategy = "hash"))]
fn partition_graph(
    graph: PyRef<'_, PyGraphFrame>,
    n_shards: usize,
    strategy: &str,
) -> PyResult<PyPartitionedGraph> {
    graph.partition(n_shards, strategy)
}

#[pyfunction]
fn _configure_mojo_runtime(path: &str) -> PyResult<()> {
    configure_mojo_runtime(Path::new(path)).map_err(gf_error_to_py_err)
}

#[pymodule]
#[pyo3(name = "_lynxes")]
fn _lynxes(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", lynxes_core::version())?;
    m.add("String", "String")?;
    m.add("Int", "Int")?;
    m.add("Float", "Float")?;
    m.add("Bool", "Bool")?;
    m.add("Date", "Date")?;
    m.add("DateTime", "DateTime")?;
    m.add("Duration", "Duration")?;
    m.add("StringView", "StringView")?;
    m.add("Any", "Any")?;

    m.add_class::<PyNodeFrame>()?;
    m.add_class::<PyEdgeFrame>()?;
    m.add_class::<PyGraphFrame>()?;
    m.add_class::<PyMutableGraphFrame>()?;
    m.add_class::<PyLazyGraphFrame>()?;
    m.add_class::<PyExpr>()?;
    m.add_class::<PyAggExpr>()?;
    m.add_class::<PyStrExprNamespace>()?;
    m.add_class::<PyPatternNode>()?;
    m.add_class::<PyPatternEdge>()?;
    m.add_class::<PySampledSubgraph>()?;
    m.add_class::<PyPartitionedGraph>()?;

    m.add_function(wrap_pyfunction!(col, m)?)?;
    m.add_function(wrap_pyfunction!(node, m)?)?;
    m.add_function(wrap_pyfunction!(edge, m)?)?;
    m.add_function(wrap_pyfunction!(graph, m)?)?;
    m.add_function(wrap_pyfunction!(count, m)?)?;
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(list, m)?)?;
    m.add_function(wrap_pyfunction!(first, m)?)?;
    m.add_function(wrap_pyfunction!(last, m)?)?;
    m.add_function(wrap_pyfunction!(read_csv_native_py, m)?)?;
    m.add_function(wrap_pyfunction!(read_gf, m)?)?;
    m.add_function(wrap_pyfunction!(read_gfb_py, m)?)?;
    m.add_function(wrap_pyfunction!(read_parquet_graph_py, m)?)?;
    m.add_function(wrap_pyfunction!(write_gf, m)?)?;
    m.add_function(wrap_pyfunction!(write_gfb_py, m)?)?;
    m.add_function(wrap_pyfunction!(write_parquet_graph_py, m)?)?;
    m.add_function(wrap_pyfunction!(write_rdf, m)?)?;
    m.add_function(wrap_pyfunction!(write_owl, m)?)?;
    m.add_function(wrap_pyfunction!(read_neo4j, m)?)?;
    m.add_function(wrap_pyfunction!(read_arangodb, m)?)?;
    m.add_function(wrap_pyfunction!(read_sparql, m)?)?;
    m.add_function(wrap_pyfunction!(partition_graph, m)?)?;
    m.add_function(wrap_pyfunction!(_configure_mojo_runtime, m)?)?;
    m.add("read_gfb", m.getattr("read_gfb_py")?)?;
    m.add("read_parquet_graph", m.getattr("read_parquet_graph_py")?)?;
    m.add("write_gfb", m.getattr("write_gfb_py")?)?;
    m.add("write_parquet_graph", m.getattr("write_parquet_graph_py")?)?;
    Ok(())
}

#[derive(Clone, Copy)]
enum FramePreviewView {
    Table,
    Head,
    Tail,
}

fn frame_order_name(sorted: bool, descending: bool) -> &'static str {
    match (sorted, descending) {
        (false, _) => "Stable",
        (true, false) => "SortedAsc",
        (true, true) => "SortedDesc",
    }
}

fn render_frame_preview(
    header: &str,
    batch: &RecordBatch,
    view: FramePreviewView,
    rows: usize,
    sort_by: Option<String>,
    descending: bool,
    width: Option<usize>,
) -> PyResult<String> {
    let columns: Vec<String> = batch
        .schema_ref()
        .fields()
        .iter()
        .map(|field| field.name().to_owned())
        .collect();
    let ordered_rows = build_frame_row_order(batch, sort_by.as_deref(), descending)?;
    let (top_rows, bottom_rows, omitted_rows) = partition_frame_rows(&ordered_rows, view, rows);

    let mut measured_rows = top_rows.clone();
    measured_rows.extend(bottom_rows.iter().copied());
    let widths = measure_frame_column_widths(batch, &columns, &measured_rows, width);

    let mut out = String::new();
    out.push_str(header);
    out.push('\n');
    for (column, width) in columns.iter().zip(widths.iter().copied()) {
        out.push_str(&format!(
            "{:<width$} ",
            truncate_display(column, width),
            width = width
        ));
    }
    out.push('\n');
    for width in &widths {
        out.push_str(&format!("{:-<width$} ", "", width = *width));
    }
    out.push('\n');

    for row in &top_rows {
        push_frame_row(&mut out, batch, *row, &widths);
    }
    if omitted_rows > 0 {
        out.push_str(&format!("... {} rows omitted ...\n", omitted_rows));
    }
    for row in &bottom_rows {
        push_frame_row(&mut out, batch, *row, &widths);
    }

    Ok(out)
}

fn partition_frame_rows(
    ordered_rows: &[usize],
    view: FramePreviewView,
    max_rows: usize,
) -> (Vec<usize>, Vec<usize>, usize) {
    let max_rows = max_rows.max(1);
    let total = ordered_rows.len();
    match view {
        FramePreviewView::Head => {
            let top = ordered_rows
                .iter()
                .take(max_rows)
                .copied()
                .collect::<Vec<_>>();
            let omitted = total.saturating_sub(top.len());
            (top, Vec::new(), omitted)
        }
        FramePreviewView::Tail => {
            let bottom = ordered_rows
                .iter()
                .rev()
                .take(max_rows)
                .copied()
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>();
            let omitted = total.saturating_sub(bottom.len());
            (Vec::new(), bottom, omitted)
        }
        FramePreviewView::Table => {
            if total <= max_rows {
                (ordered_rows.to_vec(), Vec::new(), 0)
            } else {
                let top_len = max_rows / 2;
                let bottom_len = max_rows - top_len;
                let top = ordered_rows
                    .iter()
                    .take(top_len)
                    .copied()
                    .collect::<Vec<_>>();
                let bottom = ordered_rows
                    .iter()
                    .rev()
                    .take(bottom_len)
                    .copied()
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>();
                let omitted = total.saturating_sub(top.len() + bottom.len());
                (top, bottom, omitted)
            }
        }
    }
}

fn build_frame_row_order(
    batch: &RecordBatch,
    sort_by: Option<&str>,
    descending: bool,
) -> PyResult<Vec<usize>> {
    let mut rows: Vec<usize> = (0..batch.num_rows()).collect();
    let Some(sort_by) = sort_by else {
        return Ok(rows);
    };

    let column_idx = batch
        .schema_ref()
        .index_of(sort_by)
        .map_err(|_| PyKeyError::new_err(format!("column not found: {sort_by}")))?;
    let column = batch.column(column_idx);
    rows.sort_by(|a, b| {
        compare_frame_cells(column.as_ref(), *a, *b, descending).then_with(|| a.cmp(b))
    });
    Ok(rows)
}

fn compare_frame_cells(array: &dyn Array, a: usize, b: usize, descending: bool) -> Ordering {
    let ord = match array.data_type() {
        DataType::Utf8 => {
            let values = array
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("Utf8 array downcasts to StringArray");
            compare_nullable(array, a, b, || values.value(a).cmp(values.value(b)))
        }
        DataType::Boolean => {
            let values = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .expect("Boolean array downcasts to BooleanArray");
            compare_nullable(array, a, b, || values.value(a).cmp(&values.value(b)))
        }
        DataType::Int8 => {
            let values = array
                .as_any()
                .downcast_ref::<Int8Array>()
                .expect("Int8 array downcasts to Int8Array");
            compare_nullable(array, a, b, || values.value(a).cmp(&values.value(b)))
        }
        DataType::Int16 => {
            let values = array
                .as_any()
                .downcast_ref::<Int16Array>()
                .expect("Int16 array downcasts to Int16Array");
            compare_nullable(array, a, b, || values.value(a).cmp(&values.value(b)))
        }
        DataType::Int32 => {
            let values = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32 array downcasts to Int32Array");
            compare_nullable(array, a, b, || values.value(a).cmp(&values.value(b)))
        }
        DataType::Int64 => {
            let values = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64 array downcasts to Int64Array");
            compare_nullable(array, a, b, || values.value(a).cmp(&values.value(b)))
        }
        DataType::UInt32 => {
            let values = array
                .as_any()
                .downcast_ref::<UInt32Array>()
                .expect("UInt32 array downcasts to UInt32Array");
            compare_nullable(array, a, b, || values.value(a).cmp(&values.value(b)))
        }
        DataType::UInt64 => {
            let values = array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .expect("UInt64 array downcasts to UInt64Array");
            compare_nullable(array, a, b, || values.value(a).cmp(&values.value(b)))
        }
        DataType::Float32 => {
            let values = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .expect("Float32 array downcasts to Float32Array");
            compare_nullable(array, a, b, || values.value(a).total_cmp(&values.value(b)))
        }
        DataType::Float64 => {
            let values = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("Float64 array downcasts to Float64Array");
            compare_nullable(array, a, b, || values.value(a).total_cmp(&values.value(b)))
        }
        _ => compare_nullable(array, a, b, || {
            render_frame_cell(array, a).cmp(&render_frame_cell(array, b))
        }),
    };

    if descending {
        ord.reverse()
    } else {
        ord
    }
}

fn compare_nullable<F>(array: &dyn Array, a: usize, b: usize, cmp_values: F) -> Ordering
where
    F: FnOnce() -> Ordering,
{
    match (array.is_null(a), array.is_null(b)) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => cmp_values(),
    }
}

fn measure_frame_column_widths(
    batch: &RecordBatch,
    columns: &[String],
    rows: &[usize],
    width: Option<usize>,
) -> Vec<usize> {
    let mut widths = columns
        .iter()
        .map(|name| name.chars().count())
        .collect::<Vec<_>>();
    for &row in rows {
        for (idx, column) in batch.columns().iter().enumerate() {
            widths[idx] = widths[idx].max(render_frame_cell(column.as_ref(), row).chars().count());
        }
    }

    if let Some(total_width) = width {
        let usable = total_width.saturating_sub(columns.len());
        let per_column_cap = (usable / columns.len().max(1)).max(4);
        for measured in &mut widths {
            *measured = (*measured).min(per_column_cap);
        }
    }

    widths
}

fn push_frame_row(out: &mut String, batch: &RecordBatch, row: usize, widths: &[usize]) {
    for (column, width) in batch.columns().iter().zip(widths.iter().copied()) {
        out.push_str(&format!(
            "{:<width$} ",
            truncate_display(&render_frame_cell(column.as_ref(), row), width),
            width = width
        ));
    }
    out.push('\n');
}

fn truncate_display(value: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let chars = value.chars().collect::<Vec<_>>();
    if chars.len() <= width {
        return value.to_owned();
    }
    if width == 1 {
        return "…".to_owned();
    }
    chars[..width - 1].iter().collect::<String>() + "…"
}

fn render_frame_cell(array: &dyn Array, row: usize) -> String {
    if array.is_null(row) {
        return "-".to_owned();
    }
    arrow::util::display::array_value_to_string(array, row)
        .unwrap_or_else(|_| "<invalid>".to_owned())
        .replace('\n', "\\n")
}

fn render_node_frame_info(frame: &NodeFrame) -> String {
    let batch = frame.to_record_batch();
    let labels = collect_node_labels(batch);
    let user_columns = frame_user_columns(batch, &NODE_RESERVED_COLUMNS);

    let mut out = String::new();
    out.push_str("NodeFrame info\n");
    out.push_str(&format!("  rows: {}\n", frame.len()));
    out.push_str(&format!("  columns: {}\n", batch.num_columns()));
    out.push_str(&format!(
        "  reserved columns: {}\n",
        NODE_RESERVED_COLUMNS.join(", ")
    ));
    out.push_str(&format!(
        "  labels: {}\n",
        if labels.is_empty() {
            "(none)".to_owned()
        } else {
            labels.join(", ")
        }
    ));
    out.push_str(&format!(
        "  user columns: {}\n",
        if user_columns.is_empty() {
            "(none)".to_owned()
        } else {
            user_columns.join(", ")
        }
    ));
    out
}

fn render_edge_frame_info(frame: &EdgeFrame) -> String {
    let batch = frame.to_record_batch();
    let mut edge_types = frame
        .edge_types()
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    edge_types.sort();
    let user_columns = frame_user_columns(batch, &EDGE_RESERVED_COLUMNS);

    let mut out = String::new();
    out.push_str("EdgeFrame info\n");
    out.push_str(&format!("  rows: {}\n", frame.len()));
    out.push_str(&format!("  columns: {}\n", batch.num_columns()));
    out.push_str(&format!("  unique node ids: {}\n", frame.node_count()));
    out.push_str(&format!(
        "  edge types: {}\n",
        if edge_types.is_empty() {
            "(none)".to_owned()
        } else {
            edge_types.join(", ")
        }
    ));
    out.push_str(&format!(
        "  user columns: {}\n",
        if user_columns.is_empty() {
            "(none)".to_owned()
        } else {
            user_columns.join(", ")
        }
    ));
    out
}

fn render_frame_schema(kind: &str, batch: &RecordBatch, reserved: &[&str]) -> String {
    let mut out = String::new();
    out.push_str(&format!("{kind} schema\n"));
    for field in batch.schema_ref().fields() {
        out.push_str(&format!(
            "  {:<14} {:<24} {:<8} {}\n",
            field.name(),
            field.data_type(),
            if field.is_nullable() {
                "nullable"
            } else {
                "required"
            },
            if reserved.contains(&field.name().as_str()) {
                "reserved"
            } else {
                "user"
            }
        ));
    }
    out
}

fn render_frame_glimpse(
    kind: &str,
    batch: &RecordBatch,
    rows: usize,
    sort_by: Option<String>,
    descending: bool,
    width: Option<usize>,
) -> PyResult<String> {
    let ordered_rows = build_frame_row_order(batch, sort_by.as_deref(), descending)?;
    let sampled_rows = ordered_rows
        .into_iter()
        .take(rows.max(1))
        .collect::<Vec<_>>();
    let sample_width = width.map(|w| (w / 3).max(8));

    let mut out = String::new();
    out.push_str(&format!(
        "{kind} glimpse (rows sampled: {})\n",
        sampled_rows.len()
    ));
    for (field, column) in batch.schema_ref().fields().iter().zip(batch.columns()) {
        let samples = sampled_rows
            .iter()
            .map(|row| render_frame_cell(column.as_ref(), *row))
            .map(|value| {
                sample_width
                    .map(|limit| truncate_display(&value, limit))
                    .unwrap_or(value)
            })
            .collect::<Vec<_>>();
        out.push_str(&format!(
            "  {:<12} {:<18} {}\n",
            field.name(),
            field.data_type(),
            if samples.is_empty() {
                "-".to_owned()
            } else {
                samples.join(" | ")
            }
        ));
    }
    Ok(out)
}

fn describe_node_frame(frame: &NodeFrame, mode: &str) -> PyResult<String> {
    match mode {
        "all" => {
            let mut out = String::new();
            out.push_str(&describe_node_frame(frame, "types")?);
            out.push('\n');
            out.push_str(&describe_node_frame(frame, "attrs")?);
            out.push('\n');
            out.push_str(&describe_node_frame(frame, "structure")?);
            Ok(out)
        }
        "types" => Ok(render_node_frame_types(frame)),
        "attrs" => Ok(render_frame_attr_describe(
            "node",
            frame.to_record_batch(),
            &NODE_RESERVED_COLUMNS,
        )),
        "structure" => Ok(render_node_frame_structure(frame)),
        other => Err(PyValueError::new_err(format!(
            "unsupported describe mode: {other}; expected one of: all, types, attrs, structure"
        ))),
    }
}

fn describe_edge_frame(frame: &EdgeFrame, mode: &str) -> PyResult<String> {
    match mode {
        "all" => {
            let mut out = String::new();
            out.push_str(&describe_edge_frame(frame, "types")?);
            out.push('\n');
            out.push_str(&describe_edge_frame(frame, "attrs")?);
            out.push('\n');
            out.push_str(&describe_edge_frame(frame, "structure")?);
            Ok(out)
        }
        "types" => Ok(render_edge_frame_types(frame)),
        "attrs" => Ok(render_frame_attr_describe(
            "edge",
            frame.to_record_batch(),
            &EDGE_RESERVED_COLUMNS,
        )),
        "structure" => Ok(render_edge_frame_structure(frame)),
        other => Err(PyValueError::new_err(format!(
            "unsupported describe mode: {other}; expected one of: all, types, attrs, structure"
        ))),
    }
}

fn render_node_frame_types(frame: &NodeFrame) -> String {
    let labels = collect_node_labels(frame.to_record_batch());
    let mut out = String::new();
    out.push_str("Types\n");
    out.push_str(&format!(
        "  labels: {}\n",
        if labels.is_empty() {
            "(none)".to_owned()
        } else {
            labels.join(", ")
        }
    ));
    out.push_str("  fields\n");
    for field in frame.to_record_batch().schema_ref().fields() {
        out.push_str(&format!("    {:<14} {}\n", field.name(), field.data_type()));
    }
    out
}

fn render_edge_frame_types(frame: &EdgeFrame) -> String {
    let mut edge_types = frame
        .edge_types()
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    edge_types.sort();
    let mut out = String::new();
    out.push_str("Types\n");
    out.push_str(&format!(
        "  edge types: {}\n",
        if edge_types.is_empty() {
            "(none)".to_owned()
        } else {
            edge_types.join(", ")
        }
    ));
    out.push_str("  fields\n");
    for field in frame.to_record_batch().schema_ref().fields() {
        out.push_str(&format!("    {:<14} {}\n", field.name(), field.data_type()));
    }
    out
}

fn render_node_frame_structure(frame: &NodeFrame) -> String {
    let labels = collect_node_labels(frame.to_record_batch());
    let mut out = String::new();
    out.push_str("Structure\n");
    out.push_str(&format!("  rows: {}\n", frame.len()));
    out.push_str(&format!(
        "  columns: {}\n",
        frame.to_record_batch().num_columns()
    ));
    out.push_str(&format!("  distinct labels: {}\n", labels.len()));
    out
}

fn render_edge_frame_structure(frame: &EdgeFrame) -> String {
    let mut edge_types = frame
        .edge_types()
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    edge_types.sort();
    let mut out = String::new();
    out.push_str("Structure\n");
    out.push_str(&format!("  rows: {}\n", frame.len()));
    out.push_str(&format!(
        "  columns: {}\n",
        frame.to_record_batch().num_columns()
    ));
    out.push_str(&format!("  unique node ids: {}\n", frame.node_count()));
    out.push_str(&format!("  edge type count: {}\n", edge_types.len()));
    out
}

fn render_frame_attr_describe(kind: &str, batch: &RecordBatch, reserved: &[&str]) -> String {
    let user_fields = batch
        .schema_ref()
        .fields()
        .iter()
        .filter(|field| !reserved.contains(&field.name().as_str()))
        .collect::<Vec<_>>();

    let mut out = String::new();
    out.push_str("Attributes\n");
    if user_fields.is_empty() {
        out.push_str(&format!("  {kind}: (none)\n"));
        return out;
    }

    out.push_str(&format!("  {kind}\n"));
    for field in user_fields {
        let column = batch
            .column_by_name(field.name())
            .expect("field present in record batch");
        let mut non_null_count = 0usize;
        let mut null_count = 0usize;
        let mut distinct = BTreeSet::new();
        let mut samples = Vec::new();

        for row in 0..batch.num_rows() {
            if column.is_null(row) {
                null_count += 1;
                continue;
            }
            non_null_count += 1;
            let value = render_frame_cell(column.as_ref(), row);
            distinct.insert(value.clone());
            if samples.len() < 3 && !samples.contains(&value) {
                samples.push(value);
            }
        }

        out.push_str(&format!(
            "    {:<18} {:<16} non-null={} null={} distinct={} samples={}\n",
            format!("{kind}.{}", field.name()),
            field.data_type(),
            non_null_count,
            null_count,
            distinct.len(),
            if samples.is_empty() {
                "-".to_owned()
            } else {
                samples.join(" | ")
            }
        ));
    }
    out
}

fn frame_user_columns(batch: &RecordBatch, reserved: &[&str]) -> Vec<String> {
    batch
        .schema_ref()
        .fields()
        .iter()
        .map(|field| field.name().to_owned())
        .filter(|name| !reserved.contains(&name.as_str()))
        .collect()
}

fn collect_node_labels(batch: &RecordBatch) -> Vec<String> {
    let Some(column) = batch.column_by_name(COL_NODE_LABEL) else {
        return Vec::new();
    };
    let Some(labels) = column.as_any().downcast_ref::<ListArray>() else {
        return Vec::new();
    };

    let mut values = BTreeSet::new();
    for row in 0..labels.len() {
        if labels.is_null(row) {
            continue;
        }
        let item = labels.value(row);
        let Some(strings) = item.as_any().downcast_ref::<StringArray>() else {
            continue;
        };
        for idx in 0..strings.len() {
            if strings.is_null(idx) {
                continue;
            }
            values.insert(strings.value(idx).to_owned());
        }
    }

    values.into_iter().collect()
}

fn render_python_display_slice(slice: &DisplaySlice) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "GraphFrame(rows={}, nodes={}, edges={}, isolated={}, order={})\n",
        slice.graph_summary.projected_row_count,
        slice.graph_summary.node_count,
        slice.graph_summary.edge_count,
        slice.graph_summary.isolated_node_count,
        slice.order_name
    ));

    let widths: Vec<usize> = slice
        .columns
        .iter()
        .map(|column| column.width.max(column.name.len()))
        .collect();

    for (column, width) in slice.columns.iter().zip(widths.iter().copied()) {
        out.push_str(&format!("{:<width$} ", column.name, width = width));
    }
    out.push('\n');
    for width in &widths {
        out.push_str(&format!("{:-<width$} ", "", width = *width));
    }
    out.push('\n');

    for row in &slice.top_rows {
        push_display_row(&mut out, row.values.clone(), &slice.columns, &widths);
    }
    if slice.omitted_rows > 0 {
        out.push_str(&format!("... {} rows omitted ...\n", slice.omitted_rows));
    }
    for row in &slice.bottom_rows {
        push_display_row(&mut out, row.values.clone(), &slice.columns, &widths);
    }
    out
}

fn push_display_row(
    out: &mut String,
    values: BTreeMap<String, String>,
    columns: &[lynxes_core::DisplayColumn],
    widths: &[usize],
) {
    for ((column, width), value) in columns.iter().zip(widths.iter().copied()).zip(
        columns
            .iter()
            .map(|column| values.get(&column.name).cloned().unwrap_or_default()),
    ) {
        out.push_str(&format!(
            "{:<width$} ",
            value,
            width = width.max(column.name.len())
        ));
    }
    out.push('\n');
}

fn render_python_info(info: &GraphInfo) -> String {
    let mut out = String::new();
    out.push_str("Graph info\n");
    out.push_str(&format!("  nodes: {}\n", info.summary.node_count));
    out.push_str(&format!("  edges: {}\n", info.summary.edge_count));
    out.push_str(&format!(
        "  isolated nodes: {}\n",
        info.summary.isolated_node_count
    ));
    out.push_str(&format!("  directedness: {}\n", info.summary.directedness));
    out.push_str(&format!("  self loops: {}\n", info.self_loops));
    out.push_str(&format!("  multi-edge keys: {}\n", info.multi_edge_pairs));
    out.push_str(&format!(
        "  schema: {}\n\n",
        if info.schema_present {
            "declared"
        } else {
            "observed"
        }
    ));
    out.push_str(&format!(
        "Node labels: {}\n",
        if info.node_labels.is_empty() {
            "(none)".to_owned()
        } else {
            info.node_labels.join(", ")
        }
    ));
    out.push_str(&format!(
        "Edge types: {}\n",
        if info.edge_types.is_empty() {
            "(none)".to_owned()
        } else {
            info.edge_types.join(", ")
        }
    ));
    out.push_str(&format!(
        "Node attrs: {}\n",
        if info.node_attribute_keys.is_empty() {
            "(none)".to_owned()
        } else {
            info.node_attribute_keys.join(", ")
        }
    ));
    out.push_str(&format!(
        "Edge attrs: {}\n",
        if info.edge_attribute_keys.is_empty() {
            "(none)".to_owned()
        } else {
            info.edge_attribute_keys.join(", ")
        }
    ));
    out
}

fn render_python_schema(schema: &SchemaSummary) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "Schema ({})\n",
        if schema.declared {
            "declared"
        } else {
            "observed"
        }
    ));
    out.push_str(&format!(
        "Node labels: {}\n",
        if schema.node_labels.is_empty() {
            "(none)".to_owned()
        } else {
            schema.node_labels.join(", ")
        }
    ));
    out.push_str(&format!(
        "Edge types: {}\n\n",
        if schema.edge_types.is_empty() {
            "(none)".to_owned()
        } else {
            schema.edge_types.join(", ")
        }
    ));
    out.push_str("Node fields\n");
    for field in &schema.node_fields {
        out.push_str(&format!(
            "  {:<14} {:<18} {:<8} {}\n",
            field.name,
            field.dtype,
            if field.nullable {
                "nullable"
            } else {
                "required"
            },
            if field.reserved { "reserved" } else { "user" }
        ));
    }
    out.push_str("\nEdge fields\n");
    for field in &schema.edge_fields {
        out.push_str(&format!(
            "  {:<14} {:<18} {:<8} {}\n",
            field.name,
            field.dtype,
            if field.nullable {
                "nullable"
            } else {
                "required"
            },
            if field.reserved { "reserved" } else { "user" }
        ));
    }
    out
}

fn render_python_glimpse(glimpse: &GlimpseSummary) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "Glimpse (rows sampled: {})\n",
        glimpse.rows_sampled
    ));
    for column in &glimpse.columns {
        out.push_str(&format!(
            "  {:<12} {:<18} {}\n",
            column.name,
            column.dtype,
            if column.samples.is_empty() {
                "-".to_owned()
            } else {
                column.samples.join(" | ")
            }
        ));
    }
    out
}

fn describe_graph(graph: &GraphFrame, mode: &str) -> PyResult<String> {
    match mode {
        "all" => {
            let mut out = String::new();
            out.push_str(&describe_graph(graph, "types")?);
            out.push('\n');
            out.push_str(&describe_graph(graph, "attrs")?);
            out.push('\n');
            out.push_str(&describe_graph(graph, "structure")?);
            Ok(out)
        }
        "types" => Ok(render_describe_types(&graph.display_schema())),
        "attrs" => Ok(render_describe_attrs(&graph.display_attr_stats())),
        "structure" => Ok(render_describe_structure(
            &graph
                .display_structure_stats()
                .map_err(gf_error_to_py_err)?,
        )),
        other => Err(PyValueError::new_err(format!(
            "unsupported describe mode: {other}; expected one of: all, types, attrs, structure"
        ))),
    }
}

fn render_describe_types(schema: &SchemaSummary) -> String {
    let mut out = String::new();
    out.push_str("Types\n");
    out.push_str(&format!(
        "  node labels: {}\n",
        if schema.node_labels.is_empty() {
            "(none)".to_owned()
        } else {
            schema.node_labels.join(", ")
        }
    ));
    out.push_str(&format!(
        "  edge types: {}\n",
        if schema.edge_types.is_empty() {
            "(none)".to_owned()
        } else {
            schema.edge_types.join(", ")
        }
    ));
    out
}

fn render_describe_attrs(summary: &AttrStatsSummary) -> String {
    let mut out = String::new();
    out.push_str("Attributes\n");
    if summary.node_attrs.is_empty() {
        out.push_str("  node: (none)\n");
    } else {
        out.push_str("  node\n");
        for stat in &summary.node_attrs {
            out.push_str(&format!(
                "    {:<18} {:<16} non-null={} null={} distinct={} samples={}\n",
                stat.qualified_name,
                stat.dtype,
                stat.non_null_count,
                stat.null_count,
                stat.distinct_count,
                if stat.samples.is_empty() {
                    "-".to_owned()
                } else {
                    stat.samples.join(" | ")
                }
            ));
        }
    }
    if summary.edge_attrs.is_empty() {
        out.push_str("  edge: (none)\n");
    } else {
        out.push_str("  edge\n");
        for stat in &summary.edge_attrs {
            out.push_str(&format!(
                "    {:<18} {:<16} non-null={} null={} distinct={} samples={}\n",
                stat.qualified_name,
                stat.dtype,
                stat.non_null_count,
                stat.null_count,
                stat.distinct_count,
                if stat.samples.is_empty() {
                    "-".to_owned()
                } else {
                    stat.samples.join(" | ")
                }
            ));
        }
    }
    out
}

fn render_describe_structure(stats: &StructureStats) -> String {
    let mut out = String::new();
    out.push_str("Structure\n");
    out.push_str(&format!("  density: {:.4}\n", stats.density));
    out.push_str(&format!(
        "  average out-degree: {:.2}\n",
        stats.average_out_degree
    ));
    out.push_str(&format!(
        "  average in-degree: {:.2}\n",
        stats.average_in_degree
    ));
    out.push_str(&format!("  median degree: {:.2}\n", stats.median_degree));
    out.push_str(&format!("  max degree: {}\n", stats.max_degree));
    out.push_str(&format!(
        "  connected components: {}\n",
        stats.connected_components
    ));
    out.push_str(&format!(
        "  largest component share: {:.2}%\n",
        stats.largest_component_share * 100.0
    ));
    out
}

fn extract_boolean_mask(mask: &Bound<'_, PyAny>) -> PyResult<BooleanArray> {
    if let Ok(values) = mask.extract::<Vec<Option<bool>>>() {
        return Ok(BooleanArray::from(values));
    }

    if let Ok(values) = mask.extract::<Vec<bool>>() {
        return Ok(BooleanArray::from(values));
    }

    if let Ok(PyArrowType(array_data)) = mask.extract::<PyArrowType<ArrayData>>() {
        let array = make_array(array_data);
        let array = array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| {
                PyTypeError::new_err(
                    "filter mask must be a boolean sequence or pyarrow.BooleanArray",
                )
            })?;
        return Ok(array.clone());
    }

    Err(PyTypeError::new_err(
        "filter mask must be a boolean sequence or pyarrow.BooleanArray",
    ))
}

fn gf_error_to_py_err(err: GFError) -> PyErr {
    let message = err.to_string();
    match err {
        GFError::NodeNotFound { .. }
        | GFError::EdgeNotFound { .. }
        | GFError::ColumnNotFound { .. }
        | GFError::InvalidPatternAlias { .. } => PyKeyError::new_err(message),

        GFError::ReservedColumnType { .. }
        | GFError::TypeMismatch { .. }
        | GFError::CannotInferType { .. }
        | GFError::TypeInferenceFailed { .. }
        | GFError::InvalidType { .. }
        | GFError::InvalidCast { .. }
        | GFError::DefaultTypeMismatch { .. } => PyTypeError::new_err(message),

        GFError::MissingReservedColumn { .. }
        | GFError::ReservedColumnName { .. }
        | GFError::DuplicateNodeId { .. }
        | GFError::DanglingEdge { .. }
        | GFError::InvalidDirection { .. }
        | GFError::SchemaMismatch { .. }
        | GFError::LengthMismatch { .. }
        | GFError::MissingRequiredField { .. }
        | GFError::UniqueViolation { .. }
        | GFError::CircularInheritance { .. }
        | GFError::SchemaValidation { .. }
        | GFError::ParseError { .. }
        | GFError::InvalidConfig { .. }
        | GFError::NegativeWeight { .. }
        | GFError::DomainMismatch { .. } => PyValueError::new_err(message),

        GFError::UnsupportedOperation { .. } => PyNotImplementedError::new_err(message),
        GFError::IoError(_) => PyOSError::new_err(message),
        GFError::ConnectorError { .. } => PyRuntimeError::new_err(message),
    }
}

fn build_edge_node_ids(frame: &EdgeFrame) -> Vec<String> {
    let batch = frame.to_record_batch();
    let src_col = batch
        .column_by_name(COL_EDGE_SRC)
        .expect("validated EdgeFrame has _src")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("validated EdgeFrame _src is Utf8");
    let dst_col = batch
        .column_by_name(COL_EDGE_DST)
        .expect("validated EdgeFrame has _dst")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("validated EdgeFrame _dst is Utf8");

    let mut node_ids = vec![String::new(); frame.node_count()];
    for row in 0..frame.len() {
        for id in [src_col.value(row), dst_col.value(row)] {
            if let Some(idx) = frame.node_row_idx(id) {
                if node_ids[idx as usize].is_empty() {
                    node_ids[idx as usize] = id.to_owned();
                }
            }
        }
    }

    node_ids
}

fn python_to_direction(value: &str) -> PyResult<Direction> {
    match value {
        "out" => Ok(Direction::Out),
        "in" => Ok(Direction::In),
        "both" => Ok(Direction::Both),
        "none" => Ok(Direction::None),
        other => Err(PyValueError::new_err(format!(
            "invalid direction: {other}; expected one of: out, in, both, none"
        ))),
    }
}

fn normalize_edge_type_spec(edge_type: Option<&Bound<'_, PyAny>>) -> PyResult<EdgeTypeSpec> {
    let Some(edge_type) = edge_type else {
        return Ok(EdgeTypeSpec::Any);
    };

    if edge_type.is_none() {
        return Ok(EdgeTypeSpec::Any);
    }

    if let Ok(value) = edge_type.extract::<String>() {
        return Ok(EdgeTypeSpec::Single(value));
    }

    if let Ok(values) = edge_type.extract::<Vec<String>>() {
        return Ok(match values.len() {
            0 => EdgeTypeSpec::Any,
            1 => EdgeTypeSpec::Single(values.into_iter().next().unwrap()),
            _ => EdgeTypeSpec::Multiple(values),
        });
    }

    Err(PyTypeError::new_err(
        "edge_type must be None, a string, or a sequence of strings",
    ))
}

fn shortest_path_edge_type(edge_type: Option<String>) -> EdgeTypeSpec {
    match edge_type {
        Some(edge_type) => EdgeTypeSpec::Single(edge_type),
        None => EdgeTypeSpec::Any,
    }
}

fn expr_from_col_name(name: &str) -> Expr {
    if let Some((alias, field)) = name.split_once('.') {
        if !alias.is_empty() && !field.is_empty() {
            return Expr::PatternCol {
                alias: alias.to_owned(),
                field: field.to_owned(),
            };
        }
    }

    Expr::Col {
        name: name.to_owned(),
    }
}

fn normalize_agg_expr_input(value: &Bound<'_, PyAny>) -> PyResult<Expr> {
    if let Ok(expr) = value.extract::<PyRef<'_, PyExpr>>() {
        return Ok(expr.inner.clone());
    }

    if let Ok(name) = value.extract::<String>() {
        return Ok(expr_from_col_name(&name));
    }

    Err(PyTypeError::new_err(
        "aggregate helpers expect an Expr or column name string",
    ))
}

fn expr_from_py_operand(value: &Bound<'_, PyAny>) -> PyResult<Expr> {
    if let Ok(expr) = value.extract::<PyRef<'_, PyExpr>>() {
        return Ok(expr.inner.clone());
    }

    Ok(Expr::Literal {
        value: scalar_from_py_any(value)?,
    })
}

fn scalar_from_py_any(value: &Bound<'_, PyAny>) -> PyResult<ScalarValue> {
    if value.is_none() {
        return Ok(ScalarValue::Null);
    }

    if let Ok(value) = value.extract::<bool>() {
        return Ok(ScalarValue::Bool(value));
    }

    if let Ok(value) = value.extract::<i64>() {
        return Ok(ScalarValue::Int(value));
    }

    if let Ok(value) = value.extract::<f64>() {
        return Ok(ScalarValue::Float(value));
    }

    if let Ok(value) = value.extract::<String>() {
        return Ok(ScalarValue::String(value));
    }

    if let Ok(values) = value.downcast::<PyList>() {
        return scalar_list_from_iter(values.iter());
    }

    if let Ok(values) = value.downcast::<PyTuple>() {
        return scalar_list_from_iter(values.iter());
    }

    Err(PyTypeError::new_err(
        "expected an Expr or a supported literal (None, bool, int, float, str, homogeneous list)",
    ))
}

fn scalar_list_from_iter<'py, I>(iter: I) -> PyResult<ScalarValue>
where
    I: IntoIterator<Item = Bound<'py, PyAny>>,
{
    let values: Vec<ScalarValue> = iter
        .into_iter()
        .map(|item| scalar_from_py_any(&item))
        .collect::<PyResult<_>>()?;

    ensure_homogeneous_scalar_list(&values)?;
    Ok(ScalarValue::List(values))
}

fn ensure_homogeneous_scalar_list(values: &[ScalarValue]) -> PyResult<()> {
    if let Some(first) = values.first() {
        let first_tag = scalar_type_tag(first);
        if values
            .iter()
            .any(|value| scalar_type_tag(value) != first_tag)
        {
            return Err(PyTypeError::new_err(
                "list literals must be homogeneous to lower into ScalarValue::List",
            ));
        }
    }

    Ok(())
}

fn scalar_type_tag(value: &ScalarValue) -> &'static str {
    match value {
        ScalarValue::Null => "null",
        ScalarValue::String(_) => "string",
        ScalarValue::Int(_) => "int",
        ScalarValue::Float(_) => "float",
        ScalarValue::Bool(_) => "bool",
        ScalarValue::List(_) => "list",
    }
}

fn extract_dtype(dtype: &Bound<'_, PyAny>) -> PyResult<DataType> {
    let dtype = dtype
        .extract::<String>()
        .map_err(|_| PyTypeError::new_err("dtype must be a Lynxes dtype marker or string"))?;

    match dtype.as_str() {
        "String" => Ok(DataType::Utf8),
        "StringView" | "Utf8View" => Ok(DataType::Utf8View),
        "Int" => Ok(DataType::Int64),
        "Float" => Ok(DataType::Float64),
        "Bool" => Ok(DataType::Boolean),
        "Null" => Ok(DataType::Null),
        other => Err(PyTypeError::new_err(format!(
            "unsupported Lynxes dtype marker: {other}"
        ))),
    }
}

fn path_from_py_any(path: &Bound<'_, PyAny>) -> PyResult<PathBuf> {
    if let Ok(path) = path.extract::<PathBuf>() {
        return Ok(path);
    }

    if let Ok(path) = path.extract::<String>() {
        return Ok(PathBuf::from(path));
    }

    Err(PyTypeError::new_err(
        "path arguments must be str or os.PathLike[str]",
    ))
}

fn csv_delimiter_byte(delimiter: &str) -> PyResult<u8> {
    let bytes = delimiter.as_bytes();
    if bytes.len() != 1 {
        return Err(PyValueError::new_err(
            "delimiter must be a single-byte character",
        ));
    }
    Ok(bytes[0])
}

fn csv_schema_overrides_from_py(
    schema_overrides: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<(String, DataType)>> {
    let Some(schema_overrides) = schema_overrides else {
        return Ok(Vec::new());
    };
    if schema_overrides.is_none() {
        return Ok(Vec::new());
    }

    let dict = schema_overrides
        .downcast::<pyo3::types::PyDict>()
        .map_err(|_| {
            PyTypeError::new_err("schema_overrides must be a dict[str, Lynxes dtype marker]")
        })?;
    let mut out = Vec::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let name = key
            .extract::<String>()
            .map_err(|_| PyTypeError::new_err("schema_overrides keys must be strings"))?;
        out.push((name, extract_dtype(&value)?));
    }
    Ok(out)
}

fn record_batch_to_py_rows(batch: &RecordBatch, py: Python<'_>) -> PyResult<PyObject> {
    let rows = PyList::empty_bound(py);
    let schema = batch.schema_ref();

    for row_idx in 0..batch.num_rows() {
        let row = PyDict::new_bound(py);
        for (col_idx, field) in schema.fields().iter().enumerate() {
            let value = array_value_to_py_object(batch.column(col_idx).as_ref(), row_idx, py)?;
            row.set_item(field.name(), value)?;
        }
        rows.append(row)?;
    }

    Ok(rows.into_py(py))
}

fn array_value_to_py_object(
    array: &dyn Array,
    row_idx: usize,
    py: Python<'_>,
) -> PyResult<PyObject> {
    if array.is_null(row_idx) {
        return Ok(py.None());
    }

    match array.data_type() {
        DataType::Utf8 => {
            let array = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| PyRuntimeError::new_err("Utf8 column has non-StringArray storage"))?;
            Ok(array.value(row_idx).into_py(py))
        }
        DataType::LargeUtf8 => {
            let array = array.as_any().downcast_ref::<LargeStringArray>().ok_or_else(|| {
                PyRuntimeError::new_err("LargeUtf8 column has non-LargeStringArray storage")
            })?;
            Ok(array.value(row_idx).into_py(py))
        }
        DataType::Utf8View => {
            let array = array.as_any().downcast_ref::<StringViewArray>().ok_or_else(|| {
                PyRuntimeError::new_err("Utf8View column has non-StringViewArray storage")
            })?;
            Ok(array.value(row_idx).into_py(py))
        }
        DataType::Int8 => {
            let array = array
                .as_any()
                .downcast_ref::<Int8Array>()
                .ok_or_else(|| PyRuntimeError::new_err("Int8 column has non-Int8Array storage"))?;
            Ok(array.value(row_idx).into_py(py))
        }
        DataType::Int16 => {
            let array = array
                .as_any()
                .downcast_ref::<Int16Array>()
                .ok_or_else(|| PyRuntimeError::new_err("Int16 column has non-Int16Array storage"))?;
            Ok(array.value(row_idx).into_py(py))
        }
        DataType::Int32 => {
            let array = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| PyRuntimeError::new_err("Int32 column has non-Int32Array storage"))?;
            Ok(array.value(row_idx).into_py(py))
        }
        DataType::Int64 => {
            let array = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| PyRuntimeError::new_err("Int64 column has non-Int64Array storage"))?;
            Ok(array.value(row_idx).into_py(py))
        }
        DataType::UInt32 => {
            let array = array.as_any().downcast_ref::<UInt32Array>().ok_or_else(|| {
                PyRuntimeError::new_err("UInt32 column has non-UInt32Array storage")
            })?;
            Ok(array.value(row_idx).into_py(py))
        }
        DataType::UInt64 => {
            let array = array.as_any().downcast_ref::<UInt64Array>().ok_or_else(|| {
                PyRuntimeError::new_err("UInt64 column has non-UInt64Array storage")
            })?;
            Ok(array.value(row_idx).into_py(py))
        }
        DataType::Float32 => {
            let array = array.as_any().downcast_ref::<Float32Array>().ok_or_else(|| {
                PyRuntimeError::new_err("Float32 column has non-Float32Array storage")
            })?;
            Ok(array.value(row_idx).into_py(py))
        }
        DataType::Float64 => {
            let array = array.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
                PyRuntimeError::new_err("Float64 column has non-Float64Array storage")
            })?;
            Ok(array.value(row_idx).into_py(py))
        }
        DataType::Boolean => {
            let array = array.as_any().downcast_ref::<BooleanArray>().ok_or_else(|| {
                PyRuntimeError::new_err("Boolean column has non-BooleanArray storage")
            })?;
            Ok(array.value(row_idx).into_py(py))
        }
        DataType::List(_) => {
            let array = array
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| PyRuntimeError::new_err("List column has non-ListArray storage"))?;
            let values = array.value(row_idx);
            let list = PyList::empty_bound(py);
            for value_idx in 0..values.len() {
                list.append(array_value_to_py_object(values.as_ref(), value_idx, py)?)?;
            }
            Ok(list.into_py(py))
        }
        other => Err(PyTypeError::new_err(format!(
            "to_rows() does not support Arrow dtype {other:?} yet; use to_pyarrow().to_pylist() as a fallback"
        ))),
    }
}

#[derive(Debug, Clone, Copy)]
enum FrameKind {
    Node,
    Edge,
}

fn graph_from_py_mappings(
    nodes: &Bound<'_, PyAny>,
    edges: &Bound<'_, PyAny>,
) -> PyResult<PyGraphFrame> {
    let node_batch = record_batch_from_py_mapping(nodes, FrameKind::Node)?;
    let edge_batch = record_batch_from_py_mapping(edges, FrameKind::Edge)?;
    let nodes = NodeFrame::from_record_batch(node_batch).map_err(gf_error_to_py_err)?;
    let edges = EdgeFrame::from_record_batch(edge_batch).map_err(gf_error_to_py_err)?;
    let graph = GraphFrame::new(nodes, edges).map_err(gf_error_to_py_err)?;
    Ok(PyGraphFrame::new(graph))
}

fn record_batch_from_py_mapping(
    data: &Bound<'_, PyAny>,
    frame_kind: FrameKind,
) -> PyResult<RecordBatch> {
    let dict = data
        .downcast::<pyo3::types::PyDict>()
        .map_err(|_| PyTypeError::new_err("expected a dict[str, list] column mapping"))?;

    let mut fields = Vec::with_capacity(dict.len());
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(dict.len());
    let mut expected_len: Option<usize> = None;

    for (key, value) in dict.iter() {
        let name = key
            .extract::<String>()
            .map_err(|_| PyTypeError::new_err("column mapping keys must be strings"))?;
        let column_values = py_column_from_any(&value, &name, frame_kind)?;
        let len = column_values.len();

        if let Some(expected) = expected_len {
            if len != expected {
                return Err(PyValueError::new_err(format!(
                    "all columns must have the same length (column {name} has {len}, expected {expected})"
                )));
            }
        } else {
            expected_len = Some(len);
        }

        let data_type = infer_column_dtype(&name, frame_kind, &column_values)?;
        let nullable = column_values.iter().any(py_scalar_is_null)
            || column_values.iter().any(|value| match value {
                ScalarValue::List(items) => items.iter().any(matches_scalar_null),
                _ => false,
            });
        let array = build_array_from_scalar_values(&column_values, &data_type)?;
        fields.push(Field::new(&name, data_type, nullable));
        arrays.push(array);
    }

    RecordBatch::try_new(Arc::new(Schema::new(Fields::from(fields))), arrays)
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

fn record_batch_from_pyarrow_input(input: &Bound<'_, PyAny>) -> PyResult<RecordBatch> {
    if let Ok(batch) = input.extract::<PyArrowType<RecordBatch>>() {
        return Ok(batch.0);
    }

    if input.hasattr("combine_chunks")? && input.hasattr("to_batches")? {
        let combined = input.call_method0("combine_chunks")?;
        return single_record_batch_from_pyarrow_table(&combined);
    }

    Err(PyTypeError::new_err(
        "NodeFrame.from_arrow expects a pyarrow.RecordBatch or pyarrow.Table",
    ))
}

fn single_record_batch_from_pyarrow_table(table: &Bound<'_, PyAny>) -> PyResult<RecordBatch> {
    let batches = table.call_method0("to_batches")?;
    let batches = batches
        .downcast::<PyList>()
        .map_err(|_| PyTypeError::new_err("pyarrow.Table.to_batches() did not return a list"))?;

    if batches.len() != 1 {
        return Err(PyValueError::new_err(format!(
            "expected combine_chunks().to_batches() to produce one RecordBatch, got {}",
            batches.len()
        )));
    }

    batches
        .get_item(0)?
        .extract::<PyArrowType<RecordBatch>>()
        .map(|batch| batch.0)
        .map_err(|err| PyTypeError::new_err(format!("failed to read pyarrow RecordBatch: {err}")))
}

fn node_feature_columns(
    frame: &NodeFrame,
    include: Option<Vec<String>>,
    exclude_reserved: bool,
    numeric_only: bool,
) -> PyResult<Vec<String>> {
    let schema = frame.schema();
    let names = match include {
        Some(include) => {
            for name in &include {
                if schema.field_with_name(name).is_err() {
                    return Err(PyKeyError::new_err(format!("column not found: {name}")));
                }
            }
            include
        }
        None => frame
            .column_names()
            .into_iter()
            .map(str::to_owned)
            .collect(),
    };

    let mut out = Vec::new();
    for name in names {
        if exclude_reserved && is_reserved_node_column(&name) {
            continue;
        }
        let field = schema
            .field_with_name(&name)
            .map_err(|_| PyKeyError::new_err(format!("column not found: {name}")))?;
        if numeric_only && !is_numeric_arrow_type(field.data_type()) {
            continue;
        }
        out.push(name);
    }
    Ok(out)
}

fn resolve_feature_columns_for_export(
    frame: &NodeFrame,
    columns: Option<Vec<String>>,
) -> PyResult<Vec<String>> {
    let columns = match columns {
        Some(columns) => columns,
        None => node_feature_columns(frame, None, true, true)?,
    };

    for column in &columns {
        let field = frame
            .schema()
            .field_with_name(column)
            .map_err(|_| PyKeyError::new_err(format!("column not found: {column}")))?;
        if !is_numeric_arrow_type(field.data_type()) {
            return Err(PyTypeError::new_err(format!(
                "column {column} has non-numeric type {:?}; choose numeric feature columns",
                field.data_type()
            )));
        }
    }
    Ok(columns)
}

fn is_reserved_node_column(name: &str) -> bool {
    name.starts_with('_')
}

fn is_numeric_arrow_type(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
    )
}

fn selected_node_batch_to_pyarrow(
    frame: &NodeFrame,
    indices: Option<&Bound<'_, PyAny>>,
    py: Python<'_>,
) -> PyResult<PyObject> {
    let batch = if let Some(indices) = indices {
        let row_ids = extract_row_indices(indices, frame.len())?;
        frame.gather_rows(&row_ids).map_err(gf_error_to_py_err)?
    } else {
        frame.to_record_batch().clone()
    };

    batch
        .to_pyarrow(py)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

fn select_pyarrow_columns(batch: &Bound<'_, PyAny>, columns: &[String]) -> PyResult<PyObject> {
    let py = batch.py();
    let columns = PyList::new_bound(py, columns);
    batch
        .call_method1("select", (columns,))
        .map(|selected| selected.unbind())
}

fn extract_row_indices(indices: &Bound<'_, PyAny>, len: usize) -> PyResult<Vec<u32>> {
    if let Ok(values) = extract_row_indices_via_numpy(indices, len) {
        return Ok(values);
    }

    let normalized = if indices.hasattr("detach")? {
        indices
            .call_method0("detach")?
            .call_method0("cpu")?
            .call_method0("tolist")?
    } else if indices.hasattr("to_pylist")? {
        indices.call_method0("to_pylist")?
    } else if indices.hasattr("tolist")? {
        indices.call_method0("tolist")?
    } else {
        indices.clone()
    };

    let values = normalized.extract::<Vec<i64>>().map_err(|_| {
        PyTypeError::new_err(
            "indices must be a one-dimensional sequence, numpy array, pyarrow array, or torch tensor of integers",
        )
    })?;

    values
        .into_iter()
        .map(|idx| {
            if idx < 0 || idx as usize >= len {
                Err(PyIndexError::new_err(format!(
                    "row index {idx} is out of bounds for NodeFrame of length {len}"
                )))
            } else {
                Ok(idx as u32)
            }
        })
        .collect()
}

fn extract_row_indices_via_numpy(indices: &Bound<'_, PyAny>, len: usize) -> PyResult<Vec<u32>> {
    let py = indices.py();
    let numpy = py.import_bound("numpy")?;
    let source = if indices.hasattr("detach")? {
        indices
            .call_method0("detach")?
            .call_method0("cpu")?
            .call_method0("numpy")?
    } else if indices.hasattr("to_numpy")? {
        indices.call_method0("to_numpy")?
    } else {
        indices.clone()
    };

    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("dtype", numpy.getattr("int64")?)?;
    let array = numpy.call_method("asarray", (source,), Some(&kwargs))?;
    let ndim = array.getattr("ndim")?.extract::<usize>()?;
    if ndim != 1 {
        return Err(PyTypeError::new_err("indices must be one-dimensional"));
    }

    let bytes_obj = array.call_method0("tobytes")?;
    let bytes = bytes_obj.downcast::<PyBytes>()?;
    let raw = bytes.as_bytes();
    if raw.len() % std::mem::size_of::<i64>() != 0 {
        return Err(PyTypeError::new_err("indices buffer is not int64-aligned"));
    }

    raw.chunks_exact(std::mem::size_of::<i64>())
        .map(|chunk| {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(chunk);
            let idx = i64::from_ne_bytes(buf);
            if idx < 0 || idx as usize >= len {
                Err(PyIndexError::new_err(format!(
                    "row index {idx} is out of bounds for NodeFrame of length {len}"
                )))
            } else {
                Ok(idx as u32)
            }
        })
        .collect()
}

fn py_column_from_any(
    value: &Bound<'_, PyAny>,
    name: &str,
    frame_kind: FrameKind,
) -> PyResult<Vec<ScalarValue>> {
    let iter = if let Ok(values) = value.downcast::<PyList>() {
        values.iter().collect::<Vec<_>>()
    } else if let Ok(values) = value.downcast::<PyTuple>() {
        values.iter().collect::<Vec<_>>()
    } else {
        return Err(PyTypeError::new_err(format!(
            "column {name} must be a list or tuple"
        )));
    };

    iter.into_iter()
        .map(|item| scalar_from_py_any_with_context(&item, name, frame_kind))
        .collect()
}

fn scalar_from_py_any_with_context(
    value: &Bound<'_, PyAny>,
    name: &str,
    frame_kind: FrameKind,
) -> PyResult<ScalarValue> {
    match reserved_dtype(name, frame_kind) {
        Some(DataType::Int8) => {
            if value.is_none() {
                Ok(ScalarValue::Null)
            } else {
                value
                    .extract::<i8>()
                    .map(|v| ScalarValue::Int(v as i64))
                    .map_err(|_| {
                        PyTypeError::new_err(format!(
                            "column {name} must contain int8-compatible values"
                        ))
                    })
            }
        }
        _ => scalar_from_py_any(value),
    }
}

fn reserved_dtype(name: &str, frame_kind: FrameKind) -> Option<DataType> {
    match (frame_kind, name) {
        (FrameKind::Node, "_id") => Some(DataType::Utf8),
        (FrameKind::Node, "_label") => Some(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Utf8,
            true,
        )))),
        (FrameKind::Edge, "_src") => Some(DataType::Utf8),
        (FrameKind::Edge, "_dst") => Some(DataType::Utf8),
        (FrameKind::Edge, "_type") => Some(DataType::Utf8),
        (FrameKind::Edge, "_direction") => Some(DataType::Int8),
        _ => None,
    }
}

fn infer_column_dtype(
    name: &str,
    frame_kind: FrameKind,
    values: &[ScalarValue],
) -> PyResult<DataType> {
    if let Some(dtype) = reserved_dtype(name, frame_kind) {
        return Ok(dtype);
    }

    let first_non_null = values
        .iter()
        .find(|value| !py_scalar_is_null(value))
        .ok_or_else(|| {
            PyTypeError::new_err(format!(
                "cannot infer type for empty/null-only column {name}"
            ))
        })?;

    match first_non_null {
        ScalarValue::String(_) => Ok(DataType::Utf8),
        ScalarValue::Int(_) => Ok(DataType::Int64),
        ScalarValue::Float(_) => Ok(DataType::Float64),
        ScalarValue::Bool(_) => Ok(DataType::Boolean),
        ScalarValue::List(items) => {
            let inner = items
                .iter()
                .find(|value| !matches_scalar_null(value))
                .ok_or_else(|| {
                    PyTypeError::new_err(format!("cannot infer inner list type for column {name}"))
                })?;
            let inner_type = match inner {
                ScalarValue::String(_) => DataType::Utf8,
                ScalarValue::Int(_) => DataType::Int64,
                ScalarValue::Float(_) => DataType::Float64,
                ScalarValue::Bool(_) => DataType::Boolean,
                ScalarValue::List(_) | ScalarValue::Null => {
                    return Err(PyTypeError::new_err(format!(
                        "column {name} uses an unsupported nested list shape"
                    )))
                }
            };
            Ok(DataType::List(Arc::new(Field::new(
                "item", inner_type, true,
            ))))
        }
        ScalarValue::Null => Err(PyTypeError::new_err(format!(
            "cannot infer type for null-only column {name}"
        ))),
    }
}

fn build_array_from_scalar_values(values: &[ScalarValue], dtype: &DataType) -> PyResult<ArrayRef> {
    match dtype {
        DataType::Utf8 => Ok(Arc::new(StringArray::from(
            values
                .iter()
                .map(|value| match value {
                    ScalarValue::String(value) => Ok(Some(value.clone())),
                    ScalarValue::Null => Ok(None),
                    other => Err(PyTypeError::new_err(format!(
                        "expected string column, found {:?}",
                        other
                    ))),
                })
                .collect::<PyResult<Vec<_>>>()?,
        ))),
        DataType::Int64 => Ok(Arc::new(Int64Array::from(
            values
                .iter()
                .map(|value| match value {
                    ScalarValue::Int(value) => Ok(Some(*value)),
                    ScalarValue::Null => Ok(None),
                    other => Err(PyTypeError::new_err(format!(
                        "expected int column, found {:?}",
                        other
                    ))),
                })
                .collect::<PyResult<Vec<_>>>()?,
        ))),
        DataType::Float64 => Ok(Arc::new(Float64Array::from(
            values
                .iter()
                .map(|value| match value {
                    ScalarValue::Float(value) => Ok(Some(*value)),
                    ScalarValue::Int(value) => Ok(Some(*value as f64)),
                    ScalarValue::Null => Ok(None),
                    other => Err(PyTypeError::new_err(format!(
                        "expected float column, found {:?}",
                        other
                    ))),
                })
                .collect::<PyResult<Vec<_>>>()?,
        ))),
        DataType::Boolean => Ok(Arc::new(BooleanArray::from(
            values
                .iter()
                .map(|value| match value {
                    ScalarValue::Bool(value) => Ok(Some(*value)),
                    ScalarValue::Null => Ok(None),
                    other => Err(PyTypeError::new_err(format!(
                        "expected bool column, found {:?}",
                        other
                    ))),
                })
                .collect::<PyResult<Vec<_>>>()?,
        ))),
        DataType::Int8 => Ok(Arc::new(Int8Array::from(
            values
                .iter()
                .map(|value| match value {
                    ScalarValue::Int(value) => i8::try_from(*value)
                        .map(Some)
                        .map_err(|_| PyTypeError::new_err("int8 column value is out of range")),
                    ScalarValue::Null => Ok(None),
                    other => Err(PyTypeError::new_err(format!(
                        "expected int8 column, found {:?}",
                        other
                    ))),
                })
                .collect::<PyResult<Vec<_>>>()?,
        ))),
        DataType::List(field) => match field.data_type() {
            DataType::Utf8 => Ok(Arc::new(build_list_array::<StringBuilder, _>(
                values,
                StringBuilder::new(),
                |builder, value| match value {
                    ScalarValue::String(value) => {
                        builder.append_value(value);
                        Ok(())
                    }
                    ScalarValue::Null => {
                        builder.append_null();
                        Ok(())
                    }
                    other => Err(PyTypeError::new_err(format!(
                        "expected list[string] column, found {:?}",
                        other
                    ))),
                },
            )?)),
            DataType::Int64 => Ok(Arc::new(build_list_array::<Int64Builder, _>(
                values,
                Int64Builder::new(),
                |builder, value| match value {
                    ScalarValue::Int(value) => {
                        builder.append_value(*value);
                        Ok(())
                    }
                    ScalarValue::Null => {
                        builder.append_null();
                        Ok(())
                    }
                    other => Err(PyTypeError::new_err(format!(
                        "expected list[int] column, found {:?}",
                        other
                    ))),
                },
            )?)),
            DataType::Float64 => Ok(Arc::new(build_list_array::<Float64Builder, _>(
                values,
                Float64Builder::new(),
                |builder, value| match value {
                    ScalarValue::Float(value) => {
                        builder.append_value(*value);
                        Ok(())
                    }
                    ScalarValue::Int(value) => {
                        builder.append_value(*value as f64);
                        Ok(())
                    }
                    ScalarValue::Null => {
                        builder.append_null();
                        Ok(())
                    }
                    other => Err(PyTypeError::new_err(format!(
                        "expected list[float] column, found {:?}",
                        other
                    ))),
                },
            )?)),
            DataType::Boolean => Ok(Arc::new(build_list_array::<BooleanBuilder, _>(
                values,
                BooleanBuilder::new(),
                |builder, value| match value {
                    ScalarValue::Bool(value) => {
                        builder.append_value(*value);
                        Ok(())
                    }
                    ScalarValue::Null => {
                        builder.append_null();
                        Ok(())
                    }
                    other => Err(PyTypeError::new_err(format!(
                        "expected list[bool] column, found {:?}",
                        other
                    ))),
                },
            )?)),
            other => Err(PyTypeError::new_err(format!(
                "unsupported list element type: {other:?}"
            ))),
        },
        other => Err(PyTypeError::new_err(format!(
            "unsupported inferred column type: {other:?}"
        ))),
    }
}

fn build_list_array<B, F>(
    values: &[ScalarValue],
    value_builder: B,
    mut append_value: F,
) -> PyResult<arrow::array::ListArray>
where
    B: arrow::array::ArrayBuilder,
    F: FnMut(&mut B, &ScalarValue) -> PyResult<()>,
{
    let mut builder = ListBuilder::new(value_builder);
    for value in values {
        match value {
            ScalarValue::List(items) => {
                for item in items {
                    append_value(builder.values(), item)?;
                }
                builder.append(true);
            }
            ScalarValue::Null => builder.append(false),
            other => {
                return Err(PyTypeError::new_err(format!(
                    "expected list column, found {:?}",
                    other
                )))
            }
        }
    }
    Ok(builder.finish())
}

fn py_scalar_is_null(value: &ScalarValue) -> bool {
    matches!(value, ScalarValue::Null)
}

fn matches_scalar_null(value: &ScalarValue) -> bool {
    matches!(value, ScalarValue::Null)
}

fn edge_attrs_from_py_any(
    attrs: Option<&Bound<'_, PyAny>>,
) -> PyResult<HashMap<String, ScalarValue>> {
    let Some(attrs) = attrs else {
        return Ok(HashMap::new());
    };
    let dict = attrs.downcast::<pyo3::types::PyDict>().map_err(|_| {
        PyTypeError::new_err("edge attrs must be a dict[str, scalar | list[scalar]]")
    })?;

    let mut values = HashMap::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let name = key
            .extract::<String>()
            .map_err(|_| PyTypeError::new_err("edge attrs keys must be strings"))?;
        if name.starts_with('_') {
            return Err(PyValueError::new_err(format!(
                "edge attrs cannot override reserved column {name}; use dedicated parameters instead"
            )));
        }
        values.insert(name, scalar_from_py_any(&value)?);
    }

    Ok(values)
}

fn edge_row_from_schema(
    schema: &Schema,
    src: &str,
    dst: &str,
    edge_type: Option<&str>,
    direction: Direction,
    attrs: Option<&Bound<'_, PyAny>>,
) -> PyResult<EdgeFrame> {
    let attrs = edge_attrs_from_py_any(attrs)?;

    for name in attrs.keys() {
        if schema.field_with_name(name).is_err() {
            return Err(PyKeyError::new_err(format!(
                "edge attrs contains unknown column {name}"
            )));
        }
    }

    let mut arrays = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        let value = match field.name().as_str() {
            COL_EDGE_SRC => ScalarValue::String(src.to_owned()),
            COL_EDGE_DST => ScalarValue::String(dst.to_owned()),
            COL_EDGE_TYPE => ScalarValue::String(edge_type.unwrap_or("__delta__").to_owned()),
            COL_EDGE_DIRECTION => ScalarValue::Int(direction.as_i8() as i64),
            name => attrs.get(name).cloned().unwrap_or(ScalarValue::Null),
        };

        if matches!(value, ScalarValue::Null) && !field.is_nullable() {
            return Err(PyValueError::new_err(format!(
                "edge column {} is non-nullable; provide it explicitly when adding an edge",
                field.name()
            )));
        }

        arrays.push(build_array_from_scalar_values(&[value], field.data_type())?);
    }

    let batch = RecordBatch::try_new(Arc::new(schema.clone()), arrays)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    EdgeFrame::from_record_batch(batch).map_err(gf_error_to_py_err)
}

fn register_pattern_node_constraint(
    constraints: &mut BTreeMap<String, PatternNodeConstraint>,
    node: &PyPatternNode,
) -> PyResult<()> {
    if !node.props.is_empty() {
        return Err(PyNotImplementedError::new_err(format!(
            "match_pattern: node alias '{}' uses props, but PatternNode.props is not implemented yet",
            node.alias
        )));
    }

    let constraint = PatternNodeConstraint {
        label: node.label.clone(),
    };
    match constraints.get(&node.alias) {
        Some(existing) if existing == &constraint => Ok(()),
        Some(existing) => Err(PyValueError::new_err(format!(
            "match_pattern: alias '{}' has conflicting node constraints: {:?} vs {:?}",
            node.alias, existing, constraint
        ))),
        None => {
            constraints.insert(node.alias.clone(), constraint);
            Ok(())
        }
    }
}

/// Convert a Python list of alternating `PatternNode`/`PatternEdge`/`PatternNode` …
/// into a `Pattern` for `LazyGraphFrame::match_pattern`.
///
/// Expects an odd-length list with at least 3 elements:
///   [node, edge, node, edge, node, …]
fn pattern_from_py_steps(steps: &Bound<'_, PyAny>) -> PyResult<Pattern> {
    let list = steps.downcast::<PyList>().map_err(|_| {
        PyTypeError::new_err("match_pattern: steps must be a list of alternating PatternNode / PatternEdge / PatternNode ...")
    })?;

    let len = list.len();
    if len < 3 || len % 2 == 0 {
        return Err(PyValueError::new_err(
            "match_pattern: steps must have an odd length ≥ 3 ([node, edge, node, ...])",
        ));
    }

    let mut pattern_steps: Vec<PatternStep> = Vec::with_capacity(len / 2);
    let mut node_constraints = BTreeMap::new();
    let mut step_constraints = Vec::with_capacity(len / 2);
    let mut i = 0usize;
    while i + 2 <= len - 1 {
        let from_node = list
            .get_item(i)?
            .extract::<PyRef<'_, PyPatternNode>>()
            .map_err(|_| {
                PyTypeError::new_err(format!(
                    "match_pattern: item {i} must be a PatternNode (got {:?})",
                    list.get_item(i).unwrap()
                ))
            })?;

        let pat_edge = list
            .get_item(i + 1)?
            .extract::<PyRef<'_, PyPatternEdge>>()
            .map_err(|_| {
                PyTypeError::new_err(format!(
                    "match_pattern: item {} must be a PatternEdge (got {:?})",
                    i + 1,
                    list.get_item(i + 1).unwrap()
                ))
            })?;

        let to_node = list
            .get_item(i + 2)?
            .extract::<PyRef<'_, PyPatternNode>>()
            .map_err(|_| {
                PyTypeError::new_err(format!(
                    "match_pattern: item {} must be a PatternNode (got {:?})",
                    i + 2,
                    list.get_item(i + 2).unwrap()
                ))
            })?;

        register_pattern_node_constraint(&mut node_constraints, &from_node)?;
        register_pattern_node_constraint(&mut node_constraints, &to_node)?;

        let edge_type = match &pat_edge.edge_type {
            Some(t) => EdgeTypeSpec::Single(t.clone()),
            None => EdgeTypeSpec::Any,
        };
        let max_hops = pat_edge.max_hops.unwrap_or(pat_edge.min_hops);

        if pat_edge.alias.is_some() && max_hops > 1 {
            return Err(PyNotImplementedError::new_err(format!(
                "match_pattern: edge alias on step {} is only supported for single-hop edges",
                i / 2
            )));
        }

        pattern_steps.push(PatternStep {
            from_alias: from_node.alias.clone(),
            edge_alias: pat_edge.alias.clone(),
            edge_type,
            direction: Direction::Out,
            to_alias: to_node.alias.clone(),
        });
        step_constraints.push(PatternStepConstraint {
            optional: pat_edge.optional,
            min_hops: pat_edge.min_hops,
            max_hops,
        });

        i += 2;
    }

    Pattern::with_constraints(pattern_steps, node_constraints, step_constraints)
        .map_err(gf_error_to_py_err)
}

fn write_gf_impl(graph: &GraphFrame, path: &PathBuf) -> PyResult<()> {
    core_write_gf(graph, path).map_err(gf_error_to_py_err)
}

fn unsupported_write_impl(method: &str, path: &Path) -> PyResult<()> {
    Err(PyNotImplementedError::new_err(format!(
        "{method} is not implemented in lynxes-core yet (requested path: {})",
        path.display()
    )))
}
