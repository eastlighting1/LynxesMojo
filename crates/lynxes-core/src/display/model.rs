use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayView {
    Table,
    Head,
    Tail,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DisplayRowKind {
    Edge,
    Node,
}

#[derive(Debug, Clone)]
pub struct DisplayOptions {
    pub view: DisplayView,
    pub max_rows: usize,
    pub width: Option<usize>,
    pub sort_by: Option<String>,
    pub expand_attrs: bool,
    pub attrs: Vec<String>,
}

impl Default for DisplayOptions {
    fn default() -> Self {
        Self {
            view: DisplayView::Table,
            max_rows: 10,
            width: None,
            sort_by: None,
            expand_attrs: false,
            attrs: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DisplayColumn {
    pub name: String,
    pub width: usize,
}

#[derive(Debug, Clone)]
pub struct DisplayRow {
    pub stable_index: usize,
    pub kind: DisplayRowKind,
    pub values: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct GraphSummary {
    pub projected_row_count: usize,
    pub node_count: usize,
    pub edge_count: usize,
    pub isolated_node_count: usize,
    pub node_type_count: usize,
    pub edge_type_count: usize,
    pub directedness: String,
}

#[derive(Debug, Clone)]
pub struct DisplaySlice {
    pub order_name: String,
    pub columns: Vec<DisplayColumn>,
    pub top_rows: Vec<DisplayRow>,
    pub bottom_rows: Vec<DisplayRow>,
    pub omitted_rows: usize,
    pub graph_summary: GraphSummary,
}

#[derive(Debug, Clone)]
pub struct GraphInfo {
    pub summary: GraphSummary,
    pub self_loops: usize,
    pub multi_edge_pairs: usize,
    pub node_labels: Vec<String>,
    pub edge_types: Vec<String>,
    pub node_attribute_keys: Vec<String>,
    pub edge_attribute_keys: Vec<String>,
    pub schema_present: bool,
}

#[derive(Debug, Clone)]
pub struct SchemaFieldSummary {
    pub name: String,
    pub dtype: String,
    pub nullable: bool,
    pub reserved: bool,
}

#[derive(Debug, Clone)]
pub struct SchemaSummary {
    pub declared: bool,
    pub node_labels: Vec<String>,
    pub edge_types: Vec<String>,
    pub node_fields: Vec<SchemaFieldSummary>,
    pub edge_fields: Vec<SchemaFieldSummary>,
}

#[derive(Debug, Clone)]
pub struct GlimpseColumn {
    pub name: String,
    pub dtype: String,
    pub samples: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct GlimpseSummary {
    pub rows_sampled: usize,
    pub columns: Vec<GlimpseColumn>,
}

#[derive(Debug, Clone)]
pub struct AttributeStats {
    pub qualified_name: String,
    pub dtype: String,
    pub non_null_count: usize,
    pub null_count: usize,
    pub distinct_count: usize,
    pub samples: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct AttrStatsSummary {
    pub node_attrs: Vec<AttributeStats>,
    pub edge_attrs: Vec<AttributeStats>,
}

#[derive(Debug, Clone)]
pub struct StructureStats {
    pub density: f64,
    pub average_out_degree: f64,
    pub average_in_degree: f64,
    pub median_degree: f64,
    pub max_degree: usize,
    pub connected_components: usize,
    pub largest_component_share: f64,
}
