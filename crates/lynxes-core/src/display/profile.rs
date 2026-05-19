use std::collections::{BTreeSet, HashMap, HashSet};

use arrow_array::{Array, Int8Array, ListArray, StringArray, UInt32Array};
use arrow_schema::Field;

use crate::display::model::{
    AttrStatsSummary, AttributeStats, GraphInfo, GraphSummary, SchemaFieldSummary, SchemaSummary,
    StructureStats,
};
use crate::{
    Direction, GraphFrame, COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE,
    COL_NODE_LABEL, EDGE_RESERVED_COLUMNS, NODE_RESERVED_COLUMNS,
};

pub(crate) fn graph_summary(graph: &GraphFrame) -> GraphSummary {
    let node_labels = collect_node_labels(graph);
    let edge_types = collect_edge_types(graph);
    let isolated = isolated_node_ids(graph);
    GraphSummary {
        projected_row_count: graph.edge_count() + isolated.len(),
        node_count: graph.node_count(),
        edge_count: graph.edge_count(),
        isolated_node_count: isolated.len(),
        node_type_count: node_labels.len(),
        edge_type_count: edge_types.len(),
        directedness: directedness(graph),
    }
}

pub(crate) fn graph_info(graph: &GraphFrame) -> GraphInfo {
    let summary = graph_summary(graph);
    let mut edge_pairs: HashMap<(String, String, String), usize> = HashMap::new();
    let edge_batch = graph.edges().to_record_batch();
    let src_col = edge_batch
        .column_by_name(COL_EDGE_SRC)
        .expect("_src exists")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("_src is Utf8");
    let dst_col = edge_batch
        .column_by_name(COL_EDGE_DST)
        .expect("_dst exists")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("_dst is Utf8");
    let typ_col = edge_batch
        .column_by_name(COL_EDGE_TYPE)
        .expect("_type exists")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("_type is Utf8");

    let mut self_loops = 0usize;
    for row in 0..graph.edge_count() {
        let src = src_col.value(row);
        let dst = dst_col.value(row);
        if src == dst {
            self_loops += 1;
        }
        *edge_pairs
            .entry((
                src.to_owned(),
                dst.to_owned(),
                typ_col.value(row).to_owned(),
            ))
            .or_insert(0) += 1;
    }
    let multi_edge_pairs = edge_pairs.values().filter(|&&count| count > 1).count();

    GraphInfo {
        summary,
        self_loops,
        multi_edge_pairs,
        node_labels: collect_node_labels(graph).into_iter().collect(),
        edge_types: collect_edge_types(graph).into_iter().collect(),
        node_attribute_keys: graph
            .nodes()
            .schema()
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .filter(|name| !NODE_RESERVED_COLUMNS.contains(&name.as_str()))
            .collect(),
        edge_attribute_keys: graph
            .edges()
            .schema()
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .filter(|name| !EDGE_RESERVED_COLUMNS.contains(&name.as_str()))
            .collect(),
        schema_present: graph.schema().is_some(),
    }
}

pub(crate) fn schema_summary(graph: &GraphFrame) -> SchemaSummary {
    SchemaSummary {
        declared: graph.schema().is_some(),
        node_labels: collect_node_labels(graph).into_iter().collect(),
        edge_types: collect_edge_types(graph).into_iter().collect(),
        node_fields: fields_summary(graph.nodes().schema().fields(), &NODE_RESERVED_COLUMNS),
        edge_fields: fields_summary(graph.edges().schema().fields(), &EDGE_RESERVED_COLUMNS),
    }
}

pub(crate) fn attr_stats_summary(graph: &GraphFrame) -> AttrStatsSummary {
    AttrStatsSummary {
        node_attrs: attr_stats_from_batch(
            graph.nodes().to_record_batch(),
            "node",
            &NODE_RESERVED_COLUMNS,
        ),
        edge_attrs: attr_stats_from_batch(
            graph.edges().to_record_batch(),
            "edge",
            &EDGE_RESERVED_COLUMNS,
        ),
    }
}

pub(crate) fn structure_stats(graph: &GraphFrame) -> crate::Result<StructureStats> {
    let node_ids: Vec<&str> = graph.nodes().id_column().iter().flatten().collect();
    let mut out_degrees = Vec::with_capacity(node_ids.len());
    let mut in_degrees = Vec::with_capacity(node_ids.len());
    let mut total_degrees = Vec::with_capacity(node_ids.len());

    for node_id in &node_ids {
        let out_degree = graph.out_degree(node_id)?;
        let in_degree = graph.in_degree(node_id)?;
        out_degrees.push(out_degree);
        in_degrees.push(in_degree);
        total_degrees.push(out_degree + in_degree);
    }
    total_degrees.sort_unstable();
    let median_degree = if total_degrees.is_empty() {
        0.0
    } else if total_degrees.len() % 2 == 1 {
        total_degrees[total_degrees.len() / 2] as f64
    } else {
        let hi = total_degrees.len() / 2;
        (total_degrees[hi - 1] + total_degrees[hi]) as f64 / 2.0
    };
    let max_degree = total_degrees.last().copied().unwrap_or(0);

    let cc = graph.connected_components()?;
    let component_ids = cc
        .to_record_batch()
        .column_by_name("component_id")
        .expect("connected_components output has component_id")
        .as_any()
        .downcast_ref::<UInt32Array>()
        .expect("component_id is UInt32");
    let mut sizes = HashMap::<u32, usize>::new();
    for row in 0..component_ids.len() {
        *sizes.entry(component_ids.value(row)).or_insert(0) += 1;
    }

    Ok(StructureStats {
        density: graph.density(),
        average_out_degree: average_usize(&out_degrees),
        average_in_degree: average_usize(&in_degrees),
        median_degree,
        max_degree,
        connected_components: sizes.len(),
        largest_component_share: sizes
            .values()
            .copied()
            .max()
            .map(|size| size as f64 / graph.node_count().max(1) as f64)
            .unwrap_or(0.0),
    })
}

pub(crate) fn collect_node_labels(graph: &GraphFrame) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let Some(column) = graph
        .nodes()
        .to_record_batch()
        .column_by_name(COL_NODE_LABEL)
    else {
        return out;
    };
    let labels = column
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("_label is List<Utf8>");
    for row in 0..labels.len() {
        let values = labels.value(row);
        let values = values
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("_label values are Utf8");
        for idx in 0..values.len() {
            if !values.is_null(idx) {
                out.insert(values.value(idx).to_owned());
            }
        }
    }
    out
}

pub(crate) fn collect_edge_types(graph: &GraphFrame) -> BTreeSet<String> {
    graph
        .edges()
        .edge_types()
        .into_iter()
        .map(str::to_owned)
        .collect()
}

pub(crate) fn isolated_node_ids(graph: &GraphFrame) -> Vec<String> {
    let edge_batch = graph.edges().to_record_batch();
    let src_col = edge_batch
        .column_by_name(COL_EDGE_SRC)
        .expect("_src exists")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("_src is Utf8");
    let dst_col = edge_batch
        .column_by_name(COL_EDGE_DST)
        .expect("_dst exists")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("_dst is Utf8");
    let mut seen = HashSet::with_capacity(graph.edge_count().saturating_mul(2));
    for row in 0..graph.edge_count() {
        seen.insert(src_col.value(row));
        seen.insert(dst_col.value(row));
    }
    graph
        .nodes()
        .id_column()
        .iter()
        .flatten()
        .filter(|id| !seen.contains(id))
        .map(str::to_owned)
        .collect()
}

fn directedness(graph: &GraphFrame) -> String {
    let Some(direction_col) = graph
        .edges()
        .to_record_batch()
        .column_by_name(COL_EDGE_DIRECTION)
    else {
        return "unknown".to_owned();
    };
    let direction_col = direction_col
        .as_any()
        .downcast_ref::<Int8Array>()
        .expect("_direction is Int8");
    let mut saw_directed = false;
    let mut saw_undirected = false;
    for row in 0..direction_col.len() {
        let code = direction_col.value(row);
        match Direction::try_from(code) {
            Ok(Direction::Out | Direction::In | Direction::Both) => saw_directed = true,
            Ok(Direction::None) => saw_undirected = true,
            Err(_) => {}
        }
    }
    match (saw_directed, saw_undirected) {
        (true, true) => "mixed".to_owned(),
        (true, false) => "directed".to_owned(),
        (false, true) => "undirected".to_owned(),
        (false, false) => "empty".to_owned(),
    }
}

fn fields_summary(fields: &[std::sync::Arc<Field>], reserved: &[&str]) -> Vec<SchemaFieldSummary> {
    fields
        .iter()
        .map(|field| SchemaFieldSummary {
            name: field.name().clone(),
            dtype: format!("{:?}", field.data_type()),
            nullable: field.is_nullable(),
            reserved: reserved.contains(&field.name().as_str()),
        })
        .collect()
}

fn attr_stats_from_batch(
    batch: &arrow_array::RecordBatch,
    prefix: &str,
    reserved: &[&str],
) -> Vec<AttributeStats> {
    batch
        .schema_ref()
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, field)| !reserved.contains(&field.name().as_str()))
        .map(|(idx, field)| {
            let array = batch.column(idx);
            let non_null_count = array.len() - array.null_count();
            let null_count = array.null_count();
            let mut distinct = HashSet::new();
            let mut samples = Vec::new();
            for row in 0..array.len() {
                if array.is_null(row) {
                    continue;
                }
                let value = arrow::util::display::array_value_to_string(array.as_ref(), row)
                    .unwrap_or_else(|_| "<invalid>".to_owned());
                distinct.insert(value.clone());
                if samples.len() < 3 && !samples.contains(&value) {
                    samples.push(value);
                }
            }
            AttributeStats {
                qualified_name: format!("{prefix}.{}", field.name()),
                dtype: format!("{:?}", field.data_type()),
                non_null_count,
                null_count,
                distinct_count: distinct.len(),
                samples,
            }
        })
        .collect()
}

fn average_usize(values: &[usize]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().copied().sum::<usize>() as f64 / values.len() as f64
    }
}
