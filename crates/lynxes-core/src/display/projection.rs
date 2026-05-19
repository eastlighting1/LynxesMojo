use std::collections::{BTreeMap, BTreeSet};

use arrow_array::{Array, ListArray, StringArray};

use crate::display::model::{
    DisplayOptions, DisplayRow, DisplayRowKind, DisplaySlice, DisplayView, GlimpseColumn,
    GlimpseSummary,
};
use crate::display::profile::{
    attr_stats_summary, graph_info, graph_summary, isolated_node_ids, schema_summary,
    structure_stats,
};
use crate::display::summary::format_cell_value;
use crate::display::width::{layout_rows, ColumnSpec, TruncateStrategy};
use crate::{
    GraphFrame, COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE, COL_NODE_LABEL, EDGE_RESERVED_COLUMNS,
    NODE_RESERVED_COLUMNS,
};

impl GraphFrame {
    pub fn display_slice(&self, options: DisplayOptions) -> crate::Result<DisplaySlice> {
        let summary = graph_summary(self);
        let promoted = promoted_attrs(self, &options);
        let mut rows = project_rows(self, &options, &promoted)?;

        if let Some(sort_by) = options.sort_by.as_ref() {
            rows.sort_by(|a, b| {
                let av = a.values.get(sort_by).cloned().unwrap_or_default();
                let bv = b.values.get(sort_by).cloned().unwrap_or_default();
                av.cmp(&bv)
                    .then_with(|| a.stable_index.cmp(&b.stable_index))
            });
        }

        let total = rows.len();
        let (mut top_rows, mut bottom_rows, omitted_rows) = match options.view {
            DisplayView::Head => {
                let top = rows.into_iter().take(options.max_rows).collect::<Vec<_>>();
                let omitted = total.saturating_sub(top.len());
                (top, Vec::new(), omitted)
            }
            DisplayView::Tail => {
                let bottom = rows
                    .into_iter()
                    .rev()
                    .take(options.max_rows)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>();
                let omitted = total.saturating_sub(bottom.len());
                (Vec::new(), bottom, omitted)
            }
            DisplayView::Table => {
                if total <= options.max_rows {
                    (rows, Vec::new(), 0)
                } else {
                    let top_len = options.max_rows / 2;
                    let bottom_len = options.max_rows - top_len;
                    let top = rows.iter().take(top_len).cloned().collect::<Vec<_>>();
                    let bottom = rows
                        .iter()
                        .rev()
                        .take(bottom_len)
                        .cloned()
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect::<Vec<_>>();
                    let omitted = total.saturating_sub(top.len() + bottom.len());
                    (top, bottom, omitted)
                }
            }
        };

        let columns = layout_rows(
            options.width,
            column_specs(&promoted),
            &mut top_rows,
            &mut bottom_rows,
        );

        Ok(DisplaySlice {
            order_name: if options.sort_by.is_some() {
                "Sorted".to_owned()
            } else {
                "StableDerived".to_owned()
            },
            columns,
            top_rows,
            bottom_rows,
            omitted_rows,
            graph_summary: summary,
        })
    }

    pub fn display_info(&self) -> crate::display::GraphInfo {
        graph_info(self)
    }

    pub fn display_schema(&self) -> crate::display::SchemaSummary {
        schema_summary(self)
    }

    pub fn display_glimpse(&self, options: DisplayOptions) -> crate::Result<GlimpseSummary> {
        let slice = self.display_slice(DisplayOptions {
            view: DisplayView::Head,
            ..options
        })?;
        let rows = slice.top_rows;
        let columns = slice
            .columns
            .iter()
            .map(|column| GlimpseColumn {
                name: column.name.clone(),
                dtype: column_dtype(self, &column.name),
                samples: rows
                    .iter()
                    .filter_map(|row| row.values.get(&column.name).cloned())
                    .collect(),
            })
            .collect();
        Ok(GlimpseSummary {
            rows_sampled: rows.len(),
            columns,
        })
    }

    pub fn display_attr_stats(&self) -> crate::display::AttrStatsSummary {
        attr_stats_summary(self)
    }

    pub fn display_structure_stats(&self) -> crate::Result<crate::display::StructureStats> {
        structure_stats(self)
    }
}

fn project_rows(
    graph: &GraphFrame,
    options: &DisplayOptions,
    promoted: &[String],
) -> crate::Result<Vec<DisplayRow>> {
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
    let rel_col = edge_batch
        .column_by_name(COL_EDGE_TYPE)
        .expect("_type exists")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("_type is Utf8");

    let isolated = isolated_node_ids(graph);
    let mut rows = Vec::with_capacity(graph.edge_count() + isolated.len());
    for row in 0..graph.edge_count() {
        let src = src_col.value(row);
        let dst = dst_col.value(row);
        let rel = rel_col.value(row);
        let src_type = node_type(graph, src).unwrap_or_else(|| "-".to_owned());
        let dst_type = node_type(graph, dst).unwrap_or_else(|| "-".to_owned());
        let promoted_values =
            promoted_row_values(graph, promoted, Some(row), Some(src), Some(dst), false);
        let attrs = attr_summary(
            graph,
            options,
            promoted,
            Some(row),
            Some(src),
            Some(dst),
            false,
        );

        let mut values = BTreeMap::new();
        values.insert("#".to_owned(), row.to_string());
        values.insert("kind".to_owned(), "EDGE".to_owned());
        values.insert("src".to_owned(), src.to_owned());
        values.insert("rel".to_owned(), rel.to_owned());
        values.insert("dst".to_owned(), dst.to_owned());
        values.insert("src_type".to_owned(), src_type);
        values.insert("dst_type".to_owned(), dst_type);
        for (name, value) in promoted.iter().zip(promoted_values) {
            values.insert(name.clone(), value.unwrap_or_else(|| "-".to_owned()));
        }
        values.insert("attrs".to_owned(), attrs);
        rows.push(DisplayRow {
            stable_index: row,
            kind: DisplayRowKind::Edge,
            values,
        });
    }

    for (offset, node_id) in isolated.iter().enumerate() {
        let promoted_values = promoted_row_values(graph, promoted, None, Some(node_id), None, true);
        let attrs = attr_summary(graph, options, promoted, None, Some(node_id), None, true);
        let mut values = BTreeMap::new();
        values.insert("#".to_owned(), (graph.edge_count() + offset).to_string());
        values.insert("kind".to_owned(), "NODE".to_owned());
        values.insert("src".to_owned(), node_id.clone());
        values.insert("rel".to_owned(), "-".to_owned());
        values.insert("dst".to_owned(), "-".to_owned());
        values.insert(
            "src_type".to_owned(),
            node_type(graph, node_id).unwrap_or_else(|| "-".to_owned()),
        );
        values.insert("dst_type".to_owned(), "-".to_owned());
        for (name, value) in promoted.iter().zip(promoted_values) {
            values.insert(name.clone(), value.unwrap_or_else(|| "-".to_owned()));
        }
        values.insert("attrs".to_owned(), attrs);
        rows.push(DisplayRow {
            stable_index: graph.edge_count() + offset,
            kind: DisplayRowKind::Node,
            values,
        });
    }

    Ok(rows)
}

fn promoted_attrs(graph: &GraphFrame, options: &DisplayOptions) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for attr in &options.attrs {
        if seen.insert(attr.clone()) {
            out.push(attr.clone());
        }
    }
    if options.expand_attrs {
        for candidate in ["weight", "name", "timestamp", "status", "role"] {
            if seen.contains(candidate) {
                continue;
            }
            let exists = graph.edges().schema().field_with_name(candidate).is_ok()
                || graph.nodes().schema().field_with_name(candidate).is_ok();
            if exists {
                seen.insert(candidate.to_owned());
                out.push(candidate.to_owned());
            }
        }
    }
    out
}

fn promoted_row_values(
    graph: &GraphFrame,
    promoted: &[String],
    edge_row: Option<usize>,
    src_id: Option<&str>,
    dst_id: Option<&str>,
    node_only: bool,
) -> Vec<Option<String>> {
    promoted
        .iter()
        .map(|name| {
            if let Some(edge_row) = edge_row {
                value_from_origin(graph.edges().to_record_batch(), name, edge_row)
                    .or_else(|| src_id.and_then(|id| node_value(graph, id, name)))
                    .or_else(|| dst_id.and_then(|id| node_value(graph, id, name)))
            } else if node_only {
                src_id.and_then(|id| node_value(graph, id, name))
            } else {
                None
            }
        })
        .collect()
}

fn attr_summary(
    graph: &GraphFrame,
    options: &DisplayOptions,
    promoted: &[String],
    edge_row: Option<usize>,
    src_id: Option<&str>,
    dst_id: Option<&str>,
    node_only: bool,
) -> String {
    let mut tokens = Vec::new();
    let mut used_keys = BTreeSet::new();

    let edge_names = graph
        .edges()
        .schema()
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .filter(|name| !EDGE_RESERVED_COLUMNS.contains(&name.as_str()))
        .collect::<Vec<_>>();
    let node_names = graph
        .nodes()
        .schema()
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .filter(|name| !NODE_RESERVED_COLUMNS.contains(&name.as_str()))
        .collect::<Vec<_>>();

    for name in &options.attrs {
        if promoted.contains(name) {
            continue;
        }
        if let Some(value) = edge_row
            .and_then(|row| value_from_origin(graph.edges().to_record_batch(), name, row))
            .or_else(|| src_id.and_then(|id| node_value(graph, id, name)))
            .or_else(|| dst_id.and_then(|id| node_value(graph, id, name)))
        {
            tokens.push(format!("{name}={value}"));
            used_keys.insert(name.clone());
        }
    }

    if let Some(edge_row) = edge_row {
        for name in edge_names {
            if used_keys.contains(&name) || promoted.contains(&name) {
                continue;
            }
            if let Some(value) = value_from_origin(graph.edges().to_record_batch(), &name, edge_row)
            {
                tokens.push(format!("{name}={value}"));
                used_keys.insert(name);
            }
        }
    }

    for (prefix, node_id) in [("src", src_id), ("dst", dst_id)] {
        if node_only && prefix == "dst" {
            continue;
        }
        let Some(node_id) = node_id else { continue };
        for name in &node_names {
            if used_keys.contains(name) || promoted.contains(name) {
                continue;
            }
            if let Some(value) = node_value(graph, node_id, name) {
                let duplicate = src_id
                    .filter(|_| prefix == "dst")
                    .and_then(|src| node_value(graph, src, name))
                    .is_some()
                    || edge_row
                        .and_then(|row| {
                            value_from_origin(graph.edges().to_record_batch(), name, row)
                        })
                        .is_some();
                if duplicate {
                    tokens.push(format!("{prefix}.{name}={value}"));
                } else {
                    tokens.push(format!("{name}={value}"));
                    used_keys.insert(name.clone());
                }
            }
        }
    }

    if tokens.is_empty() {
        "-".to_owned()
    } else {
        tokens.join(", ")
    }
}

fn value_from_origin(batch: &arrow_array::RecordBatch, name: &str, row: usize) -> Option<String> {
    batch
        .column_by_name(name)
        .and_then(|array| format_cell_value(array.as_ref(), row))
}

fn node_value(graph: &GraphFrame, node_id: &str, name: &str) -> Option<String> {
    let row = graph.nodes().row_index(node_id)? as usize;
    let batch = graph.nodes().to_record_batch();
    let column = batch.column_by_name(name)?;
    format_cell_value(column.as_ref(), row)
}

fn node_type(graph: &GraphFrame, node_id: &str) -> Option<String> {
    let row = graph.nodes().row_index(node_id)? as usize;
    let batch = graph.nodes().to_record_batch();
    let labels = batch
        .column_by_name(COL_NODE_LABEL)?
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("_label is List<Utf8>");
    let label_values = labels.value(row);
    let values = label_values
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("_label values are Utf8");
    match values.len() {
        0 => Some("-".to_owned()),
        1 => Some(values.value(0).to_owned()),
        n => Some(format!("{}+{}", values.value(0), n - 1)),
    }
}

fn column_specs(promoted: &[String]) -> Vec<ColumnSpec> {
    let mut specs = vec![
        ColumnSpec {
            name: "#".to_owned(),
            priority: 0,
            min_width: 2,
            max_width: 6,
            strategy: TruncateStrategy::Right,
        },
        ColumnSpec {
            name: "kind".to_owned(),
            priority: 3,
            min_width: 4,
            max_width: 4,
            strategy: TruncateStrategy::Right,
        },
        ColumnSpec {
            name: "src".to_owned(),
            priority: 0,
            min_width: 3,
            max_width: 18,
            strategy: TruncateStrategy::Right,
        },
        ColumnSpec {
            name: "rel".to_owned(),
            priority: 0,
            min_width: 3,
            max_width: 18,
            strategy: TruncateStrategy::Middle,
        },
        ColumnSpec {
            name: "dst".to_owned(),
            priority: 0,
            min_width: 3,
            max_width: 18,
            strategy: TruncateStrategy::Right,
        },
        ColumnSpec {
            name: "src_type".to_owned(),
            priority: 2,
            min_width: 8,
            max_width: 14,
            strategy: TruncateStrategy::Right,
        },
        ColumnSpec {
            name: "dst_type".to_owned(),
            priority: 2,
            min_width: 8,
            max_width: 14,
            strategy: TruncateStrategy::Right,
        },
    ];
    for name in promoted {
        specs.push(ColumnSpec {
            name: name.clone(),
            priority: 1,
            min_width: name.len().max(6),
            max_width: 16,
            strategy: TruncateStrategy::Right,
        });
    }
    specs.push(ColumnSpec {
        name: "attrs".to_owned(),
        priority: 0,
        min_width: 12,
        max_width: 48,
        strategy: TruncateStrategy::Attrs,
    });
    specs
}

fn column_dtype(graph: &GraphFrame, name: &str) -> String {
    match name {
        "#" => "usize".to_owned(),
        "kind" | "src" | "rel" | "dst" | "src_type" | "dst_type" | "attrs" => "Utf8".to_owned(),
        other => graph
            .edges()
            .schema()
            .field_with_name(other)
            .or_else(|_| graph.nodes().schema().field_with_name(other))
            .map(|field| format!("{:?}", field.data_type()))
            .unwrap_or_else(|_| "Utf8".to_owned()),
    }
}
