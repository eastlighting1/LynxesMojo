use std::collections::BTreeMap;

use crate::display::model::{DisplayColumn, DisplayRow};
use crate::display::summary::{ellipsis_middle, ellipsis_right, join_attr_tokens};

const MIN_ATTRS_WIDTH: usize = 12;

#[derive(Clone)]
pub(crate) struct ColumnSpec {
    pub name: String,
    pub priority: usize,
    pub min_width: usize,
    pub max_width: usize,
    pub strategy: TruncateStrategy,
}

#[derive(Clone, Copy)]
pub(crate) enum TruncateStrategy {
    Right,
    Middle,
    Attrs,
}

pub(crate) fn layout_rows(
    width: Option<usize>,
    specs: Vec<ColumnSpec>,
    top_rows: &mut [DisplayRow],
    bottom_rows: &mut [DisplayRow],
) -> Vec<DisplayColumn> {
    let sample_rows: Vec<&DisplayRow> = top_rows.iter().chain(bottom_rows.iter()).collect();
    let mut visible = specs;
    let target_width = width.unwrap_or(100);

    loop {
        let widths = compute_widths(&visible, &sample_rows, target_width);
        if total_table_width(&widths) <= target_width || visible.len() <= 5 {
            apply_truncation(&visible, &widths, top_rows);
            apply_truncation(&visible, &widths, bottom_rows);
            return visible
                .into_iter()
                .zip(widths)
                .map(|(spec, width)| DisplayColumn {
                    name: spec.name,
                    width,
                })
                .collect();
        }

        let drop_idx = visible
            .iter()
            .enumerate()
            .filter(|(_, spec)| {
                spec.name != "#"
                    && spec.name != "src"
                    && spec.name != "rel"
                    && spec.name != "dst"
                    && spec.name != "attrs"
            })
            .max_by_key(|(_, spec)| spec.priority)
            .map(|(idx, _)| idx);

        if let Some(idx) = drop_idx {
            visible.remove(idx);
        } else {
            break;
        }
    }

    let widths = compute_widths(&visible, &sample_rows, target_width);
    apply_truncation(&visible, &widths, top_rows);
    apply_truncation(&visible, &widths, bottom_rows);
    visible
        .into_iter()
        .zip(widths)
        .map(|(spec, width)| DisplayColumn {
            name: spec.name,
            width,
        })
        .collect()
}

fn compute_widths(specs: &[ColumnSpec], rows: &[&DisplayRow], target_width: usize) -> Vec<usize> {
    let mut widths = Vec::with_capacity(specs.len());
    let mut fixed = 0usize;
    let mut attrs_idx = None;

    for (idx, spec) in specs.iter().enumerate() {
        let mut width = spec.name.chars().count();
        for row in rows {
            if let Some(value) = row.values.get(&spec.name) {
                width = width.max(value.chars().count());
            }
        }
        width = width.clamp(spec.min_width, spec.max_width);
        if spec.name == "attrs" {
            attrs_idx = Some(idx);
        } else {
            fixed += width;
        }
        widths.push(width);
    }

    let separators = specs.len().saturating_sub(1) * 3;
    if let Some(idx) = attrs_idx {
        let available = target_width
            .saturating_sub(fixed)
            .saturating_sub(separators)
            .max(MIN_ATTRS_WIDTH);
        widths[idx] = widths[idx].max(MIN_ATTRS_WIDTH).min(available);
    }

    widths
}

fn total_table_width(widths: &[usize]) -> usize {
    widths.iter().sum::<usize>() + widths.len().saturating_sub(1) * 3
}

fn apply_truncation(specs: &[ColumnSpec], widths: &[usize], rows: &mut [DisplayRow]) {
    for row in rows {
        let mut next = BTreeMap::new();
        for (spec, width) in specs.iter().zip(widths.iter().copied()) {
            let value = row
                .values
                .get(&spec.name)
                .cloned()
                .unwrap_or_else(String::new);
            let clipped = match spec.strategy {
                TruncateStrategy::Right => ellipsis_right(&value, width),
                TruncateStrategy::Middle => ellipsis_middle(&value, width),
                TruncateStrategy::Attrs => {
                    if value == "-" {
                        value
                    } else {
                        let tokens = value.split(", ").map(str::to_owned).collect();
                        join_attr_tokens(tokens, Some(width))
                    }
                }
            };
            next.insert(spec.name.clone(), clipped);
        }
        row.values = next;
    }
}
