/// Serialise a `GraphFrame` back to the `.gf` text format.
///
/// The output is human-readable and round-trippable through `parse_gf`:
///
/// ```text
/// (alice: Person { age: 30, name: "Alice" })
/// (acme: Company { founded: 1990 })
/// alice -[KNOWS]-> bob
/// alice <-[REPORTS_TO]- acme
/// alice <-[PARTNER_OF]-> bob
/// alice --[COLOCATED_WITH]-- charlie
/// ```
///
/// # Format rules
///
/// | Direction | Arrow syntax |
/// |-----------|-------------|
/// | Out       | `src -[TYPE]-> dst` |
/// | In        | `src <-[TYPE]- dst` |
/// | Both      | `src <-[TYPE]-> dst` |
/// | None      | `src --[TYPE]-- dst` |
///
/// Properties are omitted when null.  Strings are double-quoted with `\"`
/// and `\\` escaping.  Floats always include a decimal point.
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::Path;

use arrow_array::{Array, BooleanArray, Float64Array, Int64Array, ListArray, StringArray};
use arrow_schema::DataType;

use lynxes_core::{
    Direction, GFError, GraphFrame, Result, COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC,
    COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
};

/// Write `graph` to `path` in `.gf` text format.
///
/// Creates or overwrites the file at `path`.
pub fn write_gf<P: AsRef<Path>>(graph: &GraphFrame, path: P) -> Result<()> {
    let text = serialise_graph(graph)?;
    fs::write(path, text).map_err(GFError::IoError)
}

// ── Serialisation ─────────────────────────────────────────────────────────────

fn serialise_graph(graph: &GraphFrame) -> Result<String> {
    let mut out = String::new();

    // ── Nodes ────────────────────────────────────────────────────────────────
    let node_batch = graph.nodes().to_record_batch();
    let n_nodes = node_batch.num_rows();

    let id_col = node_batch
        .column_by_name(COL_NODE_ID)
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| col_err(COL_NODE_ID))?;

    let label_col = node_batch
        .column_by_name(COL_NODE_LABEL)
        .and_then(|c| c.as_any().downcast_ref::<ListArray>())
        .ok_or_else(|| col_err(COL_NODE_LABEL))?;

    // Columns that are not structural metadata.
    let node_schema = node_batch.schema();
    let prop_cols: Vec<(&str, &arrow_array::ArrayRef)> = node_schema
        .fields()
        .iter()
        .zip(node_batch.columns())
        .filter(|(f, _)| f.name() != COL_NODE_ID && f.name() != COL_NODE_LABEL)
        .map(|(f, arr)| (f.name().as_str(), arr))
        .collect();

    for row in 0..n_nodes {
        let id = id_col.value(row);

        // Labels: list of strings.
        let labels = label_strings(label_col, row);

        // Properties.
        let props = prop_string_for_row(&prop_cols, row)?;

        write!(out, "({id}").unwrap();
        if !labels.is_empty() {
            write!(out, ": {}", labels.join(" ")).unwrap();
        }
        if props.is_empty() {
            out.push(')');
        } else {
            write!(out, " {{ {props} }})").unwrap();
        }
        out.push('\n');
    }

    if n_nodes > 0 {
        out.push('\n');
    }

    // ── Edges ────────────────────────────────────────────────────────────────
    let edge_batch = graph.edges().to_record_batch();
    let n_edges = edge_batch.num_rows();

    let src_col = edge_batch
        .column_by_name(COL_EDGE_SRC)
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| col_err(COL_EDGE_SRC))?;

    let dst_col = edge_batch
        .column_by_name(COL_EDGE_DST)
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| col_err(COL_EDGE_DST))?;

    let type_col = edge_batch
        .column_by_name(COL_EDGE_TYPE)
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| col_err(COL_EDGE_TYPE))?;

    let dir_col = edge_batch
        .column_by_name(COL_EDGE_DIRECTION)
        .ok_or_else(|| col_err(COL_EDGE_DIRECTION))?;

    let edge_schema = edge_batch.schema();
    let edge_prop_cols: Vec<(&str, &arrow_array::ArrayRef)> = edge_schema
        .fields()
        .iter()
        .zip(edge_batch.columns())
        .filter(|(f, _)| {
            !matches!(
                f.name().as_str(),
                COL_EDGE_SRC | COL_EDGE_DST | COL_EDGE_TYPE | COL_EDGE_DIRECTION
            )
        })
        .map(|(f, arr)| (f.name().as_str(), arr))
        .collect();

    for row in 0..n_edges {
        let src = src_col.value(row);
        let dst = dst_col.value(row);
        let etype = type_col.value(row);

        // Read the raw i8 direction byte.
        let dir_i8 = read_i8(dir_col, row)?;
        let direction = Direction::try_from(dir_i8)?;

        let (lhs, rhs) = match direction {
            Direction::Out => ("-[", "]->"),
            Direction::In => ("<-[", "]-"),
            Direction::Both => ("<-[", "]->"),
            Direction::None => ("--[", "]--"),
        };

        let props = prop_string_for_row(&edge_prop_cols, row)?;
        if props.is_empty() {
            writeln!(out, "{src} {lhs}{etype}{rhs} {dst}").unwrap();
        } else {
            writeln!(out, "{src} {lhs}{etype}{rhs} {dst} {{ {props} }}").unwrap();
        }
    }

    Ok(out)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn col_err(name: &str) -> GFError {
    GFError::ColumnNotFound {
        column: name.to_owned(),
    }
}

/// Extract label strings from a `ListArray<Utf8>` row.
fn label_strings(col: &ListArray, row: usize) -> Vec<String> {
    let offsets = col.offsets();
    let start = offsets[row] as usize;
    let end = offsets[row + 1] as usize;

    let values = col
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("label list values must be Utf8");

    (start..end)
        .filter_map(|i| {
            if values.is_null(i) {
                None
            } else {
                Some(values.value(i).to_owned())
            }
        })
        .collect()
}

/// Read an `i8` from a column that may be `Int8Array` or `Int64Array`.
fn read_i8(col: &arrow_array::ArrayRef, row: usize) -> Result<i8> {
    use arrow_array::Int8Array;
    if let Some(a) = col.as_any().downcast_ref::<Int8Array>() {
        return Ok(a.value(row));
    }
    if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
        return Ok(a.value(row) as i8);
    }
    Err(GFError::TypeMismatch {
        message: format!("direction column has unexpected type {:?}", col.data_type()),
    })
}

/// Build the `key: value, ...` property string for one row.
///
/// Null values are silently omitted.
fn prop_string_for_row(cols: &[(&str, &arrow_array::ArrayRef)], row: usize) -> Result<String> {
    let mut parts: Vec<String> = Vec::new();

    for &(name, arr) in cols {
        if arr.is_null(row) {
            continue;
        }
        let val = format_value(arr, row)?;
        parts.push(format!("{name}: {val}"));
    }

    Ok(parts.join(", "))
}

/// Format a single array cell as a `.gf`-compatible value literal.
fn format_value(arr: &arrow_array::ArrayRef, row: usize) -> Result<String> {
    match arr.data_type() {
        DataType::Utf8 => {
            let s = arr
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(row);
            Ok(format!("\"{}\"", escape_string(s)))
        }
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
            let v = arr
                .as_any()
                .downcast_ref::<Int64Array>()
                .map(|a| a.value(row))
                .or_else(|| {
                    use arrow_array::{Int16Array, Int32Array, Int8Array};
                    arr.as_any()
                        .downcast_ref::<Int8Array>()
                        .map(|a| a.value(row) as i64)
                        .or_else(|| {
                            arr.as_any()
                                .downcast_ref::<Int16Array>()
                                .map(|a| a.value(row) as i64)
                        })
                        .or_else(|| {
                            arr.as_any()
                                .downcast_ref::<Int32Array>()
                                .map(|a| a.value(row) as i64)
                        })
                })
                .ok_or_else(|| GFError::TypeMismatch {
                    message: format!("unexpected integer type {:?}", arr.data_type()),
                })?;
            Ok(v.to_string())
        }
        DataType::Float32 | DataType::Float64 => {
            use arrow_array::Float32Array;
            let f = if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
                a.value(row)
            } else if let Some(a) = arr.as_any().downcast_ref::<Float32Array>() {
                a.value(row) as f64
            } else {
                return Err(GFError::TypeMismatch {
                    message: format!("unexpected float type {:?}", arr.data_type()),
                });
            };
            // Always emit a decimal point so the parser treats this as a float.
            if f.fract() == 0.0 && f.is_finite() {
                Ok(format!("{f:.1}"))
            } else {
                Ok(format!("{f}"))
            }
        }
        DataType::Boolean => {
            let b = arr
                .as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap()
                .value(row);
            Ok(if b { "true" } else { "false" }.to_owned())
        }
        DataType::List(_) => {
            // Nested list — emit as a JSON-style array of quoted strings.
            let list = arr.as_any().downcast_ref::<ListArray>().unwrap();
            let labels = label_strings(list, row);
            let quoted: Vec<String> = labels
                .iter()
                .map(|s| format!("\"{}\"", escape_string(s)))
                .collect();
            Ok(format!("[{}]", quoted.join(", ")))
        }
        other => Err(GFError::TypeMismatch {
            message: format!("unsupported column type for .gf serialisation: {other:?}"),
        }),
    }
}

/// Escape a string for inclusion in a `.gf` double-quoted literal.
fn escape_string(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use lynxes_core::{EdgeFrame, GraphFrame, NodeFrame};

    use crate::parse_gf;
    use arrow_array::{
        builder::{ListBuilder, StringBuilder},
        ArrayRef, Int64Array, Int8Array, RecordBatch, StringArray,
    };
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use std::sync::Arc;

    fn small_graph() -> GraphFrame {
        let mut lb = ListBuilder::new(StringBuilder::new());
        lb.values().append_value("Person");
        lb.append(true);
        lb.values().append_value("Company");
        lb.append(true);

        let node_schema = Arc::new(ArrowSchema::new(vec![
            Field::new(COL_NODE_ID, DataType::Utf8, false),
            Field::new(
                COL_NODE_LABEL,
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                false,
            ),
            Field::new("age", DataType::Int64, true),
        ]));
        let nodes = NodeFrame::from_record_batch(
            RecordBatch::try_new(
                node_schema,
                vec![
                    Arc::new(StringArray::from(vec!["alice", "acme"])) as ArrayRef,
                    Arc::new(lb.finish()) as ArrayRef,
                    Arc::new(Int64Array::from(vec![Some(30), None])) as ArrayRef,
                ],
            )
            .unwrap(),
        )
        .unwrap();

        use lynxes_core::{COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE};
        let edge_schema = Arc::new(ArrowSchema::new(vec![
            Field::new(COL_EDGE_SRC, DataType::Utf8, false),
            Field::new(COL_EDGE_DST, DataType::Utf8, false),
            Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
            Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
        ]));
        let edges = EdgeFrame::from_record_batch(
            RecordBatch::try_new(
                edge_schema,
                vec![
                    Arc::new(StringArray::from(vec!["alice"])) as ArrayRef,
                    Arc::new(StringArray::from(vec!["acme"])) as ArrayRef,
                    Arc::new(StringArray::from(vec!["WORKS_AT"])) as ArrayRef,
                    Arc::new(Int8Array::from(vec![0i8])) as ArrayRef, // Out
                ],
            )
            .unwrap(),
        )
        .unwrap();

        GraphFrame::new(nodes, edges).unwrap()
    }

    #[test]
    fn serialise_emits_node_and_edge_lines() {
        let gf_text = serialise_graph(&small_graph()).unwrap();
        assert!(
            gf_text.contains("(alice: Person"),
            "missing alice node: {gf_text}"
        );
        assert!(
            gf_text.contains("(acme: Company"),
            "missing acme node: {gf_text}"
        );
        assert!(
            gf_text.contains("-[WORKS_AT]->"),
            "missing out-edge: {gf_text}"
        );
    }

    #[test]
    fn serialise_omits_null_properties() {
        let gf_text = serialise_graph(&small_graph()).unwrap();
        // alice has age=30, acme has age=null → only alice should have age prop.
        assert!(
            gf_text.contains("age: 30"),
            "alice should have age: {gf_text}"
        );
        let acme_line = gf_text.lines().find(|l| l.contains("(acme:")).unwrap();
        assert!(
            !acme_line.contains("age"),
            "acme must not have age: {acme_line}"
        );
    }

    #[test]
    fn direction_arrows_map_correctly() {
        use lynxes_core::{COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE};

        let mut lb_n = ListBuilder::new(StringBuilder::new());
        for _ in 0..2 {
            lb_n.values().append_value("N");
            lb_n.append(true);
        }

        let ns = Arc::new(ArrowSchema::new(vec![
            Field::new(COL_NODE_ID, DataType::Utf8, false),
            Field::new(
                COL_NODE_LABEL,
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                false,
            ),
        ]));
        let nodes = NodeFrame::from_record_batch(
            RecordBatch::try_new(
                ns,
                vec![
                    Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef,
                    Arc::new(lb_n.finish()) as ArrayRef,
                ],
            )
            .unwrap(),
        )
        .unwrap();

        let es = Arc::new(ArrowSchema::new(vec![
            Field::new(COL_EDGE_SRC, DataType::Utf8, false),
            Field::new(COL_EDGE_DST, DataType::Utf8, false),
            Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
            Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
        ]));
        // Emit one row per direction.
        let dirs: Vec<i8> = vec![0, 1, 2, 3]; // Out, In, Both, None
        let n = dirs.len();
        let edges = EdgeFrame::from_record_batch(
            RecordBatch::try_new(
                es,
                vec![
                    Arc::new(StringArray::from(vec!["a"; n])) as ArrayRef,
                    Arc::new(StringArray::from(vec!["b"; n])) as ArrayRef,
                    Arc::new(StringArray::from(vec!["E"; n])) as ArrayRef,
                    Arc::new(Int8Array::from(dirs)) as ArrayRef,
                ],
            )
            .unwrap(),
        )
        .unwrap();

        let graph = GraphFrame::new(nodes, edges).unwrap();
        let text = serialise_graph(&graph).unwrap();

        assert!(text.contains("-[E]->"), "Out arrow missing");
        assert!(text.contains("<-[E]-"), "In arrow missing");
        assert!(text.contains("<-[E]->"), "Both arrow missing");
        assert!(text.contains("--[E]--"), "Undirected arrow missing");
    }

    #[test]
    fn write_gf_creates_parseable_file() {
        let graph = small_graph();
        let path = std::env::temp_dir().join(format!("gf-writer-test-{}.gf", std::process::id()));
        write_gf(&graph, &path).unwrap();
        let source = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        // Must parse without error.
        let doc = parse_gf(&source).expect("written .gf must be parseable");
        assert_eq!(doc.nodes.len(), graph.node_count());
        assert_eq!(doc.edges.len(), graph.edge_count());
    }

    #[test]
    fn escape_string_handles_special_chars() {
        assert_eq!(escape_string(r#"say "hi""#), r#"say \"hi\""#);
        assert_eq!(escape_string(r"back\slash"), r"back\\slash");
        assert_eq!(escape_string("plain"), "plain");
    }
}
