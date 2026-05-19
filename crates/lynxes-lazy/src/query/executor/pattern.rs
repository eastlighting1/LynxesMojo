use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{Field, Schema as ArrowSchema};
use hashbrown::{HashMap, HashSet};

use lynxes_core::{Direction, EdgeFrame, GFError, GraphFrame, NodeFrame, Result, COL_NODE_LABEL};
use lynxes_plan::{
    EdgeTypeSpec, Expr, Pattern, PatternStep, PatternStepConstraint, StringOp, UnaryOp,
};

use super::{
    bind_pattern_alias, bind_pattern_alias_null, build_edge_node_ids, build_value_array,
    cast_value, convert_scalar, evaluate_binary_values, read_column_value, PatternAliasKind,
    PatternBindingRow, PatternBindings, Value,
};
pub(crate) fn matches_edge_type(edge_type: &str, spec: &EdgeTypeSpec) -> bool {
    match spec {
        EdgeTypeSpec::Single(value) => edge_type == value,
        EdgeTypeSpec::Multiple(values) => values.iter().any(|value| value == edge_type),
        EdgeTypeSpec::Any => true,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PatternCandidate {
    node_idx: u32,
    edge_row: Option<u32>,
}

pub(crate) fn pattern_node_matches_constraint(
    graph: &GraphFrame,
    edge_node_ids: &[String],
    node_idx: u32,
    alias: &str,
    pattern: &Pattern,
) -> Result<bool> {
    let Some(constraint) = pattern.node_constraint(alias) else {
        return Ok(true);
    };
    let Some(label) = constraint.label.as_deref() else {
        return Ok(true);
    };

    let node_id = edge_node_ids
        .get(node_idx as usize)
        .map(String::as_str)
        .ok_or_else(|| GFError::InvalidConfig {
            message: format!(
                "pattern alias '{alias}' is not a valid edge-local node index: {node_idx}"
            ),
        })?;
    let node_row = graph
        .nodes()
        .row_index(node_id)
        .ok_or_else(|| GFError::NodeNotFound {
            id: node_id.to_owned(),
        })?;

    match read_column_value(
        graph.nodes().to_record_batch(),
        node_row as usize,
        COL_NODE_LABEL,
    )? {
        Value::List(values) => Ok(values
            .iter()
            .any(|value| matches!(value, Value::String(candidate) if candidate == label))),
        other => Err(GFError::TypeMismatch {
            message: format!("node label column must be a list, got {other:?}"),
        }),
    }
}

pub(crate) fn bind_optional_pattern_step(
    row: &PatternBindingRow,
    step: &PatternStep,
) -> Result<Option<PatternBindingRow>> {
    let mut next = row.clone();
    if bind_pattern_alias_null(&mut next, &step.to_alias).is_err() {
        return Ok(None);
    }
    if let Some(edge_alias) = step.edge_alias.as_deref() {
        if bind_pattern_alias_null(&mut next, edge_alias).is_err() {
            return Ok(None);
        }
    }
    Ok(Some(next))
}

pub(crate) fn single_hop_pattern_candidates(
    edges: &EdgeFrame,
    from_idx: u32,
    step: &PatternStep,
) -> Vec<PatternCandidate> {
    let mut candidates = Vec::new();

    if matches!(step.direction, Direction::Out | Direction::Both) {
        for (&dst_idx, &edge_row) in edges
            .out_neighbors(from_idx)
            .iter()
            .zip(edges.out_edge_ids(from_idx).iter())
        {
            if matches_edge_type(edges.edge_type_at(edge_row), &step.edge_type) {
                candidates.push(PatternCandidate {
                    node_idx: dst_idx,
                    edge_row: Some(edge_row),
                });
            }
        }
    }

    if matches!(step.direction, Direction::In | Direction::Both) {
        for (&src_idx, &edge_row) in edges
            .in_neighbors(from_idx)
            .iter()
            .zip(edges.in_edge_ids(from_idx).iter())
        {
            if matches_edge_type(edges.edge_type_at(edge_row), &step.edge_type) {
                candidates.push(PatternCandidate {
                    node_idx: src_idx,
                    edge_row: Some(edge_row),
                });
            }
        }
    }

    candidates
}

struct VariableHopCollector<'a> {
    edges: &'a EdgeFrame,
    step: &'a PatternStep,
    constraint: &'a PatternStepConstraint,
    seen: HashSet<u32>,
    output: Vec<PatternCandidate>,
}

impl<'a> VariableHopCollector<'a> {
    fn new(
        edges: &'a EdgeFrame,
        step: &'a PatternStep,
        constraint: &'a PatternStepConstraint,
    ) -> Self {
        Self {
            edges,
            step,
            constraint,
            seen: HashSet::new(),
            output: Vec::new(),
        }
    }

    fn collect_from(&mut self, current_idx: u32, depth: u32, visited: &mut HashSet<u32>) {
        if depth >= self.constraint.min_hops && self.seen.insert(current_idx) {
            self.output.push(PatternCandidate {
                node_idx: current_idx,
                edge_row: None,
            });
        }
        if depth == self.constraint.max_hops {
            return;
        }

        for candidate in single_hop_pattern_candidates(self.edges, current_idx, self.step) {
            if !visited.insert(candidate.node_idx) {
                continue;
            }

            self.collect_from(candidate.node_idx, depth + 1, visited);
            visited.remove(&candidate.node_idx);
        }
    }

    fn finish(self) -> Vec<PatternCandidate> {
        self.output
    }
}

pub(crate) fn variable_hop_pattern_candidates(
    edges: &EdgeFrame,
    from_idx: u32,
    step: &PatternStep,
    constraint: &PatternStepConstraint,
) -> Vec<PatternCandidate> {
    let mut collector = VariableHopCollector::new(edges, step, constraint);
    let mut visited = HashSet::new();
    visited.insert(from_idx);
    collector.collect_from(from_idx, 0, &mut visited);
    collector.finish()
}

pub(crate) fn pattern_candidates(
    edges: &EdgeFrame,
    from_idx: u32,
    step: &PatternStep,
    constraint: &PatternStepConstraint,
) -> Vec<PatternCandidate> {
    if constraint.min_hops == 1 && constraint.max_hops == 1 {
        return single_hop_pattern_candidates(edges, from_idx, step);
    }

    variable_hop_pattern_candidates(edges, from_idx, step, constraint)
}

/// Expands one typed pattern step over an existing binding table.
///
/// Each input row must already bind `step.from_alias` to an `EdgeFrame` local
/// compact node index. The executor looks up matching edges from that node and
/// emits zero or more output rows. `step.to_alias` is bound to the neighboring
/// node index and `step.edge_alias`, when present, is bound to the traversed
/// edge row id.
///
/// Alias collisions are handled row-locally: if rebinding an alias would
/// assign a different value, that candidate row is dropped. This preserves the
/// "same alias means same graph element" rule without aborting the whole
/// pattern execution.
#[allow(dead_code)]
pub(crate) fn execute_pattern_step(
    graph: &GraphFrame,
    pattern: &Pattern,
    step_index: usize,
    step: &PatternStep,
    input: &PatternBindings,
) -> Result<PatternBindings> {
    let edges = graph.edges();
    let edge_node_ids = build_edge_node_ids(edges)?;
    let step_constraint = pattern.step_constraint(step_index);
    let mut output = Vec::new();

    for row in input {
        let from_idx =
            row.get(&step.from_alias)
                .copied()
                .ok_or_else(|| GFError::InvalidConfig {
                    message: format!(
                        "pattern step requires alias '{}' to be bound before execution",
                        step.from_alias
                    ),
                })?;

        let Some(from_idx) = from_idx else {
            if step_constraint.optional {
                if let Some(next) = bind_optional_pattern_step(row, step)? {
                    output.push(next);
                }
            }
            continue;
        };

        let mut matched = false;
        for candidate in pattern_candidates(edges, from_idx, step, step_constraint) {
            if !pattern_node_matches_constraint(
                graph,
                &edge_node_ids,
                candidate.node_idx,
                &step.to_alias,
                pattern,
            )? {
                continue;
            }

            let mut next = row.clone();

            if bind_pattern_alias(&mut next, &step.to_alias, candidate.node_idx).is_err() {
                continue;
            }
            if let Some(edge_alias) = step.edge_alias.as_deref() {
                let Some(edge_row) = candidate.edge_row else {
                    continue;
                };
                if bind_pattern_alias(&mut next, edge_alias, edge_row).is_err() {
                    continue;
                }
            }

            matched = true;
            output.push(next);
        }

        if step_constraint.optional && !matched {
            if let Some(next) = bind_optional_pattern_step(row, step)? {
                output.push(next);
            }
        }
    }

    Ok(output)
}

#[allow(dead_code)]
pub(crate) fn execute_pattern_steps(
    graph: &GraphFrame,
    pattern: &Pattern,
    seed_bindings: &PatternBindings,
) -> Result<PatternBindings> {
    let mut current = seed_bindings.clone();

    for (step_index, step) in pattern.steps.iter().enumerate() {
        if current.is_empty() {
            break;
        }
        current = execute_pattern_step(graph, pattern, step_index, step, &current)?;
    }

    Ok(current)
}

#[allow(dead_code)]
pub(crate) fn apply_pattern_where(
    graph: &GraphFrame,
    bindings: &PatternBindings,
    where_: Option<&Expr>,
) -> Result<PatternBindings> {
    let Some(where_) = where_ else {
        return Ok(bindings.clone());
    };

    let mut filtered = Vec::with_capacity(bindings.len());
    for row in bindings {
        match evaluate_pattern_expr(graph, row, where_)? {
            Value::Bool(true) => filtered.push(row.clone()),
            Value::Bool(false) => {}
            other => {
                return Err(GFError::TypeMismatch {
                    message: format!(
                        "pattern where predicate must evaluate to bool, got {other:?}"
                    ),
                });
            }
        }
    }

    Ok(filtered)
}

#[allow(dead_code)]
pub(crate) fn evaluate_pattern_expr(
    graph: &GraphFrame,
    binding: &PatternBindingRow,
    expr: &Expr,
) -> Result<Value> {
    match expr {
        Expr::Col { name } => Err(GFError::UnsupportedOperation {
            message: format!(
                "plain column reference '{name}' is not supported in PatternMatch where clauses"
            ),
        }),
        Expr::Literal { value } => Ok(convert_scalar(value)),
        Expr::BinaryOp { left, op, right } => evaluate_binary_values(
            evaluate_pattern_expr(graph, binding, left)?,
            op,
            evaluate_pattern_expr(graph, binding, right)?,
        ),
        Expr::UnaryOp { op, expr } => {
            let value = evaluate_pattern_expr(graph, binding, expr)?;
            match (op, value) {
                (UnaryOp::Neg, Value::Int(value)) => Ok(Value::Int(-value)),
                (UnaryOp::Neg, Value::Float(value)) => Ok(Value::Float(-value)),
                (_, other) => Err(GFError::TypeMismatch {
                    message: format!("unsupported unary expression operand: {other:?}"),
                }),
            }
        }
        Expr::ListContains { expr, item } => {
            let list = evaluate_pattern_expr(graph, binding, expr)?;
            let item = evaluate_pattern_expr(graph, binding, item)?;
            match list {
                Value::List(values) => Ok(Value::Bool(values.iter().any(|value| value == &item))),
                other => Err(GFError::TypeMismatch {
                    message: format!("ListContains expects a list operand, got {other:?}"),
                }),
            }
        }
        Expr::Cast { expr, dtype } => {
            cast_value(evaluate_pattern_expr(graph, binding, expr)?, dtype)
        }
        Expr::And { left, right } => {
            let left = evaluate_pattern_expr(graph, binding, left)?;
            let right = evaluate_pattern_expr(graph, binding, right)?;
            match (left, right) {
                (Value::Bool(left), Value::Bool(right)) => Ok(Value::Bool(left && right)),
                (left, right) => Err(GFError::TypeMismatch {
                    message: format!(
                        "boolean op expects bool operands, got {left:?} and {right:?}"
                    ),
                }),
            }
        }
        Expr::Or { left, right } => {
            let left = evaluate_pattern_expr(graph, binding, left)?;
            let right = evaluate_pattern_expr(graph, binding, right)?;
            match (left, right) {
                (Value::Bool(left), Value::Bool(right)) => Ok(Value::Bool(left || right)),
                (left, right) => Err(GFError::TypeMismatch {
                    message: format!(
                        "boolean op expects bool operands, got {left:?} and {right:?}"
                    ),
                }),
            }
        }
        Expr::Not { expr } => match evaluate_pattern_expr(graph, binding, expr)? {
            Value::Bool(value) => Ok(Value::Bool(!value)),
            other => Err(GFError::TypeMismatch {
                message: format!("Not expects bool, got {other:?}"),
            }),
        },
        Expr::PatternCol { alias, field } => read_pattern_field_value(graph, binding, alias, field),
        Expr::StringOp { op, expr, pattern } => {
            let subject = evaluate_pattern_expr(graph, binding, expr)?;
            let pat = evaluate_pattern_expr(graph, binding, pattern)?;
            match (subject, pat) {
                (Value::String(s), Value::String(p)) => Ok(Value::Bool(match op {
                    StringOp::Contains => s.contains(p.as_str()),
                    StringOp::StartsWith => s.starts_with(p.as_str()),
                    StringOp::EndsWith => s.ends_with(p.as_str()),
                })),
                (s, p) => Err(GFError::TypeMismatch {
                    message: format!("StringOp expects string operands, got {s:?} and {p:?}"),
                }),
            }
        }
    }
}

#[allow(dead_code)]
pub(crate) fn read_pattern_field_value(
    graph: &GraphFrame,
    binding: &PatternBindingRow,
    alias: &str,
    field: &str,
) -> Result<Value> {
    let bound = binding
        .get(alias)
        .copied()
        .ok_or_else(|| GFError::InvalidConfig {
            message: format!("pattern predicate requires alias '{alias}' to be bound"),
        })?;
    let Some(bound) = bound else {
        return Ok(Value::Null);
    };

    let node_has_field = graph.nodes().schema().field_with_name(field).is_ok();
    let edge_has_field = graph.edges().schema().field_with_name(field).is_ok();

    match (node_has_field, edge_has_field) {
        (true, false) => {
            let edge_node_ids = build_edge_node_ids(graph.edges())?;
            let node_id = edge_node_ids
                .get(bound as usize)
                .map(String::as_str)
                .ok_or_else(|| GFError::InvalidConfig {
                    message: format!(
                        "pattern alias '{alias}' is not a valid edge-local node index: {bound}"
                    ),
                })?;
            let node_row = graph.nodes().row_index(node_id).ok_or_else(|| GFError::NodeNotFound {
                id: node_id.to_owned(),
            })?;
            read_column_value(graph.nodes().to_record_batch(), node_row as usize, field)
        }
        (false, true) => {
            if bound as usize >= graph.edges().len() {
                return Err(GFError::InvalidConfig {
                    message: format!(
                        "pattern alias '{alias}' is not a valid edge row index: {bound}"
                    ),
                });
            }
            read_column_value(graph.edges().to_record_batch(), bound as usize, field)
        }
        (true, true) => Err(GFError::InvalidConfig {
            message: format!(
                "pattern field reference '{alias}.{field}' is ambiguous because '{field}' exists on both nodes and edges"
            ),
        }),
        (false, false) => Err(GFError::ColumnNotFound {
            column: field.to_owned(),
        }),
    }
}

#[allow(dead_code)]
pub(crate) fn collect_pattern_aliases(
    pattern: &[PatternStep],
) -> Result<Vec<(String, PatternAliasKind)>> {
    let mut aliases = Vec::new();
    let mut kinds = HashMap::<String, PatternAliasKind>::new();

    let mut register = |alias: &str, kind: PatternAliasKind| -> Result<()> {
        match kinds.get(alias).copied() {
            Some(existing) if existing == kind => Ok(()),
            Some(existing) => Err(GFError::InvalidConfig {
                message: format!(
                    "pattern alias '{alias}' is used as both {:?} and {:?}",
                    existing, kind
                ),
            }),
            None => {
                kinds.insert(alias.to_owned(), kind);
                aliases.push((alias.to_owned(), kind));
                Ok(())
            }
        }
    };

    for step in pattern {
        register(&step.from_alias, PatternAliasKind::Node)?;
        if let Some(edge_alias) = step.edge_alias.as_deref() {
            register(edge_alias, PatternAliasKind::Edge)?;
        }
        register(&step.to_alias, PatternAliasKind::Node)?;
    }

    Ok(aliases)
}

#[allow(dead_code)]
pub(crate) fn materialize_pattern_bindings(
    graph: &GraphFrame,
    pattern: &[PatternStep],
    bindings: &PatternBindings,
) -> Result<RecordBatch> {
    let aliases = collect_pattern_aliases(pattern)?;
    let mut fields = Vec::new();
    let mut columns = Vec::new();

    for (alias, kind) in aliases {
        let schema = match kind {
            PatternAliasKind::Node => graph.nodes().schema(),
            PatternAliasKind::Edge => graph.edges().schema(),
        };

        for field in schema.fields() {
            let field = field.as_ref();
            let qualified_name = format!("{alias}.{}", field.name());
            let values = bindings
                .iter()
                .map(|row| read_pattern_field_value(graph, row, &alias, field.name()))
                .collect::<Result<Vec<_>>>()?;
            let nullable =
                field.is_nullable() || values.iter().any(|value| matches!(value, Value::Null));
            fields.push(Field::new(
                &qualified_name,
                field.data_type().clone(),
                nullable,
            ));
            columns.push(build_value_array(field.data_type(), values)?);
        }
    }

    RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), columns)
        .map_err(|error| GFError::IoError(std::io::Error::other(error)))
}

pub(crate) fn validate_pattern_support(pattern: &Pattern) -> Result<()> {
    for (index, step) in pattern.steps.iter().enumerate() {
        let constraint = pattern.step_constraint(index);
        if constraint.min_hops == 0 {
            return Err(GFError::InvalidConfig {
                message: format!("pattern step {index} must use min_hops >= 1"),
            });
        }
        if constraint.max_hops < constraint.min_hops {
            return Err(GFError::InvalidConfig {
                message: format!(
                    "pattern step {index} has max_hops {} smaller than min_hops {}",
                    constraint.max_hops, constraint.min_hops
                ),
            });
        }
        if step.edge_alias.is_some() && constraint.max_hops > 1 {
            return Err(GFError::UnsupportedOperation {
                message: format!(
                    "pattern step {index} cannot bind an edge alias across multi-hop traversal"
                ),
            });
        }
    }

    Ok(())
}

#[allow(dead_code)]
pub(crate) fn execute_pattern_match(
    graph: &GraphFrame,
    anchors: &NodeFrame,
    pattern: &Pattern,
    where_: Option<&Expr>,
) -> Result<RecordBatch> {
    if pattern.is_empty() {
        return Err(GFError::InvalidConfig {
            message: "PatternMatch requires at least one step".to_owned(),
        });
    }
    validate_pattern_support(pattern)?;

    let first_step = &pattern.steps[0];
    let edge_node_ids = build_edge_node_ids(graph.edges())?;
    let mut seed_bindings = PatternBindings::new();
    for anchor_id in anchors.id_column().iter().flatten() {
        let Some(edge_local_idx) = graph.edges().node_row_idx(anchor_id) else {
            continue;
        };
        if !pattern_node_matches_constraint(
            graph,
            &edge_node_ids,
            edge_local_idx,
            &first_step.from_alias,
            pattern,
        )? {
            continue;
        }

        let mut row = PatternBindingRow::new();
        bind_pattern_alias(&mut row, &first_step.from_alias, edge_local_idx)?;
        seed_bindings.push(row);
    }

    let bindings = execute_pattern_steps(graph, pattern, &seed_bindings)?;
    let filtered = apply_pattern_where(graph, &bindings, where_)?;
    materialize_pattern_bindings(graph, &pattern.steps, &filtered)
}
