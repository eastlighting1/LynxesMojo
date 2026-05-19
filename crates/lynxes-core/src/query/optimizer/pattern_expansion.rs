use crate::{Expr, LogicalPlan, OptimizerPass};

/// Pattern-aware predicate pushdown for `LogicalPlan::PatternMatch`.
///
/// Today the logical pattern AST has no per-step `pre_filter` field, so this
/// pass performs the safe subset that the current plan shape can express:
/// conjuncts that depend only on the pattern root alias are rewritten into
/// plain node predicates and pushed into the `input` node frontier. Remaining
/// conjuncts stay in `where_` and are still evaluated by the executor.
#[derive(Debug, Default, Clone, Copy)]
pub struct PatternExpansion;

impl OptimizerPass for PatternExpansion {
    fn name(&self) -> &'static str {
        "PatternExpansion"
    }

    fn optimize(&self, plan: LogicalPlan) -> LogicalPlan {
        optimize_plan(plan)
    }
}

fn optimize_plan(plan: LogicalPlan) -> LogicalPlan {
    match plan {
        LogicalPlan::Scan { .. } => plan,
        LogicalPlan::Cache { input, key } => LogicalPlan::Cache {
            input: Box::new(optimize_plan(*input)),
            key,
        },
        LogicalPlan::Hint { input, hint } => LogicalPlan::Hint {
            input: Box::new(optimize_plan(*input)),
            hint,
        },
        LogicalPlan::FilterNodes { input, predicate } => LogicalPlan::FilterNodes {
            input: Box::new(optimize_plan(*input)),
            predicate,
        },
        LogicalPlan::FilterEdges { input, predicate } => LogicalPlan::FilterEdges {
            input: Box::new(optimize_plan(*input)),
            predicate,
        },
        LogicalPlan::ProjectNodes { input, columns } => LogicalPlan::ProjectNodes {
            input: Box::new(optimize_plan(*input)),
            columns,
        },
        LogicalPlan::ProjectEdges { input, columns } => LogicalPlan::ProjectEdges {
            input: Box::new(optimize_plan(*input)),
            columns,
        },
        LogicalPlan::Expand {
            input,
            edge_type,
            hops,
            direction,
            pre_filter,
        } => LogicalPlan::Expand {
            input: Box::new(optimize_plan(*input)),
            edge_type,
            hops,
            direction,
            pre_filter,
        },
        LogicalPlan::Traverse { input, pattern } => LogicalPlan::Traverse {
            input: Box::new(optimize_plan(*input)),
            pattern,
        },
        LogicalPlan::PatternMatch {
            input,
            pattern,
            where_,
        } => rewrite_pattern_match(optimize_plan(*input), pattern, where_),
        LogicalPlan::AggregateNeighbors {
            input,
            edge_type,
            agg,
        } => LogicalPlan::AggregateNeighbors {
            input: Box::new(optimize_plan(*input)),
            edge_type,
            agg,
        },
        LogicalPlan::Sort {
            input,
            by,
            descending,
        } => LogicalPlan::Sort {
            input: Box::new(optimize_plan(*input)),
            by,
            descending,
        },
        LogicalPlan::Limit { input, n } => LogicalPlan::Limit {
            input: Box::new(optimize_plan(*input)),
            n,
        },
    }
}

fn rewrite_pattern_match(
    mut input: LogicalPlan,
    pattern: crate::Pattern,
    where_: Option<Expr>,
) -> LogicalPlan {
    let Some(where_) = where_ else {
        return LogicalPlan::PatternMatch {
            input: Box::new(input),
            pattern,
            where_: None,
        };
    };
    let Some(root_alias) = pattern.steps.first().map(|step| step.from_alias.as_str()) else {
        return LogicalPlan::PatternMatch {
            input: Box::new(input),
            pattern,
            where_: Some(where_),
        };
    };

    let mut pushed = Vec::new();
    let mut remaining = Vec::new();

    for conjunct in split_conjuncts(where_) {
        if expr_depends_only_on_alias(&conjunct, root_alias) {
            if let Some(rewritten) = rewrite_pattern_expr_for_alias(&conjunct, root_alias) {
                pushed.push(rewritten);
                continue;
            }
        }
        remaining.push(conjunct);
    }

    for predicate in pushed {
        input = rewrite_filter_nodes(input, predicate);
    }

    LogicalPlan::PatternMatch {
        input: Box::new(input),
        pattern,
        where_: combine_conjuncts(remaining),
    }
}

fn rewrite_filter_nodes(input: LogicalPlan, predicate: Expr) -> LogicalPlan {
    match input {
        LogicalPlan::FilterNodes {
            input: inner,
            predicate: existing,
        } => LogicalPlan::FilterNodes {
            input: inner,
            predicate: Expr::And {
                left: Box::new(existing),
                right: Box::new(predicate),
            },
        },
        other => LogicalPlan::FilterNodes {
            input: Box::new(other),
            predicate,
        },
    }
}

fn split_conjuncts(expr: Expr) -> Vec<Expr> {
    match expr {
        Expr::And { left, right } => {
            let mut parts = split_conjuncts(*left);
            parts.extend(split_conjuncts(*right));
            parts
        }
        other => vec![other],
    }
}

fn combine_conjuncts(mut exprs: Vec<Expr>) -> Option<Expr> {
    if exprs.is_empty() {
        return None;
    }
    let first = exprs.remove(0);
    Some(exprs.into_iter().fold(first, |left, right| Expr::And {
        left: Box::new(left),
        right: Box::new(right),
    }))
}

fn expr_depends_only_on_alias(expr: &Expr, alias: &str) -> bool {
    let mut saw_pattern_col = false;
    let mut valid = true;
    collect_alias_usage(expr, alias, &mut saw_pattern_col, &mut valid);
    saw_pattern_col && valid
}

fn collect_alias_usage(expr: &Expr, alias: &str, saw_pattern_col: &mut bool, valid: &mut bool) {
    match expr {
        Expr::Col { .. } => {
            *valid = false;
        }
        Expr::Literal { .. } => {}
        Expr::BinaryOp { left, right, .. }
        | Expr::And { left, right }
        | Expr::Or { left, right } => {
            collect_alias_usage(left, alias, saw_pattern_col, valid);
            collect_alias_usage(right, alias, saw_pattern_col, valid);
        }
        Expr::UnaryOp { expr, .. } | Expr::Not { expr } | Expr::Cast { expr, .. } => {
            collect_alias_usage(expr, alias, saw_pattern_col, valid);
        }
        Expr::ListContains { expr, item } => {
            collect_alias_usage(expr, alias, saw_pattern_col, valid);
            collect_alias_usage(item, alias, saw_pattern_col, valid);
        }
        Expr::PatternCol { alias: used, .. } => {
            *saw_pattern_col = true;
            if used != alias {
                *valid = false;
            }
        }
        Expr::StringOp { expr, pattern, .. } => {
            collect_alias_usage(expr, alias, saw_pattern_col, valid);
            collect_alias_usage(pattern, alias, saw_pattern_col, valid);
        }
    }
}

fn rewrite_pattern_expr_for_alias(expr: &Expr, alias: &str) -> Option<Expr> {
    match expr {
        Expr::Col { .. } => None,
        Expr::Literal { value } => Some(Expr::Literal {
            value: value.clone(),
        }),
        Expr::BinaryOp { left, op, right } => Some(Expr::BinaryOp {
            left: Box::new(rewrite_pattern_expr_for_alias(left, alias)?),
            op: op.clone(),
            right: Box::new(rewrite_pattern_expr_for_alias(right, alias)?),
        }),
        Expr::UnaryOp { op, expr } => Some(Expr::UnaryOp {
            op: op.clone(),
            expr: Box::new(rewrite_pattern_expr_for_alias(expr, alias)?),
        }),
        Expr::ListContains { expr, item } => Some(Expr::ListContains {
            expr: Box::new(rewrite_pattern_expr_for_alias(expr, alias)?),
            item: Box::new(rewrite_pattern_expr_for_alias(item, alias)?),
        }),
        Expr::Cast { expr, dtype } => Some(Expr::Cast {
            expr: Box::new(rewrite_pattern_expr_for_alias(expr, alias)?),
            dtype: dtype.clone(),
        }),
        Expr::And { left, right } => Some(Expr::And {
            left: Box::new(rewrite_pattern_expr_for_alias(left, alias)?),
            right: Box::new(rewrite_pattern_expr_for_alias(right, alias)?),
        }),
        Expr::Or { left, right } => Some(Expr::Or {
            left: Box::new(rewrite_pattern_expr_for_alias(left, alias)?),
            right: Box::new(rewrite_pattern_expr_for_alias(right, alias)?),
        }),
        Expr::Not { expr } => Some(Expr::Not {
            expr: Box::new(rewrite_pattern_expr_for_alias(expr, alias)?),
        }),
        Expr::PatternCol {
            alias: used_alias,
            field,
        } if used_alias == alias => Some(Expr::Col {
            name: field.clone(),
        }),
        Expr::PatternCol { .. } => None,
        Expr::StringOp { op, expr, pattern } => Some(Expr::StringOp {
            op: op.clone(),
            expr: Box::new(rewrite_pattern_expr_for_alias(expr, alias)?),
            pattern: Box::new(rewrite_pattern_expr_for_alias(pattern, alias)?),
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{
        query::{Connector, EdgeTypeSpec, Pattern, PatternStep, ScalarValue},
        BinaryOp, Direction,
    };

    use super::*;

    #[derive(Debug)]
    struct DummyConnector;
    impl Connector for DummyConnector {}

    fn scan() -> LogicalPlan {
        LogicalPlan::Scan {
            source: Arc::new(DummyConnector),
            node_columns: None,
            edge_columns: None,
        }
    }

    fn root_age_predicate() -> Expr {
        Expr::BinaryOp {
            left: Box::new(Expr::PatternCol {
                alias: "a".to_owned(),
                field: "age".to_owned(),
            }),
            op: BinaryOp::Gt,
            right: Box::new(Expr::Literal {
                value: ScalarValue::Int(30),
            }),
        }
    }

    fn second_alias_age_predicate() -> Expr {
        Expr::BinaryOp {
            left: Box::new(Expr::PatternCol {
                alias: "b".to_owned(),
                field: "age".to_owned(),
            }),
            op: BinaryOp::Gt,
            right: Box::new(Expr::Literal {
                value: ScalarValue::Int(30),
            }),
        }
    }

    fn pattern() -> Pattern {
        Pattern::new(vec![PatternStep {
            from_alias: "a".to_owned(),
            edge_alias: Some("e".to_owned()),
            edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
            direction: Direction::Out,
            to_alias: "b".to_owned(),
        }])
    }

    #[test]
    fn pushes_root_alias_predicate_into_input_filternodes() {
        let plan = LogicalPlan::PatternMatch {
            input: Box::new(scan()),
            pattern: pattern(),
            where_: Some(root_age_predicate()),
        };

        let optimized = PatternExpansion.optimize(plan);

        let LogicalPlan::PatternMatch { input, where_, .. } = optimized else {
            panic!("expected pattern match");
        };
        assert!(where_.is_none());
        let LogicalPlan::FilterNodes { predicate, .. } = *input else {
            panic!("expected pushed FilterNodes input");
        };
        assert_eq!(
            predicate,
            Expr::BinaryOp {
                left: Box::new(Expr::Col {
                    name: "age".to_owned()
                }),
                op: BinaryOp::Gt,
                right: Box::new(Expr::Literal {
                    value: ScalarValue::Int(30)
                }),
            }
        );
    }

    #[test]
    fn preserves_non_root_alias_predicates_in_where_clause() {
        let plan = LogicalPlan::PatternMatch {
            input: Box::new(scan()),
            pattern: pattern(),
            where_: Some(Expr::And {
                left: Box::new(root_age_predicate()),
                right: Box::new(second_alias_age_predicate()),
            }),
        };

        let optimized = PatternExpansion.optimize(plan);

        let LogicalPlan::PatternMatch { input, where_, .. } = optimized else {
            panic!("expected pattern match");
        };
        let LogicalPlan::FilterNodes { predicate, .. } = *input else {
            panic!("expected pushed FilterNodes input");
        };
        assert_eq!(
            predicate,
            Expr::BinaryOp {
                left: Box::new(Expr::Col {
                    name: "age".to_owned()
                }),
                op: BinaryOp::Gt,
                right: Box::new(Expr::Literal {
                    value: ScalarValue::Int(30)
                }),
            }
        );
        assert_eq!(where_, Some(second_alias_age_predicate()));
    }

    #[test]
    fn does_not_push_plain_column_expressions() {
        let plain = Expr::BinaryOp {
            left: Box::new(Expr::Col {
                name: "age".to_owned(),
            }),
            op: BinaryOp::Gt,
            right: Box::new(Expr::Literal {
                value: ScalarValue::Int(30),
            }),
        };
        let plan = LogicalPlan::PatternMatch {
            input: Box::new(scan()),
            pattern: pattern(),
            where_: Some(plain.clone()),
        };

        let optimized = PatternExpansion.optimize(plan);

        let LogicalPlan::PatternMatch { input, where_, .. } = optimized else {
            panic!("expected pattern match");
        };
        assert!(matches!(*input, LogicalPlan::Scan { .. }));
        assert_eq!(where_, Some(plain));
    }
}
