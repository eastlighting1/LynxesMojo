#[cfg(test)]
mod executor_tests {
    use std::sync::Arc;

    use super::super::{
        apply_pattern_where, bind_pattern_alias, execute, execute_pattern_step,
        execute_pattern_steps, materialize_pattern_bindings, string_array, ExecutionValue,
        PatternBindingRow, PatternBindings,
    };
    use arrow_array::{
        builder::{ListBuilder, StringBuilder},
        ArrayRef, Float64Array, Int64Array, Int8Array, ListArray, RecordBatch, StringArray,
    };
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lynxes_core::{
        Direction, EdgeFrame, EdgeTypeSpec, GFError, GraphFrame, NodeFrame, Optimizer,
        OptimizerOptions, Pattern, PatternStep, COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC,
        COL_EDGE_TYPE, COL_NODE_ID, COL_NODE_LABEL,
    };
    use lynxes_plan::{
        AggExpr, BinaryOp, Connector, ExecutionHint, Expr, LogicalPlan, PartitionStrategy,
        ScalarValue,
    };

    fn labels_array(values: &[&[&str]]) -> ListArray {
        let mut builder = ListBuilder::new(StringBuilder::new());
        for labels in values {
            for label in *labels {
                builder.values().append_value(label);
            }
            builder.append(true);
        }
        builder.finish()
    }

    fn demo_graph() -> GraphFrame {
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
                    Arc::new(StringArray::from(vec!["alice", "bob", "charlie", "acme"]))
                        as ArrayRef,
                    Arc::new(labels_array(&[
                        &["Person"],
                        &["Person"],
                        &["Person"],
                        &["Company"],
                    ])) as ArrayRef,
                    Arc::new(Int64Array::from(vec![Some(30), Some(40), Some(20), None]))
                        as ArrayRef,
                ],
            )
            .unwrap(),
        )
        .unwrap();

        let edge_schema = Arc::new(ArrowSchema::new(vec![
            Field::new(COL_EDGE_SRC, DataType::Utf8, false),
            Field::new(COL_EDGE_DST, DataType::Utf8, false),
            Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
            Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
            Field::new("weight", DataType::Int64, true),
        ]));
        let edges = EdgeFrame::from_record_batch(
            RecordBatch::try_new(
                edge_schema,
                vec![
                    Arc::new(StringArray::from(vec!["alice", "alice", "bob"])) as ArrayRef,
                    Arc::new(StringArray::from(vec!["bob", "charlie", "acme"])) as ArrayRef,
                    Arc::new(StringArray::from(vec!["KNOWS", "KNOWS", "WORKS_AT"])) as ArrayRef,
                    Arc::new(Int8Array::from(vec![0i8, 0, 0])) as ArrayRef,
                    Arc::new(Int64Array::from(vec![Some(1), Some(2), Some(3)])) as ArrayRef,
                ],
            )
            .unwrap(),
        )
        .unwrap();

        GraphFrame::new(nodes, edges).unwrap()
    }

    fn scan(source: Arc<GraphFrame>) -> LogicalPlan {
        #[derive(Debug)]
        struct DummyConnector;
        impl Connector for DummyConnector {}

        let _ = source;
        LogicalPlan::Scan {
            source: Arc::new(DummyConnector),
            node_columns: None,
            edge_columns: None,
        }
    }

    #[test]
    fn filter_nodes_project_sort_limit_executes() {
        let graph = Arc::new(demo_graph());
        let plan = LogicalPlan::Limit {
            input: Box::new(LogicalPlan::Sort {
                input: Box::new(LogicalPlan::ProjectNodes {
                    input: Box::new(LogicalPlan::FilterNodes {
                        input: Box::new(scan(graph.clone())),
                        predicate: Expr::BinaryOp {
                            left: Box::new(Expr::Col {
                                name: "age".to_owned(),
                            }),
                            op: BinaryOp::Gt,
                            right: Box::new(Expr::Literal {
                                value: ScalarValue::Int(25),
                            }),
                        },
                    }),
                    columns: vec!["age".to_owned()],
                }),
                by: "age".to_owned(),
                descending: true,
            }),
            n: 1,
        };

        let result = execute(&plan, graph).unwrap();
        let ExecutionValue::Nodes(nodes) = result else {
            panic!("expected node result");
        };

        assert_eq!(nodes.len(), 1);
        assert_eq!(
            nodes.column_names(),
            vec![COL_NODE_ID, COL_NODE_LABEL, "age"]
        );
        assert_eq!(nodes.id_column().value(0), "bob");
    }

    #[test]
    fn filter_edges_project_sort_limit_executes() {
        let graph = Arc::new(demo_graph());
        let plan = LogicalPlan::Limit {
            input: Box::new(LogicalPlan::Sort {
                input: Box::new(LogicalPlan::ProjectEdges {
                    input: Box::new(LogicalPlan::FilterEdges {
                        input: Box::new(scan(graph.clone())),
                        predicate: Expr::BinaryOp {
                            left: Box::new(Expr::Col {
                                name: COL_EDGE_TYPE.to_owned(),
                            }),
                            op: BinaryOp::Eq,
                            right: Box::new(Expr::Literal {
                                value: ScalarValue::String("KNOWS".to_owned()),
                            }),
                        },
                    }),
                    columns: vec!["weight".to_owned()],
                }),
                by: "weight".to_owned(),
                descending: true,
            }),
            n: 1,
        };

        let result = execute(&plan, graph).unwrap();
        let ExecutionValue::Edges(edges) = result else {
            panic!("expected edge result");
        };

        assert_eq!(edges.len(), 1);
        let weight = edges
            .column("weight")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(weight.value(0), 2);
    }

    #[test]
    fn expand_from_filtered_nodes_returns_traversed_subgraph() {
        let graph = Arc::new(demo_graph());
        let plan = LogicalPlan::Expand {
            input: Box::new(LogicalPlan::FilterNodes {
                input: Box::new(scan(graph.clone())),
                predicate: Expr::BinaryOp {
                    left: Box::new(Expr::Col {
                        name: COL_NODE_ID.to_owned(),
                    }),
                    op: BinaryOp::Eq,
                    right: Box::new(Expr::Literal {
                        value: ScalarValue::String("alice".to_owned()),
                    }),
                },
            }),
            edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
            hops: 1,
            direction: Direction::Out,
            pre_filter: None,
        };

        let result = execute(&plan, graph).unwrap();
        let ExecutionValue::Graph(graph) = result else {
            panic!("expected graph result");
        };

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
        assert!(graph.nodes().row_index("alice").is_some());
        assert!(graph.nodes().row_index("bob").is_some());
        assert!(graph.nodes().row_index("charlie").is_some());
    }

    #[test]
    fn traverse_executes_pattern_steps_in_order() {
        let graph = Arc::new(demo_graph());
        let plan = LogicalPlan::Traverse {
            input: Box::new(LogicalPlan::FilterNodes {
                input: Box::new(scan(graph.clone())),
                predicate: Expr::BinaryOp {
                    left: Box::new(Expr::Col {
                        name: COL_NODE_ID.to_owned(),
                    }),
                    op: BinaryOp::Eq,
                    right: Box::new(Expr::Literal {
                        value: ScalarValue::String("alice".to_owned()),
                    }),
                },
            }),
            pattern: vec![
                PatternStep {
                    from_alias: "a".to_owned(),
                    edge_alias: Some("e1".to_owned()),
                    edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
                    direction: Direction::Out,
                    to_alias: "b".to_owned(),
                },
                PatternStep {
                    from_alias: "b".to_owned(),
                    edge_alias: Some("e2".to_owned()),
                    edge_type: EdgeTypeSpec::Single("WORKS_AT".to_owned()),
                    direction: Direction::Out,
                    to_alias: "c".to_owned(),
                },
            ],
        };

        let result = execute(&plan, graph).unwrap();
        let ExecutionValue::Graph(graph) = result else {
            panic!("expected graph result");
        };

        assert_eq!(graph.node_count(), 4);
        assert_eq!(graph.edge_count(), 3);
        assert!(graph.nodes().row_index("alice").is_some());
        assert!(graph.nodes().row_index("bob").is_some());
        assert!(graph.nodes().row_index("charlie").is_some());
        assert!(graph.nodes().row_index("acme").is_some());
    }

    #[test]
    fn aggregate_neighbors_count_appends_node_column() {
        let graph = Arc::new(demo_graph());
        let plan = LogicalPlan::AggregateNeighbors {
            input: Box::new(scan(graph.clone())),
            edge_type: "KNOWS".to_owned(),
            agg: AggExpr::Count,
        };

        let result = execute(&plan, graph).unwrap();
        let ExecutionValue::Nodes(nodes) = result else {
            panic!("expected node result");
        };

        let counts = nodes
            .column("count")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(counts.value(0), 2);
        assert_eq!(counts.value(1), 0);
        assert_eq!(counts.value(2), 0);
        assert_eq!(counts.value(3), 0);
    }

    #[test]
    fn aggregate_neighbors_mean_can_read_edge_columns() {
        let graph = Arc::new(demo_graph());
        let plan = LogicalPlan::AggregateNeighbors {
            input: Box::new(LogicalPlan::FilterNodes {
                input: Box::new(scan(graph.clone())),
                predicate: Expr::BinaryOp {
                    left: Box::new(Expr::Col {
                        name: COL_NODE_ID.to_owned(),
                    }),
                    op: BinaryOp::Eq,
                    right: Box::new(Expr::Literal {
                        value: ScalarValue::String("alice".to_owned()),
                    }),
                },
            }),
            edge_type: "KNOWS".to_owned(),
            agg: AggExpr::Mean {
                expr: Expr::Col {
                    name: "weight".to_owned(),
                },
            },
        };

        let result = execute(&plan, graph).unwrap();
        let ExecutionValue::Nodes(nodes) = result else {
            panic!("expected node result");
        };

        let mean = nodes
            .column("mean")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(nodes.len(), 1);
        assert!((mean.value(0) - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn aggregate_neighbors_alias_overrides_output_column_name() {
        let graph = Arc::new(demo_graph());
        let plan = LogicalPlan::AggregateNeighbors {
            input: Box::new(scan(graph.clone())),
            edge_type: "KNOWS".to_owned(),
            agg: AggExpr::Alias {
                expr: Box::new(AggExpr::Count),
                name: "friend_count".to_owned(),
            },
        };

        let result = execute(&plan, graph).unwrap();
        let ExecutionValue::Nodes(nodes) = result else {
            panic!("expected node result");
        };

        // Column must be named "friend_count", not "count".
        assert!(
            nodes.column("count").is_none(),
            "bare 'count' column should not exist"
        );
        let counts = nodes
            .column("friend_count")
            .expect("alias column 'friend_count' must exist")
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(counts.value(0), 2); // alice knows bob + charlie
    }

    // ???? OPT-002: EarlyTermination hint tests ??????????????????????????????????????????????????????????????????

    /// `LimitAware { n=2 }` wrapping an Expand should stop BFS once the visited
    /// set reaches 2 nodes (the seed "alice" counts as 1 before any hops).
    #[test]
    fn limit_aware_expand_stops_early() {
        // demo_graph: alice?萸븄b, alice?萸뻞arlie, bob?萸밹me
        // Without a limit all 4 nodes would be reachable from alice in 2 hops.
        // Seed the expansion from alice only (FilterNodes before Expand).
        let graph = Arc::new(demo_graph());
        let alice_only = Box::new(LogicalPlan::FilterNodes {
            input: Box::new(scan(graph.clone())),
            predicate: Expr::BinaryOp {
                left: Box::new(Expr::Col {
                    name: COL_NODE_ID.to_owned(),
                }),
                op: BinaryOp::Eq,
                right: Box::new(Expr::Literal {
                    value: ScalarValue::String("alice".to_owned()),
                }),
            },
        });
        let hint_plan = LogicalPlan::Hint {
            hint: ExecutionHint::LimitAware { n: 2 },
            input: Box::new(LogicalPlan::Expand {
                input: alice_only,
                edge_type: EdgeTypeSpec::Any,
                hops: 3,
                direction: Direction::Out,
                pre_filter: None,
            }),
        };

        let result = execute(&hint_plan, graph).unwrap();
        let ExecutionValue::Graph(g) = result else {
            panic!("expected graph result");
        };
        // visited starts at {alice}; first out-neighbor admission makes len = 2,
        // triggering break 'expand immediately.
        assert_eq!(
            g.node_count(),
            2,
            "LimitAware(2) with 1-node seed must return exactly 2 nodes"
        );
        assert!(
            g.nodes().row_index("alice").is_some(),
            "alice must be in result"
        );
    }

    /// `LimitAware { n }` where `n` exceeds total reachable nodes should return
    /// the same graph as an unrestricted Expand.
    #[test]
    fn limit_aware_expand_no_stop_when_n_exceeds_reachable() {
        let graph = Arc::new(demo_graph());

        // Seed from alice so both plans share the same 1-node frontier.
        let alice_filter = || {
            Box::new(LogicalPlan::FilterNodes {
                input: Box::new(scan(graph.clone())),
                predicate: Expr::BinaryOp {
                    left: Box::new(Expr::Col {
                        name: COL_NODE_ID.to_owned(),
                    }),
                    op: BinaryOp::Eq,
                    right: Box::new(Expr::Literal {
                        value: ScalarValue::String("alice".to_owned()),
                    }),
                },
            })
        };

        let unrestricted = LogicalPlan::Expand {
            input: alice_filter(),
            edge_type: EdgeTypeSpec::Any,
            hops: 3,
            direction: Direction::Out,
            pre_filter: None,
        };
        let hint_plan = LogicalPlan::Hint {
            hint: ExecutionHint::LimitAware { n: 100 },
            input: Box::new(LogicalPlan::Expand {
                input: alice_filter(),
                edge_type: EdgeTypeSpec::Any,
                hops: 3,
                direction: Direction::Out,
                pre_filter: None,
            }),
        };

        let base = match execute(&unrestricted, graph.clone()).unwrap() {
            ExecutionValue::Graph(g) => g,
            _ => panic!("expected graph"),
        };
        let limited = match execute(&hint_plan, graph).unwrap() {
            ExecutionValue::Graph(g) => g,
            _ => panic!("expected graph"),
        };

        assert_eq!(base.node_count(), limited.node_count());
        assert_eq!(base.edge_count(), limited.edge_count());
    }

    /// `LimitAware { n=2 }` wrapping a Traverse should stop before completing
    /// all pattern steps once the visited set grows past `n`.
    #[test]
    fn limit_aware_traverse_stops_early() {
        use lynxes_core::Direction as Dir;

        // demo_graph seeded from alice only.
        // Pattern: alice -KNOWS-> {bob, charlie} -WORKS_AT-> {acme}
        // Without limit: visited = {alice, bob, charlie, acme} (4 nodes).
        // With LimitAware n=2: after step 1 visited = {alice, bob, charlie} (3 ??2),
        // so the step-level break fires and acme is never reached.
        let graph = Arc::new(demo_graph());
        let alice_only = Box::new(LogicalPlan::FilterNodes {
            input: Box::new(scan(graph.clone())),
            predicate: Expr::BinaryOp {
                left: Box::new(Expr::Col {
                    name: COL_NODE_ID.to_owned(),
                }),
                op: BinaryOp::Eq,
                right: Box::new(Expr::Literal {
                    value: ScalarValue::String("alice".to_owned()),
                }),
            },
        });
        let hint_plan = LogicalPlan::Hint {
            hint: ExecutionHint::LimitAware { n: 2 },
            input: Box::new(LogicalPlan::Traverse {
                input: alice_only,
                pattern: vec![
                    PatternStep {
                        from_alias: "a".into(),
                        edge_alias: None,
                        edge_type: EdgeTypeSpec::Single("KNOWS".into()),
                        direction: Dir::Out,
                        to_alias: "b".into(),
                    },
                    PatternStep {
                        from_alias: "b".into(),
                        edge_alias: None,
                        edge_type: EdgeTypeSpec::Single("WORKS_AT".into()),
                        direction: Dir::Out,
                        to_alias: "c".into(),
                    },
                ],
            }),
        };

        let result = execute(&hint_plan, graph).unwrap();
        let ExecutionValue::Graph(g) = result else {
            panic!("expected graph result");
        };
        // Step-level stop fires after step 1; acme (reachable only at step 2)
        // must not appear in the result.
        assert!(
            g.nodes().row_index("acme").is_none(),
            "acme must not be reached under LimitAware(2)"
        );
        assert!(
            g.nodes().row_index("alice").is_some(),
            "alice must be in result"
        );
    }

    /// `TopK { n=2 }` over a Sort should return only the top 2 rows in correct order.
    #[test]
    fn top_k_sort_returns_k_rows_in_correct_order() {
        // demo_graph nodes have ages: alice=30, bob=40, charlie=20, acme=null
        // Sort descending by age: bob(40) > alice(30) > charlie(20) > acme(null)
        let graph = Arc::new(demo_graph());
        let hint_plan = LogicalPlan::Hint {
            hint: ExecutionHint::TopK { n: 2 },
            input: Box::new(LogicalPlan::Sort {
                input: Box::new(LogicalPlan::FilterNodes {
                    input: Box::new(scan(graph.clone())),
                    // Keep only nodes with a non-null age.
                    predicate: Expr::BinaryOp {
                        left: Box::new(Expr::Col {
                            name: "age".to_owned(),
                        }),
                        op: BinaryOp::Gt,
                        right: Box::new(Expr::Literal {
                            value: ScalarValue::Int(0),
                        }),
                    },
                }),
                by: "age".to_owned(),
                descending: true,
            }),
        };

        let result = execute(&hint_plan, graph).unwrap();
        let ExecutionValue::Nodes(nodes) = result else {
            panic!("expected node result");
        };

        assert_eq!(nodes.len(), 2, "TopK(2) must return exactly 2 rows");
        // Row 0 should be the highest age (bob=40), row 1 the next (alice=30).
        assert_eq!(nodes.id_column().value(0), "bob");
        assert_eq!(nodes.id_column().value(1), "alice");
    }

    /// `TopK { n }` where `n >= num_rows` must behave identically to a full Sort.
    #[test]
    fn top_k_sort_full_result_when_k_exceeds_rows() {
        let graph = Arc::new(demo_graph());
        let sort_plan = LogicalPlan::Sort {
            input: Box::new(LogicalPlan::FilterNodes {
                input: Box::new(scan(graph.clone())),
                predicate: Expr::BinaryOp {
                    left: Box::new(Expr::Col {
                        name: "age".to_owned(),
                    }),
                    op: BinaryOp::Gt,
                    right: Box::new(Expr::Literal {
                        value: ScalarValue::Int(0),
                    }),
                },
            }),
            by: "age".to_owned(),
            descending: false,
        };
        let hint_plan = LogicalPlan::Hint {
            hint: ExecutionHint::TopK { n: 999 },
            input: Box::new(sort_plan.clone()),
        };

        let base = match execute(&sort_plan, graph.clone()).unwrap() {
            ExecutionValue::Nodes(n) => n,
            _ => panic!("expected nodes"),
        };
        let top = match execute(&hint_plan, graph).unwrap() {
            ExecutionValue::Nodes(n) => n,
            _ => panic!("expected nodes"),
        };

        assert_eq!(base.len(), top.len());
        for i in 0..base.len() {
            assert_eq!(base.id_column().value(i), top.id_column().value(i));
        }
    }

    /// `PartitionParallel` hint is a no-op at this stage ??the plan beneath
    /// it should execute normally.
    #[test]
    fn partition_parallel_hint_falls_through() {
        let graph = Arc::new(demo_graph());
        let hint_plan = LogicalPlan::Hint {
            hint: ExecutionHint::PartitionParallel {
                strategy: PartitionStrategy::ExpandFrontier,
            },
            input: Box::new(LogicalPlan::FilterNodes {
                input: Box::new(scan(graph.clone())),
                predicate: Expr::BinaryOp {
                    left: Box::new(Expr::Col {
                        name: COL_NODE_ID.to_owned(),
                    }),
                    op: BinaryOp::Eq,
                    right: Box::new(Expr::Literal {
                        value: ScalarValue::String("alice".to_owned()),
                    }),
                },
            }),
        };

        let result = execute(&hint_plan, graph).unwrap();
        let ExecutionValue::Nodes(nodes) = result else {
            panic!("expected node result");
        };
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes.id_column().value(0), "alice");
    }

    // ???? OPT-003: PartitionParallel executor tests ??????????????????????????????????????????????????????????

    /// `PartitionParallel` + `Expand` must produce the same result as a serial
    /// `Expand` ??correctness is the baseline requirement for any parallelism.
    #[test]
    fn partition_parallel_expand_matches_serial() {
        let graph = Arc::new(demo_graph());

        let serial_plan = LogicalPlan::Expand {
            input: Box::new(scan(graph.clone())),
            edge_type: EdgeTypeSpec::Any,
            hops: 2,
            direction: Direction::Out,
            pre_filter: None,
        };
        let parallel_plan = LogicalPlan::Hint {
            hint: ExecutionHint::PartitionParallel {
                strategy: PartitionStrategy::ExpandFrontier,
            },
            input: Box::new(LogicalPlan::Expand {
                input: Box::new(scan(graph.clone())),
                edge_type: EdgeTypeSpec::Any,
                hops: 2,
                direction: Direction::Out,
                pre_filter: None,
            }),
        };

        let serial_g = match execute(&serial_plan, graph.clone()).unwrap() {
            ExecutionValue::Graph(g) => g,
            _ => panic!("expected graph"),
        };
        let parallel_g = match execute(&parallel_plan, graph).unwrap() {
            ExecutionValue::Graph(g) => g,
            _ => panic!("expected graph"),
        };

        // Same set of node IDs.
        let mut serial_ids: Vec<&str> = serial_g.nodes().id_column().iter().flatten().collect();
        let mut parallel_ids: Vec<&str> = parallel_g.nodes().id_column().iter().flatten().collect();
        serial_ids.sort_unstable();
        parallel_ids.sort_unstable();
        assert_eq!(serial_ids, parallel_ids, "node sets must match");

        // Same edge count (parallel may deduplicate the same way serial does).
        assert_eq!(
            serial_g.edge_count(),
            parallel_g.edge_count(),
            "edge counts must match"
        );
    }

    /// When the frontier is tiny (below the 2 ??n_threads threshold) the
    /// parallel path falls back to serial ??the result must still be correct.
    #[test]
    fn partition_parallel_expand_serial_fallback_is_correct() {
        // Single-node frontier: always below the threshold.
        let graph = Arc::new(demo_graph());
        let alice_only = Box::new(LogicalPlan::FilterNodes {
            input: Box::new(scan(graph.clone())),
            predicate: Expr::BinaryOp {
                left: Box::new(Expr::Col {
                    name: COL_NODE_ID.to_owned(),
                }),
                op: BinaryOp::Eq,
                right: Box::new(Expr::Literal {
                    value: ScalarValue::String("alice".to_owned()),
                }),
            },
        });

        let plan = LogicalPlan::Hint {
            hint: ExecutionHint::PartitionParallel {
                strategy: PartitionStrategy::ExpandFrontier,
            },
            input: Box::new(LogicalPlan::Expand {
                input: alice_only,
                edge_type: EdgeTypeSpec::Any,
                hops: 2,
                direction: Direction::Out,
                pre_filter: None,
            }),
        };

        let result = execute(&plan, graph).unwrap();
        let ExecutionValue::Graph(g) = result else {
            panic!("expected graph result");
        };

        // alice?萸븄b?萸밹me and alice?萸뻞arlie; all 4 nodes reachable in 2 hops.
        assert_eq!(g.node_count(), 4);
        for id in ["alice", "bob", "charlie", "acme"] {
            assert!(
                g.nodes().row_index(id).is_some(),
                "{id} missing from result"
            );
        }
    }

    /// Expanding with `PartitionParallel` and a type filter must honour the
    /// edge-type filter ??same as serial.  We seed from alice only so that the
    /// filter is exercised: alice??bob,charlie} are KNOWS, bob?萸밹me is WORKS_AT.
    #[test]
    fn partition_parallel_expand_respects_edge_type_filter() {
        let graph = Arc::new(demo_graph());

        let alice_seed = || {
            Box::new(LogicalPlan::FilterNodes {
                input: Box::new(scan(graph.clone())),
                predicate: Expr::BinaryOp {
                    left: Box::new(Expr::Col {
                        name: COL_NODE_ID.to_owned(),
                    }),
                    op: BinaryOp::Eq,
                    right: Box::new(Expr::Literal {
                        value: ScalarValue::String("alice".to_owned()),
                    }),
                },
            })
        };

        let serial_plan = LogicalPlan::Expand {
            input: alice_seed(),
            edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
            hops: 2,
            direction: Direction::Out,
            pre_filter: None,
        };
        let parallel_plan = LogicalPlan::Hint {
            hint: ExecutionHint::PartitionParallel {
                strategy: PartitionStrategy::ExpandFrontier,
            },
            input: Box::new(LogicalPlan::Expand {
                input: alice_seed(),
                edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
                hops: 2,
                direction: Direction::Out,
                pre_filter: None,
            }),
        };

        let serial_g = match execute(&serial_plan, graph.clone()).unwrap() {
            ExecutionValue::Graph(g) => g,
            _ => panic!(),
        };
        let parallel_g = match execute(&parallel_plan, graph).unwrap() {
            ExecutionValue::Graph(g) => g,
            _ => panic!(),
        };

        // alice -KNOWS-> bob, charlie; bob -KNOWS-> none; acme unreachable via KNOWS.
        let mut s: Vec<&str> = serial_g.nodes().id_column().iter().flatten().collect();
        let mut p: Vec<&str> = parallel_g.nodes().id_column().iter().flatten().collect();
        s.sort_unstable();
        p.sort_unstable();
        assert_eq!(
            s, p,
            "edge-type-filtered results must match between serial and parallel"
        );
        assert!(s.contains(&"alice") && s.contains(&"bob") && s.contains(&"charlie"));
        assert!(
            !s.contains(&"acme"),
            "acme must not appear (no KNOWS edge from alice's neighbourhood)"
        );
    }

    fn assert_alias_value(row: &PatternBindingRow, alias: &str, value: Option<u32>) {
        assert_eq!(row.get(alias), Some(&value));
    }

    fn make_pattern(steps: Vec<PatternStep>) -> Pattern {
        Pattern::new(steps)
    }

    #[test]
    fn pattern_binding_row_accepts_fresh_aliases() {
        let mut row = PatternBindingRow::new();

        bind_pattern_alias(&mut row, "a", 0).unwrap();
        bind_pattern_alias(&mut row, "b", 3).unwrap();

        assert_alias_value(&row, "a", Some(0));
        assert_alias_value(&row, "b", Some(3));
    }

    #[test]
    fn pattern_binding_row_allows_same_alias_same_value() {
        let mut row = PatternBindingRow::new();

        bind_pattern_alias(&mut row, "a", 2).unwrap();
        bind_pattern_alias(&mut row, "a", 2).unwrap();

        assert_eq!(row.len(), 1);
        assert_alias_value(&row, "a", Some(2));
    }

    #[test]
    fn pattern_binding_row_rejects_conflicting_alias_rebind() {
        let mut row = PatternBindingRow::new();
        bind_pattern_alias(&mut row, "a", 1).unwrap();

        let err = bind_pattern_alias(&mut row, "a", 4).unwrap_err();

        assert!(
            matches!(err, GFError::InvalidConfig { message } if message.contains("pattern alias 'a'"))
        );
        assert_alias_value(&row, "a", Some(1));
    }

    #[test]
    fn pattern_bindings_is_vector_of_binding_rows() {
        let mut bindings: PatternBindings = Vec::new();
        let mut row = PatternBindingRow::new();
        bind_pattern_alias(&mut row, "seed", 7).unwrap();
        bindings.push(row);

        assert_eq!(bindings.len(), 1);
        assert_alias_value(&bindings[0], "seed", Some(7));
    }

    #[test]
    fn execute_pattern_step_builds_outbound_typed_bindings() {
        let graph = demo_graph();
        let step = PatternStep {
            from_alias: "a".to_owned(),
            edge_alias: Some("e".to_owned()),
            edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
            direction: Direction::Out,
            to_alias: "b".to_owned(),
        };

        let mut seed = PatternBindingRow::new();
        bind_pattern_alias(&mut seed, "a", 0).unwrap();
        let pattern = make_pattern(vec![step.clone()]);
        let bindings = execute_pattern_step(&graph, &pattern, 0, &step, &vec![seed]).unwrap();

        assert_eq!(bindings.len(), 2);
        assert_alias_value(&bindings[0], "a", Some(0));
        assert_alias_value(&bindings[0], "e", Some(0));
        assert_alias_value(&bindings[0], "b", Some(1));
        assert_alias_value(&bindings[1], "a", Some(0));
        assert_alias_value(&bindings[1], "e", Some(1));
        assert_alias_value(&bindings[1], "b", Some(2));
    }

    #[test]
    fn execute_pattern_step_supports_inbound_typed_bindings() {
        let graph = demo_graph();
        let step = PatternStep {
            from_alias: "c".to_owned(),
            edge_alias: Some("e".to_owned()),
            edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
            direction: Direction::In,
            to_alias: "a".to_owned(),
        };

        let mut seed = PatternBindingRow::new();
        bind_pattern_alias(&mut seed, "c", 2).unwrap();
        let pattern = make_pattern(vec![step.clone()]);
        let bindings = execute_pattern_step(&graph, &pattern, 0, &step, &vec![seed]).unwrap();

        assert_eq!(bindings.len(), 1);
        assert_alias_value(&bindings[0], "c", Some(2));
        assert_alias_value(&bindings[0], "e", Some(1));
        assert_alias_value(&bindings[0], "a", Some(0));
    }

    #[test]
    fn execute_pattern_step_drops_rows_on_alias_conflict() {
        let graph = demo_graph();
        let step = PatternStep {
            from_alias: "a".to_owned(),
            edge_alias: None,
            edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
            direction: Direction::Out,
            to_alias: "b".to_owned(),
        };

        let mut seed = PatternBindingRow::new();
        bind_pattern_alias(&mut seed, "a", 0).unwrap();
        bind_pattern_alias(&mut seed, "b", 9).unwrap();
        let pattern = make_pattern(vec![step.clone()]);
        let bindings = execute_pattern_step(&graph, &pattern, 0, &step, &vec![seed]).unwrap();

        assert!(bindings.is_empty());
    }

    #[test]
    fn execute_pattern_steps_chains_two_hops_across_aliases() {
        let graph = demo_graph();
        let steps = vec![
            PatternStep {
                from_alias: "a".to_owned(),
                edge_alias: Some("e1".to_owned()),
                edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
                direction: Direction::Out,
                to_alias: "b".to_owned(),
            },
            PatternStep {
                from_alias: "b".to_owned(),
                edge_alias: Some("e2".to_owned()),
                edge_type: EdgeTypeSpec::Single("WORKS_AT".to_owned()),
                direction: Direction::Out,
                to_alias: "c".to_owned(),
            },
        ];

        let mut seed = PatternBindingRow::new();
        bind_pattern_alias(&mut seed, "a", 0).unwrap();
        let pattern = make_pattern(steps);
        let bindings = execute_pattern_steps(&graph, &pattern, &vec![seed]).unwrap();

        assert_eq!(bindings.len(), 1);
        assert_alias_value(&bindings[0], "a", Some(0));
        assert_alias_value(&bindings[0], "b", Some(1));
        assert_alias_value(&bindings[0], "c", Some(3));
        assert_alias_value(&bindings[0], "e1", Some(0));
        assert_alias_value(&bindings[0], "e2", Some(2));
    }

    #[test]
    fn execute_pattern_steps_returns_empty_when_later_hop_has_no_match() {
        let graph = demo_graph();
        let steps = vec![
            PatternStep {
                from_alias: "a".to_owned(),
                edge_alias: None,
                edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
                direction: Direction::Out,
                to_alias: "b".to_owned(),
            },
            PatternStep {
                from_alias: "b".to_owned(),
                edge_alias: None,
                edge_type: EdgeTypeSpec::Single("WORKS_AT".to_owned()),
                direction: Direction::Out,
                to_alias: "c".to_owned(),
            },
            PatternStep {
                from_alias: "c".to_owned(),
                edge_alias: None,
                edge_type: EdgeTypeSpec::Single("WORKS_AT".to_owned()),
                direction: Direction::Out,
                to_alias: "d".to_owned(),
            },
        ];

        let mut seed = PatternBindingRow::new();
        bind_pattern_alias(&mut seed, "a", 0).unwrap();
        let pattern = make_pattern(steps);
        let bindings = execute_pattern_steps(&graph, &pattern, &vec![seed]).unwrap();

        assert!(bindings.is_empty());
    }

    #[test]
    fn execute_pattern_steps_preserves_aliases_from_previous_hops() {
        let graph = demo_graph();
        let steps = vec![
            PatternStep {
                from_alias: "a".to_owned(),
                edge_alias: None,
                edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
                direction: Direction::Out,
                to_alias: "b".to_owned(),
            },
            PatternStep {
                from_alias: "b".to_owned(),
                edge_alias: None,
                edge_type: EdgeTypeSpec::Single("WORKS_AT".to_owned()),
                direction: Direction::Out,
                to_alias: "c".to_owned(),
            },
        ];

        let mut seed = PatternBindingRow::new();
        bind_pattern_alias(&mut seed, "seed", 42).unwrap();
        bind_pattern_alias(&mut seed, "a", 0).unwrap();
        let pattern = make_pattern(steps);
        let bindings = execute_pattern_steps(&graph, &pattern, &vec![seed]).unwrap();

        assert_eq!(bindings.len(), 1);
        assert_alias_value(&bindings[0], "seed", Some(42));
        assert_alias_value(&bindings[0], "a", Some(0));
        assert_alias_value(&bindings[0], "b", Some(1));
        assert_alias_value(&bindings[0], "c", Some(3));
    }

    #[test]
    fn execute_pattern_steps_supports_exact_multi_hop_constraints() {
        let graph = demo_graph();
        let pattern = Pattern::with_constraints(
            vec![PatternStep {
                from_alias: "a".to_owned(),
                edge_alias: None,
                edge_type: EdgeTypeSpec::Any,
                direction: Direction::Out,
                to_alias: "c".to_owned(),
            }],
            std::collections::BTreeMap::new(),
            vec![lynxes_core::PatternStepConstraint {
                optional: false,
                min_hops: 2,
                max_hops: 2,
            }],
        )
        .unwrap();

        let mut seed = PatternBindingRow::new();
        bind_pattern_alias(&mut seed, "a", 0).unwrap();
        let bindings = execute_pattern_steps(&graph, &pattern, &vec![seed]).unwrap();

        assert_eq!(bindings.len(), 1);
        assert_alias_value(&bindings[0], "a", Some(0));
        assert_alias_value(&bindings[0], "c", Some(3));
    }

    #[test]
    fn execute_pattern_steps_respects_node_label_constraints() {
        let graph = demo_graph();
        let pattern = Pattern::with_constraints(
            vec![PatternStep {
                from_alias: "a".to_owned(),
                edge_alias: None,
                edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
                direction: Direction::Out,
                to_alias: "b".to_owned(),
            }],
            std::collections::BTreeMap::from([(
                "b".to_owned(),
                lynxes_core::PatternNodeConstraint {
                    label: Some("Company".to_owned()),
                },
            )]),
            vec![lynxes_core::PatternStepConstraint::default()],
        )
        .unwrap();

        let mut seed = PatternBindingRow::new();
        bind_pattern_alias(&mut seed, "a", 0).unwrap();
        let bindings = execute_pattern_steps(&graph, &pattern, &vec![seed]).unwrap();

        assert!(bindings.is_empty());
    }

    #[test]
    fn execute_pattern_steps_emit_nulls_for_optional_unmatched_hops() {
        let graph = demo_graph();
        let pattern = Pattern::with_constraints(
            vec![PatternStep {
                from_alias: "a".to_owned(),
                edge_alias: Some("e".to_owned()),
                edge_type: EdgeTypeSpec::Single("WORKS_AT".to_owned()),
                direction: Direction::Out,
                to_alias: "b".to_owned(),
            }],
            std::collections::BTreeMap::new(),
            vec![lynxes_core::PatternStepConstraint {
                optional: true,
                min_hops: 1,
                max_hops: 1,
            }],
        )
        .unwrap();

        let mut seed = PatternBindingRow::new();
        bind_pattern_alias(&mut seed, "a", 2).unwrap();
        let bindings = execute_pattern_steps(&graph, &pattern, &vec![seed]).unwrap();

        assert_eq!(bindings.len(), 1);
        assert_alias_value(&bindings[0], "a", Some(2));
        assert_alias_value(&bindings[0], "b", None);
        assert_alias_value(&bindings[0], "e", None);
    }

    #[test]
    fn apply_pattern_where_filters_bindings_by_node_alias_field() {
        let graph = demo_graph();
        let steps = vec![PatternStep {
            from_alias: "a".to_owned(),
            edge_alias: None,
            edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
            direction: Direction::Out,
            to_alias: "b".to_owned(),
        }];

        let mut seed = PatternBindingRow::new();
        bind_pattern_alias(&mut seed, "a", 0).unwrap();
        let pattern = make_pattern(steps);
        let bindings = execute_pattern_steps(&graph, &pattern, &vec![seed]).unwrap();
        let predicate = Expr::BinaryOp {
            left: Box::new(Expr::PatternCol {
                alias: "b".to_owned(),
                field: "age".to_owned(),
            }),
            op: BinaryOp::Gt,
            right: Box::new(Expr::Literal {
                value: ScalarValue::Int(30),
            }),
        };

        let filtered = apply_pattern_where(&graph, &bindings, Some(&predicate)).unwrap();

        assert_eq!(filtered.len(), 1);
        assert_alias_value(&filtered[0], "a", Some(0));
        assert_alias_value(&filtered[0], "b", Some(1));
    }

    #[test]
    fn apply_pattern_where_filters_bindings_by_edge_alias_field() {
        let graph = demo_graph();
        let steps = vec![
            PatternStep {
                from_alias: "a".to_owned(),
                edge_alias: Some("e1".to_owned()),
                edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
                direction: Direction::Out,
                to_alias: "b".to_owned(),
            },
            PatternStep {
                from_alias: "b".to_owned(),
                edge_alias: Some("e2".to_owned()),
                edge_type: EdgeTypeSpec::Single("WORKS_AT".to_owned()),
                direction: Direction::Out,
                to_alias: "c".to_owned(),
            },
        ];

        let mut seed = PatternBindingRow::new();
        bind_pattern_alias(&mut seed, "a", 0).unwrap();
        let pattern = make_pattern(steps);
        let bindings = execute_pattern_steps(&graph, &pattern, &vec![seed]).unwrap();
        let predicate = Expr::BinaryOp {
            left: Box::new(Expr::PatternCol {
                alias: "e2".to_owned(),
                field: COL_EDGE_TYPE.to_owned(),
            }),
            op: BinaryOp::Eq,
            right: Box::new(Expr::Literal {
                value: ScalarValue::String("WORKS_AT".to_owned()),
            }),
        };

        let filtered = apply_pattern_where(&graph, &bindings, Some(&predicate)).unwrap();

        assert_eq!(filtered.len(), 1);
        assert_alias_value(&filtered[0], "e2", Some(2));
        assert_alias_value(&filtered[0], "c", Some(3));
    }

    #[test]
    fn apply_pattern_where_returns_all_bindings_when_predicate_is_none() {
        let graph = demo_graph();
        let steps = vec![PatternStep {
            from_alias: "a".to_owned(),
            edge_alias: None,
            edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
            direction: Direction::Out,
            to_alias: "b".to_owned(),
        }];

        let mut seed = PatternBindingRow::new();
        bind_pattern_alias(&mut seed, "a", 0).unwrap();
        let pattern = make_pattern(steps);
        let bindings = execute_pattern_steps(&graph, &pattern, &vec![seed]).unwrap();

        let filtered = apply_pattern_where(&graph, &bindings, None).unwrap();

        assert_eq!(filtered, bindings);
    }

    #[test]
    fn materialize_pattern_bindings_emits_alias_prefixed_columns_in_pattern_order() {
        let graph = demo_graph();
        let pattern = vec![
            PatternStep {
                from_alias: "a".to_owned(),
                edge_alias: Some("e1".to_owned()),
                edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
                direction: Direction::Out,
                to_alias: "b".to_owned(),
            },
            PatternStep {
                from_alias: "b".to_owned(),
                edge_alias: Some("e2".to_owned()),
                edge_type: EdgeTypeSpec::Single("WORKS_AT".to_owned()),
                direction: Direction::Out,
                to_alias: "c".to_owned(),
            },
        ];

        let mut seed = PatternBindingRow::new();
        bind_pattern_alias(&mut seed, "a", 0).unwrap();
        let pattern = make_pattern(pattern);
        let bindings = execute_pattern_steps(&graph, &pattern, &vec![seed]).unwrap();
        let batch = materialize_pattern_bindings(&graph, &pattern.steps, &bindings).unwrap();

        let schema = batch.schema();
        let column_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(
            column_names,
            vec![
                "a._id",
                "a._label",
                "a.age",
                "e1._src",
                "e1._dst",
                "e1._type",
                "e1._direction",
                "e1.weight",
                "b._id",
                "b._label",
                "b.age",
                "e2._src",
                "e2._dst",
                "e2._type",
                "e2._direction",
                "e2.weight",
                "c._id",
                "c._label",
                "c.age",
            ]
        );
        assert_eq!(batch.num_rows(), 1);

        let a_id = batch
            .column_by_name("a._id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let b_id = batch
            .column_by_name("b._id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let c_id = batch
            .column_by_name("c._id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let e1_type = batch
            .column_by_name("e1._type")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let e2_weight = batch
            .column_by_name("e2.weight")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        assert_eq!(a_id.value(0), "alice");
        assert_eq!(b_id.value(0), "bob");
        assert_eq!(c_id.value(0), "acme");
        assert_eq!(e1_type.value(0), "KNOWS");
        assert_eq!(e2_weight.value(0), 3);
    }

    #[test]
    fn materialize_pattern_bindings_supports_empty_result_with_full_schema() {
        let graph = demo_graph();
        let pattern = vec![
            PatternStep {
                from_alias: "a".to_owned(),
                edge_alias: Some("e1".to_owned()),
                edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
                direction: Direction::Out,
                to_alias: "b".to_owned(),
            },
            PatternStep {
                from_alias: "b".to_owned(),
                edge_alias: Some("e2".to_owned()),
                edge_type: EdgeTypeSpec::Single("WORKS_AT".to_owned()),
                direction: Direction::Out,
                to_alias: "c".to_owned(),
            },
        ];

        let empty: PatternBindings = Vec::new();
        let pattern = make_pattern(pattern);
        let batch = materialize_pattern_bindings(&graph, &pattern.steps, &empty).unwrap();

        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.num_columns(), 19);
        assert_eq!(batch.schema().field(0).name(), "a._id");
        assert_eq!(batch.schema().field(18).name(), "c.age");
    }

    #[test]
    fn materialize_pattern_bindings_rejects_alias_kind_conflicts() {
        let graph = demo_graph();
        let pattern = vec![PatternStep {
            from_alias: "x".to_owned(),
            edge_alias: Some("x".to_owned()),
            edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
            direction: Direction::Out,
            to_alias: "b".to_owned(),
        }];

        let empty: PatternBindings = Vec::new();
        let pattern = make_pattern(pattern);
        let err = materialize_pattern_bindings(&graph, &pattern.steps, &empty).unwrap_err();

        assert!(
            matches!(err, GFError::InvalidConfig { message } if message.contains("used as both"))
        );
    }

    #[test]
    fn pattern_match_executes_collect_path_for_two_hop_pattern() {
        let graph = Arc::new(demo_graph());
        let plan = LogicalPlan::PatternMatch {
            input: Box::new(LogicalPlan::FilterNodes {
                input: Box::new(scan(graph.clone())),
                predicate: Expr::BinaryOp {
                    left: Box::new(Expr::Col {
                        name: COL_NODE_ID.to_owned(),
                    }),
                    op: BinaryOp::Eq,
                    right: Box::new(Expr::Literal {
                        value: ScalarValue::String("alice".to_owned()),
                    }),
                },
            }),
            pattern: Pattern::new(vec![
                PatternStep {
                    from_alias: "a".to_owned(),
                    edge_alias: Some("e1".to_owned()),
                    edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
                    direction: Direction::Out,
                    to_alias: "b".to_owned(),
                },
                PatternStep {
                    from_alias: "b".to_owned(),
                    edge_alias: Some("e2".to_owned()),
                    edge_type: EdgeTypeSpec::Single("WORKS_AT".to_owned()),
                    direction: Direction::Out,
                    to_alias: "c".to_owned(),
                },
            ]),
            where_: None,
        };

        let result = execute(&plan, graph).unwrap();
        let ExecutionValue::PatternRows(batch) = result else {
            panic!("expected pattern-row result");
        };

        assert_eq!(batch.num_rows(), 1);
        let schema = batch.schema();
        assert!(schema.column_with_name("a._id").is_some());
        assert!(schema.column_with_name("b._id").is_some());
        assert!(schema.column_with_name("c._id").is_some());
        assert!(schema.column_with_name("e1._type").is_some());
        assert!(schema.column_with_name("e2.weight").is_some());

        let a_ids = string_array(&batch, "a._id").unwrap();
        let b_ids = string_array(&batch, "b._id").unwrap();
        let c_ids = string_array(&batch, "c._id").unwrap();
        let e1_types = string_array(&batch, "e1._type").unwrap();
        let e2_weight = batch
            .column_by_name("e2.weight")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        assert_eq!(a_ids.value(0), "alice");
        assert_eq!(b_ids.value(0), "bob");
        assert_eq!(c_ids.value(0), "acme");
        assert_eq!(e1_types.value(0), "KNOWS");
        assert_eq!(e2_weight.value(0), 3);
    }

    #[test]
    fn pattern_match_executes_with_where_filter() {
        let graph = Arc::new(demo_graph());
        let plan = LogicalPlan::PatternMatch {
            input: Box::new(LogicalPlan::FilterNodes {
                input: Box::new(scan(graph.clone())),
                predicate: Expr::BinaryOp {
                    left: Box::new(Expr::Col {
                        name: COL_NODE_ID.to_owned(),
                    }),
                    op: BinaryOp::Eq,
                    right: Box::new(Expr::Literal {
                        value: ScalarValue::String("alice".to_owned()),
                    }),
                },
            }),
            pattern: Pattern::new(vec![PatternStep {
                from_alias: "a".to_owned(),
                edge_alias: Some("e".to_owned()),
                edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
                direction: Direction::Out,
                to_alias: "b".to_owned(),
            }]),
            where_: Some(Expr::BinaryOp {
                left: Box::new(Expr::PatternCol {
                    alias: "b".to_owned(),
                    field: "age".to_owned(),
                }),
                op: BinaryOp::Gt,
                right: Box::new(Expr::Literal {
                    value: ScalarValue::Int(30),
                }),
            }),
        };

        let result = execute(&plan, graph).unwrap();
        let ExecutionValue::PatternRows(batch) = result else {
            panic!("expected pattern-row result");
        };

        assert_eq!(batch.num_rows(), 1);
        let b_ids = string_array(&batch, "b._id").unwrap();
        assert_eq!(b_ids.value(0), "bob");
    }

    #[test]
    fn kg_typed_one_step_pattern_executes() {
        let graph = Arc::new(demo_graph());
        let plan = LogicalPlan::PatternMatch {
            input: Box::new(LogicalPlan::FilterNodes {
                input: Box::new(scan(graph.clone())),
                predicate: Expr::BinaryOp {
                    left: Box::new(Expr::Col {
                        name: COL_NODE_ID.to_owned(),
                    }),
                    op: BinaryOp::Eq,
                    right: Box::new(Expr::Literal {
                        value: ScalarValue::String("alice".to_owned()),
                    }),
                },
            }),
            pattern: Pattern::new(vec![PatternStep {
                from_alias: "a".to_owned(),
                edge_alias: Some("e".to_owned()),
                edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
                direction: Direction::Out,
                to_alias: "b".to_owned(),
            }]),
            where_: None,
        };

        let result = execute(&plan, graph).unwrap();
        let ExecutionValue::PatternRows(batch) = result else {
            panic!("expected pattern-row result");
        };

        assert_eq!(batch.num_rows(), 2);
        let a_ids = string_array(&batch, "a._id").unwrap();
        let b_ids = string_array(&batch, "b._id").unwrap();
        let e_types = string_array(&batch, "e._type").unwrap();
        assert_eq!(a_ids.value(0), "alice");
        assert_eq!(a_ids.value(1), "alice");
        assert_eq!(b_ids.value(0), "bob");
        assert_eq!(b_ids.value(1), "charlie");
        assert_eq!(e_types.value(0), "KNOWS");
        assert_eq!(e_types.value(1), "KNOWS");
    }

    #[test]
    fn kg_two_hop_multi_step_pattern_executes() {
        let graph = Arc::new(demo_graph());
        let plan = LogicalPlan::PatternMatch {
            input: Box::new(LogicalPlan::FilterNodes {
                input: Box::new(scan(graph.clone())),
                predicate: Expr::BinaryOp {
                    left: Box::new(Expr::Col {
                        name: COL_NODE_ID.to_owned(),
                    }),
                    op: BinaryOp::Eq,
                    right: Box::new(Expr::Literal {
                        value: ScalarValue::String("alice".to_owned()),
                    }),
                },
            }),
            pattern: Pattern::new(vec![
                PatternStep {
                    from_alias: "a".to_owned(),
                    edge_alias: Some("e1".to_owned()),
                    edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
                    direction: Direction::Out,
                    to_alias: "b".to_owned(),
                },
                PatternStep {
                    from_alias: "b".to_owned(),
                    edge_alias: Some("e2".to_owned()),
                    edge_type: EdgeTypeSpec::Single("WORKS_AT".to_owned()),
                    direction: Direction::Out,
                    to_alias: "c".to_owned(),
                },
            ]),
            where_: None,
        };

        let result = execute(&plan, graph).unwrap();
        let ExecutionValue::PatternRows(batch) = result else {
            panic!("expected pattern-row result");
        };

        assert_eq!(batch.num_rows(), 1);
        assert_eq!(string_array(&batch, "a._id").unwrap().value(0), "alice");
        assert_eq!(string_array(&batch, "b._id").unwrap().value(0), "bob");
        assert_eq!(string_array(&batch, "c._id").unwrap().value(0), "acme");
    }

    #[test]
    fn kg_pattern_expansion_pushdown_preserves_result_set() {
        let graph = Arc::new(demo_graph());
        let plan = LogicalPlan::PatternMatch {
            input: Box::new(scan(graph.clone())),
            pattern: Pattern::new(vec![PatternStep {
                from_alias: "a".to_owned(),
                edge_alias: Some("e".to_owned()),
                edge_type: EdgeTypeSpec::Single("KNOWS".to_owned()),
                direction: Direction::Out,
                to_alias: "b".to_owned(),
            }]),
            where_: Some(Expr::And {
                left: Box::new(Expr::BinaryOp {
                    left: Box::new(Expr::PatternCol {
                        alias: "a".to_owned(),
                        field: "age".to_owned(),
                    }),
                    op: BinaryOp::Gt,
                    right: Box::new(Expr::Literal {
                        value: ScalarValue::Int(25),
                    }),
                }),
                right: Box::new(Expr::BinaryOp {
                    left: Box::new(Expr::PatternCol {
                        alias: "b".to_owned(),
                        field: "age".to_owned(),
                    }),
                    op: BinaryOp::Gt,
                    right: Box::new(Expr::Literal {
                        value: ScalarValue::Int(30),
                    }),
                }),
            }),
        };

        let baseline_plan = Optimizer::new(OptimizerOptions {
            pattern_expansion: false,
            ..OptimizerOptions::default()
        })
        .run(plan.clone());
        let optimized_plan = Optimizer::default().run(plan);

        let baseline = execute(&baseline_plan, graph.clone()).unwrap();
        let optimized = execute(&optimized_plan, graph).unwrap();

        let ExecutionValue::PatternRows(baseline_batch) = baseline else {
            panic!("expected pattern-row result");
        };
        let ExecutionValue::PatternRows(optimized_batch) = optimized else {
            panic!("expected pattern-row result");
        };

        assert_eq!(baseline_batch.schema(), optimized_batch.schema());
        assert_eq!(baseline_batch.num_rows(), optimized_batch.num_rows());
        assert_eq!(baseline_batch.num_columns(), optimized_batch.num_columns());
        for idx in 0..baseline_batch.num_columns() {
            assert_eq!(
                format!("{:?}", baseline_batch.column(idx)),
                format!("{:?}", optimized_batch.column(idx))
            );
        }
    }
}
