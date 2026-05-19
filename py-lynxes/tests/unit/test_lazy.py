import pytest

import lynxes as gf


class TestFilterNodes:
    def test_age_gt_filter(self, graph):
        nf = graph.lazy().filter_nodes(gf.col("age") > 25).collect_nodes()
        assert nf.len() == 4

    def test_age_eq_filter(self, graph):
        nf = graph.lazy().filter_nodes(gf.col("age") == 30).collect_nodes()
        assert nf.len() == 1

    def test_chained_filter(self, graph):
        nf = (
            graph.lazy()
            .filter_nodes(gf.col("age") > 25)
            .filter_nodes(gf.col("age") < 35)
            .collect_nodes()
        )
        assert nf.len() == 2

    def test_filter_yields_empty_when_no_match(self, graph):
        nf = graph.lazy().filter_nodes(gf.col("age") > 9999).collect_nodes()
        assert nf.is_empty()

    def test_label_contains_filter(self, graph):
        nf = graph.lazy().filter_nodes(gf.col("_label").contains("Person")).collect_nodes()
        assert nf.len() == 4

    def test_label_contains_company(self, graph):
        nf = graph.lazy().filter_nodes(gf.col("_label").contains("Company")).collect_nodes()
        assert nf.len() == 1


class TestExpand:
    def test_one_hop_out_from_alice(self, graph):
        result = (
            graph.lazy()
            .filter_nodes(gf.col("_id") == "alice")
            .expand(hops=1, direction="out")
            .collect()
        )
        assert result.node_count() >= 2

    def test_two_hops_reaches_charlie(self, graph):
        result = (
            graph.lazy()
            .filter_nodes(gf.col("_id") == "alice")
            .expand(hops=2, direction="out")
            .collect()
        )
        node_ids = set(result.nodes().ids())
        assert "charlie" in node_ids

    def test_expand_with_edge_type_filter(self, graph):
        result = (
            graph.lazy()
            .filter_nodes(gf.col("_id") == "alice")
            .expand(edge_type="KNOWS", hops=2, direction="out")
            .collect()
        )
        node_ids = set(result.nodes().ids())
        assert "acme" not in node_ids

    def test_expand_in_direction(self, graph):
        result = (
            graph.lazy()
            .filter_nodes(gf.col("_id") == "charlie")
            .expand(hops=1, direction="in")
            .collect()
        )
        node_ids = set(result.nodes().ids())
        assert "bob" in node_ids


class TestCollectionDomains:
    def test_collect_on_node_domain_raises_value_error(self, graph):
        lazy = graph.lazy().filter_nodes(gf.col("_id") == "alice")

        with pytest.raises(ValueError, match="graph-domain plan"):
            lazy.collect()

    def test_collect_edges_on_node_domain_raises_value_error(self, graph):
        lazy = graph.lazy().filter_nodes(gf.col("_id") == "alice")

        with pytest.raises(ValueError, match="node-domain or pattern-row plan"):
            lazy.collect_edges()


class TestAggExprAlias:
    def test_count_alias_changes_column_name(self, graph):
        result = (
            graph.lazy()
            .aggregate_neighbors("KNOWS", gf.count().alias("friend_count"))
            .collect_nodes()
        )
        cols = result.column_names()
        assert "friend_count" in cols
        assert "count" not in cols

    def test_alias_preserves_values(self, graph):
        result = (
            graph.lazy()
            .filter_nodes(gf.col("_id") == "alice")
            .aggregate_neighbors("KNOWS", gf.count().alias("n_friends"))
            .collect_nodes()
        )
        rb = result.to_pyarrow()
        col = rb.column("n_friends")
        assert col[0].as_py() == 2

    def test_sum_alias(self, graph):
        result = (
            graph.lazy()
            .filter_nodes(gf.col("_id") == "alice")
            .aggregate_neighbors("KNOWS", gf.count().alias("c"))
            .collect_nodes()
        )
        assert "c" in result.column_names()


class TestMatchPattern:
    def test_match_pattern_returns_lazy(self, graph):
        lazy = graph.lazy().match_pattern(
            [
                gf.node("a", "Person"),
                gf.edge("KNOWS"),
                gf.node("b", "Person"),
            ]
        )
        assert type(lazy).__name__ == "LazyGraphFrame"

    def test_match_pattern_explain_contains_pattern_match(self, graph):
        lazy = graph.lazy().match_pattern(
            [
                gf.node("a"),
                gf.edge(),
                gf.node("b"),
            ]
        )
        assert "PatternMatch" in lazy.explain()

    def test_match_pattern_collect_returns_record_batch(self, graph):
        import pyarrow as pa

        result = (
            graph.lazy()
            .match_pattern(
                [
                    gf.node("a", "Person"),
                    gf.edge("KNOWS"),
                    gf.node("b", "Person"),
                ]
            )
            .collect()
        )

        assert isinstance(result, pa.RecordBatch)
        assert "a._id" in result.schema.names
        assert "b._id" in result.schema.names

    def test_match_pattern_edge_alias_materializes_edge_columns(self, graph):
        result = (
            graph.lazy()
            .match_pattern(
                [
                    gf.node("a", "Person"),
                    gf.edge("KNOWS", alias="e"),
                    gf.node("b", "Person"),
                ]
            )
            .collect()
        )

        assert "e._type" in result.schema.names
        assert set(result.column("e._type").to_pylist()) == {"KNOWS"}

    def test_match_pattern_invalid_steps_raises(self, graph):
        with pytest.raises((TypeError, ValueError)):
            graph.lazy().match_pattern([gf.node("a"), gf.node("b")])

    def test_match_pattern_rejects_unimplemented_node_props(self, graph):
        with pytest.raises(NotImplementedError):
            graph.lazy().match_pattern(
                [
                    gf.node("a", props=["age"]),
                    gf.edge("KNOWS"),
                    gf.node("b"),
                ]
            )

    def test_match_pattern_with_where_clause(self, graph):
        lazy = graph.lazy().match_pattern(
            [
                gf.node("a", "Person"),
                gf.edge("KNOWS"),
                gf.node("b", "Person"),
            ],
            where_=gf.col("a.age") > 25,
        )
        assert "PatternMatch" in lazy.explain()

    def test_match_pattern_collect_with_where_clause_filters_rows(self, graph):
        result = (
            graph.lazy()
            .match_pattern(
                [
                    gf.node("a", "Person"),
                    gf.edge("KNOWS"),
                    gf.node("b", "Person"),
                ],
                where_=gf.col("a.age") > 25,
            )
            .collect()
        )

        a_ids = result.column("a._id").to_pylist()
        assert a_ids == ["alice", "alice"]

    def test_match_pattern_respects_to_node_label_constraints(self, graph):
        result = (
            graph.lazy()
            .match_pattern(
                [
                    gf.node("a", "Person"),
                    gf.edge("KNOWS"),
                    gf.node("b", "Company"),
                ]
            )
            .collect()
        )

        assert result.num_rows == 0

    def test_match_pattern_supports_exact_multi_hop_steps(self, graph):
        result = (
            graph.lazy()
            .filter_nodes(gf.col("_id") == "alice")
            .match_pattern(
                [
                    gf.node("a", "Person"),
                    gf.edge(min_hops=2),
                    gf.node("c", "Company"),
                ]
            )
            .collect()
        )

        assert result.column("a._id").to_pylist() == ["alice"]
        assert result.column("c._id").to_pylist() == ["acme"]

    def test_match_pattern_optional_step_materializes_null_aliases(self, graph):
        result = (
            graph.lazy()
            .filter_nodes(gf.col("_id") == "acme")
            .match_pattern(
                [
                    gf.node("a", "Company"),
                    gf.edge("KNOWS", alias="e", optional=True),
                    gf.node("b"),
                ]
            )
            .collect()
        )

        assert result.column("a._id").to_pylist() == ["acme"]
        assert result.column("b._id").to_pylist() == [None]
        assert result.column("e._type").to_pylist() == [None]
