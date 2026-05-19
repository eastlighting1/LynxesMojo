class TestDisplay:
    def test_graph_repr_contains_summary_and_rows(self, graph):
        rendered = repr(graph)
        assert "GraphFrame(rows=" in rendered
        assert "src" in rendered
        assert "alice" in rendered

    def test_head_renders_requested_slice(self, graph):
        rendered = graph.head(2, attrs=["age"])
        assert "age" in rendered
        assert "alice" in rendered

    def test_info_mentions_graph_stats(self, graph):
        rendered = graph.info()
        assert "Graph info" in rendered
        assert "self loops" in rendered
        assert "Node attrs" in rendered

    def test_schema_mentions_reserved_columns(self, graph):
        rendered = graph.schema()
        assert "Schema (" in rendered
        assert "_id" in rendered
        assert "_src" in rendered

    def test_glimpse_and_describe_structure_render(self, graph):
        glimpse = graph.glimpse(2)
        describe = graph.describe("structure")
        assert "Glimpse" in glimpse
        assert "rows sampled" in glimpse
        assert "Structure" in describe
        assert "connected components" in describe

    def test_describe_attrs_renders_stats(self, graph):
        rendered = graph.describe("attrs")
        assert "Attributes" in rendered
        assert "distinct=" in rendered
        assert "node.age" in rendered

    def test_nodeframe_display_helpers_render_algorithm_output(self, graph):
        ranks = graph.pagerank()

        assert "NodeFrame(rows=" in repr(ranks)
        assert "pagerank" in ranks.head(3, sort_by="pagerank", descending=True)
        assert "NodeFrame info" in ranks.info()
        assert "NodeFrame schema" in ranks.schema()
        assert "NodeFrame glimpse" in ranks.glimpse(2)
        assert "Attributes" in ranks.describe("attrs")

    def test_edgeframe_display_helpers_render_preview(self, graph):
        edges = graph.edges()

        assert "EdgeFrame(rows=" in repr(edges)
        assert "_src" in edges.head(2)
        assert "EdgeFrame info" in edges.info()
        assert "EdgeFrame schema" in edges.schema()
        assert "EdgeFrame glimpse" in edges.glimpse(2)
        assert "Structure" in edges.describe("structure")
