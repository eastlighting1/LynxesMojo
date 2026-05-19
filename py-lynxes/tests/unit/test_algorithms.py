class TestAlgorithms:
    def test_pagerank_returns_node_frame(self, graph):
        result = graph.pagerank()
        assert type(result).__name__ == "NodeFrame"

    def test_pagerank_has_pagerank_column(self, graph):
        result = graph.pagerank()
        assert "pagerank" in result.column_names()

    def test_pagerank_count_matches_graph(self, graph):
        result = graph.pagerank()
        assert result.len() == graph.node_count()

    def test_shortest_path_alice_to_charlie(self, graph):
        path = graph.shortest_path("alice", "charlie")
        assert isinstance(path, list)
        assert path[0] == "alice"
        assert path[-1] == "charlie"
        assert len(path) == 3

    def test_shortest_path_to_self(self, graph):
        path = graph.shortest_path("alice", "alice")
        assert path == ["alice"]

    def test_connected_components_returns_node_frame(self, graph):
        result = graph.connected_components()
        assert type(result).__name__ == "NodeFrame"

    def test_connected_components_has_column(self, graph):
        result = graph.connected_components()
        assert "component_id" in result.column_names()

    def test_community_detection_column_values_returns_python_ints(self, graph):
        result = graph.community_detection()
        community_ids = result.column_values("community_id")

        assert len(community_ids) == graph.node_count()
        assert all(isinstance(value, int) for value in community_ids)
