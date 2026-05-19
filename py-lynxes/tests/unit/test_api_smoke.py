import pytest

import lynxes as gf


class TestReadGf:
    def test_node_count(self, graph):
        assert graph.node_count() == 5

    def test_edge_count(self, graph):
        assert graph.edge_count() == 4

    def test_missing_file_raises_os_error(self):
        with pytest.raises(OSError):
            gf.read_gf("/nonexistent/path/that/does/not/exist.gf")

    def test_returns_graph_frame_type(self, graph):
        assert type(graph).__name__ == "GraphFrame"
