import pytest

import lynxes as gf


class TestErrors:
    def test_read_gf_missing_file_raises_os_error(self):
        with pytest.raises(OSError):
            gf.read_gf("/this/does/not/exist.gf")

    def test_read_gfb_missing_file_raises_os_error(self):
        with pytest.raises(OSError):
            gf.read_gfb("/this/does/not/exist.gfb")

    def test_shortest_path_missing_node_raises(self, graph):
        with pytest.raises((KeyError, RuntimeError, ValueError)):
            graph.shortest_path("alice", "does_not_exist")
