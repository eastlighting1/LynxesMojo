import sys
from pathlib import Path

import pytest

import lynxes as lx

MOJO_LIB = Path(lx.__file__).parent / ".libs" / "liblynxes_mojo_kernels.so"


@pytest.mark.skipif(
    sys.platform != "linux" or not MOJO_LIB.exists(), reason="requires bundled Mojo .so"
)
def test_structural_features_computes_degrees_by_edge_type():
    graph = lx.graph(
        nodes={
            "_id": ["alice", "bob", "carol", "solo"],
            "_label": [["Person"], ["Person"], ["Person"], ["Person"]],
        },
        edges={
            "_src": ["alice", "alice", "bob", "carol"],
            "_dst": ["bob", "carol", "alice", "carol"],
            "_type": ["KNOWS", "LIKES", "KNOWS", "KNOWS"],
            "_direction": [0, 0, 0, 0],
        },
    )

    features = graph.structural_features("KNOWS")

    assert features.column_names()[-3:] == ["out_degree", "in_degree", "total_degree"]
    assert features.column_values("_id") == ["alice", "bob", "carol", "solo"]
    assert features.column_values("out_degree") == [1, 1, 1, 0]
    assert features.column_values("in_degree") == [1, 1, 1, 0]
    assert features.column_values("total_degree") == [2, 2, 2, 0]


@pytest.mark.skipif(
    sys.platform != "linux" or not MOJO_LIB.exists(), reason="requires bundled Mojo .so"
)
def test_aggregate_neighbors_count_uses_mojo_degree_kernel():
    graph = lx.graph(
        nodes={
            "_id": ["alice", "bob", "carol"],
            "_label": [["Person"], ["Person"], ["Person"]],
        },
        edges={
            "_src": ["alice", "alice", "bob"],
            "_dst": ["bob", "carol", "carol"],
            "_type": ["KNOWS", "KNOWS", "LIKES"],
            "_direction": [0, 0, 0],
        },
    )

    nodes = graph.lazy().aggregate_neighbors("KNOWS", lx.count()).collect_nodes()

    assert nodes.column_values("count") == [2, 0, 0]


@pytest.mark.skipif(
    sys.platform == "linux" and MOJO_LIB.exists(), reason="Mojo runtime is available"
)
def test_structural_features_requires_mojo_runtime():
    graph = lx.graph(
        nodes={"_id": ["solo"], "_label": [["Person"]]},
        edges={"_src": [], "_dst": [], "_type": [], "_direction": []},
    )
    if not hasattr(graph, "structural_features"):
        pytest.skip("requires rebuilt native extension with Mojo API")

    with pytest.raises(Exception, match="Mojo"):
        graph.structural_features()
