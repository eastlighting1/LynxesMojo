import pytest


def test_to_coo(graph):
    """Test to_coo and torch.frombuffer zero-copy pipeline."""
    src, dst = graph.to_coo()

    assert len(src) > 0
    assert len(dst) == len(src)

    # Verify we can zero-copy into torch if available
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    src_buf = src.buffers()[1]
    dst_buf = dst.buffers()[1]

    # Use torch.frombuffer to create tensor without copying
    src_tensor = torch.frombuffer(src_buf, dtype=torch.int64)
    dst_tensor = torch.frombuffer(dst_buf, dtype=torch.int64)

    assert src_tensor.shape[0] == len(src)
    assert dst_tensor.shape[0] == len(dst)

    # Basic data integrity check
    assert int(src_tensor[0]) == src[0].as_py()
    assert int(dst_tensor[0]) == dst[0].as_py()


def test_sample_neighbors(graph):
    """Test sample_neighbors result shape and content."""
    # Seed nodes from the example graph
    seed_nodes = ["alice", "bob"]
    hops = 2
    fan_out = [2, 2]

    subgraph = graph.sample_neighbors(seed_nodes=seed_nodes, hops=hops, fan_out=fan_out)

    assert len(subgraph.node_indices) >= len(seed_nodes)
    assert len(subgraph.node_row_ids) == len(subgraph.node_indices)

    assert len(subgraph.edge_src) == len(subgraph.edge_dst)
    assert len(subgraph.edge_row_ids) == len(subgraph.edge_src)

    # Subgraph edges should be valid indices into the node_indices array
    num_nodes = len(subgraph.node_indices)
    for src_idx, dst_idx in zip(subgraph.edge_src, subgraph.edge_dst):
        assert src_idx < num_nodes
        assert dst_idx < num_nodes


def test_random_walk(graph):
    """Test random_walk."""
    start_nodes = ["alice", "bob"]
    length = 5
    walks_per_node = 2

    walks = graph.random_walk(
        start_nodes=start_nodes,
        length=length,
        walks_per_node=walks_per_node,
    )

    # We should have walks_per_node lists of walks per start node
    assert len(walks) == len(start_nodes) * walks_per_node

    assert isinstance(walks, list)
    if len(walks) > 0:
        assert isinstance(walks[0], list)
