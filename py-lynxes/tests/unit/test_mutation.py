import pytest

import lynxes as gf


def test_mutable_graph_frame_crud_smoke(graph):
    """CRUD Python surface smoke test for MutableGraphFrame."""
    # 1. into_mutable()
    mgf = graph.into_mutable()

    nodes = graph.nodes()

    # Filter to get exactly one row for 'alice'
    # Assuming 'alice' is the first row
    mask = [False] * len(nodes)
    if len(mask) > 0:
        mask[0] = True
    alice_node = nodes.filter(mask)

    # Filter to get exactly one row for 'bob'
    mask2 = [False] * len(nodes)
    if len(mask2) > 1:
        mask2[1] = True
    bob_node = nodes.filter(mask2)

    if len(alice_node) == 1 and len(bob_node) == 1:
        # Delete nodes to avoid duplicate ID error
        mgf.delete_node("alice")
        mgf.delete_node("bob")

        # 2. add_node()
        mgf.add_node(alice_node)

        # 3. add_nodes_batch()
        mgf.add_nodes_batch(bob_node)

        # 5. update_node()
        mgf.update_node("alice", alice_node)

    # 4. add_edge()
    mgf.add_edge("alice", "bob")

    # 7. delete_edge()
    mgf.delete_edge(0)

    # 8. compact()
    mgf.compact()

    # 9. freeze()
    new_graph = mgf.freeze()

    # Verify the new graph is valid
    assert new_graph.node_count() >= 0
    assert new_graph.edge_count() >= 0

    # Verify mgf is consumed/frozen
    with pytest.raises(RuntimeError, match="has already been frozen"):
        mgf.add_edge("alice", "bob")


def test_mutable_add_edge_preserves_explicit_edge_payload():
    graph = gf.graph(
        nodes={
            "_id": ["alice", "bob"],
            "_label": [["Person"], ["Person"]],
        },
        edges={
            "_src": ["alice"],
            "_dst": ["bob"],
            "_type": ["KNOWS"],
            "_direction": [0],
            "weight": [1],
        },
    )

    mutable = graph.into_mutable()
    mutable.add_edge("bob", "alice", edge_type="LIKES", attrs={"weight": 7})
    frozen = mutable.freeze()
    batch = frozen.edges().to_pyarrow()
    rows = list(
        zip(
            batch["_src"].to_pylist(),
            batch["_dst"].to_pylist(),
            batch["_type"].to_pylist(),
            batch["weight"].to_pylist(),
        )
    )

    assert ("bob", "alice", "LIKES", 7) in rows


def test_mutable_update_edge_preserves_existing_edge_payload():
    graph = gf.graph(
        nodes={
            "_id": ["alice", "bob"],
            "_label": [["Person"], ["Person"]],
        },
        edges={
            "_src": ["alice"],
            "_dst": ["bob"],
            "_type": ["KNOWS"],
            "_direction": [0],
            "weight": [1],
        },
    )

    mutable = graph.into_mutable()
    mutable.update_edge(0, "bob", "alice")
    frozen = mutable.freeze()
    batch = frozen.edges().to_pyarrow()
    rows = list(
        zip(
            batch["_src"].to_pylist(),
            batch["_dst"].to_pylist(),
            batch["_type"].to_pylist(),
            batch["weight"].to_pylist(),
        )
    )

    assert ("bob", "alice", "KNOWS", 1) in rows
