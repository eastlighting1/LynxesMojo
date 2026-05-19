use lynxes_core::CsrIndex;

#[test]
fn empty_graph_builds_correctly() {
    let csr = CsrIndex::build(&[], &[], 0);
    assert_eq!(csr.node_count(), 0);
    assert_eq!(csr.edge_count(), 0);
}

#[test]
fn single_node_no_edges() {
    let csr = CsrIndex::build(&[], &[], 1);
    assert_eq!(csr.node_count(), 1);
    assert_eq!(csr.edge_count(), 0);
}

#[test]
fn design_doc_example_builds_correctly() {
    let src = [0u32, 0, 1, 2, 2];
    let dst = [1u32, 2, 2, 0, 2];
    let csr = CsrIndex::build(&src, &dst, 3);

    assert_eq!(csr.neighbors(0), &[1u32, 2]);
    assert_eq!(csr.neighbors(1), &[2u32]);
    assert_eq!(csr.neighbors(2), &[0u32, 2]);
    assert_eq!(csr.edge_ids(2), &[3u32, 4]);
}

#[test]
fn isolated_nodes_have_zero_degree() {
    let csr = CsrIndex::build(&[1], &[0], 3);
    assert_eq!(csr.degree(0), 0);
    assert_eq!(csr.degree(1), 1);
    assert_eq!(csr.degree(2), 0);
}

#[test]
fn multi_edges_same_src_dst_are_preserved() {
    let csr = CsrIndex::build(&[0, 0], &[1, 1], 2);
    assert_eq!(csr.degree(0), 2);
    assert_eq!(csr.neighbors(0), &[1u32, 1]);
}

#[test]
fn out_of_bounds_returns_empty_or_zero() {
    let csr = CsrIndex::build(&[0], &[1], 2);
    assert_eq!(csr.neighbors(9), &[] as &[u32]);
    assert_eq!(csr.edge_ids(9), &[] as &[u32]);
    assert_eq!(csr.degree(9), 0);
}

#[test]
fn neighbors_and_edge_ids_are_parallel() {
    let csr = CsrIndex::build(&[0, 0, 1], &[2, 1, 0], 3);
    for node in 0..3u32 {
        assert_eq!(csr.neighbors(node).len(), csr.edge_ids(node).len());
        assert_eq!(csr.degree(node), csr.neighbors(node).len());
    }
}

#[test]
fn reverse_build_swaps_direction() {
    let src = [0u32, 0, 1, 2];
    let dst = [1u32, 2, 2, 0];
    let reverse = CsrIndex::build_reverse(&src, &dst, 3);

    assert_eq!(reverse.neighbors(0), &[2u32]);
    assert_eq!(reverse.neighbors(1), &[0u32]);
    assert_eq!(reverse.neighbors(2), &[0u32, 1]);
}

#[test]
fn reverse_build_matches_serial_swap_build() {
    let src = [0u32, 4, 2, 2, 1, 4, 0, 3, 3, 1, 5, 5];
    let dst = [4u32, 1, 0, 5, 5, 2, 3, 1, 0, 2, 4, 3];

    let reverse = CsrIndex::build_reverse(&src, &dst, 6);
    let serial = CsrIndex::build(&dst, &src, 6);

    for node in 0..6u32 {
        assert_eq!(reverse.neighbors(node), serial.neighbors(node));
        assert_eq!(reverse.edge_ids(node), serial.edge_ids(node));
        assert_eq!(reverse.degree(node), serial.degree(node));
    }
}

#[test]
#[should_panic(expected = "src_rows and dst_rows must have equal length")]
fn mismatched_src_dst_panics() {
    CsrIndex::build(&[0u32, 1], &[1u32], 2);
}
