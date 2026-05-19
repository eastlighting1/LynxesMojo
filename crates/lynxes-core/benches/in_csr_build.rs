use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use lynxes_core::CsrIndex;

fn make_reverse_inputs(node_count: u32, edge_count: usize) -> (Vec<u32>, Vec<u32>) {
    let mut src_rows = Vec::with_capacity(edge_count);
    let mut dst_rows = Vec::with_capacity(edge_count);

    // Deterministic pseudo-random-ish graph with enough skew to resemble
    // real inbound adjacency hot spots while staying reproducible.
    for i in 0..edge_count as u32 {
        let src = i % node_count;
        let dst = ((i.wrapping_mul(1_103_515_245).wrapping_add(12_345)) % node_count)
            ^ ((i / 97) % node_count);
        src_rows.push(src);
        dst_rows.push(dst % node_count);
    }

    (src_rows, dst_rows)
}

fn build_reverse_serial(src_rows: &[u32], dst_rows: &[u32], node_count: usize) -> CsrIndex {
    CsrIndex::build(dst_rows, src_rows, node_count)
}

fn bench_in_csr_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("in_csr_build");
    group.sample_size(10);

    for &(node_count, edge_count) in &[(100_000u32, 100_000usize), (1_000_000u32, 1_000_000usize)] {
        let (src_rows, dst_rows) = make_reverse_inputs(node_count, edge_count);

        group.bench_with_input(
            BenchmarkId::new("serial", edge_count),
            &edge_count,
            |b, _| {
                b.iter(|| {
                    let csr = build_reverse_serial(
                        black_box(&src_rows),
                        black_box(&dst_rows),
                        black_box(node_count as usize),
                    );
                    black_box(csr.degree(0))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("parallel", edge_count),
            &edge_count,
            |b, _| {
                b.iter(|| {
                    let csr = CsrIndex::build_reverse(
                        black_box(&src_rows),
                        black_box(&dst_rows),
                        black_box(node_count as usize),
                    );
                    black_box(csr.degree(0))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_in_csr_build);
criterion_main!(benches);
