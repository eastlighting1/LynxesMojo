// Benchmark: CSR neighbor lookup vs. linear scan over EdgeFrame
//
// Compares O(degree) CSR lookup against a naive O(E) linear scan for
// single-node neighbor queries on graphs of 100K and 1M nodes.

use std::sync::Arc;
use std::time::Duration;

use arrow_array::{Int8Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use lynxes_core::{
    CsrIndex, EdgeFrame, COL_EDGE_DIRECTION, COL_EDGE_DST, COL_EDGE_SRC, COL_EDGE_TYPE,
};

// ── Graph generators ──────────────────────────────────────────────────────────

/// Build a random-ish edge list for `n` nodes where each node i has an edge to (i+1) % n.
/// Also adds a hub node (0) with edges to the first `hub_degree` nodes.
fn make_edge_frame(n: u32, hub_degree: u32) -> EdgeFrame {
    let mut srcs: Vec<String> = Vec::new();
    let mut dsts: Vec<String> = Vec::new();

    // Ring edges
    for i in 0..n {
        srcs.push(i.to_string());
        dsts.push(((i + 1) % n).to_string());
    }

    // Hub edges: node 0 → first hub_degree nodes
    let actual_hub = hub_degree.min(n);
    for j in 1..actual_hub {
        srcs.push("0".to_owned());
        dsts.push(j.to_string());
    }

    let edge_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(COL_EDGE_SRC, DataType::Utf8, false),
        Field::new(COL_EDGE_DST, DataType::Utf8, false),
        Field::new(COL_EDGE_TYPE, DataType::Utf8, false),
        Field::new(COL_EDGE_DIRECTION, DataType::Int8, false),
    ]));

    let len = srcs.len();
    let batch = RecordBatch::try_new(
        edge_schema,
        vec![
            Arc::new(StringArray::from(srcs)) as Arc<dyn arrow_array::Array>,
            Arc::new(StringArray::from(dsts)) as Arc<dyn arrow_array::Array>,
            Arc::new(StringArray::from(vec!["E"; len])) as Arc<dyn arrow_array::Array>,
            Arc::new(Int8Array::from(vec![0i8; len])) as Arc<dyn arrow_array::Array>,
        ],
    )
    .unwrap();

    EdgeFrame::from_record_batch(batch).unwrap()
}

/// Build a CSR directly from src/dst index pairs (0-based row indices).
fn make_csr(n: u32, hub_degree: u32) -> CsrIndex {
    let mut src_rows: Vec<u32> = Vec::new();
    let mut dst_rows: Vec<u32> = Vec::new();

    for i in 0..n {
        src_rows.push(i);
        dst_rows.push((i + 1) % n);
    }

    let actual_hub = hub_degree.min(n);
    for j in 1..actual_hub {
        src_rows.push(0);
        dst_rows.push(j);
    }

    CsrIndex::build(&src_rows, &dst_rows, n as usize)
}

// ── Linear scan baseline ──────────────────────────────────────────────────────

/// Simulate a linear scan: iterate all edges and collect those with matching src.
fn linear_neighbors<'a>(
    srcs: &'a [u32],
    dsts: &'a [u32],
    target: u32,
) -> impl Iterator<Item = u32> + 'a {
    srcs.iter()
        .zip(dsts.iter())
        .filter_map(move |(&s, &d)| if s == target { Some(d) } else { None })
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

fn full_bench_enabled() -> bool {
    std::env::var_os("LYNXES_FULL_BENCH").is_some_and(|value| value == "1")
}

fn bench_csr_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("csr_neighbor_lookup");
    if !full_bench_enabled() {
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(2));
    }

    for &n in &[100_000u32, 1_000_000u32] {
        let hub_degree = 100;
        let csr = make_csr(n, hub_degree);

        group.bench_with_input(BenchmarkId::new("csr", n), &n, |b, _| {
            b.iter(|| {
                // Look up neighbors of the hub node (node 0 has hub_degree + 1 neighbors)
                let neighbors = black_box(csr.neighbors(black_box(0)));
                black_box(neighbors.len())
            });
        });
    }

    group.finish();
}

fn bench_linear_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_scan_neighbor_lookup");
    if !full_bench_enabled() {
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(2));
    }

    for &n in &[100_000u32, 1_000_000u32] {
        let hub_degree = 100u32;
        // Build flat src/dst arrays for linear scan
        let mut src_rows: Vec<u32> = (0..n).collect();
        let mut dst_rows: Vec<u32> = (1..=n).map(|i| i % n).collect();
        for j in 1..hub_degree.min(n) {
            src_rows.push(0);
            dst_rows.push(j);
        }

        group.bench_with_input(BenchmarkId::new("linear", n), &n, |b, _| {
            b.iter(|| {
                let count = black_box(
                    linear_neighbors(black_box(&src_rows), black_box(&dst_rows), black_box(0))
                        .count(),
                );
                black_box(count)
            });
        });
    }

    group.finish();
}

fn bench_edge_frame_csr_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_frame_csr_lookup");
    if !full_bench_enabled() {
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(2));
    }

    // Only 100K for EdgeFrame (string-based, slower to build)
    let n = 100_000u32;
    let hub_degree = 100;
    let frame = make_edge_frame(n, hub_degree);
    let hub_idx = frame.node_row_idx("0").unwrap();

    group.bench_function("edge_frame_100k", |b| {
        b.iter(|| {
            let neighbors = black_box(frame.out_neighbors(black_box(hub_idx)));
            black_box(neighbors.len())
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_csr_lookup,
    bench_linear_scan,
    bench_edge_frame_csr_lookup,
);
criterion_main!(benches);
