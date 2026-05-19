#![cfg(target_os = "linux")]

mod common;

use arrow_array::Int64Array;
use common::sample_graph;
use lynxes_core::{configure_mojo_runtime, GFError};

fn configure_from_env() {
    if let Some(path) = std::env::var_os("LYNXES_MOJO_LIB") {
        configure_mojo_runtime(path.to_string_lossy().as_ref()).unwrap();
    }
}

#[test]
fn mojo_structural_features_compute_all_edge_degrees() {
    configure_from_env();
    let graph = sample_graph();

    let features = match graph.structural_features(None) {
        Ok(features) => features,
        Err(GFError::UnsupportedOperation { message }) if message.contains("Mojo runtime") => {
            panic!("LYNXES_MOJO_LIB must point to a built liblynxes_mojo_kernels.so for this test")
        }
        Err(err) => panic!("unexpected structural_features error: {err}"),
    };

    let out_degree = features
        .column("out_degree")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let in_degree = features
        .column("in_degree")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let total_degree = features
        .column("total_degree")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();

    assert_eq!(features.id_column().value(0), "alice");
    assert_eq!(features.id_column().value(4), "acme");
    assert_eq!(out_degree.values(), &[2, 1, 0, 1, 0]);
    assert_eq!(in_degree.values(), &[1, 1, 2, 0, 1]);
    assert_eq!(total_degree.values(), &[3, 2, 2, 1, 1]);
}

#[test]
fn mojo_structural_features_respect_edge_type_filter() {
    configure_from_env();
    let graph = sample_graph();

    let features = graph.structural_features(Some("KNOWS")).unwrap();
    let out_degree = features
        .column("out_degree")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let in_degree = features
        .column("in_degree")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();

    assert_eq!(out_degree.values(), &[1, 1, 0, 0, 0]);
    assert_eq!(in_degree.values(), &[0, 1, 1, 0, 0]);
}
