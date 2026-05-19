import multiprocessing as mp
import pickle

import numpy as np
import pyarrow as pa
import pytest

import lynxes as lx


def _labels(values):
    return pa.array(values, type=pa.list_(pa.string()))


def _frame():
    return lx.NodeFrame.from_dict(
        {
            "_id": ["n0", "n1", "n2"],
            "_label": [["Person"], ["Person"], ["Company"]],
            "age": [10, 20, 30],
            "score": [1.5, 2.5, 3.5],
            "name": ["a", "b", "c"],
        }
    )


def test_feature_columns_excludes_reserved_and_non_numeric_by_default():
    frame = _frame()

    assert frame.feature_columns() == ["age", "score"]
    assert frame.feature_columns(numeric_only=False) == ["age", "score", "name"]
    assert frame.feature_columns(include=["_id", "score", "name"], numeric_only=False) == [
        "score",
        "name",
    ]


def test_to_numpy_uses_feature_columns_and_honors_explicit_order():
    frame = _frame()

    default = frame.to_numpy()
    explicit = frame.to_numpy(columns=["score", "age"], dtype="float32")

    np.testing.assert_allclose(default, np.array([[10, 1.5], [20, 2.5], [30, 3.5]]))
    assert explicit.shape == (3, 2)
    assert explicit.dtype == np.float32
    np.testing.assert_allclose(explicit, np.array([[1.5, 10], [2.5, 20], [3.5, 30]]))


def test_to_numpy_indices_accept_sequence_numpy_and_pyarrow():
    frame = _frame()

    expected = np.array([[30], [10]])
    np.testing.assert_allclose(frame.to_numpy(columns=["age"], indices=[2, 0]), expected)
    np.testing.assert_allclose(
        frame.to_numpy(columns=["age"], indices=np.array([2, 0], dtype=np.int64)),
        expected,
    )
    np.testing.assert_allclose(
        frame.to_numpy(columns=["age"], indices=pa.array([2, 0], type=pa.uint32())),
        expected,
    )


def test_to_numpy_shape_rules_for_single_empty_rows_and_empty_columns():
    frame = _frame()

    assert frame.to_numpy(columns=["age"]).shape == (3, 1)
    assert frame.to_numpy(columns=["age"], indices=[]).shape == (0, 1)
    assert frame.to_numpy(columns=[]).shape == (3, 0)


def test_take_preserves_schema_and_supports_empty_selection():
    frame = _frame()

    taken = frame.take([2, 0])
    empty = frame.take([])

    assert taken.ids() == ["n2", "n0"]
    assert taken.column_names() == frame.column_names()
    assert empty.column_names() == frame.column_names()
    assert len(empty) == 0


def test_export_errors_are_actionable():
    frame = _frame()

    with pytest.raises(KeyError, match="column not found"):
        frame.to_numpy(columns=["missing"])
    with pytest.raises(TypeError, match="non-numeric"):
        frame.to_numpy(columns=["name"])
    with pytest.raises(IndexError, match="out of bounds"):
        frame.take([3])


def test_from_arrow_table_with_chunks_keeps_all_rows():
    left = pa.table(
        {
            "_id": pa.array(["n0", "n1"]),
            "_label": _labels([["Person"], ["Person"]]),
            "x": pa.array([1, 2], type=pa.int64()),
        }
    )
    right = pa.table(
        {
            "_id": pa.array(["n2"]),
            "_label": _labels([["Person"]]),
            "x": pa.array([3], type=pa.int64()),
        }
    )
    table = pa.concat_tables([left, right])

    frame = lx.NodeFrame.from_arrow(table)

    assert frame.ids() == ["n0", "n1", "n2"]
    np.testing.assert_allclose(frame.to_numpy(), np.array([[1], [2], [3]]))


def test_to_tensor_matches_numpy_when_torch_is_installed():
    torch = pytest.importorskip("torch")
    frame = _frame()

    tensor = frame.to_tensor(columns=["score", "age"], indices=[2, 0], dtype="float32")

    assert tuple(tensor.shape) == (2, 2)
    assert tensor.dtype == torch.float32
    np.testing.assert_allclose(tensor.cpu().numpy(), np.array([[3.5, 30], [1.5, 10]]))


def _pickle_roundtrip_worker(payload):
    frame = pickle.loads(payload)
    return frame.ids(), frame.to_numpy(columns=["age"]).tolist()


def test_nodeframe_pickle_roundtrip_supports_spawn_worker_policy():
    frame = _frame()
    payload = pickle.dumps(frame)

    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        ids, ages = pool.apply(_pickle_roundtrip_worker, (payload,))

    assert ids == ["n0", "n1", "n2"]
    assert ages == [[10], [20], [30]]
