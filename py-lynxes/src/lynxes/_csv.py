"""CSV reader front-end for Lynxes Python APIs."""

from lynxes._lynxes import NodeFrame
from lynxes._lynxes import read_csv_native_py as _read_csv_native


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.csv as csv
    except ImportError as exc:
        raise ImportError(
            "lynxes.read_csv(engine='pyarrow') requires pyarrow. Install pyarrow to use it."
        ) from exc

    return pa, pc, csv


def _string_array_from_column(column, *, name):
    pa, pc, _ = _require_pyarrow()
    if column.null_count:
        raise ValueError(f"{name} cannot contain null values")
    return pc.cast(column.combine_chunks(), pa.string())


def _label_array(label, rows):
    pa, _, _ = _require_pyarrow()
    offsets = pa.array(range(rows + 1), type=pa.int32())
    values = pa.array([label] * rows, type=pa.string())
    return pa.ListArray.from_arrays(offsets, values)


def _singleton_label_array(column, *, name):
    pa, pc, _ = _require_pyarrow()
    if column.null_count:
        raise ValueError(f"{name} cannot contain null values")
    values = pc.cast(column.combine_chunks(), pa.string())
    offsets = pa.array(range(len(values) + 1), type=pa.int32())
    return pa.ListArray.from_arrays(offsets, values)


def _csv_table_to_node_frame(table, *, label=None, id_col=None, id_prefix=None, columns=None):
    pa, _, _ = _require_pyarrow()
    column_names = table.column_names
    rows = table.num_rows

    if id_col is not None:
        if id_col not in column_names:
            raise ValueError(f"id_col {id_col!r} not found in CSV columns")
        id_array = _string_array_from_column(table[id_col], name=id_col)
    elif "_id" in column_names:
        id_array = _string_array_from_column(table["_id"], name="_id")
    else:
        prefix = id_prefix or "row"
        id_array = pa.array((f"{prefix}_{idx}" for idx in range(rows)), type=pa.string())

    if label is not None:
        label_array = _label_array(label, rows)
    elif "_label" in column_names:
        label_array = _singleton_label_array(table["_label"], name="_label")
    else:
        raise ValueError("read_csv requires label=... unless the CSV contains a _label column")

    arrays = [id_array, label_array]
    names = ["_id", "_label"]
    output_names = columns if columns is not None else column_names
    for name in output_names:
        if name in {"_id", "_label"}:
            continue
        if name.startswith("_"):
            raise ValueError(f"CSV column {name!r} is reserved by NodeFrame")
        if name not in column_names:
            raise ValueError(f"column {name!r} not found in CSV columns")
        arrays.append(table[name].combine_chunks())
        names.append(name)

    batch = pa.RecordBatch.from_arrays(arrays, names=names)
    return NodeFrame.from_arrow(batch)


def read_csv(
    path,
    *,
    label=None,
    id_col=None,
    id_prefix=None,
    columns=None,
    schema_overrides=None,
    engine="native",
    infer_schema_rows=None,
    batch_size=65536,
    has_header=True,
    delimiter=",",
    read_options=None,
    parse_options=None,
    convert_options=None,
):
    """Read a CSV file into a NodeFrame."""
    if engine == "native":
        if read_options is not None or parse_options is not None or convert_options is not None:
            raise ValueError("pyarrow CSV options require engine='pyarrow'")
        return _read_csv_native(
            path,
            label=label,
            id_col=id_col,
            id_prefix=id_prefix,
            columns=columns,
            schema_overrides=schema_overrides,
            infer_schema_rows=infer_schema_rows,
            batch_size=batch_size,
            has_header=has_header,
            delimiter=delimiter,
        )

    if engine != "pyarrow":
        raise ValueError("engine must be 'native' or 'pyarrow'")
    if infer_schema_rows is not None or batch_size != 65536 or not has_header or delimiter != ",":
        raise ValueError("native CSV options require engine='native'")

    pa, _, csv = _require_pyarrow()
    if convert_options is None:
        include_columns = _pyarrow_include_columns(columns, id_col=id_col, label=label)
        column_types = _pyarrow_schema_overrides(pa, schema_overrides)
        if include_columns is not None or column_types:
            convert_options = csv.ConvertOptions(
                include_columns=include_columns,
                column_types=column_types,
            )
    elif columns is not None or schema_overrides:
        raise ValueError(
            "columns/schema_overrides cannot be combined with explicit pyarrow convert_options"
        )
    table = csv.read_csv(
        path,
        read_options=read_options,
        parse_options=parse_options,
        convert_options=convert_options,
    )
    return _csv_table_to_node_frame(
        table,
        label=label,
        id_col=id_col,
        id_prefix=id_prefix,
        columns=columns,
    )


def _pyarrow_include_columns(columns, *, id_col, label):
    if columns is None:
        return None

    include_columns = []
    if id_col is not None:
        include_columns.append(id_col)
    elif "_id" in columns:
        include_columns.append("_id")
    if label is None:
        include_columns.append("_label")
    for name in columns:
        if name not in include_columns:
            include_columns.append(name)
    return include_columns


def _pyarrow_schema_overrides(pa, schema_overrides):
    if not schema_overrides:
        return {}

    marker_to_type = {
        "String": pa.string(),
        "StringView": getattr(pa, "string_view", pa.string)(),
        "Utf8View": getattr(pa, "string_view", pa.string)(),
        "Int": pa.int64(),
        "Float": pa.float64(),
        "Bool": pa.bool_(),
    }
    out = {}
    for name, dtype in schema_overrides.items():
        try:
            out[name] = marker_to_type[dtype]
        except KeyError as exc:
            raise ValueError(f"unsupported schema override dtype for {name!r}: {dtype!r}") from exc
    return out


def _node_frame_read_csv(
    _cls,
    path,
    *,
    label=None,
    id_col=None,
    id_prefix=None,
    columns=None,
    schema_overrides=None,
    engine="native",
    infer_schema_rows=None,
    batch_size=65536,
    has_header=True,
    delimiter=",",
    read_options=None,
    parse_options=None,
    convert_options=None,
):
    return read_csv(
        path,
        label=label,
        id_col=id_col,
        id_prefix=id_prefix,
        columns=columns,
        schema_overrides=schema_overrides,
        engine=engine,
        infer_schema_rows=infer_schema_rows,
        batch_size=batch_size,
        has_header=has_header,
        delimiter=delimiter,
        read_options=read_options,
        parse_options=parse_options,
        convert_options=convert_options,
    )


NodeFrame.read_csv = classmethod(_node_frame_read_csv)

__all__ = ["read_csv"]
