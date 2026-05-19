from std.memory import UnsafePointer


@export("lynxes_structural_degrees_i64", ABI="C")
fn lynxes_structural_degrees_i64(
    node_count: Int,
    node_to_edge_idx: UnsafePointer[Int64, ImmutExternalOrigin],
    out_offsets: UnsafePointer[UInt32, ImmutExternalOrigin],
    out_edge_ids: UnsafePointer[UInt32, ImmutExternalOrigin],
    in_offsets: UnsafePointer[UInt32, ImmutExternalOrigin],
    in_edge_ids: UnsafePointer[UInt32, ImmutExternalOrigin],
    edge_allowed: UnsafePointer[UInt8, ImmutExternalOrigin],
    out_degree: UnsafePointer[Int64, MutExternalOrigin],
    in_degree: UnsafePointer[Int64, MutExternalOrigin],
    total_degree: UnsafePointer[Int64, MutExternalOrigin],
) -> Int32:
    for row in range(node_count):
        var edge_idx_i64 = node_to_edge_idx[row]
        if edge_idx_i64 < 0:
            out_degree[row] = Int64(0)
            in_degree[row] = Int64(0)
            total_degree[row] = Int64(0)
            continue

        var edge_idx = Int(edge_idx_i64)

        var out_count = 0
        var out_start = Int(out_offsets[edge_idx])
        var out_end = Int(out_offsets[edge_idx + 1])
        for pos in range(out_start, out_end):
            var edge_row = Int(out_edge_ids[pos])
            if edge_allowed[edge_row] != UInt8(0):
                out_count += 1

        var in_count = 0
        var in_start = Int(in_offsets[edge_idx])
        var in_end = Int(in_offsets[edge_idx + 1])
        for pos in range(in_start, in_end):
            var edge_row = Int(in_edge_ids[pos])
            if edge_allowed[edge_row] != UInt8(0):
                in_count += 1

        out_degree[row] = Int64(out_count)
        in_degree[row] = Int64(in_count)
        total_degree[row] = Int64(out_count + in_count)

    return Int32(0)
