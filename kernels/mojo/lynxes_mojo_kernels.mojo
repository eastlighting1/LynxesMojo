from std.memory import UnsafePointer


@export("lynxes_structural_degrees_i64", ABI="C")
fn lynxes_structural_degrees_i64(
    node_count: Int,
    node_to_edge_idx: UnsafePointer[mut=False, Int64, _],
    out_offsets: UnsafePointer[mut=False, UInt32, _],
    out_edge_ids: UnsafePointer[mut=False, UInt32, _],
    in_offsets: UnsafePointer[mut=False, UInt32, _],
    in_edge_ids: UnsafePointer[mut=False, UInt32, _],
    edge_allowed: UnsafePointer[mut=False, UInt8, _],
    out_degree: UnsafePointer[mut=True, Int64, _],
    in_degree: UnsafePointer[mut=True, Int64, _],
    total_degree: UnsafePointer[mut=True, Int64, _],
) -> Int32:
    for row in range(node_count):
        let edge_idx_i64 = node_to_edge_idx[row]
        if edge_idx_i64 < 0:
            out_degree[row] = Int64(0)
            in_degree[row] = Int64(0)
            total_degree[row] = Int64(0)
            continue

        let edge_idx = Int(edge_idx_i64)

        var out_count = 0
        let out_start = Int(out_offsets[edge_idx])
        let out_end = Int(out_offsets[edge_idx + 1])
        for pos in range(out_start, out_end):
            let edge_row = Int(out_edge_ids[pos])
            if edge_allowed[edge_row] != UInt8(0):
                out_count += 1

        var in_count = 0
        let in_start = Int(in_offsets[edge_idx])
        let in_end = Int(in_offsets[edge_idx + 1])
        for pos in range(in_start, in_end):
            let edge_row = Int(in_edge_ids[pos])
            if edge_allowed[edge_row] != UInt8(0):
                in_count += 1

        out_degree[row] = Int64(out_count)
        in_degree[row] = Int64(in_count)
        total_degree[row] = Int64(out_count + in_count)

    return Int32(0)
