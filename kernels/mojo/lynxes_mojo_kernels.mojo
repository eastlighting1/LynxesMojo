from memory import LegacyUnsafePointer


@export("lynxes_structural_degrees_i64", ABI="C")
fn lynxes_structural_degrees_i64(
    node_count: Int,
    node_to_edge_idx: LegacyUnsafePointer[Int64],
    out_offsets: LegacyUnsafePointer[UInt32],
    out_edge_ids: LegacyUnsafePointer[UInt32],
    in_offsets: LegacyUnsafePointer[UInt32],
    in_edge_ids: LegacyUnsafePointer[UInt32],
    edge_allowed: LegacyUnsafePointer[UInt8],
    out_degree: LegacyUnsafePointer[Int64],
    in_degree: LegacyUnsafePointer[Int64],
    total_degree: LegacyUnsafePointer[Int64],
) -> Int32:
    for row in range(node_count):
        let edge_idx_i64 = node_to_edge_idx.load(row)[0]
        if edge_idx_i64 < 0:
            out_degree.store(row, Int64(0))
            in_degree.store(row, Int64(0))
            total_degree.store(row, Int64(0))
            continue

        let edge_idx = Int(edge_idx_i64)

        var out_count = 0
        let out_start = Int(out_offsets.load(edge_idx)[0])
        let out_end = Int(out_offsets.load(edge_idx + 1)[0])
        for pos in range(out_start, out_end):
            let edge_row = Int(out_edge_ids.load(pos)[0])
            if edge_allowed.load(edge_row)[0] != UInt8(0):
                out_count += 1

        var in_count = 0
        let in_start = Int(in_offsets.load(edge_idx)[0])
        let in_end = Int(in_offsets.load(edge_idx + 1)[0])
        for pos in range(in_start, in_end):
            let edge_row = Int(in_edge_ids.load(pos)[0])
            if edge_allowed.load(edge_row)[0] != UInt8(0):
                in_count += 1

        out_degree.store(row, Int64(out_count))
        in_degree.store(row, Int64(in_count))
        total_degree.store(row, Int64(out_count + in_count))

    return Int32(0)
