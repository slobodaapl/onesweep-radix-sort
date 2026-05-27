#[repr(C)]
#[derive(Clone, Copy)]
struct OneSweepRadixSortControl {
    count: u32, shift: u32, tile_count: u32, descending: u32,
    has_values: u32, segment_count: u32, segmented: u32, scan_ring_tile_count: u32,
    scan_ring_tile_mask: u32, scan_lookback_limit: u32,
}

fn encode_metal_contract(encoder: &metal::ComputeCommandEncoderRef, pipelines: [&metal::ComputePipelineStateRef; 4], buffers: [&metal::BufferRef; 12], mut pc: OneSweepRadixSortControl) {
    // Create pipelines for one selected workgroup size: 64, 128, 256, or 512.
    for (i, buffer) in buffers.iter().enumerate() {
        encoder.set_buffer(i as u64, Some(buffer), 0);
    }
    for pass in 0..4 {
        pc.shift = pass * 8;
        encoder.set_bytes(12, std::mem::size_of::<OneSweepRadixSortControl>() as u64, &pc as *const _ as *const _);
        for pipeline in pipelines {
            encoder.set_compute_pipeline_state(pipeline);
            // Dispatch clear, histogram, bin-base, scatter sizes from the contract; insert buffer memory barriers between kernels.
        }
    }
}
