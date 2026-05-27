#[repr(C)]
#[derive(Clone, Copy)]
struct OneSweepRadixSortControl {
    count: u32, shift: u32, tile_count: u32, descending: u32,
    has_values: u32, segment_count: u32, segmented: u32, scan_ring_tile_count: u32,
    scan_ring_tile_mask: u32, scan_lookback_limit: u32,
}

fn record_dx12_contract(command_list: &windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList, mut pc: OneSweepRadixSortControl) {
    // Create pipelines for one selected workgroup size: 64, 128, 256, or 512.
    for pass in 0..4 {
        pc.shift = pass * 8;
        unsafe {
            command_list.SetComputeRoot32BitConstants(1, 10, &pc as *const _ as *const _, 0);
            // Bind one descriptor table containing storage buffers 0..11.
            // Dispatch clear, histogram, bin-base, scatter with UAV barriers between kernels.
        }
    }
}
