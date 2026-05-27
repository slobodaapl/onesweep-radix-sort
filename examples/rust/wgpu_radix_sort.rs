#[repr(C)]
#[derive(Clone, Copy)]
struct OneSweepRadixSortControl {
    count: u32, shift: u32, tile_count: u32, descending: u32,
    has_values: u32, segment_count: u32, segmented: u32, scan_ring_tile_count: u32,
    scan_ring_tile_mask: u32, scan_lookback_limit: u32,
}

fn encode_wgpu_contract(pass: &mut wgpu::ComputePass<'_>, bind_group: &wgpu::BindGroup, pipelines: [&wgpu::ComputePipeline; 4], workgroup_size: u32, mut pc: OneSweepRadixSortControl) {
    // workgroup_size must match the selected pipeline variant: 64, 128, 256, or 512.
    pass.set_bind_group(0, bind_group, &[]); // 12 storage buffers, bindings 0..11.
    for radix_pass in 0..4 {
        pc.shift = radix_pass * 8;
        // Write pc to the backend control buffer before each kernel.
        pass.set_pipeline(pipelines[0]);
        pass.dispatch_workgroups(((pc.scan_ring_tile_count.max(pc.segment_count) * 256) + workgroup_size - 1) / workgroup_size, 1, 1);
        pass.set_pipeline(pipelines[1]);
        pass.dispatch_workgroups(pc.tile_count, 1, 1);
        pass.set_pipeline(pipelines[2]);
        pass.dispatch_workgroups(pc.segment_count, 1, 1);
        pass.set_pipeline(pipelines[3]);
        pass.dispatch_workgroups(pc.tile_count, 1, 1);
    }
}
