struct OneSweepRadixSortControl {
  uint32_t count, shift, tile_count, descending;
  uint32_t has_values, segment_count, segmented, scan_ring_tile_count;
  uint32_t scan_ring_tile_mask, scan_lookback_limit;
};

void encode_wgpu_contract(WGPUComputePassEncoder pass,
                          WGPUBindGroup bind_group,
                          WGPUComputePipeline clear,
                          WGPUComputePipeline histogram,
                          WGPUComputePipeline bin_base,
                          WGPUComputePipeline scatter,
                          uint32_t workgroup_size,
                          const OneSweepRadixSortControl& base) {
  // workgroup_size must match the selected pipeline variant: 64, 128, 256, or 512.
  wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, nullptr); // 12 storage buffers.
  for (uint32_t radix_pass = 0; radix_pass < 4; ++radix_pass) {
    OneSweepRadixSortControl pc = base;
    pc.shift = radix_pass * 8u;
    // Upload pc to the backend's uniform/push-constant equivalent before each kernel.
    wgpuComputePassEncoderSetPipeline(pass, clear);
    wgpuComputePassEncoderDispatchWorkgroups(pass, (std::max(pc.scan_ring_tile_count, pc.segment_count) * 256u + workgroup_size - 1u) / workgroup_size, 1, 1);
    wgpuComputePassEncoderSetPipeline(pass, histogram);
    wgpuComputePassEncoderDispatchWorkgroups(pass, pc.tile_count, 1, 1);
    wgpuComputePassEncoderSetPipeline(pass, bin_base);
    wgpuComputePassEncoderDispatchWorkgroups(pass, pc.segment_count, 1, 1);
    wgpuComputePassEncoderSetPipeline(pass, scatter);
    wgpuComputePassEncoderDispatchWorkgroups(pass, pc.tile_count, 1, 1);
    // End-pass or explicit backend barriers must make storage writes visible between kernels.
  }
}
