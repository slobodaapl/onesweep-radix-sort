typedef struct OneSweepRadixSortControl {
  uint32_t count, shift, tile_count, descending;
  uint32_t has_values, segment_count, segmented, scan_ring_tile_count;
  uint32_t scan_ring_tile_mask, scan_lookback_limit;
} OneSweepRadixSortControl;

void onesweep_record_wgpu_contract(WGPUComputePassEncoder pass,
                                   WGPUBindGroup bind_group,
                                   WGPUComputePipeline clear,
                         WGPUComputePipeline histogram,
                         WGPUComputePipeline bin_base,
                         WGPUComputePipeline scatter,
                         uint32_t workgroup_size,
                         OneSweepRadixSortControl pc) {
  /* workgroup_size must match the selected pipeline variant: 64, 128, 256, or 512. */
  wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
  for (uint32_t radix_pass = 0; radix_pass < 4; ++radix_pass) {
    pc.shift = radix_pass * 8u;
    wgpuComputePassEncoderSetPipeline(pass, clear);
    uint32_t clear_items = pc.scan_ring_tile_count > pc.segment_count ? pc.scan_ring_tile_count : pc.segment_count;
    clear_items *= 256u;
    wgpuComputePassEncoderDispatchWorkgroups(pass, (clear_items + workgroup_size - 1u) / workgroup_size, 1, 1);
    wgpuComputePassEncoderSetPipeline(pass, histogram);
    wgpuComputePassEncoderDispatchWorkgroups(pass, pc.tile_count, 1, 1);
    wgpuComputePassEncoderSetPipeline(pass, bin_base);
    wgpuComputePassEncoderDispatchWorkgroups(pass, pc.segment_count, 1, 1);
    wgpuComputePassEncoderSetPipeline(pass, scatter);
    wgpuComputePassEncoderDispatchWorkgroups(pass, pc.tile_count, 1, 1);
  }
}
