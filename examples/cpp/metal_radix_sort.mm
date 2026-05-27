struct OneSweepRadixSortControl {
  uint32_t count, shift, tile_count, descending;
  uint32_t has_values, segment_count, segmented, scan_ring_tile_count;
  uint32_t scan_ring_tile_mask, scan_lookback_limit;
};

void encode_metal_contract(id<MTLComputeCommandEncoder> enc,
                           id<MTLComputePipelineState> clear,
                           id<MTLComputePipelineState> histogram,
                           id<MTLComputePipelineState> binBase,
                           id<MTLComputePipelineState> scatter,
                           const std::array<id<MTLBuffer>, 12>& buffers,
                           uint32_t workgroupSize,
                           OneSweepRadixSortControl pc) {
  // workgroupSize must match the selected pipeline variant: 64, 128, 256, or 512.
  for (NSUInteger i = 0; i < buffers.size(); ++i) {
    [enc setBuffer:buffers[i] offset:0 atIndex:i];
  }

  MTLSize threads = MTLSizeMake(workgroupSize, 1, 1);
  for (uint32_t pass = 0; pass < 4; ++pass) {
    pc.shift = pass * 8u;
    [enc setBytes:&pc length:sizeof(pc) atIndex:12];
    [enc setComputePipelineState:clear];
    [enc dispatchThreadgroups:MTLSizeMake((std::max(pc.scan_ring_tile_count, pc.segment_count) * 256u + workgroupSize - 1u) / workgroupSize, 1, 1) threadsPerThreadgroup:threads];
    // memoryBarrierWithScope:MTLBarrierScopeBuffers after each kernel.
    [enc setComputePipelineState:histogram];
    [enc dispatchThreadgroups:MTLSizeMake(pc.tile_count, 1, 1) threadsPerThreadgroup:threads];
    [enc setComputePipelineState:binBase];
    [enc dispatchThreadgroups:MTLSizeMake(pc.segment_count, 1, 1) threadsPerThreadgroup:threads];
    [enc setComputePipelineState:scatter];
    [enc dispatchThreadgroups:MTLSizeMake(pc.tile_count, 1, 1) threadsPerThreadgroup:threads];
  }
}
