struct OneSweepRadixSortControl {
  uint32_t count;
  uint32_t shift;
  uint32_t tile_count;
  uint32_t descending;
  uint32_t has_values;
  uint32_t segment_count;
  uint32_t segmented;
  uint32_t scan_ring_tile_count;
  uint32_t scan_ring_tile_mask;
  uint32_t scan_lookback_limit;
};

void record_dx12_contract(ID3D12GraphicsCommandList* cmd,
                          ID3D12RootSignature* root,
                          ID3D12PipelineState* clear,
                          ID3D12PipelineState* histogram,
                          ID3D12PipelineState* bin_base,
                          ID3D12PipelineState* scatter,
                          D3D12_GPU_DESCRIPTOR_HANDLE srv_uav_table,
                          uint32_t workgroup_size,
                          OneSweepRadixSortControl pc) {
  // workgroup_size must match the selected pipeline variant: 64, 128, 256, or 512.
  cmd->SetComputeRootSignature(root);
  cmd->SetComputeRootDescriptorTable(0, srv_uav_table); // 12 buffers, bindings 0..11.

  for (uint32_t pass = 0; pass < 4; ++pass) {
    pc.shift = pass * 8u;
    cmd->SetComputeRoot32BitConstants(1, 10, &pc, 0);
    cmd->SetPipelineState(clear);
    cmd->Dispatch((std::max(pc.scan_ring_tile_count, pc.segment_count) * 256u + workgroup_size - 1u) / workgroup_size, 1, 1);
    // UAV barrier covering sort buffers/scratch.
    cmd->SetPipelineState(histogram);
    cmd->Dispatch(pc.tile_count, 1, 1);
    // UAV barrier.
    cmd->SetPipelineState(bin_base);
    cmd->Dispatch(pc.segment_count, 1, 1);
    // UAV barrier.
    cmd->SetPipelineState(scatter);
    cmd->Dispatch(pc.tile_count, 1, 1);
    // UAV barrier, then swap input/output descriptors for the next pass.
  }
}
