typedef struct OneSweepRadixSortControl {
  uint32_t count, shift, tile_count, descending;
  uint32_t has_values, segment_count, segmented, scan_ring_tile_count;
  uint32_t scan_ring_tile_mask, scan_lookback_limit;
} OneSweepRadixSortControl;

void onesweep_record_vulkan_contract(VkCommandBuffer cmd,
                                     VkPipelineLayout layout,
                                     VkDescriptorSet set,
                                     VkPipeline clear,
                                     VkPipeline histogram,
                                     VkPipeline bin_base,
                                     VkPipeline scatter,
                                     uint32_t workgroup_size,
                                     OneSweepRadixSortControl pc) {
  /* workgroup_size must match the selected compiled shader variant: 64, 128, 256, or 512. */
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, NULL);
  for (uint32_t pass = 0; pass < 4; ++pass) {
    pc.shift = pass * 8u;
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, clear);
    uint32_t clear_items = pc.scan_ring_tile_count > pc.segment_count ? pc.scan_ring_tile_count : pc.segment_count;
    clear_items *= 256u;
    vkCmdDispatch(cmd, (clear_items + workgroup_size - 1u) / workgroup_size, 1, 1);
    /* compute shader write -> read/write barrier */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, histogram);
    vkCmdDispatch(cmd, pc.tile_count, 1, 1);
    /* barrier */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bin_base);
    vkCmdDispatch(cmd, pc.segment_count, 1, 1);
    /* barrier */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, scatter);
    vkCmdDispatch(cmd, pc.tile_count, 1, 1);
    /* barrier and swap input/output descriptors */
  }
}
