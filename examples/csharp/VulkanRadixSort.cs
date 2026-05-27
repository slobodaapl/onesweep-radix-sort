using System.Runtime.InteropServices;

[StructLayout(LayoutKind.Sequential)]
struct OneSweepRadixSortControl {
    public uint Count, Shift, TileCount, Descending;
    public uint HasValues, SegmentCount, Segmented, ScanRingTileCount;
    public uint ScanRingTileMask, ScanLookbackLimit;
}

static void RecordVulkanContract(VkCommandBuffer cmd, VkPipelineLayout layout, VkDescriptorSet set, VkPipeline clear, VkPipeline histogram, VkPipeline binBase, VkPipeline scatter, uint workgroupSize, OneSweepRadixSortControl pc) {
    // workgroupSize must match the selected shader variant: 64, 128, 256, or 512.
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, ref set, 0, nint.Zero);
    for (uint pass = 0; pass < 4; pass++) {
        pc.Shift = pass * 8;
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint)Marshal.SizeOf<OneSweepRadixSortControl>(), ref pc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, clear);
        uint clearItems = Math.Max(pc.ScanRingTileCount, pc.SegmentCount) * 256;
        vkCmdDispatch(cmd, (clearItems + workgroupSize - 1) / workgroupSize, 1, 1);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, histogram);
        vkCmdDispatch(cmd, pc.TileCount, 1, 1);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, binBase);
        vkCmdDispatch(cmd, pc.SegmentCount, 1, 1);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, scatter);
        vkCmdDispatch(cmd, pc.TileCount, 1, 1);
    }
}
