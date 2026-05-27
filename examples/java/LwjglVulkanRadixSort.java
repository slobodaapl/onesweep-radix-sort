import java.nio.ByteBuffer;

final class LwjglVulkanRadixSort {
  static final class Control {
    int count, shift, tileCount, descending;
    int hasValues, segmentCount, segmented, scanRingTileCount;
    int scanRingTileMask, scanLookbackLimit;

    ByteBuffer write(ByteBuffer dst) {
      dst.clear();
      dst.putInt(count).putInt(shift).putInt(tileCount).putInt(descending);
      dst.putInt(hasValues).putInt(segmentCount).putInt(segmented).putInt(scanRingTileCount);
      dst.putInt(scanRingTileMask).putInt(scanLookbackLimit);
      dst.flip();
      return dst;
    }
  }

  static void record(long cmd, long layout, long descriptorSet, long clear, long histogram, long binBase, long scatter, int workgroupSize, Control pc, ByteBuffer pushBytes) {
    // workgroupSize must match the selected shader variant: 64, 128, 256, or 512.
    VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, descriptorSet, null);
    for (int pass = 0; pass < 4; pass++) {
      pc.shift = pass * 8;
      VK10.vkCmdPushConstants(cmd, layout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, pc.write(pushBytes));
      VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, clear);
      int clearItems = Math.max(pc.scanRingTileCount, pc.segmentCount) * 256;
      VK10.vkCmdDispatch(cmd, (clearItems + workgroupSize - 1) / workgroupSize, 1, 1);
      VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, histogram);
      VK10.vkCmdDispatch(cmd, pc.tileCount, 1, 1);
      VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, binBase);
      VK10.vkCmdDispatch(cmd, pc.segmentCount, 1, 1);
      VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, scatter);
      VK10.vkCmdDispatch(cmd, pc.tileCount, 1, 1);
    }
  }
}
