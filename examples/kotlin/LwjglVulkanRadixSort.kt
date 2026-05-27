import java.nio.ByteBuffer

data class OneSweepRadixSortControl(
    var count: Int,
    var shift: Int,
    var tileCount: Int,
    var descending: Int,
    var hasValues: Int,
    var segmentCount: Int,
    var segmented: Int,
    var scanRingTileCount: Int,
    var scanRingTileMask: Int,
    var scanLookbackLimit: Int,
) {
    fun write(dst: ByteBuffer): ByteBuffer {
        dst.clear()
        dst.putInt(count).putInt(shift).putInt(tileCount).putInt(descending)
        dst.putInt(hasValues).putInt(segmentCount).putInt(segmented).putInt(scanRingTileCount)
        dst.putInt(scanRingTileMask).putInt(scanLookbackLimit)
        dst.flip()
        return dst
    }
}

fun recordLwjglVulkanContract(
    cmd: Long,
    layout: Long,
    descriptorSet: Long,
    clear: Long,
    histogram: Long,
    binBase: Long,
    scatter: Long,
    workgroupSize: Int,
    pc: OneSweepRadixSortControl,
    pushBytes: ByteBuffer,
) {
    // workgroupSize must match the selected shader variant: 64, 128, 256, or 512.
    VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, descriptorSet, null)
    repeat(4) { pass ->
        pc.shift = pass * 8
        VK10.vkCmdPushConstants(cmd, layout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, pc.write(pushBytes))
        VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, clear)
        val clearItems = maxOf(pc.scanRingTileCount, pc.segmentCount) * 256
        VK10.vkCmdDispatch(cmd, (clearItems + workgroupSize - 1) / workgroupSize, 1, 1)
        VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, histogram)
        VK10.vkCmdDispatch(cmd, pc.tileCount, 1, 1)
        VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, binBase)
        VK10.vkCmdDispatch(cmd, pc.segmentCount, 1, 1)
        VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, scatter)
        VK10.vkCmdDispatch(cmd, pc.tileCount, 1, 1)
    }
}
