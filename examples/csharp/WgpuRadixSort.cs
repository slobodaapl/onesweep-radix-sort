using System.Runtime.InteropServices;

[StructLayout(LayoutKind.Sequential)]
struct OneSweepRadixSortControl {
    public uint Count, Shift, TileCount, Descending;
    public uint HasValues, SegmentCount, Segmented, ScanRingTileCount;
    public uint ScanRingTileMask, ScanLookbackLimit;
}

static void RecordWgpuContract(WGPUComputePassEncoder pass, WGPUBindGroup bindGroup, WGPUComputePipeline clear, WGPUComputePipeline histogram, WGPUComputePipeline binBase, WGPUComputePipeline scatter, OneSweepRadixSortControl pc) {
    // Create pipelines for one selected workgroup size: 64, 128, 256, or 512.
    wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup, 0, null);
    for (uint radixPass = 0; radixPass < 4; radixPass++) {
        pc.Shift = radixPass * 8;
        // Upload pc, then dispatch clear, histogram, bin-base, scatter in order.
    }
}
