using System.Runtime.InteropServices;

[StructLayout(LayoutKind.Sequential)]
struct OneSweepRadixSortControl {
    public uint Count, Shift, TileCount, Descending;
    public uint HasValues, SegmentCount, Segmented, ScanRingTileCount;
    public uint ScanRingTileMask, ScanLookbackLimit;
}

static void RecordDx12Contract(ID3D12GraphicsCommandList cmd, OneSweepRadixSortControl pc) {
    // Create pipelines for one selected workgroup size: 64, 128, 256, or 512.
    for (uint pass = 0; pass < 4; pass++) {
        pc.Shift = pass * 8;
        // Bind descriptor table containing storage buffers 0..11.
        // Set root constants, dispatch clear/histogram/bin-base/scatter, and insert UAV barriers between kernels.
    }
}
