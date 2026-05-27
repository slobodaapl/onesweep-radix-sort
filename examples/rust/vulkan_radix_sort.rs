use vulkano::buffer::Buffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};

#[repr(C)]
#[derive(Clone, Copy)]
struct OneSweepRadixSortControl {
    count: u32,
    shift: u32,
    tile_count: u32,
    descending: u32,
    has_values: u32,
    segment_count: u32,
    segmented: u32,
    scan_ring_tile_count: u32,
    scan_ring_tile_mask: u32,
    scan_lookback_limit: u32,
}

fn descriptor_writes(
    in_keys: Buffer,
    in_values: Buffer,
    out_keys: Buffer,
    out_values: Buffer,
    tile_counts: Buffer,
    global_counts: Buffer,
    bin_bases: Buffer,
    scan_status: Buffer,
    scan_prefix: Buffer,
    segment_offsets: Buffer,
    segment_tile_offsets: Buffer,
    scan_tail: Buffer,
) -> [WriteDescriptorSet; 12] {
    [
        WriteDescriptorSet::buffer(0, in_keys),
        WriteDescriptorSet::buffer(1, in_values),
        WriteDescriptorSet::buffer(2, out_keys),
        WriteDescriptorSet::buffer(3, out_values),
        WriteDescriptorSet::buffer(4, tile_counts),
        WriteDescriptorSet::buffer(5, global_counts),
        WriteDescriptorSet::buffer(6, bin_bases),
        WriteDescriptorSet::buffer(7, scan_status),
        WriteDescriptorSet::buffer(8, scan_prefix),
        WriteDescriptorSet::buffer(9, segment_offsets),
        WriteDescriptorSet::buffer(10, segment_tile_offsets),
        WriteDescriptorSet::buffer(11, scan_tail),
    ]
}

fn record_vulkano_contract<L>(
    builder: &mut AutoCommandBufferBuilder<L>,
    descriptor_set: DescriptorSet,
    clear: ComputePipeline,
    histogram: ComputePipeline,
    bin_base: ComputePipeline,
    scatter: ComputePipeline,
    workgroup_size: u32,
    mut pc: OneSweepRadixSortControl,
) {
    // workgroup_size must match the selected Vulkano pipeline variant: 64, 128, 256, or 512.
    let layout = clear.layout().clone();
    for pass in 0..4 {
        pc.shift = pass * 8;
        builder
            .bind_pipeline_compute(clear.clone())
            .bind_descriptor_sets(PipelineBindPoint::Compute, layout.clone(), 0, descriptor_set.clone())
            .push_constants(layout.clone(), 0, pc)
            .dispatch([((pc.scan_ring_tile_count.max(pc.segment_count) * 256) + workgroup_size - 1) / workgroup_size, 1, 1])
            .unwrap();
        // Vulkano inserts required barriers from resource usage; explicit barriers may be needed
        // if using lower-level synchronization APIs or swapping descriptor ownership manually.
        builder.bind_pipeline_compute(histogram.clone()).dispatch([pc.tile_count, 1, 1]).unwrap();
        builder.bind_pipeline_compute(bin_base.clone()).dispatch([pc.segment_count, 1, 1]).unwrap();
        builder.bind_pipeline_compute(scatter.clone()).dispatch([pc.tile_count, 1, 1]).unwrap();
        // Swap input/output descriptor sets before the next pass.
    }
}
