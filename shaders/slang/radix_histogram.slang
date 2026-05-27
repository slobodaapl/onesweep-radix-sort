#include "common.slang"

groupshared uint hist[ONESWEEP_RADIX];

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID,
          uint3 group_thread_id : SV_GroupThreadID,
          uint group_index : SV_GroupIndex) {
  uint tile = group_id.x;
  uint lid = group_thread_id.x;
  uint segment = segment_for_tile(tile);
  uint local_tile = tile - segment_tile_begin(segment);
  uint global_index = segment_item_begin(segment) + local_tile * ONESWEEP_WORKGROUP_SIZE + lid;
  uint segment_end = segment_item_end(segment);

  if (lid < ONESWEEP_RADIX) {
    hist[lid] = 0u;
  }
  GroupMemoryBarrierWithGroupSync();

  if (global_index < segment_end) {
    uint d = digit_of(in_keys[global_index]);
    InterlockedAdd(hist[d], 1u);
  }
  GroupMemoryBarrierWithGroupSync();

  if (lid < ONESWEEP_RADIX) {
    uint c = hist[lid];
    tile_counts[tile_digit_index(tile, lid)] = c;
    if (c != 0u) {
      InterlockedAdd(global_counts[segment_digit_index(segment, lid)], c);
    }
  }
}
