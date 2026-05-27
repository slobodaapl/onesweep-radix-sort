#include "common.hlsli"

groupshared uint hist[ONESWEEP_RADIX];

[numthreads(ONESWEEP_WORKGROUP_SIZE_X, ONESWEEP_WORKGROUP_SIZE_Y, 1)]
void main(uint3 group_id : SV_GroupID,
          uint3 group_thread_id : SV_GroupThreadID,
          uint group_index : SV_GroupIndex) {
  uint tile = group_id.x;
  uint lid = group_index;
  uint segment = 0u;
  uint global_index = tile * ONESWEEP_WORKGROUP_SIZE + lid;
  uint segment_end = pc.count;

#ifndef ONESWEEP_FLAT_ONLY
  if (pc.segmented != 0u) {
    segment = segment_for_tile(tile);
    uint local_tile = tile - segment_tile_begin(segment);
    global_index = segment_item_begin(segment) + local_tile * ONESWEEP_WORKGROUP_SIZE + lid;
    segment_end = segment_item_end(segment);
  }
#endif

  for (uint i = lid; i < ONESWEEP_RADIX; i += ONESWEEP_WORKGROUP_SIZE) {
    hist[i] = 0u;
  }
  GroupMemoryBarrierWithGroupSync();

  uint digit = 0u;
  bool valid = global_index < segment_end;
  if (valid) {
    digit = digit_of(in_keys[global_index]);
  }

  uint4 match_mask = WaveMatch(digit);
  uint4 valid_mask = WaveActiveBallot(valid);
  match_mask.x &= valid_mask.x;
  match_mask.y &= valid_mask.y;
  match_mask.z &= valid_mask.z;
  match_mask.w &= valid_mask.w;

  if (valid && WaveGetLaneIndex() == wave_mask_first_lane(match_mask)) {
    InterlockedAdd(hist[digit], wave_mask_count(match_mask));
  }
  GroupMemoryBarrierWithGroupSync();

  for (uint i = lid; i < ONESWEEP_RADIX; i += ONESWEEP_WORKGROUP_SIZE) {
    uint c = hist[i];
    tile_counts[tile_digit_index(tile, i)] = c;
    if (c != 0u) {
      InterlockedAdd(global_counts[segment_digit_index(segment, i)], c);
    }
  }
}
