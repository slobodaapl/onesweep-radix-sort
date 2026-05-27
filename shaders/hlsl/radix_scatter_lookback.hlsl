#include "common.hlsli"

groupshared uint subgroup_counts[ONESWEEP_MAX_SUBGROUPS * ONESWEEP_RADIX];
groupshared uint tile_prefix[ONESWEEP_RADIX];

uint load_scan_status(uint index) {
  return scan_status[index];
}

uint load_scan_tail() {
  return scan_tail[0];
}

uint lookback_prefix(uint segment_tile_base, uint tile, uint digit) {
  uint acc = 0u;
  uint t = tile;
  uint ring_floor = tile > scan_lookback_limit() ? tile - scan_lookback_limit() : 0u;
  uint ring_stop = max(segment_tile_base, ring_floor);

  while (t > ring_stop) {
    --t;
    uint idx = tile_digit_index(t, digit);
    uint ring_idx = scan_ring_digit_index(t, digit);
    uint status = load_scan_status(ring_idx);

    if (status == t + 1u) {
      DeviceMemoryBarrier();
      acc += scan_prefix[ring_idx] + tile_counts[idx];
      return acc;
    }

    acc += tile_counts[idx];
  }

  while (t > segment_tile_base) {
    --t;
    acc += tile_counts[tile_digit_index(t, digit)];
  }

  return acc;
}

void wait_for_ring_slot(uint tile) {
  if (tile < pc.scan_ring_tile_count) {
    return;
  }

  uint lookback_limit = scan_lookback_limit();
  while ((load_scan_tail() & ONESWEEP_TAIL_MASK) + pc.scan_ring_tile_count <=
         tile + lookback_limit) {
  }
}

void publish_tail(uint tile) {
  while ((load_scan_tail() & ONESWEEP_TAIL_MASK) != (tile & ONESWEEP_TAIL_MASK)) {
  }
  InterlockedAdd(scan_tail[0], 1u);
}

[numthreads(ONESWEEP_WORKGROUP_SIZE_X, ONESWEEP_WORKGROUP_SIZE_Y, 1)]
void main(uint3 group_id : SV_GroupID,
          uint3 group_thread_id : SV_GroupThreadID,
          uint group_index : SV_GroupIndex) {
  uint tile = group_id.x;
  uint lid = group_index;
  uint segment = 0u;
  uint segment_start = 0u;
  uint segment_end = pc.count;
  uint segment_tile_base = 0u;
  uint global_index = tile * ONESWEEP_WORKGROUP_SIZE + lid;

#ifndef ONESWEEP_FLAT_ONLY
  if (pc.segmented != 0u) {
    segment = segment_for_tile(tile);
    segment_start = segment_item_begin(segment);
    segment_end = segment_item_end(segment);
    segment_tile_base = segment_tile_begin(segment);
    uint local_tile = tile - segment_tile_base;
    global_index = segment_start + local_tile * ONESWEEP_WORKGROUP_SIZE + lid;
  }
#endif
  bool valid = global_index < segment_end;

  uint raw_key = valid ? in_keys[global_index] : 0u;
  uint digit = valid ? digit_of(raw_key) : 0u;

  if (lid == 0u) {
    wait_for_ring_slot(tile);
  }

  GroupMemoryBarrierWithGroupSync();

  for (uint i = lid; i < ONESWEEP_MAX_SUBGROUPS * ONESWEEP_RADIX; i += ONESWEEP_WORKGROUP_SIZE) {
    subgroup_counts[i] = 0u;
  }

  GroupMemoryBarrierWithGroupSync();

  uint subgroup_id = subgroup_index_in_group(group_index);
  uint lane = WaveGetLaneIndex();
  uint4 match_mask = WaveMatch(digit);
  uint4 valid_mask = WaveActiveBallot(valid);
  match_mask.x &= valid_mask.x;
  match_mask.y &= valid_mask.y;
  match_mask.z &= valid_mask.z;
  match_mask.w &= valid_mask.w;

  uint rank_in_subgroup = wave_mask_count_before_lane(match_mask, lane);
  if (valid && rank_in_subgroup == 0u) {
    subgroup_counts[subgroup_id * ONESWEEP_RADIX + digit] = wave_mask_count(match_mask);
  }

  GroupMemoryBarrierWithGroupSync();

  uint subgroup_prefix = 0u;
  if (valid) {
    for (uint sg = 0u; sg < subgroup_id; ++sg) {
      subgroup_prefix += subgroup_counts[sg * ONESWEEP_RADIX + digit];
    }
  }
  uint local_rank = subgroup_prefix + rank_in_subgroup;

  for (uint i = lid; i < ONESWEEP_RADIX; i += ONESWEEP_WORKGROUP_SIZE) {
    uint prefix = lookback_prefix(segment_tile_base, tile, i);
    uint ring_idx = scan_ring_digit_index(tile, i);
    tile_prefix[i] = prefix;
    scan_prefix[ring_idx] = prefix;
    DeviceMemoryBarrier();
    scan_status[ring_idx] = tile + 1u;
  }

  GroupMemoryBarrierWithGroupSync();

  if (valid) {
    uint dst = segment_start + bin_bases[segment_digit_index(segment, digit)] + tile_prefix[digit] + local_rank;
    out_keys[dst] = raw_key;
    if (pc.has_values != 0u) {
      out_values[dst] = in_values[global_index];
    }
  }

  GroupMemoryBarrierWithGroupSync();

  if (lid == 0u) {
    publish_tail(tile);
  }
}
