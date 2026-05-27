#include "common.hlsli"

groupshared uint wave_totals[ONESWEEP_MAX_SUBGROUPS];

[numthreads(ONESWEEP_WORKGROUP_SIZE_X, ONESWEEP_WORKGROUP_SIZE_Y, 1)]
void main(uint3 group_id : SV_GroupID,
          uint3 group_thread_id : SV_GroupThreadID,
          uint group_index : SV_GroupIndex) {
  uint segment = group_id.x;
  uint lid = group_index;
#if ONESWEEP_WORKGROUP_SIZE >= 256
  uint idx = segment_digit_index(segment, lid);
  uint count = lid < ONESWEEP_RADIX ? global_counts[idx] : 0u;

  uint subgroup_id = subgroup_index_in_group(group_index);
  uint lane = WaveGetLaneIndex();
  uint wave_count = WaveGetLaneCount();
  uint wave_prefix = WavePrefixSum(count);
  if (lane == wave_count - 1u) {
    wave_totals[subgroup_id] = wave_prefix + count;
  }

  GroupMemoryBarrierWithGroupSync();

  uint wave_base = 0u;
  for (uint sg = 0u; sg < subgroup_id; ++sg) {
    wave_base += wave_totals[sg];
  }

  if (lid < ONESWEEP_RADIX) {
    bin_bases[idx] = wave_base + wave_prefix;
  }
#else
  if (lid == 0u) {
    uint prefix = 0u;
    for (uint digit = 0u; digit < ONESWEEP_RADIX; ++digit) {
      uint idx = segment_digit_index(segment, digit);
      uint count = global_counts[idx];
      bin_bases[idx] = prefix;
      prefix += count;
    }
  }
#endif
}
