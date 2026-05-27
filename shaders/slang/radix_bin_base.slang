#include "common.slang"

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID,
          uint3 group_thread_id : SV_GroupThreadID) {
  uint segment = group_id.x;
  uint lid = group_thread_id.x;

  if (lid == 0u) {
    uint acc = 0u;
    for (uint d = 0u; d < ONESWEEP_RADIX; ++d) {
      uint idx = segment_digit_index(segment, d);
      uint c = global_counts[idx];
      bin_bases[idx] = acc;
      acc += c;
    }
  }
}
