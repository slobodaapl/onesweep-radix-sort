#include "common.hlsli"

[numthreads(ONESWEEP_WORKGROUP_SIZE_X, ONESWEEP_WORKGROUP_SIZE_Y, 1)]
void main(uint3 group_id : SV_GroupID,
          uint group_index : SV_GroupIndex) {
  uint id = group_id.x * ONESWEEP_WORKGROUP_SIZE + group_index;
  uint scan_state_count = pc.scan_ring_tile_count * ONESWEEP_RADIX;
  uint segment_state_count = pc.segment_count * ONESWEEP_RADIX;
  uint clear_state_count = max(scan_state_count, segment_state_count);

  if (id >= clear_state_count) {
    return;
  }

  if (id < scan_state_count) {
    scan_status[id] = 0u;
  }

  if (id < segment_state_count) {
    global_counts[id] = 0u;
  }

  if (id == 0u) {
    scan_tail[0] = 0u;
  }
}
