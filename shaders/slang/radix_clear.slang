#include "common.slang"

[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
  uint id = dispatch_thread_id.x;
  uint tile_state_count = pc.tile_count * ONESWEEP_RADIX;
  uint scan_state_count = pc.scan_ring_tile_count * ONESWEEP_RADIX;
  uint segment_state_count = pc.segment_count * ONESWEEP_RADIX;

  if (id < tile_state_count) {
    tile_counts[id] = 0u;
  }

  if (id < scan_state_count) {
    scan_status[id] = 0u;
    scan_prefix[id] = 0u;
  }

  if (id < segment_state_count) {
    global_counts[id] = 0u;
    bin_bases[id] = 0u;
  }

  if (id == 0u) {
    scan_tail[0] = 0u;
  }
}
