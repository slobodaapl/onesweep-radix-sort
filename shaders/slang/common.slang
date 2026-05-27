#ifndef ONESWEEP_COMMON_HLSLI
#define ONESWEEP_COMMON_HLSLI

static const uint ONESWEEP_RADIX_BITS = 8u;
static const uint ONESWEEP_RADIX = 1u << ONESWEEP_RADIX_BITS;
static const uint ONESWEEP_RADIX_MASK = ONESWEEP_RADIX - 1u;
static const uint ONESWEEP_WORKGROUP_SIZE = 256u;
static const uint ONESWEEP_MAX_SUBGROUPS = 8u;
static const uint ONESWEEP_TAIL_BITS = 4u;
static const uint ONESWEEP_TAIL_GROUP_SIZE = 1u << ONESWEEP_TAIL_BITS;
static const uint ONESWEEP_TAIL_MASK = 0xfffffff0u;

struct PushConstants {
  uint count;
  uint shift;
  uint tile_count;
  uint descending;
  uint has_values;
  uint segment_count;
  uint segmented;
  uint scan_ring_tile_count;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<uint> in_keys;
[[vk::binding(1, 0)]] StructuredBuffer<uint> in_values;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> out_keys;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> out_values;
[[vk::binding(4, 0)]] globallycoherent RWStructuredBuffer<uint> tile_counts;
[[vk::binding(5, 0)]] globallycoherent RWStructuredBuffer<uint> global_counts;
[[vk::binding(6, 0)]] globallycoherent RWStructuredBuffer<uint> bin_bases;
[[vk::binding(7, 0)]] globallycoherent RWStructuredBuffer<uint> scan_status;
[[vk::binding(8, 0)]] globallycoherent RWStructuredBuffer<uint> scan_prefix;
[[vk::binding(9, 0)]] StructuredBuffer<uint> segment_offsets;
[[vk::binding(10, 0)]] StructuredBuffer<uint> segment_tile_offsets;
[[vk::binding(11, 0)]] globallycoherent RWStructuredBuffer<uint> scan_tail;

uint effective_key(uint raw_key) {
  return pc.descending != 0u ? ~raw_key : raw_key;
}

uint digit_of(uint raw_key) {
  return (effective_key(raw_key) >> pc.shift) & ONESWEEP_RADIX_MASK;
}

uint tile_digit_index(uint tile, uint digit) {
  return tile * ONESWEEP_RADIX + digit;
}

uint segment_digit_index(uint segment, uint digit) {
  return segment * ONESWEEP_RADIX + digit;
}

uint scan_ring_digit_index(uint tile, uint digit) {
  return (tile % pc.scan_ring_tile_count) * ONESWEEP_RADIX + digit;
}

uint scan_lookback_limit() {
  return max(1u, pc.scan_ring_tile_count >> 1u);
}

uint segment_for_tile(uint tile) {
  if (pc.segmented == 0u) {
    return 0u;
  }

  uint lo = 0u;
  uint hi = pc.segment_count;
  while (lo + 1u < hi) {
    uint mid = (lo + hi) >> 1u;
    if (segment_tile_offsets[mid] <= tile) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

uint segment_item_begin(uint segment) {
  return pc.segmented != 0u ? segment_offsets[segment] : 0u;
}

uint segment_item_end(uint segment) {
  return pc.segmented != 0u ? segment_offsets[segment + 1u] : pc.count;
}

uint segment_tile_begin(uint segment) {
  return pc.segmented != 0u ? segment_tile_offsets[segment] : 0u;
}

#endif
