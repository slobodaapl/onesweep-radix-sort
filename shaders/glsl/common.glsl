#ifndef ONESWEEP_COMMON_GLSL
#define ONESWEEP_COMMON_GLSL

const uint ONESWEEP_RADIX_BITS = 8u;
const uint ONESWEEP_RADIX = 1u << ONESWEEP_RADIX_BITS;
const uint ONESWEEP_RADIX_MASK = ONESWEEP_RADIX - 1u;
const uint ONESWEEP_WORKGROUP_SIZE = 256u;
const uint ONESWEEP_MAX_SUBGROUPS = 8u;
const uint ONESWEEP_TAIL_BITS = 4u;
const uint ONESWEEP_TAIL_GROUP_SIZE = 1u << ONESWEEP_TAIL_BITS;
const uint ONESWEEP_TAIL_MASK = 0xfffffff0u;

layout(push_constant) uniform PushConstants {
  uint count;
  uint shift;
  uint tile_count;
  uint descending;
  uint has_values;
  uint segment_count;
  uint segmented;
  uint scan_ring_tile_count;
} pc;

layout(set = 0, binding = 0, std430) readonly buffer InKeys {
  uint in_keys[];
};

layout(set = 0, binding = 1, std430) readonly buffer InValues {
  uint in_values[];
};

layout(set = 0, binding = 2, std430) writeonly buffer OutKeys {
  uint out_keys[];
};

layout(set = 0, binding = 3, std430) writeonly buffer OutValues {
  uint out_values[];
};

layout(set = 0, binding = 4, std430) coherent buffer TileCounts {
  uint tile_counts[];
};

layout(set = 0, binding = 5, std430) coherent buffer GlobalCounts {
  uint global_counts[];
};

layout(set = 0, binding = 6, std430) coherent buffer BinBases {
  uint bin_bases[];
};

layout(set = 0, binding = 7, std430) coherent buffer ScanStatus {
  uint scan_status[];
};

layout(set = 0, binding = 8, std430) coherent buffer ScanPrefix {
  uint scan_prefix[];
};

layout(set = 0, binding = 9, std430) readonly buffer SegmentOffsets {
  uint segment_offsets[];
};

layout(set = 0, binding = 10, std430) readonly buffer SegmentTileOffsets {
  uint segment_tile_offsets[];
};

layout(set = 0, binding = 11, std430) coherent buffer ScanTail {
  uint scan_tail[];
};

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
