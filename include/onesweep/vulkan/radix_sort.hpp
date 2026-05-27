#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <filesystem>
#include <string>

namespace onesweep::vulkan {

struct RadixSortCreateInfo {
  VkDevice device = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  const VkAllocationCallbacks* allocator = nullptr;

  // The supplied device must enable VK_NV_shader_subgroup_partitioned or
  // VK_EXT_shader_subgroup_partitioned. The wrapper can check physical-device
  // support, but Vulkan does not expose a direct query for enabled extensions
  // on an already-created VkDevice.

  // Directory containing compiled SPIR-V files:
  //   wg64/radix_clear.comp.spv
  //   wg128/radix_clear.comp.spv
  //   wg256/radix_clear.comp.spv
  //   wg256/radix_histogram.comp.spv
  //   ...
  //   wg512/radix_scatter_lookback.comp.spv
  std::filesystem::path spirv_directory;

  // Maximum number of keys this sorter may process without reallocating.
  // ensure_capacity may grow later if needed.
  uint32_t initial_capacity = 0;
  uint32_t initial_segments = 1;
  uint32_t initial_tiles = 0;

  // 0 selects the wrapper's backend heuristic. Supported explicit values are
  // 64, 128, 256, and 512.
  uint32_t preferred_workgroup_size = 0;

  // Experimental layout override for local tuning. Leave empty for the normal
  // 1D variant selected from preferred_workgroup_size.
  std::string preferred_workgroup_layout;
};

struct RadixSortDesc {
  VkBuffer keys = VK_NULL_HANDLE;
  VkDeviceSize key_offset = 0;

  // Optional values carried with keys. Values are uint32_t.
  // If values is VK_NULL_HANDLE, key-only sorting is performed.
  VkBuffer values = VK_NULL_HANDLE;
  VkDeviceSize value_offset = 0;

  uint32_t count = 0;
  bool descending = false;
};

struct SegmentedRadixSortDesc {
  VkBuffer keys = VK_NULL_HANDLE;
  VkDeviceSize key_offset = 0;

  VkBuffer values = VK_NULL_HANDLE;
  VkDeviceSize value_offset = 0;

  // uint32_t segment item offsets, length segment_count + 1.
  VkBuffer segment_offsets = VK_NULL_HANDLE;
  VkDeviceSize segment_offsets_offset = 0;

  // uint32_t segment tile offsets, length segment_count + 1. Each entry is the
  // prefix sum of ceil((segment_offsets[i + 1] - segment_offsets[i]) / workgroup_size()).
  VkBuffer segment_tile_offsets = VK_NULL_HANDLE;
  VkDeviceSize segment_tile_offsets_offset = 0;

  uint32_t count = 0;
  uint32_t segment_count = 0;
  uint32_t tile_count = 0;
  bool descending = false;
};

class RadixSort {
 public:
  static constexpr uint32_t kRadixBits = 8;
  static constexpr uint32_t kRadix = 1u << kRadixBits;
  static constexpr uint32_t kMinWorkgroupSize = 64;
  static constexpr uint32_t kMaxWorkgroupSize = 512;
  static constexpr uint32_t kDefaultWorkgroupSize = 256;
  static constexpr uint32_t kMaxSubgroups = kMaxWorkgroupSize / 32;
  static constexpr uint32_t kScanRingTiles = 4096;
  static constexpr uint32_t kPasses32 = 32 / kRadixBits;
  static constexpr uint32_t kProfileTimestampCount = 1 + kPasses32 * 4;

  explicit RadixSort(const RadixSortCreateInfo& info);
  ~RadixSort();

  RadixSort(const RadixSort&) = delete;
  RadixSort& operator=(const RadixSort&) = delete;

  RadixSort(RadixSort&&) = delete;
  RadixSort& operator=(RadixSort&&) = delete;

  void ensure_capacity(uint32_t max_items);

  // Records commands into cmd. The command buffer must be in recording state.
  // The input key buffer must have STORAGE_BUFFER, TRANSFER_SRC, and TRANSFER_DST usage if
  // an odd number of passes is configured. With the default 8-bit radix there are 4 passes,
  // so the final result lands back in the original key buffer without an extra copy.
  // The key format is uint32_t. Values, when provided, are uint32_t.
  void sort(VkCommandBuffer cmd, const RadixSortDesc& desc);
  void sort_profiled(VkCommandBuffer cmd,
                     const RadixSortDesc& desc,
                     VkQueryPool query_pool,
                     uint32_t first_query = 0);

  // Sorts each contiguous segment independently. Offsets buffers are uint32_t arrays.
  void sort_segmented(VkCommandBuffer cmd, const SegmentedRadixSortDesc& desc);

  [[nodiscard]] uint32_t capacity() const { return capacity_; }
  [[nodiscard]] uint32_t tile_count() const { return tile_count_; }
  [[nodiscard]] uint32_t workgroup_size() const { return workgroup_size_; }
  [[nodiscard]] const std::string& workgroup_layout() const { return workgroup_layout_; }

  [[nodiscard]] VkBuffer temp_keys_buffer() const { return temp_keys_.buffer; }
  [[nodiscard]] VkBuffer temp_values_buffer() const { return temp_values_.buffer; }
  [[nodiscard]] VkBuffer tile_counts_buffer() const { return tile_counts_.buffer; }
  [[nodiscard]] VkBuffer scan_status_buffer() const { return scan_status_.buffer; }
  [[nodiscard]] VkBuffer scan_prefix_buffer() const { return scan_prefix_.buffer; }
  [[nodiscard]] VkBuffer scan_tail_buffer() const { return scan_tail_.buffer; }

 private:
  struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
  };

  struct PushConstants {
    uint32_t count = 0;
    uint32_t shift = 0;
    uint32_t tile_count = 0;
    uint32_t descending = 0;
    uint32_t has_values = 0;
    uint32_t segment_count = 1;
    uint32_t segmented = 0;
    uint32_t scan_ring_tile_count = 1;
    uint32_t scan_ring_tile_mask = 0;
    uint32_t scan_lookback_limit = 1;
  };

  void create_static_objects();
  void destroy_static_objects();
  void destroy_scratch();
  void ensure_capacity(uint32_t max_items, uint32_t max_segments, uint32_t max_tiles);

  [[nodiscard]] Buffer create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                     VkMemoryPropertyFlags memory_properties) const;
  void destroy_buffer(Buffer& buffer) const;
  [[nodiscard]] uint32_t find_memory_type(uint32_t type_filter,
                                          VkMemoryPropertyFlags properties) const;

  [[nodiscard]] VkShaderModule load_shader_module(const std::filesystem::path& path) const;
  [[nodiscard]] VkPipeline create_compute_pipeline(const std::filesystem::path& spv_file) const;

  VkDescriptorSet allocate_and_update_descriptors(
      VkBuffer in_keys, VkDeviceSize in_key_offset,
      VkBuffer in_values, VkDeviceSize in_value_offset,
      VkBuffer out_keys, VkDeviceSize out_key_offset,
      VkBuffer out_values, VkDeviceSize out_value_offset,
      bool has_values,
      VkBuffer segment_offsets, VkDeviceSize segment_offsets_offset,
      VkBuffer segment_tile_offsets, VkDeviceSize segment_tile_offsets_offset);

  void record_sort(VkCommandBuffer cmd, const RadixSortDesc& desc,
                   VkBuffer segment_offsets, VkDeviceSize segment_offsets_offset,
                   VkBuffer segment_tile_offsets, VkDeviceSize segment_tile_offsets_offset,
                   uint32_t segment_count, uint32_t live_tile_count, bool segmented,
                   VkQueryPool profile_query_pool = VK_NULL_HANDLE,
                   uint32_t first_profile_query = 0);

  void compute_barrier(VkCommandBuffer cmd) const;
  void transfer_to_compute_barrier(VkCommandBuffer cmd) const;
  void compute_to_transfer_barrier(VkCommandBuffer cmd) const;

  VkDevice device_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  const VkAllocationCallbacks* allocator_ = nullptr;
  std::filesystem::path spirv_directory_;

  uint32_t capacity_ = 0;
  uint32_t tile_count_ = 0;
  uint32_t segment_capacity_ = 0;
  uint32_t workgroup_size_ = kDefaultWorkgroupSize;
  uint32_t max_subgroups_ = kDefaultWorkgroupSize / 32;
  std::string workgroup_layout_;

  VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;

  VkPipeline clear_pipeline_ = VK_NULL_HANDLE;
  VkPipeline histogram_pipeline_ = VK_NULL_HANDLE;
  VkPipeline flat_histogram_pipeline_ = VK_NULL_HANDLE;
  VkPipeline bin_base_pipeline_ = VK_NULL_HANDLE;
  VkPipeline scatter_pipeline_ = VK_NULL_HANDLE;
  VkPipeline flat_scatter_pipeline_ = VK_NULL_HANDLE;

  Buffer temp_keys_;
  Buffer temp_values_;
  Buffer dummy_values_;
  Buffer tile_counts_;
  Buffer global_counts_;
  Buffer bin_bases_;
  Buffer scan_status_;
  Buffer scan_prefix_;
  Buffer scan_tail_;
  Buffer default_segment_offsets_;
  Buffer default_segment_tile_offsets_;
};

}  // namespace onesweep::vulkan
