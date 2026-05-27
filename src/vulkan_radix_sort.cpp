#include "onesweep/vulkan/radix_sort.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace onesweep::vulkan {
namespace {

constexpr uint32_t ceil_div_u32(uint32_t x, uint32_t y) {
  return (x + y - 1u) / y;
}

uint32_t ceil_power_of_two_u32(uint32_t x) {
  if (x <= 1u) {
    return 1u;
  }
  --x;
  x |= x >> 1u;
  x |= x >> 2u;
  x |= x >> 4u;
  x |= x >> 8u;
  x |= x >> 16u;
  return x + 1u;
}

bool valid_workgroup_size(uint32_t workgroup_size) {
  return workgroup_size == 64u ||
         workgroup_size == 128u ||
         workgroup_size == RadixSort::kDefaultWorkgroupSize ||
         workgroup_size == RadixSort::kMaxWorkgroupSize;
}

bool valid_workgroup_layout(const std::string& layout) {
  return layout == "wg512x32";
}

uint32_t layout_workgroup_size(const std::string& layout) {
  return RadixSort::kMaxWorkgroupSize;
}

void vk_check(VkResult result, const char* what) {
  if (result != VK_SUCCESS) {
    throw std::runtime_error(std::string(what) + " failed with VkResult " + std::to_string(result));
  }
}

std::vector<char> read_binary_file(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::ate | std::ios::binary);
  if (!file) {
    throw std::runtime_error("could not open SPIR-V file: " + path.string());
  }

  const auto size = file.tellg();
  if (size <= 0 || (static_cast<std::size_t>(size) % sizeof(uint32_t)) != 0) {
    throw std::runtime_error("invalid SPIR-V file size: " + path.string());
  }

  std::vector<char> bytes(static_cast<std::size_t>(size));
  file.seekg(0);
  file.read(bytes.data(), size);
  return bytes;
}

VkDescriptorSetLayoutBinding storage_binding(uint32_t binding) {
  VkDescriptorSetLayoutBinding b{};
  b.binding = binding;
  b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  b.descriptorCount = 1;
  b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  return b;
}

bool physical_device_has_extension(VkPhysicalDevice physical_device, const char* extension_name) {
  uint32_t count = 0;
  vk_check(vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, nullptr),
           "vkEnumerateDeviceExtensionProperties count");
  std::vector<VkExtensionProperties> extensions(count);
  vk_check(vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, extensions.data()),
           "vkEnumerateDeviceExtensionProperties");
  return std::any_of(extensions.begin(), extensions.end(), [extension_name](const auto& extension) {
    return std::strcmp(extension.extensionName, extension_name) == 0;
  });
}

}  // namespace

RadixSort::RadixSort(const RadixSortCreateInfo& info)
    : device_(info.device),
      physical_device_(info.physical_device),
      allocator_(info.allocator),
      spirv_directory_(info.spirv_directory) {
  if (device_ == VK_NULL_HANDLE || physical_device_ == VK_NULL_HANDLE) {
    throw std::invalid_argument("RadixSort requires a valid VkDevice and VkPhysicalDevice");
  }

  VkPhysicalDeviceSubgroupProperties subgroup{};
  subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
  VkPhysicalDeviceProperties2 props2{};
  props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  props2.pNext = &subgroup;
  vkGetPhysicalDeviceProperties2(physical_device_, &props2);

  if ((subgroup.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) == 0 ||
      (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT) == 0 ||
      (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_PARTITIONED_BIT_EXT) == 0) {
    throw std::runtime_error("device does not expose required subgroup basic+ballot+partitioned operations");
  }
  if (!physical_device_has_extension(physical_device_, VK_NV_SHADER_SUBGROUP_PARTITIONED_EXTENSION_NAME) &&
      !physical_device_has_extension(physical_device_, VK_EXT_SHADER_SUBGROUP_PARTITIONED_EXTENSION_NAME)) {
    throw std::runtime_error("device does not expose VK_NV_shader_subgroup_partitioned or VK_EXT_shader_subgroup_partitioned");
  }
  if (subgroup.subgroupSize < 32u) {
    throw std::runtime_error("device subgroup size is too small for the 8-bit shader variant");
  }

  if (info.preferred_workgroup_size != 0u) {
    if (!valid_workgroup_size(info.preferred_workgroup_size)) {
      throw std::invalid_argument("preferred_workgroup_size must be 0, 64, 128, 256, or 512");
    }
    if (props2.properties.limits.maxComputeWorkGroupInvocations < info.preferred_workgroup_size) {
      throw std::runtime_error("device maxComputeWorkGroupInvocations is below preferred workgroup size");
    }
    workgroup_size_ = info.preferred_workgroup_size;
  } else if (props2.properties.limits.maxComputeWorkGroupInvocations >= kMaxWorkgroupSize) {
    workgroup_size_ = kMaxWorkgroupSize;
  } else if (props2.properties.limits.maxComputeWorkGroupInvocations >= kDefaultWorkgroupSize) {
    workgroup_size_ = kDefaultWorkgroupSize;
  } else if (props2.properties.limits.maxComputeWorkGroupInvocations >= 128u) {
    workgroup_size_ = 128u;
  } else if (props2.properties.limits.maxComputeWorkGroupInvocations >= kMinWorkgroupSize) {
    workgroup_size_ = 64u;
  } else {
    throw std::runtime_error("device maxComputeWorkGroupInvocations is below required workgroup size");
  }
  if (!info.preferred_workgroup_layout.empty()) {
    if (!valid_workgroup_layout(info.preferred_workgroup_layout)) {
      throw std::invalid_argument("unknown preferred_workgroup_layout");
    }
    const uint32_t layout_size = layout_workgroup_size(info.preferred_workgroup_layout);
    if (info.preferred_workgroup_size != 0u && info.preferred_workgroup_size != layout_size) {
      throw std::invalid_argument("preferred_workgroup_layout conflicts with preferred_workgroup_size");
    }
    if (props2.properties.limits.maxComputeWorkGroupInvocations < layout_size) {
      throw std::runtime_error("device maxComputeWorkGroupInvocations is below preferred workgroup layout size");
    }
    workgroup_size_ = layout_size;
    workgroup_layout_ = info.preferred_workgroup_layout;
  }
  max_subgroups_ = workgroup_size_ / 32u;

  create_static_objects();
  const uint32_t initial_capacity = info.initial_capacity == 0 ? workgroup_size_ : info.initial_capacity;
  const uint32_t initial_tiles = info.initial_tiles == 0
                                     ? ceil_div_u32(initial_capacity, workgroup_size_)
                                     : info.initial_tiles;
  ensure_capacity(initial_capacity, std::max(info.initial_segments, 1u), initial_tiles);
}

RadixSort::~RadixSort() {
  destroy_scratch();
  destroy_static_objects();
}

void RadixSort::create_static_objects() {
  std::array<VkDescriptorSetLayoutBinding, 12> bindings{};
  for (uint32_t i = 0; i < static_cast<uint32_t>(bindings.size()); ++i) {
    bindings[i] = storage_binding(i);
  }

  VkDescriptorSetLayoutCreateInfo set_info{};
  set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  set_info.bindingCount = static_cast<uint32_t>(bindings.size());
  set_info.pBindings = bindings.data();
  vk_check(vkCreateDescriptorSetLayout(device_, &set_info, allocator_, &descriptor_set_layout_),
           "vkCreateDescriptorSetLayout");

  VkPushConstantRange push_range{};
  push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push_range.offset = 0;
  push_range.size = sizeof(PushConstants);

  VkPipelineLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layout_info.setLayoutCount = 1;
  layout_info.pSetLayouts = &descriptor_set_layout_;
  layout_info.pushConstantRangeCount = 1;
  layout_info.pPushConstantRanges = &push_range;
  vk_check(vkCreatePipelineLayout(device_, &layout_info, allocator_, &pipeline_layout_),
           "vkCreatePipelineLayout");

  VkDescriptorPoolSize pool_size{};
  pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_size.descriptorCount = static_cast<uint32_t>(bindings.size()) * 4096u;

  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.maxSets = 4096u;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes = &pool_size;
  vk_check(vkCreateDescriptorPool(device_, &pool_info, allocator_, &descriptor_pool_),
           "vkCreateDescriptorPool");

  const std::filesystem::path variant_dir =
      spirv_directory_ / (workgroup_layout_.empty() ? ("wg" + std::to_string(workgroup_size_)) : workgroup_layout_);
  clear_pipeline_ = create_compute_pipeline(variant_dir / "radix_clear.comp.spv");
  histogram_pipeline_ = create_compute_pipeline(variant_dir / "radix_histogram.comp.spv");
  flat_histogram_pipeline_ = create_compute_pipeline(variant_dir / "radix_histogram_flat.comp.spv");
  bin_base_pipeline_ = create_compute_pipeline(variant_dir / "radix_bin_base.comp.spv");
  scatter_pipeline_ = create_compute_pipeline(variant_dir / "radix_scatter_lookback.comp.spv");
  flat_scatter_pipeline_ = create_compute_pipeline(variant_dir / "radix_scatter_lookback_flat.comp.spv");
}

void RadixSort::destroy_static_objects() {
  if (clear_pipeline_ != VK_NULL_HANDLE) {
    vkDestroyPipeline(device_, clear_pipeline_, allocator_);
    clear_pipeline_ = VK_NULL_HANDLE;
  }
  if (histogram_pipeline_ != VK_NULL_HANDLE) {
    vkDestroyPipeline(device_, histogram_pipeline_, allocator_);
    histogram_pipeline_ = VK_NULL_HANDLE;
  }
  if (flat_histogram_pipeline_ != VK_NULL_HANDLE) {
    vkDestroyPipeline(device_, flat_histogram_pipeline_, allocator_);
    flat_histogram_pipeline_ = VK_NULL_HANDLE;
  }
  if (bin_base_pipeline_ != VK_NULL_HANDLE) {
    vkDestroyPipeline(device_, bin_base_pipeline_, allocator_);
    bin_base_pipeline_ = VK_NULL_HANDLE;
  }
  if (scatter_pipeline_ != VK_NULL_HANDLE) {
    vkDestroyPipeline(device_, scatter_pipeline_, allocator_);
    scatter_pipeline_ = VK_NULL_HANDLE;
  }
  if (flat_scatter_pipeline_ != VK_NULL_HANDLE) {
    vkDestroyPipeline(device_, flat_scatter_pipeline_, allocator_);
    flat_scatter_pipeline_ = VK_NULL_HANDLE;
  }
  if (descriptor_pool_ != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(device_, descriptor_pool_, allocator_);
    descriptor_pool_ = VK_NULL_HANDLE;
  }
  if (pipeline_layout_ != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(device_, pipeline_layout_, allocator_);
    pipeline_layout_ = VK_NULL_HANDLE;
  }
  if (descriptor_set_layout_ != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(device_, descriptor_set_layout_, allocator_);
    descriptor_set_layout_ = VK_NULL_HANDLE;
  }
}

void RadixSort::destroy_scratch() {
  destroy_buffer(temp_keys_);
  destroy_buffer(temp_values_);
  destroy_buffer(dummy_values_);
  destroy_buffer(tile_counts_);
  destroy_buffer(global_counts_);
  destroy_buffer(bin_bases_);
  destroy_buffer(scan_status_);
  destroy_buffer(scan_prefix_);
  destroy_buffer(scan_tail_);
  destroy_buffer(default_segment_offsets_);
  destroy_buffer(default_segment_tile_offsets_);
  capacity_ = 0;
  tile_count_ = 0;
  segment_capacity_ = 0;
}

void RadixSort::ensure_capacity(uint32_t max_items) {
  ensure_capacity(max_items, 1, ceil_div_u32(std::max(max_items, workgroup_size_), workgroup_size_));
}

void RadixSort::ensure_capacity(uint32_t max_items, uint32_t max_segments, uint32_t max_tiles) {
  max_segments = std::max(max_segments, 1u);
  max_tiles = std::max(max_tiles, 1u);
  if (max_items <= capacity_ && max_segments <= segment_capacity_ && max_tiles <= tile_count_) {
    return;
  }

  vkDeviceWaitIdle(device_);
  destroy_scratch();

  capacity_ = std::max(max_items, workgroup_size_);
  tile_count_ = std::max(max_tiles, ceil_div_u32(capacity_, workgroup_size_));
  segment_capacity_ = max_segments;

  const VkDeviceSize key_bytes = static_cast<VkDeviceSize>(capacity_) * sizeof(uint32_t);
  const VkDeviceSize tile_digit_bytes =
      static_cast<VkDeviceSize>(tile_count_) * kRadix * sizeof(uint32_t);
  const VkDeviceSize scan_digit_bytes =
      static_cast<VkDeviceSize>(std::min(ceil_power_of_two_u32(tile_count_), kScanRingTiles)) *
      kRadix * sizeof(uint32_t);
  const VkDeviceSize segment_radix_bytes =
      static_cast<VkDeviceSize>(segment_capacity_) * kRadix * sizeof(uint32_t);

  constexpr VkBufferUsageFlags scratch_usage =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
      VK_BUFFER_USAGE_TRANSFER_DST_BIT;

  temp_keys_ = create_buffer(key_bytes, scratch_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  temp_values_ = create_buffer(key_bytes, scratch_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  dummy_values_ = create_buffer(sizeof(uint32_t), scratch_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  tile_counts_ = create_buffer(tile_digit_bytes, scratch_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  global_counts_ = create_buffer(segment_radix_bytes, scratch_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  bin_bases_ = create_buffer(segment_radix_bytes, scratch_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  scan_status_ = create_buffer(scan_digit_bytes, scratch_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  scan_prefix_ = create_buffer(scan_digit_bytes, scratch_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  scan_tail_ = create_buffer(sizeof(uint32_t), scratch_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  default_segment_offsets_ = create_buffer(2u * sizeof(uint32_t), scratch_usage,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  default_segment_tile_offsets_ = create_buffer(2u * sizeof(uint32_t), scratch_usage,
                                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

RadixSort::Buffer RadixSort::create_buffer(
    VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memory_properties) const {
  Buffer out{};
  out.size = size;

  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  vk_check(vkCreateBuffer(device_, &buffer_info, allocator_, &out.buffer), "vkCreateBuffer");

  VkMemoryRequirements req{};
  vkGetBufferMemoryRequirements(device_, out.buffer, &req);

  VkMemoryAllocateInfo alloc{};
  alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc.allocationSize = req.size;
  alloc.memoryTypeIndex = find_memory_type(req.memoryTypeBits, memory_properties);
  vk_check(vkAllocateMemory(device_, &alloc, allocator_, &out.memory), "vkAllocateMemory");
  vk_check(vkBindBufferMemory(device_, out.buffer, out.memory, 0), "vkBindBufferMemory");

  return out;
}

void RadixSort::destroy_buffer(Buffer& buffer) const {
  if (buffer.buffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(device_, buffer.buffer, allocator_);
    buffer.buffer = VK_NULL_HANDLE;
  }
  if (buffer.memory != VK_NULL_HANDLE) {
    vkFreeMemory(device_, buffer.memory, allocator_);
    buffer.memory = VK_NULL_HANDLE;
  }
  buffer.size = 0;
}

uint32_t RadixSort::find_memory_type(uint32_t type_filter,
                                           VkMemoryPropertyFlags properties) const {
  VkPhysicalDeviceMemoryProperties mem{};
  vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem);

  for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
    if ((type_filter & (1u << i)) != 0 &&
        (mem.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }

  throw std::runtime_error("no suitable Vulkan memory type found");
}

VkShaderModule RadixSort::load_shader_module(const std::filesystem::path& path) const {
  const std::vector<char> bytes = read_binary_file(path);

  VkShaderModuleCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  info.codeSize = bytes.size();
  info.pCode = reinterpret_cast<const uint32_t*>(bytes.data());

  VkShaderModule module = VK_NULL_HANDLE;
  vk_check(vkCreateShaderModule(device_, &info, allocator_, &module), "vkCreateShaderModule");
  return module;
}

VkPipeline RadixSort::create_compute_pipeline(const std::filesystem::path& spv_file) const {
  VkShaderModule module = load_shader_module(spv_file);

  VkPipelineShaderStageCreateInfo stage{};
  stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage.module = module;
  stage.pName = "main";

  VkComputePipelineCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  info.stage = stage;
  info.layout = pipeline_layout_;

  VkPipeline pipeline = VK_NULL_HANDLE;
  const VkResult result = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &info,
                                                   allocator_, &pipeline);
  vkDestroyShaderModule(device_, module, allocator_);
  vk_check(result, "vkCreateComputePipelines");
  return pipeline;
}

VkDescriptorSet RadixSort::allocate_and_update_descriptors(
    VkBuffer in_keys, VkDeviceSize in_key_offset,
    VkBuffer in_values, VkDeviceSize in_value_offset,
    VkBuffer out_keys, VkDeviceSize out_key_offset,
    VkBuffer out_values, VkDeviceSize out_value_offset,
    bool has_values,
    VkBuffer segment_offsets,
    VkDeviceSize segment_offsets_offset,
    VkBuffer segment_tile_offsets,
    VkDeviceSize segment_tile_offsets_offset) {
  VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
  VkDescriptorSetAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc_info.descriptorPool = descriptor_pool_;
  alloc_info.descriptorSetCount = 1;
  alloc_info.pSetLayouts = &descriptor_set_layout_;
  vk_check(vkAllocateDescriptorSets(device_, &alloc_info, &descriptor_set),
           "vkAllocateDescriptorSets");

  VkDescriptorBufferInfo in_keys_info{};
  in_keys_info.buffer = in_keys;
  in_keys_info.offset = in_key_offset;
  in_keys_info.range = VK_WHOLE_SIZE;

  VkDescriptorBufferInfo in_values_info{};
  in_values_info.buffer = has_values ? in_values : dummy_values_.buffer;
  in_values_info.offset = has_values ? in_value_offset : 0;
  in_values_info.range = VK_WHOLE_SIZE;

  VkDescriptorBufferInfo out_keys_info{};
  out_keys_info.buffer = out_keys;
  out_keys_info.offset = out_key_offset;
  out_keys_info.range = VK_WHOLE_SIZE;

  VkDescriptorBufferInfo out_values_info{};
  out_values_info.buffer = has_values ? out_values : dummy_values_.buffer;
  out_values_info.offset = has_values ? out_value_offset : 0;
  out_values_info.range = VK_WHOLE_SIZE;

  VkDescriptorBufferInfo tile_counts_info{};
  tile_counts_info.buffer = tile_counts_.buffer;
  tile_counts_info.range = tile_counts_.size;

  VkDescriptorBufferInfo global_counts_info{};
  global_counts_info.buffer = global_counts_.buffer;
  global_counts_info.range = global_counts_.size;

  VkDescriptorBufferInfo bin_bases_info{};
  bin_bases_info.buffer = bin_bases_.buffer;
  bin_bases_info.range = bin_bases_.size;

  VkDescriptorBufferInfo scan_status_info{};
  scan_status_info.buffer = scan_status_.buffer;
  scan_status_info.range = scan_status_.size;

  VkDescriptorBufferInfo scan_prefix_info{};
  scan_prefix_info.buffer = scan_prefix_.buffer;
  scan_prefix_info.range = scan_prefix_.size;

  VkDescriptorBufferInfo segment_offsets_info{};
  segment_offsets_info.buffer = segment_offsets;
  segment_offsets_info.offset = segment_offsets_offset;
  segment_offsets_info.range = VK_WHOLE_SIZE;

  VkDescriptorBufferInfo segment_tile_offsets_info{};
  segment_tile_offsets_info.buffer = segment_tile_offsets;
  segment_tile_offsets_info.offset = segment_tile_offsets_offset;
  segment_tile_offsets_info.range = VK_WHOLE_SIZE;

  VkDescriptorBufferInfo scan_tail_info{};
  scan_tail_info.buffer = scan_tail_.buffer;
  scan_tail_info.range = scan_tail_.size;

  std::array<VkDescriptorBufferInfo, 12> infos = {
      in_keys_info,      in_values_info,      out_keys_info,
      out_values_info,   tile_counts_info,    global_counts_info,
      bin_bases_info,    scan_status_info,    scan_prefix_info,
      segment_offsets_info, segment_tile_offsets_info, scan_tail_info,
  };

  std::array<VkWriteDescriptorSet, 12> writes{};
  for (uint32_t i = 0; i < static_cast<uint32_t>(writes.size()); ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = descriptor_set;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }

  vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  return descriptor_set;
}

void RadixSort::compute_barrier(VkCommandBuffer cmd) const {
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

  vkCmdPipelineBarrier(cmd,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0,
                       1,
                       &barrier,
                       0,
                       nullptr,
                       0,
                       nullptr);
}

void RadixSort::transfer_to_compute_barrier(VkCommandBuffer cmd) const {
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd,
                       VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0,
                       1,
                       &barrier,
                       0,
                       nullptr,
                       0,
                       nullptr);
}

void RadixSort::compute_to_transfer_barrier(VkCommandBuffer cmd) const {
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(cmd,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT,
                       0,
                       1,
                       &barrier,
                       0,
                       nullptr,
                       0,
                       nullptr);
}

void RadixSort::sort(VkCommandBuffer cmd, const RadixSortDesc& desc) {
  if (desc.count == 0) {
    return;
  }
  if (desc.keys == VK_NULL_HANDLE) {
    throw std::invalid_argument("RadixSortDesc::keys must be non-null");
  }

  const uint32_t live_tile_count = ceil_div_u32(desc.count, workgroup_size_);
  ensure_capacity(desc.count, 1, live_tile_count);
  record_sort(cmd,
              desc,
              default_segment_offsets_.buffer,
              0,
              default_segment_tile_offsets_.buffer,
              0,
              1,
              live_tile_count,
              false);
}

void RadixSort::sort_profiled(VkCommandBuffer cmd,
                              const RadixSortDesc& desc,
                              VkQueryPool query_pool,
                              uint32_t first_query) {
  if (query_pool == VK_NULL_HANDLE) {
    throw std::invalid_argument("sort_profiled requires a valid VkQueryPool");
  }
  if (desc.count == 0) {
    return;
  }
  if (desc.keys == VK_NULL_HANDLE) {
    throw std::invalid_argument("RadixSortDesc::keys must be non-null");
  }

  const uint32_t live_tile_count = ceil_div_u32(desc.count, workgroup_size_);
  ensure_capacity(desc.count, 1, live_tile_count);
  record_sort(cmd,
              desc,
              default_segment_offsets_.buffer,
              0,
              default_segment_tile_offsets_.buffer,
              0,
              1,
              live_tile_count,
              false,
              query_pool,
              first_query);
}

void RadixSort::sort_segmented(VkCommandBuffer cmd, const SegmentedRadixSortDesc& desc) {
  if (desc.count == 0 || desc.segment_count == 0 || desc.tile_count == 0) {
    return;
  }
  if (desc.keys == VK_NULL_HANDLE) {
    throw std::invalid_argument("SegmentedRadixSortDesc::keys must be non-null");
  }
  if (desc.segment_offsets == VK_NULL_HANDLE || desc.segment_tile_offsets == VK_NULL_HANDLE) {
    throw std::invalid_argument("segmented sort requires segment offset buffers");
  }

  ensure_capacity(desc.count, desc.segment_count, desc.tile_count);
  RadixSortDesc flat{};
  flat.keys = desc.keys;
  flat.key_offset = desc.key_offset;
  flat.values = desc.values;
  flat.value_offset = desc.value_offset;
  flat.count = desc.count;
  flat.descending = desc.descending;
  record_sort(cmd,
              flat,
              desc.segment_offsets,
              desc.segment_offsets_offset,
              desc.segment_tile_offsets,
              desc.segment_tile_offsets_offset,
              desc.segment_count,
              desc.tile_count,
              true);
}

void RadixSort::record_sort(VkCommandBuffer cmd, const RadixSortDesc& desc,
                                  VkBuffer segment_offsets, VkDeviceSize segment_offsets_offset,
                                  VkBuffer segment_tile_offsets, VkDeviceSize segment_tile_offsets_offset,
                                  uint32_t segment_count, uint32_t live_tile_count,
                                  bool segmented,
                                  VkQueryPool profile_query_pool,
                                  uint32_t first_profile_query) {

  const bool has_values = desc.values != VK_NULL_HANDLE;

  VkBuffer src_keys = desc.keys;
  VkDeviceSize src_key_offset = desc.key_offset;
  VkBuffer src_values = has_values ? desc.values : dummy_values_.buffer;
  VkDeviceSize src_value_offset = has_values ? desc.value_offset : 0;

  VkBuffer dst_keys = temp_keys_.buffer;
  VkDeviceSize dst_key_offset = 0;
  VkBuffer dst_values = has_values ? temp_values_.buffer : dummy_values_.buffer;
  VkDeviceSize dst_value_offset = 0;

  const uint32_t scan_ring_tile_count = std::min(ceil_power_of_two_u32(live_tile_count), kScanRingTiles);
  const uint32_t clear_items = std::max({
      scan_ring_tile_count * kRadix,
      segment_count * kRadix,
      1u,
  });
  const uint32_t clear_groups = ceil_div_u32(clear_items, workgroup_size_);
  uint32_t profile_query = first_profile_query;
  if (profile_query_pool != VK_NULL_HANDLE) {
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, profile_query_pool, profile_query++);
  }

  for (uint32_t pass = 0; pass < kPasses32; ++pass) {
    VkDescriptorSet descriptor_set = allocate_and_update_descriptors(
        src_keys, src_key_offset, src_values, src_value_offset,
        dst_keys, dst_key_offset, dst_values, dst_value_offset, has_values,
        segment_offsets, segment_offsets_offset,
        segment_tile_offsets, segment_tile_offsets_offset);

    PushConstants pc{};
    pc.count = desc.count;
    pc.shift = pass * kRadixBits;
    pc.tile_count = live_tile_count;
    pc.descending = desc.descending ? 1u : 0u;
    pc.has_values = has_values ? 1u : 0u;
    pc.segment_count = segment_count;
    pc.segmented = segmented ? 1u : 0u;
    pc.scan_ring_tile_count = scan_ring_tile_count;
    pc.scan_ring_tile_mask = scan_ring_tile_count - 1u;
    pc.scan_lookback_limit = std::max(1u, scan_ring_tile_count >> 1u);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1,
                            &descriptor_set, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &pc);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, clear_pipeline_);
    vkCmdDispatch(cmd, clear_groups, 1, 1);
    compute_barrier(cmd);
    if (profile_query_pool != VK_NULL_HANDLE) {
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, profile_query_pool, profile_query++);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      segmented ? histogram_pipeline_ : flat_histogram_pipeline_);
    vkCmdDispatch(cmd, live_tile_count, 1, 1);
    compute_barrier(cmd);
    if (profile_query_pool != VK_NULL_HANDLE) {
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, profile_query_pool, profile_query++);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bin_base_pipeline_);
    vkCmdDispatch(cmd, segment_count, 1, 1);
    compute_barrier(cmd);
    if (profile_query_pool != VK_NULL_HANDLE) {
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, profile_query_pool, profile_query++);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      segmented ? scatter_pipeline_ : flat_scatter_pipeline_);
    vkCmdDispatch(cmd, live_tile_count, 1, 1);
    compute_barrier(cmd);
    if (profile_query_pool != VK_NULL_HANDLE) {
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, profile_query_pool, profile_query++);
    }

    std::swap(src_keys, dst_keys);
    std::swap(src_key_offset, dst_key_offset);
    std::swap(src_values, dst_values);
    std::swap(src_value_offset, dst_value_offset);

    if (dst_keys == temp_keys_.buffer) {
      dst_key_offset = 0;
      dst_values = has_values ? temp_values_.buffer : dummy_values_.buffer;
      dst_value_offset = 0;
    } else {
      dst_key_offset = desc.key_offset;
      dst_values = has_values ? desc.values : dummy_values_.buffer;
      dst_value_offset = has_values ? desc.value_offset : 0;
    }
  }

  if (src_keys != desc.keys) {
    compute_to_transfer_barrier(cmd);

    VkBufferCopy key_copy{};
    key_copy.srcOffset = src_key_offset;
    key_copy.dstOffset = desc.key_offset;
    key_copy.size = static_cast<VkDeviceSize>(desc.count) * sizeof(uint32_t);
    vkCmdCopyBuffer(cmd, src_keys, desc.keys, 1, &key_copy);

    if (has_values) {
      VkBufferCopy value_copy{};
      value_copy.srcOffset = src_value_offset;
      value_copy.dstOffset = desc.value_offset;
      value_copy.size = static_cast<VkDeviceSize>(desc.count) * sizeof(uint32_t);
      vkCmdCopyBuffer(cmd, src_values, desc.values, 1, &value_copy);
    }

    transfer_to_compute_barrier(cmd);
  }
}

}  // namespace onesweep::vulkan
