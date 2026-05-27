#include "onesweep/vulkan/radix_sort.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef ONESWEEP_RADIX_SORT_SHADER_DIR
#define ONESWEEP_RADIX_SORT_SHADER_DIR "./shaders"
#endif

namespace {

void vk_check(VkResult result, const char* what) {
  if (result != VK_SUCCESS) {
    throw std::runtime_error(std::string(what) + " failed with VkResult " + std::to_string(result));
  }
}

struct Buffer {
  VkDevice device = VK_NULL_HANDLE;
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize size = 0;

  void destroy() {
    if (buffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device, buffer, nullptr);
      buffer = VK_NULL_HANDLE;
    }
    if (memory != VK_NULL_HANDLE) {
      vkFreeMemory(device, memory, nullptr);
      memory = VK_NULL_HANDLE;
    }
    size = 0;
  }
};

uint32_t find_memory_type(VkPhysicalDevice physical_device,
                          uint32_t type_filter,
                          VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties mem{};
  vkGetPhysicalDeviceMemoryProperties(physical_device, &mem);

  for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
    if ((type_filter & (1u << i)) != 0 &&
        (mem.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }

  throw std::runtime_error("no suitable memory type");
}

Buffer create_buffer(VkPhysicalDevice physical_device,
                     VkDevice device,
                     VkDeviceSize size,
                     VkBufferUsageFlags usage,
                     VkMemoryPropertyFlags properties) {
  Buffer out{};
  out.device = device;
  out.size = size;

  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  vk_check(vkCreateBuffer(device, &buffer_info, nullptr, &out.buffer), "vkCreateBuffer");

  VkMemoryRequirements req{};
  vkGetBufferMemoryRequirements(device, out.buffer, &req);

  VkMemoryAllocateInfo alloc{};
  alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc.allocationSize = req.size;
  alloc.memoryTypeIndex = find_memory_type(physical_device, req.memoryTypeBits, properties);
  vk_check(vkAllocateMemory(device, &alloc, nullptr, &out.memory), "vkAllocateMemory");
  vk_check(vkBindBufferMemory(device, out.buffer, out.memory, 0), "vkBindBufferMemory");

  return out;
}

void* map_buffer(VkDevice device, Buffer& buffer) {
  void* ptr = nullptr;
  vk_check(vkMapMemory(device, buffer.memory, 0, buffer.size, 0, &ptr), "vkMapMemory");
  return ptr;
}

void command_barrier(VkCommandBuffer cmd,
                     VkPipelineStageFlags src_stage,
                     VkPipelineStageFlags dst_stage,
                     VkAccessFlags src_access,
                     VkAccessFlags dst_access) {
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = src_access;
  barrier.dstAccessMask = dst_access;
  vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

VkInstance create_instance() {
  VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "onesweep_radix_sort_vulkan_u32_test";
  app.apiVersion = VK_API_VERSION_1_2;

  VkInstanceCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  info.pApplicationInfo = &app;

  VkInstance instance = VK_NULL_HANDLE;
  vk_check(vkCreateInstance(&info, nullptr, &instance), "vkCreateInstance");
  return instance;
}

struct DeviceSelection {
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  uint32_t queue_family = 0;
};

const char* subgroup_partitioned_extension(VkPhysicalDevice physical_device) {
  uint32_t count = 0;
  vk_check(vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, nullptr),
           "vkEnumerateDeviceExtensionProperties count");
  std::vector<VkExtensionProperties> extensions(count);
  vk_check(vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, extensions.data()),
           "vkEnumerateDeviceExtensionProperties");

  const auto has_extension = [&](const char* name) {
    return std::any_of(extensions.begin(), extensions.end(), [name](const auto& extension) {
      return std::strcmp(extension.extensionName, name) == 0;
    });
  };

  if (has_extension(VK_NV_SHADER_SUBGROUP_PARTITIONED_EXTENSION_NAME)) {
    return VK_NV_SHADER_SUBGROUP_PARTITIONED_EXTENSION_NAME;
  }
  if (has_extension(VK_EXT_SHADER_SUBGROUP_PARTITIONED_EXTENSION_NAME)) {
    return VK_EXT_SHADER_SUBGROUP_PARTITIONED_EXTENSION_NAME;
  }
  return nullptr;
}

DeviceSelection select_device(VkInstance instance) {
  uint32_t count = 0;
  vk_check(vkEnumeratePhysicalDevices(instance, &count, nullptr), "vkEnumeratePhysicalDevices count");
  if (count == 0) {
    throw std::runtime_error("no Vulkan physical devices found");
  }

  std::vector<VkPhysicalDevice> devices(count);
  vk_check(vkEnumeratePhysicalDevices(instance, &count, devices.data()), "vkEnumeratePhysicalDevices");

  for (VkPhysicalDevice pd : devices) {
    VkPhysicalDeviceSubgroupProperties subgroup{};
    subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroup;
    vkGetPhysicalDeviceProperties2(pd, &props2);

    if ((subgroup.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) == 0 ||
        (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT) == 0 ||
        (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_PARTITIONED_BIT_EXT) == 0 ||
        subgroup_partitioned_extension(pd) == nullptr) {
      continue;
    }

    uint32_t q_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &q_count, nullptr);
    std::vector<VkQueueFamilyProperties> queues(q_count);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &q_count, queues.data());

    for (uint32_t i = 0; i < q_count; ++i) {
      if ((queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
        std::cout << "Selected device: " << props2.properties.deviceName << "\n";
        return {pd, i};
      }
    }
  }

  throw std::runtime_error("no suitable compute device with subgroup ballot+partitioned support found");
}

VkDevice create_device(VkPhysicalDevice physical_device, uint32_t queue_family) {
  float priority = 1.0f;
  VkDeviceQueueCreateInfo qinfo{};
  qinfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  qinfo.queueFamilyIndex = queue_family;
  qinfo.queueCount = 1;
  qinfo.pQueuePriorities = &priority;

  const char* extension = subgroup_partitioned_extension(physical_device);
  if (extension == nullptr) {
    throw std::runtime_error("selected device lost subgroup partitioned extension support");
  }

  VkPhysicalDeviceShaderSubgroupPartitionedFeaturesEXT subgroup_partitioned{};
  subgroup_partitioned.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_PARTITIONED_FEATURES_EXT;
  subgroup_partitioned.shaderSubgroupPartitioned = VK_TRUE;

  VkDeviceCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  info.pNext = std::strcmp(extension, VK_EXT_SHADER_SUBGROUP_PARTITIONED_EXTENSION_NAME) == 0
                   ? &subgroup_partitioned
                   : nullptr;
  info.queueCreateInfoCount = 1;
  info.pQueueCreateInfos = &qinfo;
  info.enabledExtensionCount = 1;
  info.ppEnabledExtensionNames = &extension;

  VkDevice device = VK_NULL_HANDLE;
  vk_check(vkCreateDevice(physical_device, &info, nullptr, &device), "vkCreateDevice");
  return device;
}

}  // namespace

int main() {
  try {
    constexpr uint32_t n = 2u << 20;
    constexpr VkDeviceSize bytes = static_cast<VkDeviceSize>(n) * sizeof(uint32_t);
    const std::vector<uint32_t> segment_offsets = {0u, 17u, 530u, 4626u, n};
    const uint32_t segment_count = static_cast<uint32_t>(segment_offsets.size() - 1);
    const VkDeviceSize segment_offsets_bytes =
        static_cast<VkDeviceSize>(segment_offsets.size()) * sizeof(uint32_t);

    std::mt19937 rng(0xC0FFEEu);
    std::uniform_int_distribution<uint32_t> dist;

    std::vector<uint32_t> input(n);
    for (uint32_t& x : input) {
      x = dist(rng);
    }

    std::vector<uint32_t> expected = input;
    std::sort(expected.begin(), expected.end(), std::greater<uint32_t>{});
    std::vector<uint32_t> segmented_expected = input;
    for (std::size_t i = 0; i < segment_count; ++i) {
      std::sort(segmented_expected.begin() + segment_offsets[i],
                segmented_expected.begin() + segment_offsets[i + 1],
                std::greater<uint32_t>{});
    }

    VkInstance instance = create_instance();
    DeviceSelection selected = select_device(instance);
    VkDevice device = create_device(selected.physical_device, selected.queue_family);

    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, selected.queue_family, 0, &queue);

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = selected.queue_family;

    VkCommandPool pool = VK_NULL_HANDLE;
    vk_check(vkCreateCommandPool(device, &pool_info, nullptr, &pool), "vkCreateCommandPool");

    VkCommandBufferAllocateInfo cmd_info{};
    cmd_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_info.commandPool = pool;
    cmd_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_info.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vk_check(vkAllocateCommandBuffers(device, &cmd_info, &cmd), "vkAllocateCommandBuffers");

    Buffer upload = create_buffer(selected.physical_device, device, bytes,
                                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Buffer readback = create_buffer(selected.physical_device, device, bytes,
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Buffer segmented_readback = create_buffer(selected.physical_device, device, bytes,
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Buffer keys = create_buffer(selected.physical_device, device, bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buffer segmented_keys = create_buffer(selected.physical_device, device, bytes,
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buffer segment_offsets_upload = create_buffer(selected.physical_device, device, segment_offsets_bytes,
                                                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Buffer segment_offsets_gpu = create_buffer(selected.physical_device, device, segment_offsets_bytes,
                                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buffer segment_tile_offsets_upload = create_buffer(selected.physical_device, device, segment_offsets_bytes,
                                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Buffer segment_tile_offsets_gpu = create_buffer(selected.physical_device, device, segment_offsets_bytes,
                                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    std::memcpy(map_buffer(device, upload), input.data(), bytes);
    vkUnmapMemory(device, upload.memory);
    std::memcpy(map_buffer(device, segment_offsets_upload), segment_offsets.data(), segment_offsets_bytes);
    vkUnmapMemory(device, segment_offsets_upload.memory);

    bool all_ok = true;
    for (uint32_t workgroup_size : {64u, 128u, 256u, 512u}) {
      onesweep::vulkan::RadixSortCreateInfo sorter_info{};
      sorter_info.device = device;
      sorter_info.physical_device = selected.physical_device;
      sorter_info.spirv_directory = ONESWEEP_RADIX_SORT_SHADER_DIR;
      sorter_info.initial_capacity = n;
      sorter_info.initial_segments = segment_count;
      sorter_info.initial_tiles =
          (n + onesweep::vulkan::RadixSort::kMinWorkgroupSize - 1u) /
              onesweep::vulkan::RadixSort::kMinWorkgroupSize +
          segment_count;
      sorter_info.preferred_workgroup_size = workgroup_size;
      auto sorter = std::make_unique<onesweep::vulkan::RadixSort>(sorter_info);

      std::vector<uint32_t> segment_tile_offsets(segment_offsets.size());
      for (std::size_t i = 1; i < segment_offsets.size(); ++i) {
        const uint32_t count = segment_offsets[i] - segment_offsets[i - 1];
        const uint32_t tile_size = sorter->workgroup_size();
        segment_tile_offsets[i] = segment_tile_offsets[i - 1] + (count + tile_size - 1u) / tile_size;
      }
      const uint32_t segment_tile_count = segment_tile_offsets.back();
      std::memcpy(map_buffer(device, segment_tile_offsets_upload), segment_tile_offsets.data(),
                  segment_offsets_bytes);
      vkUnmapMemory(device, segment_tile_offsets_upload.memory);

      vk_check(vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");

      VkCommandBufferBeginInfo begin{};
      begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      vk_check(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");

      VkBufferCopy upload_copy{};
      upload_copy.size = bytes;
      vkCmdCopyBuffer(cmd, upload.buffer, keys.buffer, 1, &upload_copy);
      vkCmdCopyBuffer(cmd, upload.buffer, segmented_keys.buffer, 1, &upload_copy);
      VkBufferCopy segment_offsets_copy{};
      segment_offsets_copy.size = segment_offsets_bytes;
      vkCmdCopyBuffer(cmd, segment_offsets_upload.buffer, segment_offsets_gpu.buffer, 1,
                      &segment_offsets_copy);
      vkCmdCopyBuffer(cmd, segment_tile_offsets_upload.buffer, segment_tile_offsets_gpu.buffer, 1,
                      &segment_offsets_copy);

      command_barrier(cmd,
                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

      onesweep::vulkan::RadixSortDesc sort_desc{};
      sort_desc.keys = keys.buffer;
      sort_desc.count = n;
      sort_desc.descending = true;
      sorter->sort(cmd, sort_desc);

      onesweep::vulkan::SegmentedRadixSortDesc segmented_desc{};
      segmented_desc.keys = segmented_keys.buffer;
      segmented_desc.segment_offsets = segment_offsets_gpu.buffer;
      segmented_desc.segment_tile_offsets = segment_tile_offsets_gpu.buffer;
      segmented_desc.count = n;
      segmented_desc.segment_count = segment_count;
      segmented_desc.tile_count = segment_tile_count;
      segmented_desc.descending = true;
      sorter->sort_segmented(cmd, segmented_desc);

      command_barrier(cmd,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_ACCESS_SHADER_WRITE_BIT,
                      VK_ACCESS_TRANSFER_READ_BIT);

      VkBufferCopy download_copy{};
      download_copy.size = bytes;
      vkCmdCopyBuffer(cmd, keys.buffer, readback.buffer, 1, &download_copy);
      vkCmdCopyBuffer(cmd, segmented_keys.buffer, segmented_readback.buffer, 1, &download_copy);

      vk_check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

      VkSubmitInfo submit{};
      submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submit.commandBufferCount = 1;
      submit.pCommandBuffers = &cmd;
      vk_check(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit");
      vk_check(vkQueueWaitIdle(queue), "vkQueueWaitIdle");

      std::vector<uint32_t> output(n);
      std::memcpy(output.data(), map_buffer(device, readback), bytes);
      vkUnmapMemory(device, readback.memory);
      std::vector<uint32_t> segmented_output(n);
      std::memcpy(segmented_output.data(), map_buffer(device, segmented_readback), bytes);
      vkUnmapMemory(device, segmented_readback.memory);

      const bool ok = output == expected;
      const bool segmented_ok = segmented_output == segmented_expected;
      std::cout << "wg" << workgroup_size << (ok ? " sort verified\n" : " sort mismatch\n");
      std::cout << "wg" << workgroup_size
                << (segmented_ok ? " segmented sort verified\n" : " segmented sort mismatch\n");
      all_ok = all_ok && ok && segmented_ok;
    }

    segment_tile_offsets_gpu.destroy();
    segment_tile_offsets_upload.destroy();
    segment_offsets_gpu.destroy();
    segment_offsets_upload.destroy();
    segmented_keys.destroy();
    keys.destroy();
    segmented_readback.destroy();
    upload.destroy();
    readback.destroy();
    vkDestroyCommandPool(device, pool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    return all_ok ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
