#include "onesweep/vulkan/radix_sort.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
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
  app.pApplicationName = "onesweep_radix_sort_vulkan_u32_benchmark";
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
  float timestamp_period = 1.0f;
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
      if ((queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0 && queues[i].timestampValidBits > 0) {
        std::cout << "Selected device: " << props2.properties.deviceName << "\n";
        return {pd, i, props2.properties.limits.timestampPeriod};
      }
    }
  }

  throw std::runtime_error("no suitable compute device with subgroup ballot+partitioned and timestamps found");
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

void reset_command_buffer(VkCommandBuffer cmd) {
  vk_check(vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vk_check(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");
}

void submit_and_wait(VkQueue queue, VkCommandBuffer cmd) {
  vk_check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
  VkSubmitInfo submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;
  vk_check(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit");
  vk_check(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
}

std::vector<uint32_t> generate(uint32_t count) {
  std::mt19937 rng(0xC0FFEEu ^ count);
  std::uniform_int_distribution<uint32_t> dist;
  std::vector<uint32_t> keys(count);
  for (uint32_t& key : keys) {
    key = dist(rng);
  }
  return keys;
}

void cpu_radix_sort_descending(const std::vector<uint32_t>& input,
                               std::vector<uint32_t>& output,
                               std::vector<uint32_t>& scratch) {
  output = input;
  scratch.resize(input.size());

  constexpr uint32_t kRadix = onesweep::vulkan::RadixSort::kRadix;
  constexpr uint32_t kMask = kRadix - 1u;
  constexpr uint32_t kPasses = onesweep::vulkan::RadixSort::kPasses32;

  uint32_t counts[kPasses][kRadix] = {};
  for (uint32_t key : input) {
    const uint32_t descending_key = ~key;
    ++counts[0][descending_key & kMask];
    ++counts[1][(descending_key >> 8u) & kMask];
    ++counts[2][(descending_key >> 16u) & kMask];
    ++counts[3][(descending_key >> 24u) & kMask];
  }

  uint32_t offsets[kPasses][kRadix];
  for (uint32_t pass = 0; pass < kPasses; ++pass) {
    uint32_t sum = 0;
    for (uint32_t digit = 0; digit < kRadix; ++digit) {
      offsets[pass][digit] = sum;
      sum += counts[pass][digit];
    }
  }

  std::vector<uint32_t>* src = &output;
  std::vector<uint32_t>* dst = &scratch;

  for (uint32_t pass = 0; pass < kPasses; ++pass) {
    const uint32_t shift = pass * onesweep::vulkan::RadixSort::kRadixBits;
    uint32_t* pass_offsets = offsets[pass];
    for (uint32_t key : *src) {
      const uint32_t digit = ((~key) >> shift) & kMask;
      (*dst)[pass_offsets[digit]++] = key;
    }

    std::swap(src, dst);
  }

  if (src != &output) {
    output = *src;
  }
}

bool is_descending(const std::vector<uint32_t>& values) {
  for (std::size_t i = 1; i < values.size(); ++i) {
    if (values[i - 1] < values[i]) {
      return false;
    }
  }
  return true;
}

std::vector<std::string> split_csv(const std::string& text) {
  std::vector<std::string> out;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (!item.empty()) {
      out.push_back(item);
    }
  }
  return out;
}

void apply_workgroup_variant(const std::string& value,
                             onesweep::vulkan::RadixSortCreateInfo& info) {
  info.preferred_workgroup_size = 0;
  info.preferred_workgroup_layout.clear();
  if (value.rfind("wg", 0) == 0) {
    info.preferred_workgroup_layout = value;
  } else if (value.find_first_not_of("0123456789") != std::string::npos) {
    info.preferred_workgroup_layout = "wg" + value;
  } else {
    info.preferred_workgroup_size = static_cast<uint32_t>(std::stoul(value));
  }
}

std::string variant_name(const onesweep::vulkan::RadixSort& sorter) {
  return sorter.workgroup_layout().empty()
             ? "wg" + std::to_string(sorter.workgroup_size())
             : sorter.workgroup_layout();
}

double average(const std::vector<double>& values) {
  double sum = 0.0;
  for (double value : values) {
    sum += value;
  }
  return values.empty() ? 0.0 : sum / static_cast<double>(values.size());
}

double median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::vector<uint32_t> sizes;
    uint32_t preferred_workgroup_size = 0;
    std::string preferred_workgroup_layout;
    std::vector<std::string> compare_workgroups;
    bool kernel_breakdown = false;
    if (argc > 1) {
      for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--kernel-breakdown") == 0) {
          kernel_breakdown = true;
          continue;
        }
        if (std::strcmp(argv[i], "--compare-wg") == 0 && i + 1 < argc) {
          compare_workgroups = split_csv(argv[++i]);
          continue;
        }
        constexpr std::string_view compare_prefix = "--compare-wg=";
        const std::string compare_arg = argv[i];
        if (compare_arg.rfind(compare_prefix, 0) == 0) {
          compare_workgroups = split_csv(compare_arg.substr(compare_prefix.size()));
          continue;
        }
        if (std::strcmp(argv[i], "--wg") == 0 && i + 1 < argc) {
          const std::string value = argv[++i];
          if (value.rfind("wg", 0) == 0) {
            preferred_workgroup_layout = value;
          } else if (value.find('x') != std::string::npos) {
            preferred_workgroup_layout = "wg" + value;
          } else {
            preferred_workgroup_size = static_cast<uint32_t>(std::stoul(value));
            preferred_workgroup_layout.clear();
          }
          continue;
        }
        constexpr std::string_view wg_prefix = "--wg=";
        const std::string arg = argv[i];
        if (arg.rfind(wg_prefix, 0) == 0) {
          const std::string value = arg.substr(wg_prefix.size());
          if (value.rfind("wg", 0) == 0) {
            preferred_workgroup_layout = value;
          } else if (value.find('x') != std::string::npos) {
            preferred_workgroup_layout = "wg" + value;
          } else {
            preferred_workgroup_size = static_cast<uint32_t>(std::stoul(value));
            preferred_workgroup_layout.clear();
          }
          continue;
        }
        sizes.push_back(static_cast<uint32_t>(std::stoul(argv[i])));
      }
    } else {
      sizes = {1024u, 4096u, 8192u, 32768u, 131072u, 524288u, 2097152u};
    }
    if (sizes.empty()) {
      sizes = {1024u, 4096u, 8192u, 32768u, 131072u, 524288u, 2097152u};
    }

    constexpr int kWarmupRounds = 3;
    constexpr int kMeasuredRounds = 10;
    const uint32_t max_n = *std::max_element(sizes.begin(), sizes.end());
    const VkDeviceSize max_bytes = static_cast<VkDeviceSize>(max_n) * sizeof(uint32_t);

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

    VkQueryPoolCreateInfo query_info{};
    query_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    query_info.queryCount = onesweep::vulkan::RadixSort::kProfileTimestampCount;
    VkQueryPool query_pool = VK_NULL_HANDLE;
    vk_check(vkCreateQueryPool(device, &query_info, nullptr, &query_pool), "vkCreateQueryPool");

    Buffer upload = create_buffer(selected.physical_device, device, max_bytes,
                                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Buffer readback = create_buffer(selected.physical_device, device, max_bytes,
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Buffer keys = create_buffer(selected.physical_device, device, max_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (!compare_workgroups.empty()) {
      struct VariantRun {
        std::string name;
        std::unique_ptr<onesweep::vulkan::RadixSort> sorter;
      };

      std::vector<VariantRun> variants;
      variants.reserve(compare_workgroups.size());
      for (const std::string& wg : compare_workgroups) {
        onesweep::vulkan::RadixSortCreateInfo info{};
        info.device = device;
        info.physical_device = selected.physical_device;
        info.spirv_directory = ONESWEEP_RADIX_SORT_SHADER_DIR;
        info.initial_capacity = max_n;
        apply_workgroup_variant(wg, info);
        auto sorter = std::make_unique<onesweep::vulkan::RadixSort>(info);
        variants.push_back({variant_name(*sorter), std::move(sorter)});
      }

      std::cout << std::left
                << std::setw(12) << "size"
                << std::setw(14) << "variant"
                << std::setw(16) << "kernelAvgMs"
                << std::setw(16) << "kernelMinMs"
                << std::setw(16) << "kernelMedMs"
                << std::setw(16) << "submitAvgMs"
                << std::setw(16) << "submitMinMs"
                << std::setw(16) << "submitMedMs"
                << "\n";

      for (uint32_t n : sizes) {
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(n) * sizeof(uint32_t);
        std::vector<uint32_t> input = generate(n);
        std::vector<uint32_t> expected;
        std::vector<uint32_t> cpu_scratch;
        cpu_radix_sort_descending(input, expected, cpu_scratch);

        std::memcpy(map_buffer(device, upload), input.data(), bytes);
        vkUnmapMemory(device, upload.memory);

        VkBufferCopy copy{};
        copy.size = bytes;
        onesweep::vulkan::RadixSortDesc desc{};
        desc.keys = keys.buffer;
        desc.count = n;
        desc.descending = true;

        for (VariantRun& variant : variants) {
          reset_command_buffer(cmd);
          vkCmdCopyBuffer(cmd, upload.buffer, keys.buffer, 1, &copy);
          command_barrier(cmd,
                          VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
          variant.sorter->sort(cmd, desc);
          command_barrier(cmd,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_TRANSFER_READ_BIT);
          vkCmdCopyBuffer(cmd, keys.buffer, readback.buffer, 1, &copy);
          submit_and_wait(queue, cmd);

          std::vector<uint32_t> output(n);
          std::memcpy(output.data(), map_buffer(device, readback), bytes);
          vkUnmapMemory(device, readback.memory);
          if (output != expected) {
            const auto mismatch = std::mismatch(output.begin(), output.end(), expected.begin());
            const std::size_t index = static_cast<std::size_t>(mismatch.first - output.begin());
            throw std::runtime_error("GPU sort verification failed for " + variant.name +
                                     " size " + std::to_string(n) +
                                     " at index " + std::to_string(index) +
                                     ": gpu=" + std::to_string(output[index]) +
                                     " cpu=" + std::to_string(expected[index]));
          }
        }

        for (int i = 0; i < kWarmupRounds; ++i) {
          for (VariantRun& variant : variants) {
            reset_command_buffer(cmd);
            vkCmdCopyBuffer(cmd, upload.buffer, keys.buffer, 1, &copy);
            command_barrier(cmd,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_ACCESS_TRANSFER_WRITE_BIT,
                            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            variant.sorter->sort(cmd, desc);
            submit_and_wait(queue, cmd);
          }
        }

        std::vector<std::vector<double>> kernel_ms(variants.size());
        std::vector<std::vector<double>> submit_ms(variants.size());
        for (int round = 0; round < kMeasuredRounds; ++round) {
          for (std::size_t variant_index = 0; variant_index < variants.size(); ++variant_index) {
            VariantRun& variant = variants[variant_index];
            reset_command_buffer(cmd);
            vkCmdResetQueryPool(cmd, query_pool, 0, 2);
            vkCmdCopyBuffer(cmd, upload.buffer, keys.buffer, 1, &copy);
            command_barrier(cmd,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_ACCESS_TRANSFER_WRITE_BIT,
                            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, 0);
            variant.sorter->sort(cmd, desc);
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, 1);

            auto wall_start = std::chrono::steady_clock::now();
            submit_and_wait(queue, cmd);
            auto wall_end = std::chrono::steady_clock::now();
            submit_ms[variant_index].push_back(
                std::chrono::duration<double, std::milli>(wall_end - wall_start).count());

            uint64_t timestamps[2] = {};
            vk_check(vkGetQueryPoolResults(device,
                                           query_pool,
                                           0,
                                           2,
                                           sizeof(timestamps),
                                           timestamps,
                                           sizeof(uint64_t),
                                           VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
                     "vkGetQueryPoolResults");
            kernel_ms[variant_index].push_back(
                static_cast<double>(timestamps[1] - timestamps[0]) *
                static_cast<double>(selected.timestamp_period) / 1'000'000.0);
          }
        }

        for (std::size_t variant_index = 0; variant_index < variants.size(); ++variant_index) {
          const auto& kernels = kernel_ms[variant_index];
          const auto& submits = submit_ms[variant_index];
          std::cout << std::left
                    << std::setw(12) << n
                    << std::setw(14) << variants[variant_index].name
                    << std::setw(16) << average(kernels)
                    << std::setw(16) << *std::min_element(kernels.begin(), kernels.end())
                    << std::setw(16) << median(kernels)
                    << std::setw(16) << average(submits)
                    << std::setw(16) << *std::min_element(submits.begin(), submits.end())
                    << std::setw(16) << median(submits)
                    << "\n";
        }
      }

      variants.clear();
      keys.destroy();
      readback.destroy();
      upload.destroy();
      vkDestroyQueryPool(device, query_pool, nullptr);
      vkDestroyCommandPool(device, pool, nullptr);
      vkDestroyDevice(device, nullptr);
      vkDestroyInstance(instance, nullptr);
      return 0;
    }

    onesweep::vulkan::RadixSortCreateInfo sorter_info{};
    sorter_info.device = device;
    sorter_info.physical_device = selected.physical_device;
    sorter_info.spirv_directory = ONESWEEP_RADIX_SORT_SHADER_DIR;
    sorter_info.initial_capacity = max_n;
    sorter_info.preferred_workgroup_size = preferred_workgroup_size;
    sorter_info.preferred_workgroup_layout = preferred_workgroup_layout;
    auto sorter = std::make_unique<onesweep::vulkan::RadixSort>(sorter_info);
    std::cout << "Workgroup variant: "
              << (sorter->workgroup_layout().empty()
                      ? "wg" + std::to_string(sorter->workgroup_size())
                      : sorter->workgroup_layout())
              << "\n";

    if (kernel_breakdown) {
      if (sizes.size() != 1) {
        std::cout << "kernel breakdown uses the first requested size only\n";
      }
      const uint32_t n = sizes.front();
      const VkDeviceSize bytes = static_cast<VkDeviceSize>(n) * sizeof(uint32_t);
      std::vector<uint32_t> input = generate(n);
      std::vector<uint32_t> expected;
      std::vector<uint32_t> cpu_scratch;
      cpu_radix_sort_descending(input, expected, cpu_scratch);

      std::memcpy(map_buffer(device, upload), input.data(), bytes);
      vkUnmapMemory(device, upload.memory);

      VkBufferCopy copy{};
      copy.size = bytes;
      onesweep::vulkan::RadixSortDesc desc{};
      desc.keys = keys.buffer;
      desc.count = n;
      desc.descending = true;

      reset_command_buffer(cmd);
      vkCmdCopyBuffer(cmd, upload.buffer, keys.buffer, 1, &copy);
      command_barrier(cmd,
                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
      sorter->sort(cmd, desc);
      command_barrier(cmd,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_ACCESS_SHADER_WRITE_BIT,
                      VK_ACCESS_TRANSFER_READ_BIT);
      vkCmdCopyBuffer(cmd, keys.buffer, readback.buffer, 1, &copy);
      submit_and_wait(queue, cmd);
      std::vector<uint32_t> output(n);
      std::memcpy(output.data(), map_buffer(device, readback), bytes);
      vkUnmapMemory(device, readback.memory);
      if (output != expected) {
        throw std::runtime_error("GPU sort verification failed for size " + std::to_string(n));
      }

      for (int i = 0; i < kWarmupRounds; ++i) {
        reset_command_buffer(cmd);
        vkCmdCopyBuffer(cmd, upload.buffer, keys.buffer, 1, &copy);
        command_barrier(cmd,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        sorter->sort(cmd, desc);
        submit_and_wait(queue, cmd);
      }

      double clear_ms = 0.0;
      double histogram_ms = 0.0;
      double bin_base_ms = 0.0;
      double scatter_ms = 0.0;
      double total_ms = 0.0;
      for (int i = 0; i < kMeasuredRounds; ++i) {
        reset_command_buffer(cmd);
        vkCmdResetQueryPool(cmd, query_pool, 0, onesweep::vulkan::RadixSort::kProfileTimestampCount);
        vkCmdCopyBuffer(cmd, upload.buffer, keys.buffer, 1, &copy);
        command_barrier(cmd,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        sorter->sort_profiled(cmd, desc, query_pool);
        submit_and_wait(queue, cmd);

        uint64_t timestamps[onesweep::vulkan::RadixSort::kProfileTimestampCount] = {};
        vk_check(vkGetQueryPoolResults(device,
                                       query_pool,
                                       0,
                                       onesweep::vulkan::RadixSort::kProfileTimestampCount,
                                       sizeof(timestamps),
                                       timestamps,
                                       sizeof(uint64_t),
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
                 "vkGetQueryPoolResults");

        const double period_ms = static_cast<double>(selected.timestamp_period) / 1'000'000.0;
        for (uint32_t pass = 0; pass < onesweep::vulkan::RadixSort::kPasses32; ++pass) {
          const uint32_t base = pass * 4u;
          clear_ms += static_cast<double>(timestamps[base + 1] - timestamps[base]) * period_ms;
          histogram_ms += static_cast<double>(timestamps[base + 2] - timestamps[base + 1]) * period_ms;
          bin_base_ms += static_cast<double>(timestamps[base + 3] - timestamps[base + 2]) * period_ms;
          scatter_ms += static_cast<double>(timestamps[base + 4] - timestamps[base + 3]) * period_ms;
        }
        total_ms += static_cast<double>(timestamps[onesweep::vulkan::RadixSort::kProfileTimestampCount - 1] -
                                        timestamps[0]) * period_ms;
      }

      std::cout << std::left
                << std::setw(12) << "size"
                << std::setw(14) << "clearMs"
                << std::setw(14) << "histMs"
                << std::setw(14) << "binBaseMs"
                << std::setw(14) << "scatterMs"
                << std::setw(14) << "totalMs"
                << "\n";
      std::cout << std::left
                << std::setw(12) << n
                << std::setw(14) << (clear_ms / kMeasuredRounds)
                << std::setw(14) << (histogram_ms / kMeasuredRounds)
                << std::setw(14) << (bin_base_ms / kMeasuredRounds)
                << std::setw(14) << (scatter_ms / kMeasuredRounds)
                << std::setw(14) << (total_ms / kMeasuredRounds)
                << "\n";

      sorter.reset();
      keys.destroy();
      readback.destroy();
      upload.destroy();
      vkDestroyQueryPool(device, query_pool, nullptr);
      vkDestroyCommandPool(device, pool, nullptr);
      vkDestroyDevice(device, nullptr);
      vkDestroyInstance(instance, nullptr);
      return 0;
    }

    std::cout << std::left
              << std::setw(12) << "size"
              << std::setw(16) << "cpuRadixMs"
              << std::setw(18) << "gpuKernelMs"
              << std::setw(18) << "gpuSubmitMs"
              << std::setw(18) << "kernelSpeedup"
              << std::setw(18) << "submitSpeedup"
              << "\n";

    for (uint32_t n : sizes) {
      const VkDeviceSize bytes = static_cast<VkDeviceSize>(n) * sizeof(uint32_t);
      std::vector<uint32_t> input = generate(n);
      std::vector<uint32_t> expected;
      std::vector<uint32_t> cpu_scratch;
      cpu_radix_sort_descending(input, expected, cpu_scratch);
      if (!is_descending(expected)) {
        throw std::runtime_error("CPU radix sort failed for size " + std::to_string(n));
      }

      std::memcpy(map_buffer(device, upload), input.data(), bytes);
      vkUnmapMemory(device, upload.memory);

      reset_command_buffer(cmd);
      VkBufferCopy copy{};
      copy.size = bytes;
      vkCmdCopyBuffer(cmd, upload.buffer, keys.buffer, 1, &copy);
      command_barrier(cmd,
                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
      onesweep::vulkan::RadixSortDesc desc{};
      desc.keys = keys.buffer;
      desc.count = n;
      desc.descending = true;
      sorter->sort(cmd, desc);
      command_barrier(cmd,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_ACCESS_SHADER_WRITE_BIT,
                      VK_ACCESS_TRANSFER_READ_BIT);
      vkCmdCopyBuffer(cmd, keys.buffer, readback.buffer, 1, &copy);
      submit_and_wait(queue, cmd);

      std::vector<uint32_t> output(n);
      std::memcpy(output.data(), map_buffer(device, readback), bytes);
      vkUnmapMemory(device, readback.memory);
      if (output != expected) {
        throw std::runtime_error("GPU sort verification failed for size " + std::to_string(n));
      }

      for (int i = 0; i < kWarmupRounds; ++i) {
        std::vector<uint32_t> cpu_output;
        cpu_radix_sort_descending(input, cpu_output, cpu_scratch);
      }

      double cpu_radix_total_ms = 0.0;
      std::vector<uint32_t> cpu_output;
      for (int i = 0; i < kMeasuredRounds; ++i) {
        auto cpu_start = std::chrono::steady_clock::now();
        cpu_radix_sort_descending(input, cpu_output, cpu_scratch);
        auto cpu_end = std::chrono::steady_clock::now();
        if (cpu_output != expected) {
          throw std::runtime_error("CPU radix verification failed for size " + std::to_string(n));
        }
        cpu_radix_total_ms +=
            std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
      }

      for (int i = 0; i < kWarmupRounds; ++i) {
        reset_command_buffer(cmd);
        vkCmdCopyBuffer(cmd, upload.buffer, keys.buffer, 1, &copy);
        command_barrier(cmd,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        sorter->sort(cmd, desc);
        submit_and_wait(queue, cmd);
      }

      double gpu_total_ms = 0.0;
      double wall_total_ms = 0.0;
      for (int i = 0; i < kMeasuredRounds; ++i) {
        reset_command_buffer(cmd);
        vkCmdResetQueryPool(cmd, query_pool, 0, 2);
        vkCmdCopyBuffer(cmd, upload.buffer, keys.buffer, 1, &copy);
        command_barrier(cmd,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, 0);
        sorter->sort(cmd, desc);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, 1);

        auto wall_start = std::chrono::steady_clock::now();
        submit_and_wait(queue, cmd);
        auto wall_end = std::chrono::steady_clock::now();
        wall_total_ms += std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

        uint64_t timestamps[2] = {};
        vk_check(vkGetQueryPoolResults(device,
                                       query_pool,
                                       0,
                                       2,
                                       sizeof(timestamps),
                                       timestamps,
                                       sizeof(uint64_t),
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
                 "vkGetQueryPoolResults");
        gpu_total_ms += static_cast<double>(timestamps[1] - timestamps[0]) *
                        static_cast<double>(selected.timestamp_period) / 1'000'000.0;
      }

      const double cpu_radix_avg_ms = cpu_radix_total_ms / kMeasuredRounds;
      const double gpu_kernel_avg_ms = gpu_total_ms / kMeasuredRounds;
      const double gpu_submit_avg_ms = wall_total_ms / kMeasuredRounds;

      std::cout << std::left
                << std::setw(12) << n
                << std::setw(16) << cpu_radix_avg_ms
                << std::setw(18) << gpu_kernel_avg_ms
                << std::setw(18) << gpu_submit_avg_ms
                << std::setw(18) << (cpu_radix_avg_ms / gpu_kernel_avg_ms)
                << std::setw(18) << (cpu_radix_avg_ms / gpu_submit_avg_ms)
                << "\n";
    }

    sorter.reset();
    keys.destroy();
    readback.destroy();
    upload.destroy();
    vkDestroyQueryPool(device, query_pool, nullptr);
    vkDestroyCommandPool(device, pool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
