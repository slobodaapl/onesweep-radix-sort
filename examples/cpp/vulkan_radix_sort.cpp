#include <onesweep/vulkan/radix_sort.hpp>

void record_sort(VkCommandBuffer cmd,
                 VkBuffer keys,
                 VkBuffer values,
                 uint32_t count,
                 onesweep::vulkan::RadixSort& sorter) {
  onesweep::vulkan::RadixSortDesc desc{};
  desc.keys = keys;
  desc.values = values;
  desc.count = count;
  desc.descending = true;

  // Caller records transfer-write -> compute-read/write before this call and
  // compute-write -> transfer/read-or-consumer after it.
  sorter.sort(cmd, desc);
}
