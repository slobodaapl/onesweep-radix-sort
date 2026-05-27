# onesweep_radix_sort

`onesweep_radix_sort` is a GPU radix sort for `uint32_t` keys with optional `uint32_t` values, descending order, and segmented sorting for independent contiguous ranges.

HLSL under `shaders/hlsl/` is the shader source used by the build. GLSL and Slang ports live under `shaders/glsl/` and `shaders/slang/`. Vulkan uses SPIR-V generated from HLSL under `build/shaders/wg64/`, `build/shaders/wg128/`, `build/shaders/wg256/`, or `build/shaders/wg512/`.

## Algorithm

Each 8-bit radix pass runs:

1. Clear scratch state.
2. Build per-tile and per-segment digit histograms.
3. Build per-segment digit bases.
4. Stable scatter with wave/subgroup partitioned-match ranks and a stamped circular-buffer lookback path.

## Requirements

- Vulkan 1.2 headers/runtime for the C++ Vulkan wrapper
- Compute queue
- Subgroup/wave basic, ballot, and partitioned operations
- `VK_NV_shader_subgroup_partitioned` or `VK_EXT_shader_subgroup_partitioned` enabled on the Vulkan device
- `dxc` for HLSL-to-SPIR-V builds
- Optional `spirv-val` for generated SPIR-V validation
- C++20 compiler

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The CTest target `onesweep_radix_sort_vulkan_u32` creates a Vulkan instance/device, uploads `2^21` random keys, sorts descending on the GPU, reads the result back, and verifies against `std::sort`. It also verifies segmented descending sort against per-range `std::sort`.

Benchmark:

```bash
cmake --build build --target onesweep_radix_sort_vulkan_u32_benchmark
./build/onesweep_radix_sort_vulkan_u32_benchmark
```

It reports CPU radix time, GPU timestamped kernel time, GPU submit/wait wall time, and CPU-radix-to-GPU speedup columns for each configured input size. Use `--wg 64`, `--wg 128`, `--wg 256`, or `--wg 512` to force a compiled workgroup-size variant while benchmarking. Use `--compare-wg 64,128,256,512` to compare variants in one run.

## Vulkan C++ Wrapper

The C++ wrapper records the Vulkan sort dispatches:

```cpp
#include <onesweep/vulkan/radix_sort.hpp>

onesweep::vulkan::RadixSortCreateInfo create{};
create.device = device;
create.physical_device = physicalDevice;
create.spirv_directory = ONESWEEP_RADIX_SORT_SHADER_DIR;
create.initial_capacity = maxItemCount;
create.preferred_workgroup_size = 0; // auto-select; 64, 128, 256, or 512 may be forced

onesweep::vulkan::RadixSort sorter(create);

onesweep::vulkan::RadixSortDesc desc{};
desc.keys = keyBuffer;
desc.values = valueBuffer;       // optional, may be VK_NULL_HANDLE
desc.count = itemCount;
desc.descending = true;

sorter.sort(cmd, desc);
```

Segmented sort:

```cpp
onesweep::vulkan::SegmentedRadixSortDesc segmented{};
segmented.keys = keyBuffer;
segmented.values = valueBuffer;  // optional
segmented.segment_offsets = segmentOffsetsBuffer;
segmented.segment_tile_offsets = segmentTileOffsetsBuffer;
segmented.count = itemCount;
segmented.segment_count = segmentCount;
segmented.tile_count = totalSegmentTileCount;
segmented.descending = true;

sorter.sort_segmented(cmd, segmented);
```

Build `segment_tile_offsets` and `totalSegmentTileCount` with `sorter.workgroup_size()`. The command buffer must already be recording. Key and value buffers must be storage buffers. The default four 8-bit passes place the sorted result back in the original buffers.

## Integration Snippets

Source snippets live under `examples/` by language:

- `examples/cpp/`: Vulkan, DX12, Metal, WGPU.
- `examples/c/`: Vulkan and WGPU/native-style snippets.
- `examples/rust/`: Vulkan, DX12, Metal, WGPU.
- `examples/csharp/`: Vulkan, DX12, WGPU.
- `examples/java/`: Vulkan via LWJGL.
- `examples/kotlin/`: Vulkan via LWJGL.

These snippets show shader loading, storage-buffer bindings, control data, dispatch order, and synchronization.

## Constraints

The 8-bit scatter path requires wave/subgroup ballot and partitioned-match support. Workgroup size is selected from compiled shader variants; the Vulkan wrapper builds 64-, 128-, 256-, and 512-thread variants.
