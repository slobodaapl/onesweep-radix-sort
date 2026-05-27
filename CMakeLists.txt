cmake_minimum_required(VERSION 3.20)
project(onesweep_radix_sort LANGUAGES CXX)

enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Vulkan REQUIRED)
find_program(DXC dxc REQUIRED)
find_program(SPIRV_VAL spirv-val)

add_library(onesweep_radix_sort_vulkan STATIC
  src/vulkan_radix_sort.cpp
)

target_include_directories(onesweep_radix_sort_vulkan PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(onesweep_radix_sort_vulkan PUBLIC Vulkan::Vulkan)

target_compile_definitions(onesweep_radix_sort_vulkan PUBLIC
  ONESWEEP_RADIX_SORT_SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders"
)

set(ONESWEEP_SHADER_NAMES
  radix_clear
  radix_histogram
  radix_bin_base
  radix_scatter_lookback
)

set(ONESWEEP_WORKGROUP_VARIANTS wg64 wg128 wg256 wg512 wg512x32)
set(ONESWEEP_HLSL_SPIRV_OUTPUTS)
foreach(variant_name ${ONESWEEP_WORKGROUP_VARIANTS})
  if(variant_name STREQUAL "wg64")
    set(workgroup_size 64)
    set(workgroup_size_x 64)
    set(workgroup_size_y 1)
    set(max_subgroups 2)
  elseif(variant_name STREQUAL "wg128")
    set(workgroup_size 128)
    set(workgroup_size_x 128)
    set(workgroup_size_y 1)
    set(max_subgroups 4)
  elseif(variant_name STREQUAL "wg256")
    set(workgroup_size 256)
    set(workgroup_size_x 256)
    set(workgroup_size_y 1)
    set(max_subgroups 8)
  elseif(variant_name STREQUAL "wg512")
    set(workgroup_size 512)
    set(workgroup_size_x 512)
    set(workgroup_size_y 1)
    set(max_subgroups 16)
  elseif(variant_name STREQUAL "wg512x32")
    set(workgroup_size 512)
    set(workgroup_size_x 32)
    set(workgroup_size_y 16)
    set(max_subgroups 16)
  endif()
  foreach(shader ${ONESWEEP_SHADER_NAMES})
    set(source ${CMAKE_CURRENT_SOURCE_DIR}/shaders/hlsl/${shader}.hlsl)
    set(output ${CMAKE_CURRENT_BINARY_DIR}/shaders/${variant_name}/${shader}.comp.spv)
    list(APPEND ONESWEEP_HLSL_SPIRV_OUTPUTS ${output})
    set(validate_command)
    if(SPIRV_VAL)
      set(validate_command COMMAND ${SPIRV_VAL} --target-env vulkan1.2 ${output})
    endif()
    add_custom_command(
      OUTPUT ${output}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders/${variant_name}
      COMMAND ${DXC} -spirv -T cs_6_6 -E main -fspv-target-env=vulkan1.2
              -DONESWEEP_WORKGROUP_SIZE=${workgroup_size}
              -DONESWEEP_WORKGROUP_SIZE_X=${workgroup_size_x}
              -DONESWEEP_WORKGROUP_SIZE_Y=${workgroup_size_y}
              -DONESWEEP_MAX_SUBGROUPS=${max_subgroups}
              ${source} -Fo ${output}
      ${validate_command}
      DEPENDS ${source} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/hlsl/common.hlsli
      VERBATIM
    )
  endforeach()
  foreach(shader radix_histogram radix_scatter_lookback)
    set(source ${CMAKE_CURRENT_SOURCE_DIR}/shaders/hlsl/${shader}.hlsl)
    set(output ${CMAKE_CURRENT_BINARY_DIR}/shaders/${variant_name}/${shader}_flat.comp.spv)
    list(APPEND ONESWEEP_HLSL_SPIRV_OUTPUTS ${output})
    set(validate_command)
    if(SPIRV_VAL)
      set(validate_command COMMAND ${SPIRV_VAL} --target-env vulkan1.2 ${output})
    endif()
    add_custom_command(
      OUTPUT ${output}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders/${variant_name}
      COMMAND ${DXC} -spirv -T cs_6_6 -E main -fspv-target-env=vulkan1.2
              -DONESWEEP_WORKGROUP_SIZE=${workgroup_size}
              -DONESWEEP_WORKGROUP_SIZE_X=${workgroup_size_x}
              -DONESWEEP_WORKGROUP_SIZE_Y=${workgroup_size_y}
              -DONESWEEP_MAX_SUBGROUPS=${max_subgroups}
              -DONESWEEP_FLAT_ONLY=1
              ${source} -Fo ${output}
      ${validate_command}
      DEPENDS ${source} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/hlsl/common.hlsli
      VERBATIM
    )
  endforeach()
endforeach()

add_custom_target(onesweep_radix_sort_shaders ALL DEPENDS ${ONESWEEP_HLSL_SPIRV_OUTPUTS})
add_dependencies(onesweep_radix_sort_vulkan onesweep_radix_sort_shaders)

if(SPIRV_VAL)
  message(STATUS "spirv-val found; generated SPIR-V will be validated")
else()
  message(STATUS "spirv-val not found; generated SPIR-V validation will be skipped")
endif()

add_executable(onesweep_radix_sort_vulkan_u32_test tests/sort_u32_test.cpp)
target_link_libraries(onesweep_radix_sort_vulkan_u32_test PRIVATE onesweep_radix_sort_vulkan)
add_dependencies(onesweep_radix_sort_vulkan_u32_test onesweep_radix_sort_shaders)
add_test(NAME onesweep_radix_sort_vulkan_u32 COMMAND onesweep_radix_sort_vulkan_u32_test)

add_executable(onesweep_radix_sort_vulkan_u32_benchmark examples/benchmark_u32.cpp)
target_link_libraries(onesweep_radix_sort_vulkan_u32_benchmark PRIVATE onesweep_radix_sort_vulkan)
add_dependencies(onesweep_radix_sort_vulkan_u32_benchmark onesweep_radix_sort_shaders)
