cmake_minimum_required (VERSION 3.8)

project (vulkan)

file(GLOB_RECURSE ${PROJECT_NAME}_HEADER
				${CMAKE_CURRENT_SOURCE_DIR}/include/*/*.h
				${CMAKE_CURRENT_SOURCE_DIR}/include/*/*.hpp)

add_library(${PROJECT_NAME} INTERFACE ${${PROJECT_NAME}_HEADER})

set_target_properties(vulkan PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/VulkanRT/x64/vulkan-1.dll)

target_include_directories(${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
