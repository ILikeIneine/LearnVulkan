cmake_minimum_required (VERSION 3.8)

project ("LearnVulkan")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

OPTION(HIDE_CMD_WINDOW "Hide CMD Window, only open for debugging" ON)

#--------------------------------------------------------------------
# Hide the console window in visual studio projects
#--------------------------------------------------------------------
if(MSVC)
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup")
endif()

include_directories(${CMAKE_SOURCE_DIR}/third_party/vulkan/include)

include(FetchContent)
FetchContent_Declare(
    fetch_vk_bootstrap
    GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap
    GIT_TAG        v1.3.287 #suggest using a tag so the library doesn't update whenever new commits are pushed to a branch
)
FetchContent_MakeAvailable(fetch_vk_bootstrap)

# scoop on msvc
find_package(Vulkan REQUIRED)

# sudirectories
add_subdirectory(third_party/glfw)
add_subdirectory(third_party/glm)
add_subdirectory(third_party/ASSIMP)
add_subdirectory(third_party/VulkanMemoryAllocator)
add_subdirectory(third_party/SDL)
add_subdirectory(third_party/simdjson)
add_subdirectory(third_party/fastgltf)

add_subdirectory(projects)
set(HIDE_CMD_WINDOW OFF CACHE "Hide CMD Window, only open for debugging" BOOL FORCE)
