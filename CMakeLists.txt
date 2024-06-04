cmake_minimum_required (VERSION 3.8)

project ("LearnVulkan")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

OPTION(HIDE_CMD_WINDOW "Hide CMD Window, only open for debugging" ON)

#--------------------------------------------------------------------
# Hide the console window in visual studio projects
#--------------------------------------------------------------------
if(MSVC)
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup")
endif()

include_directories(${CMAKE_SOURCE_DIR}/third_party/vulkan/include)

IF (NOT Vulkan_FOUND)
	find_library(Vulkan_LIBRARY NAMES vulkan-1 vulkan PATHS ${CMAKE_SOURCE_DIR}/libs/VulkanRT)
	IF (Vulkan_LIBRARY)
		set(Vulkan_FOUND ON)
		MESSAGE("-- Using bundled Vulkan library version")
	ELSE()
		message(WARNING "Could not find Vulkan library. Please ensure it is installed.")
	ENDIF()
ENDIF()


# sudirectories
add_subdirectory(third_party/glfw)
add_subdirectory(third_party/glm)
add_subdirectory(third_party/ASSIMP)

add_subdirectory(projects)
set(HIDE_CMD_WINDOW OFF CACHE "Hide CMD Window, only open for debugging" BOOL FORCE)
