Project(HelloWorld)

file(GLOB_RECURSE ${PROJECT_NAME}_HEADERS 
                ${CMAKE_CURRENT_SOURCE_DIR}/headers/*.h
                ${CMAKE_CURRENT_SOURCE_DIR}/headers/*.hpp)

file(GLOB_RECURSE ${PROJECT_NAME}_SOURCES 
                ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
                ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

file(GLOB_RECURSE ${PROJECT_NAME}_SHADERS
                ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*)

# generate headers
set(PROJECT_HOME ${CMAKE_CURRENT_SOURCE_DIR})
configure_file(config.h.in ${CMAKE_CURRENT_BINARY_DIR}/gen/config.h @ONLY)

# generate shaders
add_shaders(${PROJECT_NAME}_ShaderGen ${${PROJECT_NAME}_SHADERS})

add_executable(${PROJECT_NAME} ${${PROJECT_NAME}_SOURCES} ${${PROJECT_NAME}_HEADERS})

add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_ShaderGen)

target_sources(${PROJECT_NAME} PRIVATE ${${PROJECT_NAME}_SHADERS})

target_include_directories(${PROJECT_NAME} PRIVATE 
                            ${CMAKE_CURRENT_SOURCE_DIR}/headers
                            ${CMAKE_SOURCE_DIR}/third_party
                            ${CMAKE_CURRENT_BINARY_DIR}/gen)

target_link_libraries(${PROJECT_NAME} glfw glm ${Vulkan_LIBRARY})

set_target_properties(
             ${PROJECT_NAME}
             PROPERTIES
             WIN32_EXECUTABLE 
             ${HIDE_CMD_WINDOW} 
)
