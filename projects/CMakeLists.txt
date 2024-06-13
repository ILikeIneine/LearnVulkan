set(GEN_SHADER_HOME ${CMAKE_CURRENT_BINARY_DIR}/gen/shaders)

function(add_shaders TARGET_NAME)
    set(SHADER_SOURCE_FILES ${ARGN})

    list(LENGTH SHADER_SOURCE_FILES FILE_COUNT)
    if (FILE_COUNT EQUAL 0)
        message(FATAL_ERROR "Cannot create a shaders target without any source files")
    endif()

    set(SHADER_COMMANDS)
    set(SHADER_PRODUCTS)

    foreach(SHADER_SOURCE IN LISTS SHADER_SOURCE_FILES)
        cmake_path(GET SHADER_SOURCE FILENAME SHADER_NAME)

        list(APPEND SHADER_COMMANDS COMMAND)
        list(APPEND SHADER_COMMANDS Vulkan::glslc)
        list(APPEND SHADER_COMMANDS "${SHADER_SOURCE}")
        list(APPEND SHADER_COMMANDS "-o")
        list(APPEND SHADER_COMMANDS "${GEN_SHADER_HOME}/${SHADER_NAME}.spv")

        list(APPEND SHADER_PRODUCTS "${GEN_SHADER_HOME}/${SHADER_NAME}.spv")
    endforeach()

    add_custom_target(${TARGET_NAME} ALL
        ${SHADER_COMMANDS}
        COMMENT "Compiling Shaders [${TARGET_NAME}]"
        BYPRODUCTS ${SHADER_PRODUCTS}
    )
endfunction()

add_subdirectory(HelloWorld)
