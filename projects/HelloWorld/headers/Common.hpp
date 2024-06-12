#pragma once

#include <glm/glm.hpp>

struct UniformBufferObject
{
    glm::vec2 foo;
    alignas(16) glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};