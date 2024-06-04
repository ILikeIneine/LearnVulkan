#include <iostream>
#include <stdexcept>
#include <cstdlib>

#include "Triangle.hpp"

int main() {
    auto app = HelloTriangleApplication();

    try
    {
        app.run();
    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}