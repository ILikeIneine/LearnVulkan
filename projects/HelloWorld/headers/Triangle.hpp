#pragma once

#include <vector>
#include <algorithm>
#include <optional>
#include <set>
#include <fstream>
#include <format>
#include <chrono>
#include <map>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#define GLM_FORCE_RADIANS
#include <glm/gtc/matrix_transform.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include "config.h"
#include "Vertex.hpp"
#include "Common.hpp"

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector validationLayers = { "VK_LAYER_KHRONOS_validation" };

const std::vector deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

static VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                             const VkAllocationCallbacks* pAllocator,
                                             VkDebugUtilsMessengerEXT* pDebugMessenger)
{
    if (auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(
        instance, "vkCreateDebugUtilsMessengerEXT")))
    {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }

    return VK_ERROR_EXTENSION_NOT_PRESENT;
}

static void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger,
                                          const VkAllocationCallbacks* pAllocator)
{
    if (auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT")))
    {
        func(instance, debugMessenger, pAllocator);
    }
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData)
{
    std::cerr << "\n-------------------- validation layer -------------------- \n"
              << "---------------------------------------------------------- \n" 
              << pCallbackData->pMessage << '\n';

    return VK_FALSE;
}

static std::vector<char> readFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if(!file.is_open())
    {
        throw std::runtime_error(std::format("failed to open file: {}\n", filename));
    }
    auto fileSize = static_cast<std::size_t>(file.tellg());
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    return buffer;
}

struct QueueFamilyIndices
{
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    [[nodiscard]]
    bool isComplete() const
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    void initWindow()
    {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window_ = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height)
    {
        auto app = static_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void initVulkan() {
        // init vulkan instance
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createFrameBuffers();
        createCommandPool();

        createTextureImage();
        createTextureImageView();
        createTextureSampler();

        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();

        createDescriptorPool();
        createDescriptorSets();

        createCommandBuffer();
        createSyncObjects();
    }
    
    void cleanup() const
    {
        cleanupSwapChain();

        vkDestroySampler(device_, textureSampler_, nullptr);
        vkDestroyImageView(device_, textureImageView_, nullptr);
        vkDestroyImage(device_, textureImage_, nullptr);
        vkFreeMemory(device_, textureImageMemory_, nullptr);

        vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);

        for(std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            vkDestroySemaphore(device_, renderFinishedSemaphores_[i], nullptr);
            vkDestroySemaphore(device_, imageAvailableSemaphores_[i], nullptr);
            vkDestroyFence(device_, inFlightFences_[i], nullptr);
        }

        vkDestroyPipeline(device_, graphicsPipeline_, nullptr);
        vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        vkDestroyRenderPass(device_, renderPass_, nullptr);

        vkDestroyCommandPool(device_, commandPool_, nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroyBuffer(device_, uniformBuffers_[i], nullptr);
            vkFreeMemory(device_, uniformBuffersMemory_[i], nullptr);
        }

        vkDestroyBuffer(device_, vertexBuffer_, nullptr);
        vkFreeMemory(device_, vertexBufferMemory_, nullptr);

        vkDestroyBuffer(device_, indexBuffer_, nullptr);
        vkFreeMemory(device_, indexBufferMemory_, nullptr);

        vkDestroyDevice(device_, nullptr);

        if constexpr(enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance_, debugMessenger_, nullptr);
        }

        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        vkDestroyInstance(instance_, nullptr);
        
        glfwDestroyWindow(window_);

        glfwTerminate();
    }


    void mainLoop() 
    {
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            drawFrame();
        }

        vkDeviceWaitIdle(device_);
    }


    void createSyncObjects()
    {
        VkSemaphoreCreateInfo semaphoreInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        };

        VkFenceCreateInfo fenceInfo{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
        };

        for(std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            if (vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &imageAvailableSemaphores_[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &renderFinishedSemaphores_[i]) != VK_SUCCESS ||
                vkCreateFence(device_, &fenceInfo, nullptr, &inFlightFences_[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create semaphores!");
            }
        }
    }
    void updateUniformBuffer(uint32_t currentImage) const
    {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo{};
        ubo.model = glm::rotate(glm::mat4{ 1.0 }, time * glm::radians(90.0f), glm::vec3{ 0.0,0.0,1.0 });
        ubo.view = glm::lookAt(glm::vec3{ 2.0,2.0,2.0 }, glm::vec3{ 0.0,0.0,0.0 }, glm::vec3(0.0, 0.0, 1.0));
        ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent_.width / (float)swapChainExtent_.height, 0.1f, 10.f);
        ubo.proj[1][1] *= -1;

        memcpy(uniformBuffersMapped_[currentImage], &ubo, sizeof(ubo));
    }

    void drawFrame() 
    {
        // Fences
        vkWaitForFences(device_, 1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);

        // Get an image frame in swap chain
        uint32_t image_index;
        auto result = vkAcquireNextImageKHR(device_, swapChain_, UINT64_MAX, imageAvailableSemaphores_[currentFrame_], VK_NULL_HANDLE, &image_index);
        if (result == VK_ERROR_OUT_OF_DATE_KHR)
        {
            recreateSwapChain();
            return;
        }
        if (not (result == VK_SUCCESS or result == VK_SUBOPTIMAL_KHR))
        {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        vkResetFences(device_, 1, &inFlightFences_[currentFrame_]);

        // Init and record command buffer
        updateUniformBuffer(currentFrame_);

        vkResetCommandBuffer(commandBuffers_[currentFrame_], 0);
        recordCommandBuffer(commandBuffers_[currentFrame_], image_index);

        // submit command
        VkSubmitInfo submit_info{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO};
        std::array wait_semaphores = {imageAvailableSemaphores_[currentFrame_]};
        std::array<VkPipelineStageFlags, 1> wait_stages = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submit_info.waitSemaphoreCount = 1;
        submit_info.pWaitSemaphores = wait_semaphores.data();
        submit_info.pWaitDstStageMask = wait_stages.data();

        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &commandBuffers_[currentFrame_];

        // Signals fired on finish command buffer
        std::array signal_semaphores = {renderFinishedSemaphores_[currentFrame_]};
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = signal_semaphores.data();

        if (vkQueueSubmit(graphicsQueue_, 1, &submit_info, inFlightFences_[currentFrame_]) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        VkPresentInfoKHR present_info{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = signal_semaphores.data(),
        };

        std::array swap_chains = {swapChain_};
        present_info.swapchainCount = 1;
        present_info.pSwapchains = swap_chains.data();
        present_info.pImageIndices = &image_index;
        present_info.pResults = nullptr; // Optional

        result = vkQueuePresentKHR(presentQueue_, &present_info);

        if (result == VK_ERROR_OUT_OF_DATE_KHR or result == VK_SUBOPTIMAL_KHR or framebufferResized)
        {
            framebufferResized = false;
            recreateSwapChain();
        }
        else if (result != VK_SUCCESS)
        {
            throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame_ = (currentFrame_ + 1) & MAX_FRAMES_IN_FLIGHT;
    }

     void createSurface()
    {
         if (glfwCreateWindowSurface(instance_, window_, nullptr, &surface_) != VK_SUCCESS) {
             throw std::runtime_error("failed to create window surface!");
         }
    }


    void createInstance()
    {
        VkApplicationInfo appInfo{
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = "Hello Triangle",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "No Engine",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = VK_API_VERSION_1_0,
        };

        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        VkInstanceCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &appInfo,
            .enabledLayerCount = 0,
            .enabledExtensionCount = glfwExtensionCount,
            .ppEnabledExtensionNames = glfwExtensions,
        };

        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        if constexpr (enableValidationLayers)
        {
            if (!checkValidationLayerSupport())
            {
                throw std::runtime_error("validation layers requested, but not available!");
            }
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            auto debugCreateInfo = populateDebugMessengerCreateInfo();
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
        }
        else
        {
            createInfo.enabledLayerCount = 0;
            createInfo.pNext = nullptr;
        }

        if(vkCreateInstance(&createInfo, nullptr, &instance_))
        {
            throw std::runtime_error("failed to create instance");
        }

        //uint32_t extensionCount = 0;
        //vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

        //std::vector<VkExtensionProperties> extensions(extensionCount);
        //vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

        //std::cout << "available extensions:\n";
        //for (const auto& extension : extensions) {
        //    std::cout << '\t' << extension.extensionName << '\n';
        //}
    }

    void setupDebugMessenger()
    {
        (void)this;

        if constexpr (!enableValidationLayers) return;

        auto createInfo = populateDebugMessengerCreateInfo();

        if (CreateDebugUtilsMessengerEXT(instance_, &createInfo, nullptr, &debugMessenger_) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }

    auto populateDebugMessengerCreateInfo() const -> VkDebugUtilsMessengerCreateInfoEXT
    {
        (void)this;
        VkDebugUtilsMessengerCreateInfoEXT createInfo = {
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = debugCallback,
        };
        return createInfo;
    }

    void pickPhysicalDevice()
    {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);

        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice_ = device;
                break;
            }
        }
        if (physicalDevice_ == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }

        /*
        std::multimap<std::size_t, VkPhysicalDevice> candidates;
        for (const auto& device : devices) {
            auto score = rateDeviceSuitability(device);
            candidates.insert(std::make_pair(score, device));
        }

        // Check if the best candidate is suitable at all
        if (candidates.rbegin()->first > 0) {
            physicalDevice_ = candidates.rbegin()->second;
        }
        else {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
        */
    }

    std::size_t rateDeviceSuitability(VkPhysicalDevice device) const
    {
        (void)this;
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);

        VkPhysicalDeviceFeatures deviceFeatures;
        vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

        std::size_t score = 0;

        std::cout << deviceProperties.deviceName << '\n';

        // Discrete GPUs have a significant performance advantage
        if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            score += 1000;
        }

        // Maximum possible size of textures affects graphics quality
        score += deviceProperties.limits.maxImageDimension2D;

        // Application can't function without geometry shaders
        if (!deviceFeatures.geometryShader) {
            return 0;
        }

        return score;
    }

    [[nodiscard]]
    bool isDeviceSuitable(VkPhysicalDevice device) const
    {
        QueueFamilyIndices indices = findQueueFamilies(device);
        // check 1
        bool extensionsSupported = checkDeviceExtensionSupport(device);

        // check 2
        bool swapChainAdequate = false;
        if (extensionsSupported) {
            auto swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        // check 3
        VkPhysicalDeviceFeatures supportedFeatures;
        vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

        return indices.isComplete() and extensionsSupported and swapChainAdequate and supportedFeatures.samplerAnisotropy;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device) const {
        (void)this;
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        // If all required extensions are satisfied
        return requiredExtensions.empty();
    }

    void createLogicalDevice()
    {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice_);
        if (!indices.isComplete()) { throw std::runtime_error("failed to create logical device!"); }

        // Prepare queue infos
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos{};
        std::set uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies)
        {
            VkDeviceQueueCreateInfo queueCreateInfo{
                .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .queueFamilyIndex = queueFamily,
                .queueCount = 1,
                .pQueuePriorities = &queuePriority,
            };
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures{
            .samplerAnisotropy = VK_TRUE,
        };

        VkDeviceCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
            .pQueueCreateInfos = queueCreateInfos.data(),
            .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
            .ppEnabledExtensionNames = deviceExtensions.data(),
            .pEnabledFeatures = &deviceFeatures,
        };

        if constexpr (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(device_, indices.graphicsFamily.value(), 0, &graphicsQueue_);
        vkGetDeviceQueue(device_, indices.presentFamily.value(), 0, &presentQueue_);
    }

    [[nodiscard]]
    auto findQueueFamilies(VkPhysicalDevice device) const -> QueueFamilyIndices 
    {
        (void)this;
        QueueFamilyIndices indices;
        // Assign index to queue families that could be found

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {

            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface_, &presentSupport);

            if (presentSupport) {
                indices.presentFamily = i;
            }

            if (indices.isComplete()) {
                break;
            }
            i++;
        }
        return indices;
    }

    [[nodiscard]]
    auto querySwapChainSupport(VkPhysicalDevice device) const -> SwapChainSupportDetails
    {
        SwapChainSupportDetails details;
        // Query capabilities
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface_, &details.capabilities);
        // Query formats
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &formatCount, nullptr);

        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &formatCount, details.formats.data());
        }
        // Query present mode
        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &presentModeCount, nullptr);

        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    [[nodiscard]]
    auto chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) const -> VkSurfaceFormatKHR
    {
        (void)this;
        auto pred = [](auto&& availableFormat)
        {
            return availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace ==
                VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        };

        if (const auto result = std::ranges::find_if(availableFormats, pred); result != std::end(availableFormats))
        {
            return *result;
        }

        return availableFormats[0];
    }

    [[nodiscard]]
    auto chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) const -> VkPresentModeKHR
    {
        (void)this;
        if (std::ranges::find(availablePresentModes, VK_PRESENT_MODE_MAILBOX_KHR) != std::end(availablePresentModes))
        {
            return VK_PRESENT_MODE_MAILBOX_KHR;
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    [[nodiscard]]
    auto getRequiredExtensions() const -> std::vector<const char*>
    {
        (void)this;
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if constexpr (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    void createSwapChain()
    {
        auto swap_chain_support_details = querySwapChainSupport(physicalDevice_);
        
        auto surface_format = chooseSwapSurfaceFormat(swap_chain_support_details.formats);
        auto present_mode = chooseSwapPresentMode(swap_chain_support_details.presentModes);
        auto extent = chooseSwapExtent(swap_chain_support_details.capabilities);

        auto image_count = swap_chain_support_details.capabilities.minImageCount + 1;

        if (swap_chain_support_details.capabilities.maxImageCount > 0 && image_count > swap_chain_support_details.capabilities.maxImageCount)
            image_count = swap_chain_support_details.capabilities.maxImageCount;

        VkSwapchainCreateInfoKHR create_info{
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface = surface_,
            .minImageCount = image_count,
            .imageFormat = surface_format.format,
            .imageColorSpace = surface_format.colorSpace,
            .imageExtent = extent,
            .imageArrayLayers = 1,
            .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        };

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice_);
        std::array queueFamilyIndices = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        if (indices.graphicsFamily != indices.presentFamily) {
            create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            create_info.queueFamilyIndexCount = 2;
            create_info.pQueueFamilyIndices = queueFamilyIndices.data();
        }
        else {
            create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            create_info.queueFamilyIndexCount = 0; // Optional
            create_info.pQueueFamilyIndices = nullptr; // Optional
        }

        create_info.preTransform = swap_chain_support_details.capabilities.currentTransform;
        create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

        create_info.presentMode = present_mode;
        create_info.clipped = VK_TRUE;

        create_info.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(device_, &create_info, nullptr, &swapChain_) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create swap chain!");
        }

        vkGetSwapchainImagesKHR(device_, swapChain_, &image_count, nullptr);
        swapChainImages_.resize(image_count);
        vkGetSwapchainImagesKHR(device_, swapChain_, &image_count, swapChainImages_.data());

        swapChainImageFormat_ = surface_format.format;
        swapChainExtent_ = extent;
    }

    void cleanupSwapChain() const
    {
        for (const auto framebuffer : swapChainFramebuffers_) {
            vkDestroyFramebuffer(device_, framebuffer, nullptr);
        }

        for (const auto image_view : swapChainImageViews_) {
            vkDestroyImageView(device_, image_view, nullptr);
        }

        vkDestroySwapchainKHR(device_, swapChain_, nullptr);
    }

    void recreateSwapChain()
    {
        int width{};
        int height{};
        glfwGetFramebufferSize(window_, &width, &height);
        while(width == 0 or height == 0)
        {
            glfwGetFramebufferSize(window_, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device_);

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createFrameBuffers();
    }

    VkImageView createImageView(VkImage image, VkFormat format) const
    {
        VkImageViewCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = format,
            .components = {
                .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                .a = VK_COMPONENT_SWIZZLE_IDENTITY,
            },
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };

        VkImageView imageView;
        if (vkCreateImageView(device_, &createInfo, nullptr, &imageView) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image views!");
        }
        return imageView;
    }

    void createImageViews()
    {
        swapChainImageViews_.resize(swapChainImages_.size());
        for (size_t i = 0; i < swapChainImages_.size(); i++) {
            swapChainImageViews_[i] = createImageView(swapChainImages_[i], swapChainImageFormat_);
        }
    }

    void createTextureImage()
    {
        int texWidth, texHeight, texChannels;
        auto pixels = stbi_load(RESOURCES_HOME"/textures/texture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        VkDeviceSize imageSize = static_cast<VkDeviceSize>(texWidth) * texHeight * 4;

        if(!pixels)
        {
            throw std::runtime_error("failed to load texture image!");
        }

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;

        createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device_, stagingBufferMemory, 0, imageSize, 0, &data);
        std::memcpy(data, pixels, imageSize);
        vkUnmapMemory(device_, stagingBufferMemory);

        stbi_image_free(pixels);

        createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage_, textureImageMemory_);

        transitionImageLayout(textureImage_, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        copyBufferToImage(stagingBuffer, textureImage_, texWidth, texHeight);
        transitionImageLayout(textureImage_, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        vkDestroyBuffer(device_, stagingBuffer, nullptr);
        vkFreeMemory(device_, stagingBufferMemory, nullptr);
    }


    void createTextureImageView()
    {
        textureImageView_ = createImageView(textureImage_, VK_FORMAT_R8G8B8A8_SRGB);
    }

    void createTextureSampler()
    {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice_, &properties);

        VkSamplerCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_LINEAR,
            .minFilter = VK_FILTER_LINEAR,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .mipLodBias = 0.0f,
            .anisotropyEnable = VK_TRUE,
            .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
            .compareEnable = VK_FALSE,
            .compareOp = VK_COMPARE_OP_ALWAYS,
            .minLod = 0.0f,
            .maxLod = 0.0f,
            .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
            .unnormalizedCoordinates = VK_FALSE,
        };

        if (vkCreateSampler(device_, &createInfo, nullptr, &textureSampler_)!=VK_SUCCESS)
        {
            throw std::runtime_error("failed to create texture sampler!");
        }

    }


    void createRenderPass()
    {
        VkAttachmentDescription colorAttachment {
            .format = swapChainImageFormat_,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        };

        VkAttachmentReference colorAttachmentRef {
            .attachment = 0,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };

        VkSubpassDescription subpass {
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorAttachmentRef,
        };

        VkSubpassDependency dependency{
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        };

        VkRenderPassCreateInfo renderPassInfo{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments = &colorAttachment,
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 1,
            .pDependencies = &dependency,
        };


        if (vkCreateRenderPass(device_, &renderPassInfo, nullptr, &renderPass_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    void createDescriptorSetLayout()
    {
        VkDescriptorSetLayoutBinding uboLayoutBinding{
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            .pImmutableSamplers = nullptr,
        };

        VkDescriptorSetLayoutBinding samplerLayoutBinding{
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        };

        std::array bindings = { uboLayoutBinding , samplerLayoutBinding };

        VkDescriptorSetLayoutCreateInfo layoutCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data(),
        };

        if (vkCreateDescriptorSetLayout(device_, &layoutCreateInfo, nullptr, &descriptorSetLayout_))
        {
            throw std::runtime_error("failed to create descriptor set layout!");
        }

    }

    void createGraphicsPipeline()
    {
        auto vert_shader_code = readFile(GEN_SHADER_HOME "/shader.vert.spv");
        auto frag_shader_code = readFile(GEN_SHADER_HOME "/shader.frag.spv");

        auto vertShaderModule = createShaderModule(vert_shader_code);
        auto fragShaderModule = createShaderModule(frag_shader_code);

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();


        VkPipelineShaderStageCreateInfo vertShaderStageInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertShaderModule,
            .pName = "main",
        };

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragShaderModule,
            .pName = "main",
        };

        std::array shaderStages = { vertShaderStageInfo, fragShaderStageInfo };
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDescription,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
            .pVertexAttributeDescriptions = attributeDescriptions.data(),
        };

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE,
        };

        VkViewport viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(swapChainExtent_.width),
            .height = static_cast<float>(swapChainExtent_.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };

        VkPipelineViewportStateCreateInfo viewportState{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .scissorCount = 1,
        };

        VkRect2D scissor{
            .offset = { 0, 0 },
            .extent = swapChainExtent_,
        };

        VkPipelineRasterizationStateCreateInfo rasterizer{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,

            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,

            .depthBiasEnable = VK_FALSE,
            .depthBiasConstantFactor = 0.0f, // Optional
            .depthBiasClamp = 0.0f, // Optional
            .depthBiasSlopeFactor = 0.0f, // Optional

            .lineWidth = 1.0f,
        };

        VkPipelineMultisampleStateCreateInfo multisampling{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable = VK_FALSE,
            .minSampleShading = 1.0f, // Optional
            .pSampleMask = nullptr, // Optional
            .alphaToCoverageEnable = VK_FALSE, // Optional
            .alphaToOneEnable = VK_FALSE, // Optional
        };

        VkPipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = VK_FALSE,
            .srcColorBlendFactor = VK_BLEND_FACTOR_ONE, // Optional
            .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO, // Optional
            .colorBlendOp = VK_BLEND_OP_ADD, // Optional
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE, // Optional
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO, // Optional
            .alphaBlendOp = VK_BLEND_OP_ADD, // Optional
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        };

        VkPipelineColorBlendStateCreateInfo colorBlending{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = VK_FALSE,
            .logicOp = VK_LOGIC_OP_COPY, // Optional
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment,
        };
        colorBlending.blendConstants[0] = 0.0f; // Optional
        colorBlending.blendConstants[1] = 0.0f; // Optional
        colorBlending.blendConstants[2] = 0.0f; // Optional
        colorBlending.blendConstants[3] = 0.0f; // Optional

        std::array dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };

        VkPipelineDynamicStateCreateInfo dynamicStateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data(),
        };

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1, // Optional
            .pSetLayouts = &descriptorSetLayout_, // Optional
            .pushConstantRangeCount = 0, // Optional
            .pPushConstantRanges = nullptr, // Optional
        };

        if (vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipelineLayout_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            .pStages = shaderStages.data(),
            //
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = nullptr,
            .pColorBlendState = &colorBlending,
            .pDynamicState = &dynamicStateInfo,
            //
            .layout = pipelineLayout_,
            //
            .renderPass = renderPass_,
            .subpass = 0,
            // 
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1,
        };

        if (vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline_) !=
            VK_SUCCESS)
        {
            throw std::runtime_error("failed to create graphics pipeline!");
        }
        
        vkDestroyShaderModule(device_, fragShaderModule, nullptr);
        vkDestroyShaderModule(device_, vertShaderModule, nullptr);
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer,
                      VkDeviceMemory& bufferMemory) const
    {
        VkBufferCreateInfo bufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };

        if (vkCreateBuffer(device_, &bufferCreateInfo, nullptr, &buffer) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create vertex buffer!");
        }

        VkMemoryRequirements memoryRequirements;
        vkGetBufferMemoryRequirements(device_, buffer, &memoryRequirements);

        VkMemoryAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memoryRequirements.size,
            .memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, properties)
        };

        if (vkAllocateMemory(device_, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate vertex buffer memory!");
        }

        vkBindBufferMemory(device_, buffer, bufferMemory, 0);
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) const
    {

        auto commandBuffer = beginSingleTimeCommands();

        VkBufferCopy copyRegion{ .size = size, };
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        endSingleTimeCommands(commandBuffer);

    }

    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, 
        VkImage& image, VkDeviceMemory& imageMemory ) const
    {
        VkImageCreateInfo imageInfo {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = format,
            .extent = {
                .width = width,
                .height = height,
                .depth = 1
            },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = tiling,
            .usage = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };

        if (vkCreateImage(device_, &imageInfo, nullptr, &image) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create image!");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device_, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
        };

        if (vkAllocateMemory(device_, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate image memory!");
        }

        vkBindImageMemory(device_, image, imageMemory, 0);
   
    }

    VkCommandBuffer beginSingleTimeCommands() const
    {
        (void)this;
        VkCommandBufferAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = commandPool_,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1
        };

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device_, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags =    VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };

        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer) const
    {
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer,
        };

        vkQueueSubmit(graphicsQueue_, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue_);

        vkFreeCommandBuffers(device_, commandPool_, 1, &commandBuffer);
    }

    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) const
    {
        (void)this;
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();
        VkImageMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };

        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags destinationStage;

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED and newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if(oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL and newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;

            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else
        {
            throw std::invalid_argument("unsupported layout transition!");
        }

        vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        endSingleTimeCommands(commandBuffer);
    }

    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) const
    {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();
        VkBufferImageCopy region{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,

            .imageSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },

            .imageOffset = {0,0,0},
            .imageExtent = {width, height, 1},
        };

        vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        endSingleTimeCommands(commandBuffer);
    }

    void createVertexBuffer()
    {
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device_, stagingBufferMemory, 0, bufferSize, 0, &data);
        std::memcpy(data, vertices.data(), bufferSize);
        vkUnmapMemory(device_, stagingBufferMemory);

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer_, vertexBufferMemory_);
        copyBuffer(stagingBuffer, vertexBuffer_, bufferSize);

        vkDestroyBuffer(device_, stagingBuffer, nullptr);
        vkFreeMemory(device_, stagingBufferMemory, nullptr);
    }

    void createIndexBuffer()
    {
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device_, stagingBufferMemory, 0, bufferSize, 0, &data);
        std::memcpy(data, indices.data(), bufferSize);
        vkUnmapMemory(device_, stagingBufferMemory);

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer_, indexBufferMemory_);
        copyBuffer(stagingBuffer, indexBuffer_, bufferSize);

        vkDestroyBuffer(device_, stagingBuffer, nullptr);
        vkFreeMemory(device_, stagingBufferMemory, nullptr);
    }

    void createUniformBuffers()
    {
        for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            constexpr VkDeviceSize bufferSize = sizeof(UniformBufferObject);
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers_[i], uniformBuffersMemory_[i]);

            vkMapMemory(device_, uniformBuffersMemory_[i], 0, bufferSize, 0, &uniformBuffersMapped_[i]);
        }
    }

    void createDescriptorPool()
    {
        std::array<VkDescriptorPoolSize, 2> poolSizes{};

        poolSizes[0] = {
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = MAX_FRAMES_IN_FLIGHT,
        };
        poolSizes[1] = {
            .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = MAX_FRAMES_IN_FLIGHT,
        };

        VkDescriptorPoolCreateInfo poolInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data(),
        };

        if (vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }

    void createDescriptorSets()
    {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout_);
        VkDescriptorSetAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptorPool_,
            .descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
            .pSetLayouts = layouts.data(),
        };

        if (vkAllocateDescriptorSets(device_, &allocInfo, descriptorSets_.data())!=VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        for(std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            VkDescriptorBufferInfo bufferInfo{
                .buffer = uniformBuffers_[i],
                .offset = 0,
                .range = sizeof(UniformBufferObject),
            };

            VkDescriptorImageInfo imageInfo{
                .sampler = textureSampler_,
                .imageView = textureImageView_,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };

            std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

            descriptorWrites[0] = {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorSets_[i],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = &bufferInfo,
            };

            descriptorWrites[1] = {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorSets_[i],
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &imageInfo,
            };

            vkUpdateDescriptorSets(device_, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }

    }

    [[nodiscard]]
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const
    {
        VkPhysicalDeviceMemoryProperties memory_properties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memory_properties);
        for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i)
        {
            if ((typeFilter & (1 << i)) and
                (memory_properties.memoryTypes[i].propertyFlags & properties) == properties)
            {
                return i;
            }
        }
        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createFrameBuffers()
    {
        swapChainFramebuffers_.resize(swapChainImageViews_.size());

        for (std::size_t i=0; i < swapChainImageViews_.size(); ++i)
        {
            std::array attachments = { swapChainImageViews_[i] };

            VkFramebufferCreateInfo frameBufferInfo{
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .renderPass = renderPass_,
                .attachmentCount = 1,
                .pAttachments = attachments.data(),
                .width = swapChainExtent_.width,
                .height = swapChainExtent_.height,
                .layers = 1,
            };

            if (vkCreateFramebuffer(device_, &frameBufferInfo, nullptr, &swapChainFramebuffers_[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create frame buffer!");
            }
        }
    }

    void createCommandPool()
    {
        auto queue_family_indices = findQueueFamilies(physicalDevice_);
        VkCommandPoolCreateInfo pool_create_info
        {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = queue_family_indices.graphicsFamily.value(),
        };

        if (vkCreateCommandPool(device_, &pool_create_info, nullptr, &commandPool_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    void createCommandBuffer()
    {
        commandBuffers_.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocate_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = commandPool_,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = static_cast<uint32_t>(commandBuffers_.size()),
        };

        if (vkAllocateCommandBuffers(device_, &allocate_info, commandBuffers_.data())!=VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    void  recordCommandBuffer(VkCommandBuffer command_buffer, uint32_t image_index) const
    {
        VkCommandBufferBeginInfo begin_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = 0,
            .pInheritanceInfo = nullptr,
        };

        if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo render_pass_begin_info{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = renderPass_,
            .framebuffer = swapChainFramebuffers_[image_index],
        };

        render_pass_begin_info.renderArea.offset = { 0 ,0 };
        render_pass_begin_info.renderArea.extent = swapChainExtent_;

        VkClearValue clear_color = { {{0.0f, 0.0f, 0.0f, 1.0f}} };
        render_pass_begin_info.clearValueCount = 1;
        render_pass_begin_info.pClearValues = &clear_color;

        vkCmdBeginRenderPass(command_buffer, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline_);

            VkBuffer vertexBuffers[] = { vertexBuffer_ };
            VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(command_buffer, 0, 1, vertexBuffers, offsets);
            vkCmdBindIndexBuffer(command_buffer, indexBuffer_, 0, VK_INDEX_TYPE_UINT16);
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_, 0, 1, &descriptorSets_[currentFrame_], 0, nullptr);

            VkViewport viewport{
                .x = 0.0f,
                .y = 0.0f,
                .width = static_cast<float>(swapChainExtent_.width),
                .height = static_cast<float>(swapChainExtent_.height),
                .minDepth = 0.0f,
                .maxDepth = 1.0f,
            };
            vkCmdSetViewport(command_buffer, 0, 1, &viewport);

            VkRect2D scissor{
                .offset = { 0, 0 },
                .extent = swapChainExtent_,
            };
            vkCmdSetScissor(command_buffer, 0, 1, &scissor);

            vkCmdDrawIndexed(command_buffer, static_cast<uint32_t>(indices.size()), 1, 0, 0,0);
        vkCmdEndRenderPass(command_buffer);

        if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    auto createShaderModule(const std::vector<char>& code) const -> VkShaderModule
    {
        VkShaderModuleCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = code.size(),
            .pCode = reinterpret_cast<const uint32_t*>(code.data()),
        };

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device_, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }

    auto chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) const -> VkExtent2D
    {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
        {
            return capabilities.currentExtent;
        }
        int width{}; int height{};
        glfwGetFramebufferSize(window_, &width, &height);

        VkExtent2D actualExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };

        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);

        actualExtent.height= std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

        return actualExtent;
    }

    [[nodiscard]]
    bool checkValidationLayerSupport() const
    {
        (void)this;
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers) {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }

        return true;
    }

private:
    GLFWwindow* window_{};
    VkInstance instance_{};
    VkDebugUtilsMessengerEXT debugMessenger_{};
    VkPhysicalDevice physicalDevice_{};
    VkDevice device_{};

    VkQueue graphicsQueue_{};
    VkQueue presentQueue_{};
    
    VkSurfaceKHR surface_{};
    VkSwapchainKHR swapChain_{};
    VkFormat swapChainImageFormat_{};
    VkExtent2D swapChainExtent_{};

    VkRenderPass renderPass_{};
    VkDescriptorSetLayout descriptorSetLayout_{};
    VkPipelineLayout pipelineLayout_{};

    VkPipeline graphicsPipeline_{};
    VkCommandPool commandPool_{};

    std::vector<VkImage> swapChainImages_{};
    std::vector<VkImageView> swapChainImageViews_{};
    std::vector<VkFramebuffer> swapChainFramebuffers_{};

    std::vector<VkCommandBuffer> commandBuffers_{MAX_FRAMES_IN_FLIGHT};
    std::vector<VkSemaphore> imageAvailableSemaphores_{MAX_FRAMES_IN_FLIGHT};
    std::vector<VkSemaphore> renderFinishedSemaphores_{MAX_FRAMES_IN_FLIGHT};
    std::vector<VkFence> inFlightFences_{MAX_FRAMES_IN_FLIGHT};

    bool framebufferResized = false; // hint for window adjustment

    uint32_t currentFrame_{};

    VkBuffer vertexBuffer_{};
    VkDeviceMemory vertexBufferMemory_{};
    VkBuffer indexBuffer_{};
    VkDeviceMemory indexBufferMemory_{};

    std::vector<VkBuffer> uniformBuffers_{ MAX_FRAMES_IN_FLIGHT };
    std::vector<VkDeviceMemory> uniformBuffersMemory_{ MAX_FRAMES_IN_FLIGHT };
    std::vector<void*> uniformBuffersMapped_{ MAX_FRAMES_IN_FLIGHT };
    VkDescriptorPool descriptorPool_{};
    std::vector<VkDescriptorSet> descriptorSets_{ MAX_FRAMES_IN_FLIGHT };

    VkImage textureImage_{};
    VkDeviceMemory textureImageMemory_{};
    VkImageView textureImageView_{};
    VkSampler textureSampler_{};
};
