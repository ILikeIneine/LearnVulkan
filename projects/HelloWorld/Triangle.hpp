#pragma once

#include <vector>
#include <map>

#define GLFW_INCLUDE_VULKAN
#include <optional>
#include <GLFW/glfw3.h>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;

const std::vector validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

static VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }

    return VK_ERROR_EXTENSION_NOT_PRESENT;
}

static void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

    std::cerr << "validation layer: " << pCallbackData->pMessage << '\n';

    return VK_FALSE;
}

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;

    bool isComplete() {
        return graphicsFamily.has_value();
    }
};


#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

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
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }
    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        pickPhysicalDevice();
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
        }

        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

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

    std::vector<const char*> getRequiredExtensions() const
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

    void mainLoop() const
    {
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
        }
    }

    void cleanup() const
    {
        if constexpr(enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance_, debugMessenger_, nullptr);
        }

        vkDestroyInstance(instance_, nullptr);
        glfwDestroyWindow(window_);

        glfwTerminate();
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
        /*
        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice_ = device;
                break;
            }
        }
        if (physicalDevice_ == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
        */
    }

    std::size_t rateDeviceSuitability(VkPhysicalDevice device) const
    {
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);

        VkPhysicalDeviceFeatures deviceFeatures;
        vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

        std::size_t score = 0;

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

	bool isDeviceSuitable(VkPhysicalDevice device) 
    {
        QueueFamilyIndices indices = findQueueFamilies(device);

        return indices.isComplete();
	}

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
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

            if (indices.isComplete()) {
                break;
            }
            i++;
        }
        return indices;
    }

private:
    GLFWwindow* window_ = nullptr;
    VkInstance instance_ = nullptr;
    VkDebugUtilsMessengerEXT debugMessenger_ = nullptr;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
};