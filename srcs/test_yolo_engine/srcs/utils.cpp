#include <filesystem>
#include <fstream>
#include <iostream>
#include <utils.hpp>

namespace fs = std::filesystem;

nvinfer1::ICudaEngine* createEngine(const std::string& engine_path,
                                    nvinfer1::IRuntime* runtime)
{
    std::ifstream file(engine_path, std::ios::binary);
    if (!file)
    {
        std::cerr << "Fail to load model: path doesn't exist\n";
        return nullptr;
    }
    std::cout << "Engine file loaded." << std::endl;

    // seek the end of the file to determine engine size
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    // create a file to hold engine data
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);

    return runtime->deserializeCudaEngine(engineData.data(), size);
}

void checkEngineSpecs(nvinfer1::ICudaEngine* engine)
{
    int numBindings = engine->getNbBindings();
    std::cout << "Number of bindings: " << numBindings << std::endl;

    for (int i = 0; i < numBindings; ++i)
    {
        const char* name = engine->getBindingName(i);
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        nvinfer1::DataType dtype = engine->getBindingDataType(i);
        bool isInput = engine->bindingIsInput(i);

        std::cout << "Binding index " << i << ": " << name << std::endl;
        std::cout << "  Is input: " << (isInput ? "Yes" : "No") << std::endl;
        std::cout << "  Dimensions: ";
        for (int j = 0; j < dims.nbDims; ++j)
        {
            std::cout << dims.d[j] << " ";
        }
        std::cout << std::endl;
        std::cout << "  Data type: "
                  << (dtype == nvinfer1::DataType::kFLOAT ? "FLOAT" : "Other")
                  << std::endl;
    }
}

void debugOutput(const std::vector<float>& output_data, cv::Mat& output_image)
{
    // Normalize output to 0-255 range
    cv::Mat display_output;
    cv::normalize(output_image, display_output, 0, 255, cv::NORM_MINMAX,
                  CV_8UC1);

    cv::imwrite("results/output_image.jpg", display_output);
    std::cout << "Output saved to results/output_image.jpg" << std::endl;
    float min_val = *std::min_element(output_data.begin(), output_data.end());
    float max_val = *std::max_element(output_data.begin(), output_data.end());
    std::cout << "Output value range: " << min_val << " to " << max_val
              << std::endl;

    // Apply treshold
    float threshold = 0.3;
    cv::Mat thresholded_output = output_image > threshold;
    cv::imwrite("results/thresholded_output.jpg", thresholded_output);
}

std::vector<float> loadImage(const std::string& image_path)
{
    // IMAGE LOADING/RESIZING PART --> this is now a function
    fs::create_directory("results");

    // Load color image
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty())
    {
        throw std::runtime_error("Failed to load image");
    }
    std::cout << "Image size: " << img.cols << "x" << img.rows << std::endl;

    // Resize image to 256x256
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(256, 256));

    // Preparing vector where image data will be stored
    std::vector<float> og_image(256 * 256 * 3);

    // Fill vector with pixel values (normalized)
    for (int y = 0; y < resized_img.rows; ++y)
    {
        for (int x = 0; x < resized_img.cols; ++x)
        {
            cv::Vec3b pixel = resized_img.at<cv::Vec3b>(y, x);
            // Store the pixel data in the og_image vector (RGB channels) -->
            // normalized by dividing by 255
            og_image[(y * resized_img.cols + x) * 3] =
                static_cast<float>(pixel[2]) / 255.0f; // Red channel
            og_image[(y * resized_img.cols + x) * 3 + 1] =
                static_cast<float>(pixel[1]) / 255.0f; // Green channel
            og_image[(y * resized_img.cols + x) * 3 + 2] =
                static_cast<float>(pixel[0]) / 255.0f; // Blue channel
        }
    }
    return og_image;
}
