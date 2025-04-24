#include "Benchmarker.hpp"
#include <chrono>
#include <filesystem>
#include <fmt/color.h>
#include <fmt/format.h>

bool Benchmarker::init() { return inference_engine_.initRuntime(); }

bool Benchmarker::loadEngine(const std::string& engine_path)
{
    return inference_engine_.loadEngine(engine_path);
}

void Benchmarker::runBenchmarking(const std::string& models_path,
                                  const std::string& images_path)
{
    std::filesystem::recursive_directory_iterator it_models(models_path);

    for (const auto& dir_entry : it_models)
    {
        if (dir_entry.path().string().find(".engine") == std::string::npos)
            continue;

        fmt::print("\n[{}]: {}\n",
                   fmt::format(fmt::fg(fmt::color::cyan), "TESTING"),
                   dir_entry.path().string());

        if (!inference_engine_.loadEngine(dir_entry.path().string()))
            continue;

        fmt::print("[{}]: {}....\n",
                   fmt::format(fmt::fg(fmt::color::cyan), "TESTING"),
                   "Running inference");

        int processed_pics{};
        auto start = std::chrono::high_resolution_clock::now();
        std::filesystem::recursive_directory_iterator it_images(images_path);
        for (const auto& image : it_images)
        {
            auto flat_image = loadImage(image.path().string());
            if (flat_image.empty())
            {
                fmt::print("[{}]: couldn't read {}\n",
                           fmt::format(fmt::fg(fmt::color::red), "ERROR"),
                           image.path().string());
                continue;
            }
            inference_engine_.runInference(flat_image);
            processed_pics++;
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::seconds>(end - start);

        fmt::print("[{}]: processed {} pics in {}s ({}FPS)\n",
                   fmt::format(fmt::fg(fmt::color::green), "RESULT"),
                   processed_pics, duration.count(),
                   static_cast<float>(processed_pics) / duration.count());
        inference_engine_.reset();
    }
}

std::vector<float> Benchmarker::loadImage(const std::string& image_path)
{
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty())
        return {};

    // WARN: potential mismatch size / expected input size
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
