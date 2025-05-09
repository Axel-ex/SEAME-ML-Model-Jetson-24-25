#include <Logger.hpp>
#include <chrono>
#include <filesystem>
#include <fmt/color.h>
#include <fmt/format.h>
#include <utils.hpp>

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "./test_yolo <images_path>";
        return EXIT_FAILURE;
    }
    std::string image_path = argv[1];

    InferenceEngine inference_engine;

    inference_engine.init();
    inference_engine.checkEngineSpecs();

    // Iterating over a dir and measuring the FPS
    auto dir_it = std::filesystem::directory_iterator(image_path);
    auto start = std::chrono::high_resolution_clock::now();
    int pics_count{};

    for (const auto& entry : dir_it)
    {
        if (entry.is_directory() || ((entry.path().has_extension() &&
                                      entry.path().extension() != ".jpg")))
        {
            fmt::print("[{}]: {} is not an image file",
                       fmt::format(fmt::fg(fmt::color::indian_red), "Error"),
                       entry.path().string());
            continue;
        }

        cv::Mat img = cv::imread(entry.path(), cv::IMREAD_COLOR);
        if (img.empty())
        {
            fmt::print("[{}]: reading the image: {}",
                       fmt::format(fmt::fg(fmt::color::indian_red), "Error"),
                       entry.path().string());
            continue;
        }
        cv::resize(img, img, INPUT_IMG_SIZE);
        auto flat_img = flattenImage(img);

        inference_engine.runInference(flat_img);

        YoloResult result = postProcessObjDetection(inference_engine);
        printResult(result);
        drawYoloResult(result, img);

        cv::Mat lane_mask = getLaneMask(inference_engine);
        cv::Mat colored;

        // Blend the result into a single image
        cv::applyColorMap(lane_mask, colored, cv::COLORMAP_JET);
        cv::addWeighted(img, 0.7, colored, 0.3, 0, img);
        cv::imwrite("results/" + entry.path().stem().string() + "_result.jpg",
                    img);
        pics_count++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start);
    fmt::print("[{}]: processed {} in {} ({})",
               fmt::format(fmt::fg(fmt::color::spring_green), "RESULTS"),
               pics_count, duration.count(), pics_count / duration.count());

    return EXIT_SUCCESS;
}
