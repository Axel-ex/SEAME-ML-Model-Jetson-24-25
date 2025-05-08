#include <Logger.hpp>
#include <fmt/color.h>
#include <fmt/format.h>
#include <utils.hpp>

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "./test_yolo <image_path>";
        return EXIT_FAILURE;
    }
    std::string image_path = argv[1];

    InferenceEngine inference_engine;

    inference_engine.init();
    inference_engine.checkEngineSpecs();

    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty())
    {
        fmt::print("[{}]: reading the image: {} doesnt exist",
                   fmt::format(fmt::fg(fmt::color::indian_red), "Error"),
                   image_path);
        return EXIT_FAILURE;
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
    cv::imwrite("results/yolop_result.jpg", img);

    return EXIT_SUCCESS;
}
