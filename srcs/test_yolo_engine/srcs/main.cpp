#include "ImageProcessor.hpp"
#include <InferenceEngine.hpp>
#include <Logger.hpp>
#include <utils.hpp>

const cv::Size INPUT_IMG_SIZE(256, 256); // WARN: check bindings size
const cv::Size OUTPUT_IMG_SIZE(256, 256);
const std::string IMAGE_PATH = "../../images/dark_frame_0114.jpg";

int main(int argc, char** argv)
{
    InferenceEngine inference_engine;
    // ImageProcessor image_processor(INPUT_IMG_SIZE, OUTPUT_IMG_SIZE);

    inference_engine.init();
    inference_engine.checkEngineSpecs();
    // cv::Mat img = cv::imread(IMAGE_PATH, cv::IMREAD_COLOR);
    // auto flatten = image_processor.flattenImage(img);
}
