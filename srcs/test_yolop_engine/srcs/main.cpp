#include <Logger.hpp>
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

    // cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    // cv::resize(img, img, INPUT_IMG_SIZE);
    // auto flat_img = flattenImage(img);
    //
    // inference_engine.runInference(flat_img);
    // YoloResult result = postProcess(inference_engine);
    // printResult(result);
    // saveResult(result, img);
    //
    return EXIT_SUCCESS;
}
