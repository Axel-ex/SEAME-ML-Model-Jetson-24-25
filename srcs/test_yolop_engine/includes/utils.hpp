#include "InferenceEngine.hpp"
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

const cv::Size INPUT_IMG_SIZE(640, 640); // WARN: check bindings size
const std::vector<std::string> YOLOP_CLASSES = {"car"};

struct YoloResult
{
        std::vector<cv::Rect> boxes;
        std::vector<int> class_ids;
        std::vector<float> confidences;
};

std::vector<float> flattenImage(cv::Mat& img);
YoloResult postProcessObjDetection(InferenceEngine& inference_engine);
std::string mapIdtoString(int id);
void printResult(const YoloResult& result);
void saveYoloResult(const YoloResult& result, cv::Mat& og_img);
cv::Mat getLaneMask(InferenceEngine& inference_engine);
