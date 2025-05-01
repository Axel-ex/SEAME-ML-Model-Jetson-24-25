#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class Logger : public nvinfer1::ILogger
{
    public:
        void log(Severity severity, const char* msg) noexcept override;
};

nvinfer1::ICudaEngine* createEngine(const std::string& engine_path,
                                    nvinfer1::IRuntime* runtime);
void checkEngineSpecs(nvinfer1::ICudaEngine* engine);
std::vector<float> loadImage(const std::string& img_path);
void debugOutput(const std::vector<float>& output_data, cv::Mat& output_image);
