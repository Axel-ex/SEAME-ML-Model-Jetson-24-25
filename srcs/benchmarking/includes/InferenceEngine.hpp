#pragma once

#include <NvInfer.h>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

using namespace nvinfer1;

constexpr auto ENGINE_PATH =
    "/home/axel/SEAME-ADS-Autonomous-lane-detection-24-25/model.engine";
constexpr auto INPUT_LAYER_NAME = "input_1";
constexpr auto OUTPUT_LAYER_NAME = "conv2d_14";

/**
 * @brief custom deleter for TRT objects
 *
 * @tparam T
 * @param obj
 */
template <typename T> struct TrtDeleter
{
        void operator()(T* obj) const
        {
            if (obj)
                obj->destroy();
        }
};

/**
 * @brief custom deleter for Cuda allocations
 *
 * @tparam T
 * @param ptr
 */
template <typename T> struct cudaDeleter
{
        void operator()(T* ptr) const
        {
            if (ptr)
            {
                cudaError_t error = cudaFree(ptr);
                if (error != cudaSuccess)
                    std::cerr << cudaGetErrorString(error) << "\n";
            }
        }
};

template <typename T> using TrtUniquePtr = std::unique_ptr<T, TrtDeleter<T>>;
template <typename T> using CudaUniquePtr = std::unique_ptr<T, cudaDeleter<T>>;

/**
 * @class InferenceEngine
 * @brief Encapsulates TensorRT engine creation, memory management, and
 * inference execution.
 *
 * This class manages the full lifecycle of the TensorRT engine, including
 * deserialization, context creation, device memory allocation, and inference
 * execution. The inference result is kept on the GPU to allow efficient
 * post-processing.
 */
class InferenceEngine
{
    public:
        InferenceEngine();
        ~InferenceEngine() = default;

        bool initRuntime();
        bool loadEngine(const std::string& engine_path);
        bool runInference(const std::vector<float>& flat_img) const;
        float* getOutputDevicePtr() const;

    private:
        TrtUniquePtr<IRuntime> runtime_;
        TrtUniquePtr<IExecutionContext> context_;
        TrtUniquePtr<ICudaEngine> engine_;
        size_t input_size_{1};
        size_t output_size_{1};
        CudaUniquePtr<void> d_input_;
        CudaUniquePtr<void> d_output_;

        ICudaEngine* createCudaEngine(const std::string& engine_name);
        void allocateDevices();
};
