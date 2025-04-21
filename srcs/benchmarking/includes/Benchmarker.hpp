#pragma once

#include "InferenceEngine.hpp"

class Benchmarker
{
    public:
        Benchmarker() = default;
        ~Benchmarker() = default;

        bool init();
        bool loadEngine(const std::string& engine_path);
        void runBenchmarking(const std::string& models_path,
                             const std::string& image_path);

    private:
        InferenceEngine inference_engine_;

        std::vector<float> loadImage(const std::string& image_path);
};
