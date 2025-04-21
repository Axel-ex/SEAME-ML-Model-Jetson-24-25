#include "Benchmarker.hpp"
#include <filesystem>

bool Benchmarker::init() { return inference_engine_.initRuntime(); }

bool Benchmarker::loadEngine(const std::string& engine_path)
{
    return inference_engine_.loadEngine(engine_path);
}

void Benchmarker::runBenchmarking(const std::string& models_path,
                                  const std::string& images_path) const
{
    std::filesystem::recursive_directory_iterator it(models_path);

    for (const auto& dir_entry : it)
    {
        std::cout << it->path().string() << "\n";
    }
}
