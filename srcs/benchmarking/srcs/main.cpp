#include "Benchmarker.hpp"
#include <fmt/color.h>
#include <fmt/format.h>

int main(int argc, char** argv)
{
    Benchmarker benchmarker;

    if (argc < 3)
    {
        fmt::print("{}: {}", fmt::format(fmt::fg(fmt::color::red), "USAGE"),
                   "./benchmarker <models_path> <images_path>\n");
        return EXIT_FAILURE;
    }

    const std::string models_path(argv[1]);
    const std::string images_path(argv[2]);

    if (!benchmarker.init())
        return EXIT_FAILURE;
    fmt::print("[{}]: {}\n", fmt::format(fmt::fg(fmt::color::green), "SUCESS"),
               "Runtime succefully created");
    benchmarker.runBenchmarking(models_path, images_path);

    return EXIT_SUCCESS;
}
