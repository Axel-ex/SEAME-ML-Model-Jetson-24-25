#include "Benchmarker.hpp"

int main(int argc, char** argv)
{
    Benchmarker benchmarker;

    if (argc < 3)
    {
        std::cerr << "./benchmarker <models_path> <images_path>";
        return EXIT_FAILURE;
    }
    const std::string models_path(argv[1]);
    const std::string images_path(argv[2]);

    if (!benchmarker.init())
        return EXIT_FAILURE;

    benchmarker.runBenchmarking(models_path, images_path);

    return EXIT_SUCCESS;
}
