# Overview

This repo should be cloned on the JETSON NANO. It contains code that uses the TensorRT API correctly. This should serve as a reference when developing code in C++ with TensorRT.

WARNING: Make sure you have a .engine called "correct.engine" at the root of the repo before running the program (the path is hardcoded).

### Compilation
```bash
cd srcs
cmake -Bbuild
cmake --build build
```

### Usage
```bash
./build test_engine <image_path>
```
