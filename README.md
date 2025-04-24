# TensorRT C++ Inference on Jetson Nano

This repository provides example C++ code for running inference using the **TensorRT API** on the **Jetson Nano**. It can serve as a clean reference for building high-performance inference applications using precompiled `.engine` files.

> ⚠️ **Important:**  
> Make sure you have a TensorRT engine file named `correct.engine` placed at the **root of the repository**. The path is currently hardcoded in the code for simplicity.

---

## 📁 Repo Structure

- **`benchmarking/`** — Runs performance benchmarking on multiple `.engine` files. Reports FPS and inference time.
- **`test_engine/`** — Performs inference on a single image. Useful for debugging or understanding TensorRT integration.

---

## 🛠️ Compilation

You can compile either of the tools using `cmake`:

```
cd srcs/<name>  # Replace <name> with 'benchmarking' or 'test_engine'
cmake -B build
cmake --build build
```

---

## 🚀 Usage

### Inference on a Single Image (Debug/Test Mode)

```
./build/test_engine <model_path> <image_path>
```

Example:

```
./build/test_engine ./optimized.engine ../../images/road.jpg
```

---

### Benchmark Multiple Engines

```
./build/benchmarker <models_path> <images_path>
```

Example:

```
./build/benchmarker ../../models/ ../../images/
```

This will loop through all `.engine` files in the `models_path`, run inference on all images in `images_path`, and report performance metrics like FPS.

