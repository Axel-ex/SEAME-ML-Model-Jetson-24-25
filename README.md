# TensorRT C++ Inference on Jetson Nano

This repository provides example C++ code for running inference using the **TensorRT API** on the **Jetson Nano**. It can serve as a clean reference for building high-performance inference applications using precompiled `.engine` files.

---

## 📁 Repo Structure

- **`benchmarking/`** — Runs performance benchmarking on multiple `.engine` files. Reports FPS and inference time.
- **`test_unet_engine/`** — Performs inference on a single image using Unet. Useful for debugging and understanding TensorRT C++ integration.
- **`test_yolo_engine/`** — Performs inference on a single image using Yolov5.

---

## 🛠️ Compilation

You can compile either of the tools using `cmake`:

```
cd srcs/<name>  # Replace <name> with 'benchmarking' / 'test_unet_engine' / 'test_yolo_engine'
cmake -B build
cmake --build build
```

---

## 🚀 Usage

### Inference on a Single Image (Debug/Test Mode)

```
./build/test_unet <image_path>
```

Example:

```
./build/test_unet ../../images/unet/road.jpg

./build/test_yolo ../../images/yolo/scene.jpg
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

