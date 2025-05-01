#include "InferenceEngine.hpp"
#include <opencv2/dnn.hpp>
#include <utils.hpp>

// INPUT should be [all R values] + [all G values] + [all B values] (Input
// tensor CHW format)
std::vector<float> flattenImage(cv::Mat& img)
{
    std::vector<float> flatten_img(3 * INPUT_IMG_SIZE.height *
                                   INPUT_IMG_SIZE.width);

    for (int c = 0; c < 3; ++c)
    {
        for (int y = 0; y < img.rows; ++y)
        {
            for (int x = 0; x < img.cols; ++x)
            {
                float val =
                    static_cast<float>(img.at<cv::Vec3b>(y, x)[2 - c]) / 255.0f;
                flatten_img[c * img.rows * img.cols + y * img.cols + x] = val;
            }
        }
    }
    return flatten_img;
}

// NOTE: Our output is 1 25200 85: 25200 elements (bounding / anchor boxes) each
// of them containing 85 subelements (cx, cy, w, h, obj_confidence,
// class_prob(80)). coordinates represent the center of the bounding box as well
// as its width and height 80 class due to COCO dataset

/**
 * @brief extract the inference result from GPU , post process the data to
 * filter for the predictions with highest confidence, construct a YoloResult
 * from the computations
 *
 * @param inference_engine
 * @return
 */
YoloResult postProcess(InferenceEngine& inference_engine)
{
    // Get data from gpu
    std::vector<float> host_output(inference_engine.getOuputSize() /
                                   sizeof(float));
    cudaMemcpy(host_output.data(), inference_engine.getOutputDevicePtr(),
               inference_engine.getOuputSize(), cudaMemcpyDeviceToHost);

    // Loop over the detections (Refer to check_bindings for this information)
    const int nb_elements = 25200;
    const int num_classes = 80;
    const int element_size = 5 + num_classes;

    float conf_threshold = 0.2;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    for (int i = 0; i < nb_elements; i++)
    {
        const float* element = &host_output[i * element_size];
        float elem_conf = element[4];

        if (elem_conf < conf_threshold)
            continue;

        // Find class with max score
        float max_class_prob = 0.0f;
        int class_id = -1;
        for (int c = 0; c < num_classes; ++c)
        {
            if (element[5 + c] > max_class_prob)
            {
                max_class_prob = element[5 + c];
                class_id = c;
            }
        }

        float final_conf = elem_conf * max_class_prob;
        if (final_conf < conf_threshold)
            continue;

        // YOLO box format is center_x, center_y, width, height
        float cx = element[0];
        float cy = element[1];
        float w = element[2];
        float h = element[3];

        int left = static_cast<int>(cx - w / 2.0f);
        int top = static_cast<int>(cy - h / 2.0f);
        int width = static_cast<int>(w);
        int height = static_cast<int>(h);

        boxes.emplace_back(left, top, width, height);
        confidences.push_back(final_conf);
        class_ids.push_back(class_id);
    }

    // Filter with non maximum suppression (NMS)
    std::vector<int> indices;
    float nms_treshold = 0.45f;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_treshold,
                      indices);

    YoloResult result;

    for (int idx : indices)
    {
        result.boxes.push_back(boxes[idx]);
        result.confidences.push_back(confidences[idx]);
        result.class_ids.push_back(class_ids[idx]);
    }

    return result;
}

std::string mapIdtoString(int id)
{
    if (id >= 80)
        return "Invalid id";
    return COCO_CLASSES[id];
}

void saveResult(const YoloResult& result, cv::Mat& og_image)
{
    for (int i = 0; i < result.boxes.size(); i++)
    {
        auto box = result.boxes[i];
        auto id = result.class_ids[i];
        auto confidence = result.confidences[i];

        cv::rectangle(og_image, box, cv::Scalar(255, 0, 0));

        // Compose the label
        std::string label = mapIdtoString(id);
        label += " (" + cv::format("%.2f", confidence) + ")";

        // Calculate label position
        int baseline = 0;
        cv::Size label_size =
            cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        int top = std::max(box.y, label_size.height);

        // Draw the label with backgorund
        cv::rectangle(og_image, cv::Point(box.x, top - label_size.height),
                      cv::Point(box.x + label_size.width, top + baseline),
                      cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(og_image, label, cv::Point(box.x, top),
                    cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 0, 0));
    }

    std::string output_name = "results/test_yolo_output.jpg";
    cv::imwrite(output_name, og_image);
    std::cout << "Output saved as " << output_name << std::endl;
}

void printResult(const YoloResult& result)
{
    for (int i = 0; i < result.boxes.size(); i++)
    {
        cv::Rect box = result.boxes[i];
        float conf = result.confidences[i];
        int class_id = result.class_ids[i];
        std::cout << "Detected: " << mapIdtoString(class_id) << "(" << conf
                  << ") at " << box << "\n";
    }
}
