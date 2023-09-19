#include <chrono>
#include <thread>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <fstream>

// TensorRT
#include "config.h"
#include "model.h"
#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "yolov7/main.cpp"

// ROS
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "image_transport/image_transport.hpp"
//#include "rtx_msg_interface/msg/bounding_box.hpp"
//#include "rtx_msg_interface/msg/bounding_boxes.hpp"


using namespace nvinfer1;



class Yolov7 : public rclcpp::Node
{
public:
  Yolov7(ICudaEngine* engine_);
  ~Yolov7();

private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr image);
  void initialize_model();
 

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr raw_image_subscriber_;

  IExecutionContext* context_ = nullptr;
  ICudaEngine* engine_;
  cudaStream_t stream_;

  float* device_buffers_[2];
  float* output_buffer_host_ = nullptr;

  std::vector<std::vector<Detection>> result_batch;
};

