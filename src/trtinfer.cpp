#include "trtinfer.hpp"


using namespace nvinfer1;


Yolov7::Yolov7(ICudaEngine* engine_)
: Node("yolov7")
{
    // QoS
    rclcpp::QoS system_qos = rclcpp::QoS(rclcpp::SystemDefaultsQoS());
     std::cerr << "here???\n";
    // Subscriber
    using std::placeholders::_1;
    this->raw_image_subscriber_ = this->create_subscription<sensor_msgs::msg::Image>("/image_raw", system_qos, std::bind(&Yolov7::image_callback, this, _1));
    this->engine_ = engine_;
    std::cerr << "here?\n";
    // Initialize model
    this->initialize_model();

    // Information
    RCLCPP_INFO(this->get_logger(), "[Inference] : Initialize.");
}

Yolov7::~Yolov7()
{
    // Release stream and buffers
    cudaStreamDestroy(this->stream_);
    CUDA_CHECK(cudaFree(this->device_buffers_[0]));
    CUDA_CHECK(cudaFree(this->device_buffers_[1]));
    delete[] this->output_buffer_host_;
    cuda_preprocess_destroy();

    // Destroy the engine
    delete this->context_;
    delete this->engine_;
   
}


void Yolov7::initialize_model()
{

    this->context_ = (engine_)->createExecutionContext();

    assert(this->context_);
  
    CUDA_CHECK(cudaStreamCreate(&stream_));

    cuda_preprocess_init(kMaxInputImageSize);

    // Prepare cpu and gpu buffers
    prepare_buffer(engine_, &device_buffers_[0], &device_buffers_[1], &output_buffer_host_);

}

void Yolov7::image_callback(const sensor_msgs::msg::Image::SharedPtr image)
{
    // Convert Ros2 image to OpenCV image
    cv_bridge::CvImageConstPtr cv_image = cv_bridge::toCvShare(image, "bgr8");
    if (cv_image->image.empty()) return;

    std::vector<cv::Mat> image_batch;
    image_batch.push_back(cv_image->image);
    


    // Preprocess
    cuda_batch_preprocess(image_batch, device_buffers_[0], kInputW, kInputH, stream_);

    // Inference
    rclcpp::Time tensorrt_inference_start = this->get_clock()->now();
    infer(*context_, stream_, (void**)device_buffers_, output_buffer_host_, kBatchSize);
    rclcpp::Duration tensorrt_inference_time = this->get_clock()->now() - tensorrt_inference_start;
    RCLCPP_INFO(this->get_logger(), "[Inference] :  - tensorrt inference time : %10.5lf ms.", tensorrt_inference_time.seconds() * 1000.0);
    // Postprocess
    std::vector<std::vector<Detection>> result_batch;
    batch_nms(result_batch, output_buffer_host_, image_batch.size(), kOutputSize, kConfThresh, kNmsThresh);

    // Draw image
    draw_bbox(image_batch, result_batch);

    for (size_t j = 0; j < image_batch.size(); j++)
    {
        cv::imshow("Result image", image_batch[j]);
    }

    cv::waitKey(10);
    
}


