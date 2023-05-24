#include "trtinfer.hpp"
#include "memory.h"


int main(int argc, char** argv)
{
     
    std::string model_name_ = "/home/nvidia/engine/yolov7.engine";

    IRuntime* runtime_ = nullptr;
    ICudaEngine* engine_2 = nullptr;
    IExecutionContext* context_ = nullptr;

    std::cerr << "hst\n";
    deserialize_engine(model_name_, &runtime_, &engine_2, &context_);
    std::cerr << "hst2\n";
    rclcpp::init(argc, argv);
    rclcpp::executors::MultiThreadedExecutor executor;
    auto node = std::make_shared<Yolov7>(engine_2);
     auto node1 = std::make_shared<Yolov7_2>(engine_2);
    executor.add_node(node);
    executor.add_node(node1);
    executor.spin();
    rclcpp::shutdown();


    return 0;
}

