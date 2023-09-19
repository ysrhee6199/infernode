#include "trtinfer.hpp"
#include "memory.h"


int main(int argc, char** argv)
{
     
    std::string model_name_ = "/home/avees/engine/yolov7.engine";

    IRuntime* runtime_ = nullptr;
    ICudaEngine* engine_2 = nullptr;
    IExecutionContext* context_ = nullptr;

    deserialize_engine(model_name_, &runtime_, &engine_2, &context_);
    rclcpp::init(argc, argv);
    rclcpp::executors::SingleThreadedExecutor executor;
    auto node = std::make_shared<Yolov7>(engine_2);
    executor.add_node(node)
    executor.spin();
    rclcpp::shutdown();


    return 0;
}

