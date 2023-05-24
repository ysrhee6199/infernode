cmake_minimum_required(VERSION 3.5)
project(infernode)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Enable CUDA
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("embed_platform on")
  include_directories(/usr/local/cuda/targets/aarch64-linux/include)
  link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
  message("embed_platform off")
  # cuda
  include_directories(/usr/local/cuda/include)
  link_directories(/usr/local/cuda/lib64)

  # tensorrt
  include_directories(/home/nvidia/TensorRT-8.2.5.1/include)
  link_directories(/home/nvidia/TensorRT-8.2.5.1/lib)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
#find_package(rtx_msg_interface REQUIRED)
find_package(OpenCV 4 REQUIRED)

include_directories(include)

include_directories(include/yolov7/include)
include_directories(include/yolov7/plugin)

include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB_RECURSE TENSORRT_SRCS include/yolov7/src/*.cpp include/yolov7/src/*.cu)

# Build
add_executable(infernode src/exec/trtinfer_exec.cpp ${TENSORRT_SRCS})
ament_target_dependencies(infernode
  rclcpp
  std_msgs
  sensor_msgs
  image_transport
  cv_bridge
 # rtx_msg_interface
)



add_library(infernode_library SHARED src/trtinfer.cpp)
ament_target_dependencies(infernode_library 
  rclcpp
  std_msgs
  sensor_msgs
  image_transport
  cv_bridge
  #rtx_msg_interface
)

add_library(tensorrt_plugins SHARED include/yolov7/plugin/yololayer.cu)
target_link_libraries(tensorrt_plugins nvinfer cudart)
target_link_libraries(infernode nvinfer)
target_link_libraries(infernode cudart)
target_link_libraries(infernode tensorrt_plugins)
target_link_libraries(infernode ${OpenCV_LIBS})



target_link_libraries(infernode
  infernode_library
)

install(TARGETS
  tensorrt_plugins
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS
infernode
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
