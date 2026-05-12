cmake_minimum_required(VERSION 3.0.2)
project(joint_pose_estimation)

## Compile as C++14, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)
#SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  pcl_conversions
  roscpp
  pcl_ros
  rospy
  sensor_msgs
  std_msgs
  tf2
  tf2_ros
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
set(Boost_LIBRARY_DIR /usr/lib/x86_64-linux-gnu/)

find_package(PCL REQUIRED)
find_package(OpenCV 4.2 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Yaml-cpp REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(Eigen3 REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

catkin_package(
 CATKIN_DEPENDS roscpp rospy std_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)

add_executable(estimation_node src/estimation_node.cpp src/funcs.cpp)
add_executable(realtime_node src/realtime_node.cpp src/funcs.cpp)

target_link_libraries(estimation_node ${catkin_LIBRARIES}  ${OpenCV_LIBS} ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${CERES_LIBRARIES} yaml-cpp Eigen3::Eigen)
target_link_libraries(realtime_node ${catkin_LIBRARIES}  ${OpenCV_LIBS} ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${CERES_LIBRARIES} yaml-cpp Eigen3::Eigen)
