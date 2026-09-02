cmake_minimum_required(VERSION 3.8)
project(arm_lab_bringup)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch worlds rviz config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
