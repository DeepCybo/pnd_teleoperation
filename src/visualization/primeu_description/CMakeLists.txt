cmake_minimum_required(VERSION 3.10.2)
project(primeu_description)
find_package(ament_cmake REQUIRED)
find_package(urdf REQUIRED)
find_package(xacro REQUIRED)

ament_package()

install(DIRECTORY assets urdf config DESTINATION share/${PROJECT_NAME})