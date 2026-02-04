cmake_minimum_required(VERSION 3.16)
project(project4 LANGUAGES CXX)

add_executable(project4 Project4.cpp)
target_compile_features(project4 PRIVATE cxx_std_20)

find_package(OpenMP REQUIRED)
target_link_libraries(project4 PRIVATE OpenMP::OpenMP_CXX)

