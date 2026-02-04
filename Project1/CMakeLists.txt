cmake_minimum_required(VERSION 3.16)
project(project1 LANGUAGES CXX)

add_executable(project1 project1.cpp)
target_compile_features(project1 PRIVATE cxx_std_20)

find_package(OpenMP REQUIRED)
target_link_libraries(project1 PRIVATE OpenMP::OpenMP_CXX)

