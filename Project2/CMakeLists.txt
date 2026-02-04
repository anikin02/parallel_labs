cmake_minimum_required(VERSION 3.16)
project(project2 LANGUAGES CXX)

add_executable(project2 project2.cpp)
target_compile_features(project2 PRIVATE cxx_std_20)


