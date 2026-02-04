cmake_minimum_required(VERSION 3.16)
project(project3 LANGUAGES CXX)

add_executable(project3
  entrypoint.cpp
  mod_ops.cpp
  num_threads.cpp
  performance.cpp
  randomize.cpp
  test.cpp
  vector_mod.cpp
)

target_compile_features(project3 PRIVATE cxx_std_20)
target_include_directories(project3 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenMP REQUIRED)
target_link_libraries(project3 PRIVATE OpenMP::OpenMP_CXX)


