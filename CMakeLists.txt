cmake_minimum_required(VERSION 3.17)
project(MatrixOperationsParallel)

set(CMAKE_CXX_STANDARD 14)

add_executable(MatrixOperationsParallel main.cpp)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(MatrixOperationsParallel PUBLIC OpenMP::OpenMP_CXX)
endif()