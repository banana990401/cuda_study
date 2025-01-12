cmake_minimum_required(VERSION 3.20)

project(cuda_study LANGUAGES CXX CUDA)
add_subdirectory(reduce)
add_subdirectory(device_query)
add_subdirectory(histogram)