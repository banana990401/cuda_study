cmake_minimum_required(VERSION 3.20)

project(cuda_study LANGUAGES CXX CUDA)
set(SRC_DIR reduce)
add_subdirectory(${SRC_DIR})