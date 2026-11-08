cmake_minimum_required(VERSION 3.16)

project(RingAttentionMPI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ring_attention_mpi
    src/main.cpp
    src/attention.cpp
)

target_include_directories(ring_attention_mpi PRIVATE include)