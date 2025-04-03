cmake_minimum_required(VERSION 3.10)
project(testcuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set CUDA architectures (e.g., 75 for RTX 2060/3060, adjust for your GPU)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Add the blockchain subdirectory
add_subdirectory(cppblockchain)