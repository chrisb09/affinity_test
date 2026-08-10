cmake_minimum_required(VERSION 3.18)
project(AffinityTest LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(mpi_pair_latency mpi_pair_latency.cpp)
target_link_libraries(mpi_pair_latency PRIVATE MPI::MPI_CXX)

add_executable(tcp_pair_latency tcp_pair_latency.cpp)
target_link_libraries(tcp_pair_latency PRIVATE MPI::MPI_CXX)

add_executable(mpi_gpu_transfer mpi_gpu_transfer.cu)
target_link_libraries(mpi_gpu_transfer PRIVATE MPI::MPI_CXX CUDA::cudart)
