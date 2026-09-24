cmake_minimum_required(VERSION 3.18)
project(AffinityTest LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# 01. P2P Simplex (Ping-Pong) Benchmarks
add_executable(mpi_pair_latency src/mpi_pair_latency.cpp)
target_link_libraries(mpi_pair_latency PRIVATE MPI::MPI_CXX)

add_executable(tcp_pair_latency src/tcp_pair_latency.cpp)
target_link_libraries(tcp_pair_latency PRIVATE MPI::MPI_CXX)

add_executable(uds_pair_latency src/uds_pair_latency.cpp)
target_link_libraries(uds_pair_latency PRIVATE MPI::MPI_CXX)

add_executable(mpi_gpu_transfer src/mpi_gpu_transfer.cu)
target_link_libraries(mpi_gpu_transfer PRIVATE MPI::MPI_CXX CUDA::cudart numa)

# 02. P2P Duplex (Simultaneous Bidirectional) Benchmarks
add_executable(mpi_duplex_latency src/mpi_duplex_latency.cpp)
target_link_libraries(mpi_duplex_latency PRIVATE MPI::MPI_CXX)

add_executable(tcp_duplex_latency src/tcp_duplex_latency.cpp)
target_link_libraries(tcp_duplex_latency PRIVATE MPI::MPI_CXX Threads::Threads)

add_executable(uds_duplex_latency src/uds_duplex_latency.cpp)
target_link_libraries(uds_duplex_latency PRIVATE MPI::MPI_CXX Threads::Threads)

# 03. Incast / Flooding Congestion Benchmarks
add_executable(mpi_incast_latency src/mpi_incast_latency.cpp)
target_link_libraries(mpi_incast_latency PRIVATE MPI::MPI_CXX)

add_executable(tcp_incast_latency src/tcp_incast_latency.cpp)
target_link_libraries(tcp_incast_latency PRIVATE MPI::MPI_CXX Threads::Threads)

add_executable(uds_incast_latency src/uds_incast_latency.cpp)
target_link_libraries(uds_incast_latency PRIVATE MPI::MPI_CXX Threads::Threads)

# 04. Transport Coupling Schemes Suite
add_executable(mpi_coupling_schemes src/mpi_coupling_schemes.cpp)
target_link_libraries(mpi_coupling_schemes PRIVATE MPI::MPI_CXX)

add_executable(tcp_coupling_schemes src/tcp_coupling_schemes.cpp)
target_link_libraries(tcp_coupling_schemes PRIVATE MPI::MPI_CXX Threads::Threads)

add_executable(uds_coupling_schemes src/uds_coupling_schemes.cpp)
target_link_libraries(uds_coupling_schemes PRIVATE MPI::MPI_CXX Threads::Threads)

# 05. Active Worker Count Scaling Sweep
add_executable(mpi_worker_sweep src/mpi_worker_sweep.cpp)
target_link_libraries(mpi_worker_sweep PRIVATE MPI::MPI_CXX)

add_executable(tcp_worker_sweep src/tcp_worker_sweep.cpp)
target_link_libraries(tcp_worker_sweep PRIVATE MPI::MPI_CXX Threads::Threads)
