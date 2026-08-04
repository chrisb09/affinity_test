# Affinity & Topology Benchmarks

This directory contains two C++/CUDA MPI tools designed to benchmark intra-node CPU and GPU topology placement effects.

## Benchmarks

### 1. `mpi_pair_latency` (CPU Core-to-Core Latency)
Measures the full $N \times (N-1) / 2$ rank pair ping-pong latency matrix for all MPI ranks in an allocation.
- **Output:** CSV edge-list containing min, mean, median, and max half-RTT latency ($\mu$s) alongside per-rank hostnames, active CPUs, and affinity masks.
- **Sequential Pair Scheduling:** Rank 0 coordinates test pairs while idle ranks sleep, avoiding CPU oversubscription and cgroup throttling on login nodes.

### 2. `mpi_gpu_transfer` (MPI Rank-to-GPU Host/Device Transfers)
Measures Host-to-Device (H2D) and Device-to-Host (D2H) transfer performance for each rank against specified CUDA device IDs.
- **Metrics:** 
  - Small transfer latency ($\mu$s) for a small payload (default: 4 KB).
  - Large transfer bandwidth (GB/s) for a large payload (default: 16 MB or 64 MB).
- **PCI & Device Awareness:** Records physical PCI Bus ID for each GPU to uniquely map device placement regardless of CUDA CUDA_VISIBLE_DEVICES index ordering.
- **Exclusive Device Protection:** Checks compute mode (`Compute_Mode: Exclusive_Process`) and device accessibility first to gracefully log unavailable devices rather than crash or hang.
- **Token Ring Execution:** Ranks run sequentially via token ring so only 1 rank accesses CUDA at a time, protecting shared GPUs and login node CPU quotas.

---

## Build Instructions

### Prerequisites
- CMake >= 3.18
- OpenMPI (or MPI-3 compatible toolchain)
- CUDA Toolkit 12.x (`nvcc`)

### Build
```bash
cd affinity_test
mkdir -p build
cmake -B build -S .
cmake --build build -j 4
```

This generates two binaries under `build/`:
- `build/mpi_pair_latency`
- `build/mpi_gpu_transfer`

---

## Usage Guide

### CPU Core-to-Core Latency
```bash
mpirun -mca shmem_mmap_enable_nfs_warning 0 -mca orte_tmpdir_base /tmp \
  --oversubscribe --map-by core --bind-to core \
  -np 96 ./build/mpi_pair_latency \
  -o cpu_96ranks_pair_latency.csv -w 10 -i 50
```

#### Options:
- `-o, --output <file>`: CSV output filename (default: `cpu_pair_latency.csv`)
- `-b, --bytes <bytes>`: Message payload size in bytes (default: `64`)
- `-w, --warmup <count>`: Warmup iterations per pair (default: `100`)
- `-i, --iters <count>`: Measured iterations per pair (default: `1000`)

---

### MPI Rank-to-GPU Transfers
```bash
# Test 8 NUMA domains against GPUs 0, 1, 2, 3
mpirun -mca shmem_mmap_enable_nfs_warning 0 -mca orte_tmpdir_base /tmp \
  --map-by numa --bind-to numa \
  -np 8 ./build/mpi_gpu_transfer \
  -d 0,1,2,3 -l 16777216 -o gpu_8numa_transfer.csv -w 2 -i 5
```

#### Options:
- `-o, --output <file>`: CSV output filename (default: `gpu_transfer_latency.csv`)
- `-d, --devices <ids>`: Comma-separated CUDA device IDs (default: `0,1`)
- `-s, --small-size <bytes>`: Small transfer payload size for latency in bytes (default: `4096`)
- `-l, --large-size <bytes>`: Large transfer payload size for bandwidth in bytes (default: `67108864`)
- `-w, --warmup <count>`: Warmup iterations per rank/device (default: `10`)
- `-i, --iters <count>`: Measured iterations per test (default: `50`)

---

## Heatmap Visualizations

A Python script `plot_heatmaps.py` is included to visualize both CPU latency and GPU transfer results.

```bash
python3 plot_heatmaps.py --cpu-csv cpu_96ranks_pair_latency.csv --gpu-csv gpu_8numa_transfer.csv
```

Outputs generated:
1. `cpu_latency_heatmap.png`: Symmetric $N \times N$ matrix heatmap illustrating intra-socket vs. cross-socket CPU latency.
2. `gpu_transfer_heatmap_bandwidth.png`: D2H and H2D bandwidth (GB/s) per rank/NUMA domain against available GPUs.
3. `gpu_transfer_heatmap_latency.png`: Small-transfer D2H/H2D latency ($\mu$s) per rank against available GPUs.

---

## Initial Benchmark Findings (Login Node `login23-g-1`)

### 1. CPU Core-to-Core Ping-Pong Latency (96 Cores)
- **Same Socket (CPUs 0-47 to CPUs 0-47):** Median latency **~0.68 $\mu$s**.
- **Cross Socket (CPUs 0-47 to CPUs 48-95):** Median latency **~1.60 $\mu$s** (~2.37x higher latency over the UPI cross-socket interconnect).

### 2. Rank-to-GPU Host-to-Device / Device-to-Host Bandwidth
- **GPU 0 (PCI `0000:1b:00.0`, Socket 0 / NUMA 0):**
  - **Socket 0 ranks (NUMA 0-3):** D2H Bandwidth **~41.5 GB/s**.
  - **Socket 1 ranks (NUMA 4-7):** D2H Bandwidth drops to **~18.3 - 18.9 GB/s** (56% bandwidth loss across sockets).
- **GPU 3 (PCI `0000:9d:00.0`, Socket 1 / NUMA 6):**
  - **Socket 1 ranks (NUMA 4-7):** D2H Bandwidth **~35.0 - 38.5 GB/s**.
  - **Socket 0 ranks (NUMA 0-3):** D2H Bandwidth drops to **~18.5 GB/s** (traversing cross-socket UPI link).
- **GPU 2 (`0000:ad:00.0`):** Detected in `Exclusive_Process` mode; safely reported `GPU in Exclusive/Prohibited compute mode` status without failing the benchmark run.
