# Affinity & Communication Benchmark Suite

Comprehensive C++/CUDA and MPI/TCP benchmark suite designed to evaluate hardware topology, interconnect capabilities, and one-to-many communication schemes across the CLAIX-23 cluster partition (Intel Xeon Sapphire Rapids 96 cores/node, NVIDIA GPUs, and InfiniBand fabric).

---

## Directory Structure

```
sourcecode/affinity_test/
├── CMakeLists.txt                                     # Builds all 11 benchmark executables from src/
├── README.md                                          # Documentation and usage guide
├── .gitignore
├── plot_heatmaps.py                                   # P2P Simplex, Duplex, and Incast visualization tool
├── plot_coupling_schemes.py                           # Coupling Schemes & Worker Scaling visualization tool
│
├── src/                                               # C++ / CUDA Source Files
│   ├── mpi_pair_latency.cpp                           # 01. P2P Simplex Ping-Pong (MPI)
│   ├── tcp_pair_latency.cpp                           # 01. P2P Simplex Ping-Pong (TCP IPoIB)
│   ├── uds_pair_latency.cpp                           # 01. P2P Simplex Ping-Pong (Unix Domain Sockets)
│   ├── mpi_gpu_transfer.cu                            # 01. Host-to-Device / Device-to-Host (CUDA)
│   ├── mpi_duplex_latency.cpp                         # 02. P2P Full-Duplex (MPI)
│   ├── tcp_duplex_latency.cpp                         # 02. P2P Full-Duplex (TCP IPoIB)
│   ├── uds_duplex_latency.cpp                         # 02. P2P Full-Duplex (Unix Domain Sockets)
│   ├── mpi_incast_latency.cpp                         # 03. Incast Flooding Payload Sweep (MPI)
│   ├── tcp_incast_latency.cpp                         # 03. Incast Flooding Payload Sweep (TCP IPoIB)
│   ├── uds_incast_latency.cpp                         # 03. Incast Flooding Payload Sweep (Unix Domain Sockets)
│   ├── mpi_coupling_schemes.cpp                       # 04. HPC-ML Coupling Transport Schemes (MPI)
│   ├── tcp_coupling_schemes.cpp                       # 04. HPC-ML Coupling Transport Schemes (TCP IPoIB)
│   ├── uds_coupling_schemes.cpp                       # 04. HPC-ML Coupling Transport Schemes (Unix Domain Sockets)
│   ├── mpi_worker_sweep.cpp                           # 05. Active Worker Scaling Sweep (MPI)
│   └── tcp_worker_sweep.cpp                           # 05. Active Worker Scaling Sweep (TCP IPoIB)
│
└── jobscripts/                                        # Slurm SBATCH Submission Scripts
    ├── run_cpu_devel.sbatch                           # 01. Run P2P Simplex (MPI)
    ├── run_tcp_devel.sbatch                           # 01. Run P2P Simplex (TCP)
    ├── run_gpu_c23g.sbatch                            # 01. Run GPU PCIe Transfers
    ├── run_cpu_duplex_devel.sbatch                    # 02. Run P2P Duplex (MPI)
    ├── run_tcp_duplex_devel.sbatch                    # 02. Run P2P Duplex (TCP)
    ├── run_cpu_incast_devel.sbatch                    # 03. Run Incast Sweep (MPI)
    ├── run_tcp_incast_devel.sbatch                    # 03. Run Incast Sweep (TCP)
    ├── run_coupling_schemes_devel.sbatch              # 04. Run Coupling Schemes Suite
    ├── run_worker_sweep_devel.sbatch                  # 05. Run Worker Scaling Sweep (MPI)
    ├── run_tcp_worker_sweep_devel.sbatch              # 05. Run Worker Scaling Sweep (TCP)
    └── run_uds_devel.sbatch                           # 06. Run Complete Single-Node UDS Suite
```

---

## Build Instructions

### Prerequisites
- CMake >= 3.18
- OpenMPI (MPI-3 compatible)
- CUDA Toolkit >= 12.0 (`nvcc`)
- GCC >= 11.3

### Compile All Binaries
```bash
mkdir -p build
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
```

This produces all 11 benchmark executables under `build/`:
- `build/mpi_pair_latency`
- `build/tcp_pair_latency`
- `build/mpi_gpu_transfer`
- `build/mpi_duplex_latency`
- `build/tcp_duplex_latency`
- `build/mpi_incast_latency`
- `build/tcp_incast_latency`
- `build/mpi_coupling_schemes`
- `build/tcp_coupling_schemes`
- `build/mpi_worker_sweep`
- `build/tcp_worker_sweep`

---

## Running Benchmarks (Slurm `devel` Partition)

Submit any benchmark using its dedicated jobscript from `jobscripts/`:

```bash
# 01. P2P Simplex Pair Matrices (18,336 pairs across 192 ranks)
sbatch jobscripts/run_cpu_devel.sbatch
sbatch jobscripts/run_tcp_devel.sbatch
sbatch jobscripts/run_gpu_c23g.sbatch

# 02. P2P Full-Duplex Pair Matrices (with 30-min auto-rescheduling)
sbatch jobscripts/run_cpu_duplex_devel.sbatch
sbatch jobscripts/run_tcp_duplex_devel.sbatch

# 03. Incast Congestion Payload Sweep (64 B to 4 MiB)
sbatch jobscripts/run_cpu_incast_devel.sbatch
sbatch jobscripts/run_tcp_incast_devel.sbatch

# 04. Comprehensive Coupling Schemes Suite (Gather-Scatter, Immediate, Credits)
sbatch jobscripts/run_coupling_schemes_devel.sbatch

# 05. Active Worker Scaling Sweep (W = 1 to 191)
sbatch jobscripts/run_worker_sweep_devel.sbatch
sbatch jobscripts/run_tcp_worker_sweep_devel.sbatch
```

---

## Visualization Tools

```bash
# Generate P2P Simplex, Duplex, and Incast plots
python3 plot_heatmaps.py \
  --cpu-csv data/01_p2p_simplex/cpu_192ranks_pair_latency.csv \
  --tcp-csv data/01_p2p_simplex/tcp_192ranks_pair_latency.csv \
  --gpu-csv data/01_p2p_simplex/gpu_c23g_transfer_latency.csv \
  --mpi-duplex-csv data/02_p2p_duplex/cpu_192ranks_duplex_latency.csv \
  --tcp-duplex-csv data/02_p2p_duplex/tcp_192ranks_duplex_latency.csv

# Generate Coupling Schemes and Worker Scaling curves
python3 plot_coupling_schemes.py \
  --mpi-summary data/04_coupling_schemes/mpi_coupling_summary.csv \
  --mpi-detail  data/04_coupling_schemes/mpi_coupling_details.csv \
  --tcp-summary data/04_coupling_schemes/tcp_coupling_summary.csv \
  --tcp-detail  data/04_coupling_schemes/tcp_coupling_details.csv \
  --mpi-worker-sweep data/05_worker_count_scaling/mpi_worker_sweep_summary.csv \
  --tcp-worker-sweep data/05_worker_count_scaling/tcp_worker_sweep_summary.csv \
  --output-prefix "worker_scaling"
```
