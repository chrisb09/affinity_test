#!/bin/bash
#SBATCH -J gpu_rank_transfer
#SBATCH -N 1
#SBATCH --ntasks-per-node=96
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH -p c23g
#SBATCH -A thes2181
#SBATCH -o logs/gpu_c23g_%j.out

set -euo pipefail

WORKDIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/affinity_test"
cd "$WORKDIR"
mkdir -p logs data/01_p2p_simplex

# Load OpenMPI 4.1.4 and CUDA 12.4 modules
source /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh

PYTHON_BIN="/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3"

echo "================================================================="
echo "=== System Hardware Topology & Diagnostics ==="
echo "================================================================="
echo "Node: $(hostname)"
date
echo ""
echo "--- GPU Device Status ---"
nvidia-smi
echo ""
echo "--- GPU Interconnect & NUMA Affinity Topology Matrix ---"
nvidia-smi topo -m
echo ""
echo "--- Host NUMA Hardware Configuration ---"
numactl --hardware
echo ""
echo "================================================================="
echo "=== Starting Precision GPU Rank Transfer Benchmark ==="
echo "================================================================="
echo "Partition: c23g, Account: thes2181, GPUs: 4, Ranks: 96"
echo "Payloads: Latency = 4 KiB, Bandwidth = 16 MiB, 64 MiB, 256 MiB"
echo "Iterations: 5 warmup, 50 timed iterations (interleaved/counterbalanced)"
echo "Using mpirun from: $(which mpirun)"
date

mpirun -mca shmem_mmap_enable_nfs_warning 0 -mca orte_tmpdir_base /tmp \
  --map-by core --bind-to core \
  -np 96 ./build/mpi_gpu_transfer \
  -d 0,1,2,3 \
  -s 4096 \
  -l 16777216,67108864,268435456 \
  -w 5 -i 50 \
  -o data/01_p2p_simplex/gpu_c23g_transfer_latency.csv

echo "GPU benchmark completed. Generating heatmaps..."
"$PYTHON_BIN" plot_heatmaps.py \
  --gpu-csv data/01_p2p_simplex/gpu_c23g_transfer_latency.csv

echo "Generating thesis baseline figures..."
"$PYTHON_BIN" /rwthfs/rz/cluster/home/ro092286/Master-Thesis/gitlab/sourcecode/affinity_test/plot_thesis_baselines.py || true

date
echo "=== All tasks completed successfully! ==="
