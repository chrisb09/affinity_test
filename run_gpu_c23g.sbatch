#!/bin/bash
#SBATCH -J gpu_rank_transfer
#SBATCH -N 1
#SBATCH --ntasks-per-node=96
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH -p c23g
#SBATCH -A thes2181
#SBATCH -o gpu_c23g_%j.out

set -euo pipefail

WORKDIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/affinity_test"
cd "$WORKDIR"

# Load OpenMPI 4.1.4 and CUDA 12.4 modules
source /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh

PYTHON_BIN="/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3"

echo "=== Starting GPU Rank Transfer Benchmark (Exclusive Node) ==="
echo "Partition: c23g, Account: thes2181, GPUs: 4"
echo "Using mpirun from: $(which mpirun)"
date

mpirun -mca shmem_mmap_enable_nfs_warning 0 -mca orte_tmpdir_base /tmp \
  --map-by core --bind-to core \
  -np 96 ./build/mpi_gpu_transfer \
  -d 0,1,2,3 -s 4096 -l 16777216 \
  -o gpu_c23g_transfer_latency.csv -w 5 -i 20

echo "GPU benchmark completed. Generating heatmaps..."
"$PYTHON_BIN" plot_heatmaps.py --gpu-csv gpu_c23g_transfer_latency.csv
date
