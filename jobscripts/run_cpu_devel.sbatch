#!/bin/bash
#SBATCH -J cpu_pair_latency
#SBATCH -N 2
#SBATCH --ntasks-per-node=96
#SBATCH --mem=238G
#SBATCH --time=01:00:00
#SBATCH -p devel
#SBATCH -o cpu_devel_%j.out

set -euo pipefail

WORKDIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/affinity_test"
cd "$WORKDIR"

# Load OpenMPI 4.1.4 and CUDA 12.4 modules
source /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh

PYTHON_BIN="/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3"

echo "=== Starting CPU Core Latency & Bandwidth Benchmark ==="
echo "Node count: $SLURM_JOB_NUM_NODES, Total ranks: 192"
echo "Using mpirun from: $(which mpirun)"
date

mpirun -mca shmem_mmap_enable_nfs_warning 0 -mca orte_tmpdir_base /tmp \
  --map-by core --bind-to core \
  -np 192 ./build/mpi_pair_latency \
  -o cpu_192ranks_pair_latency.csv -s 64 -l 4194304 -w 10 -i 50

echo "CPU benchmark completed. Generating heatmaps..."
"$PYTHON_BIN" plot_heatmaps.py --cpu-csv cpu_192ranks_pair_latency.csv
date
