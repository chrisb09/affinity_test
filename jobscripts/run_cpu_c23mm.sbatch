#!/bin/bash
#SBATCH -J cpu_pair_c23mm
#SBATCH -N 2
#SBATCH --ntasks-per-node=96
#SBATCH --exclusive
#SBATCH --mem=238G
#SBATCH --time=08:00:00
#SBATCH -p c23mm
#SBATCH -A rwth2150
#SBATCH -o logs/cpu_c23mm_%j.out

set -euo pipefail

WORKDIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/affinity_test"
cd "$WORKDIR"
mkdir -p logs data/01_p2p_simplex

source /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh

PYTHON_BIN="/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3"
OUTPUT_CSV="data/01_p2p_simplex/cpu_192ranks_pair_latency_c23mm.csv"

echo "================================================================="
echo "=== CPU Core Latency & Bandwidth Benchmark (c23mm) ==="
echo "================================================================="
echo "Job ID: $SLURM_JOB_ID, Nodes: $SLURM_JOB_NUM_NODES, Total Ranks: $SLURM_NTASKS"
echo "Account: rwth2150, Partition: c23mm, Exclusive: YES"
echo "Using mpirun from: $(which mpirun)"
date
echo ""

mpirun -mca shmem_mmap_enable_nfs_warning 0 -mca orte_tmpdir_base /tmp \
  --map-by core --bind-to core \
  -np 192 ./build/mpi_pair_latency \
  -o "${OUTPUT_CSV}" -s 64 -l 4194304 -w 10 -i 50

echo ""
echo "CPU benchmark completed. Generating heatmaps..."
"$PYTHON_BIN" plot_heatmaps.py --cpu-csv "${OUTPUT_CSV}" --cpu-prefix data/01_p2p_simplex/cpu_pair_c23mm || true
date
