#!/bin/bash
#SBATCH -J worker_sweep_c23mm
#SBATCH -N 2
#SBATCH --ntasks-per-node=96
#SBATCH --exclusive
#SBATCH --mem=238G
#SBATCH --time=02:00:00
#SBATCH -p c23mm
#SBATCH -A rwth2150
#SBATCH -o logs/worker_sweep_c23mm_%j.out

set -euo pipefail

WORKDIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/affinity_test"
cd "$WORKDIR"
mkdir -p logs data/05_worker_count_scaling

source /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh

PYTHON_BIN="/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3"
SWEEP_SUM_CSV="data/05_worker_count_scaling/mpi_worker_sweep_summary_c23mm.csv"
SWEEP_DET_CSV="data/05_worker_count_scaling/mpi_worker_sweep_details_c23mm.csv"

echo "================================================================="
echo "=== MPI Active Worker Scaling Sweep Benchmark (c23mm) ==="
echo "================================================================="
echo "Job:    ${SLURM_JOB_ID}"
echo "Nodes:  ${SLURM_JOB_NUM_NODES}, Ranks: ${SLURM_NTASKS}"
echo "Account: rwth2150, Partition: c23mm, Exclusive: YES"
date
echo ""

echo ">>> Running MPI Worker Count Scaling Sweep (1 to 191 Workers) ..."
mpirun -mca shmem_mmap_enable_nfs_warning 0 -mca orte_tmpdir_base /tmp \
  --map-by core --bind-to core \
  -np 192 ./build/mpi_worker_sweep \
    -o "${SWEEP_SUM_CSV}" \
    -d "${SWEEP_DET_CSV}" \
    -t 0 \
    -p 4194304 \
    -w 5 -i 50

echo ""
echo ">>> Generating Worker Scaling Curves ..."
"${PYTHON_BIN}" plot_coupling_schemes.py \
    --worker-sweep-summary "${SWEEP_SUM_CSV}" \
    --output-prefix "worker_scaling_mpi_c23mm" || true

echo "Worker scaling sweep completed successfully."
date
