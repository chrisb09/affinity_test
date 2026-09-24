#!/bin/bash
# affinity_test/run_worker_sweep_devel.sbatch
# ============================================
# Active Worker Count Scaling Sweep (W = 1 to 191) for HPC-ML Controller Communication.
# 2 nodes, 96 ranks/node (192 total).
#
# Evaluates throughput and latency scaling across W in {1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 191}
# with 50 timed iterations per pattern.

#SBATCH -J worker_sweep
#SBATCH -N 2
#SBATCH --ntasks-per-node=96
#SBATCH --mem=238G
#SBATCH --time=00:45:00
#SBATCH -p devel
#SBATCH -o /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/affinity_test/logs/worker_sweep_devel_%j.out

set -euo pipefail

WORKDIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/affinity_test"
cd "$WORKDIR"
mkdir -p "${WORKDIR}/logs"

source /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh

PYTHON_BIN="/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3"
SWEEP_SUM_CSV="mpi_worker_sweep_summary.csv"
SWEEP_DET_CSV="mpi_worker_sweep_details.csv"

echo "=== MPI Active Worker Scaling Sweep Benchmark ==="
echo "Job:    ${SLURM_JOB_ID}"
echo "Nodes:  ${SLURM_JOB_NUM_NODES}, Ranks: ${SLURM_NTASKS}"
date

echo "--- OpenMPI Active Configuration & MCA Parameters ---"
ompi_info --param btl all --level 9 2>/dev/null | grep -E "btl_vader_eager_limit|btl_vader_max_send_size|btl_openib_eager_limit" || true
ompi_info --param pml all --level 9 2>/dev/null | grep -E "pml_ucx|pml_ob1" || true

# Build binary if needed
if [[ ! -f build/mpi_worker_sweep ]] || \
   [[ mpi_worker_sweep.cpp -nt build/mpi_worker_sweep ]] || \
   [[ CMakeLists.txt       -nt build/mpi_worker_sweep ]]; then
    echo ">>> Building mpi_worker_sweep ..."
    mkdir -p build
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_VERBOSE_MAKEFILE=OFF
    cmake --build build --target mpi_worker_sweep -j 4
fi

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
    --output-prefix "worker_scaling"

echo "Worker scaling sweep completed successfully."
date
