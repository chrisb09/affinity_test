#!/bin/bash
# affinity_test/run_coupling_schemes_devel.sbatch
# ===============================================
# Comprehensive Transport Schemes Benchmark Suite for HPC-ML Coupling.
# Runs both MPI and TCP (IPoIB) implementations across 192 ranks (2 nodes).
#
# Evaluates:
#   1. Incast (191 -> 1) with Application-level ACKs
#   2. Fan-out (1 -> 191) with Worker ACKs
#   3. Full-Duplex (191 <-> 1 Simultaneous Bidirectional)
#   4. Gather-Scatter (Bulk Synchronous Coupling)
#   5. Immediate Response (Pipelined per-worker Coupling)
#   6. Serialized Reference (Zero-contention baseline)
#   7. Credit Concurrency Sweeps (Caps: 1, 2, 4, 8, 16, 32, 64)
#   8. Locality-aware Credits (1 Local + 1 Remote)
#   9. AIx Ramping Concurrency

#SBATCH -J coupling_bench
#SBATCH -N 2
#SBATCH --ntasks-per-node=96
#SBATCH --mem=238G
#SBATCH --time=01:00:00
#SBATCH -p devel
#SBATCH -o /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/affinity_test/logs/coupling_devel_%j.out

set -euo pipefail

WORKDIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/affinity_test"
cd "$WORKDIR"
mkdir -p "${WORKDIR}/logs"

source /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh

PYTHON_BIN="/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3"
MPI_SUM_CSV="mpi_coupling_summary.csv"
MPI_DET_CSV="mpi_coupling_details.csv"
TCP_SUM_CSV="tcp_coupling_summary.csv"
TCP_DET_CSV="tcp_coupling_details.csv"

echo "=== HPC-ML Coupling Transport Schemes Benchmark Suite ==="
echo "Job:    ${SLURM_JOB_ID}"
echo "Nodes:  ${SLURM_JOB_NUM_NODES}, Ranks: ${SLURM_NTASKS}"
date

# Build binaries if missing or outdated
if [[ ! -f build/mpi_coupling_schemes ]] || \
   [[ mpi_coupling_schemes.cpp -nt build/mpi_coupling_schemes ]] || \
   [[ ! -f build/tcp_coupling_schemes ]] || \
   [[ tcp_coupling_schemes.cpp -nt build/tcp_coupling_schemes ]]; then
    echo ">>> Building coupling schemes binaries ..."
    mkdir -p build
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_VERBOSE_MAKEFILE=OFF
    cmake --build build --target mpi_coupling_schemes tcp_coupling_schemes -j 4
fi

echo ""
echo ">>> 1. Running MPI Coupling Schemes Suite ..."
mpirun -mca shmem_mmap_enable_nfs_warning 0 -mca orte_tmpdir_base /tmp \
  --map-by core --bind-to core \
  -np 192 ./build/mpi_coupling_schemes \
    -o "${MPI_SUM_CSV}" \
    -d "${MPI_DET_CSV}" \
    -t 0,96 \
    -p 4194304 \
    -w 3 -i 20

echo ""
echo ">>> 2. Running TCP/IP (IPoIB) Coupling Schemes Suite ..."
mpirun -mca shmem_mmap_enable_nfs_warning 0 -mca orte_tmpdir_base /tmp \
  --map-by core --bind-to core \
  -np 192 ./build/tcp_coupling_schemes \
    --iface ib0 \
    -o "${TCP_SUM_CSV}" \
    -d "${TCP_DET_CSV}" \
    -t 0,96 \
    -p 4194304 \
    -w 3 -i 20

echo ""
echo ">>> 3. Generating Coupling Schemes Visualization & Analysis Plots ..."
"${PYTHON_BIN}" plot_coupling_schemes.py \
    --mpi-summary "${MPI_SUM_CSV}" \
    --mpi-detail  "${MPI_DET_CSV}" \
    --tcp-summary "${TCP_SUM_CSV}" \
    --tcp-detail  "${TCP_DET_CSV}" \
    --output-prefix "coupling_schemes"

echo "All coupling benchmarks completed successfully."
date
