#!/bin/bash
# affinity_test/run_tcp_worker_sweep_devel.sbatch
# ================================================
# TCP/IP (IPoIB) Active Worker Count Scaling Sweep (W = 1 to 191).
# 2 nodes, 96 ranks/node (192 total).
#
# Evaluates throughput and latency scaling across W in {1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 191}
# with 50 timed iterations per pattern.

#SBATCH -J tcp_worker_sweep
#SBATCH -N 2
#SBATCH --ntasks-per-node=96
#SBATCH --mem=238G
#SBATCH --time=00:45:00
#SBATCH -p devel
#SBATCH -o /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/affinity_test/logs/tcp_worker_sweep_devel_%j.out

set -euo pipefail

WORKDIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/affinity_test"
cd "$WORKDIR"
mkdir -p "${WORKDIR}/logs"

source /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh

PYTHON_BIN="/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3"
MPI_SWEEP_SUM="mpi_worker_sweep_summary.csv"
TCP_SWEEP_SUM="tcp_worker_sweep_summary.csv"
TCP_SWEEP_DET="tcp_worker_sweep_details.csv"

echo "=== TCP/IP (IPoIB) Active Worker Scaling Sweep Benchmark ==="
echo "Job:    ${SLURM_JOB_ID}"
echo "Nodes:  ${SLURM_JOB_NUM_NODES}, Ranks: ${SLURM_NTASKS}"
echo "iface:  ib0"
date

# Build binary if needed
if [[ ! -f build/tcp_worker_sweep ]] || \
   [[ tcp_worker_sweep.cpp -nt build/tcp_worker_sweep ]] || \
   [[ CMakeLists.txt       -nt build/tcp_worker_sweep ]]; then
    echo ">>> Building tcp_worker_sweep ..."
    mkdir -p build
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_VERBOSE_MAKEFILE=OFF
    cmake --build build --target tcp_worker_sweep -j 4
fi

echo ""
echo ">>> Running TCP/IP Worker Count Scaling Sweep (1 to 191 Workers) ..."
mpirun -mca shmem_mmap_enable_nfs_warning 0 -mca orte_tmpdir_base /tmp \
  --map-by core --bind-to core \
  -np 192 ./build/tcp_worker_sweep \
    --iface ib0 \
    -o "${TCP_SWEEP_SUM}" \
    -d "${TCP_SWEEP_DET}" \
    -t 0 \
    -p 4194304 \
    -w 5 -i 50

echo ""
echo ">>> Generating Combined MPI vs TCP Worker Scaling Curves ..."
if [[ -f "${MPI_SWEEP_SUM}" ]]; then
    "${PYTHON_BIN}" plot_coupling_schemes.py \
        --mpi-worker-sweep "${MPI_SWEEP_SUM}" \
        --tcp-worker-sweep "${TCP_SWEEP_SUM}" \
        --output-prefix "worker_scaling"
else
    "${PYTHON_BIN}" plot_coupling_schemes.py \
        --tcp-worker-sweep "${TCP_SWEEP_SUM}" \
        --output-prefix "worker_scaling"
fi

echo "TCP worker scaling sweep completed successfully."
date
