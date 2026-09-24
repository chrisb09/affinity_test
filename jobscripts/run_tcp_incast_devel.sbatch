#!/bin/bash
# affinity_test/run_tcp_incast_devel.sbatch
# ==========================================
# TCP/IP (IPoIB) Incast Flooding Benchmark (Fan-in Congestion).
# 2 nodes, 96 ranks/node (192 total).
# Sweeps payload sizes 64 B to 4 MiB targeting Rank 0 (Node 0) and Rank 96 (Node 1).

#SBATCH -J tcp_incast_bench
#SBATCH -N 2
#SBATCH --ntasks-per-node=96
#SBATCH --mem=238G
#SBATCH --time=00:30:00
#SBATCH -p devel
#SBATCH -o /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/affinity_test/logs/tcp_incast_devel_%j.out

set -euo pipefail

WORKDIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/affinity_test"
cd "$WORKDIR"
mkdir -p "${WORKDIR}/logs"

source /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh

PYTHON_BIN="/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3"
SUMMARY_CSV="tcp_incast_summary.csv"
DETAIL_CSV="tcp_incast_details.csv"
MPI_SUMMARY_CSV="mpi_incast_summary.csv"

echo "=== TCP/IP (IPoIB) Incast Flooding Benchmark ==="
echo "Job:    ${SLURM_JOB_ID}"
echo "Nodes:  ${SLURM_JOB_NUM_NODES}, Ranks: ${SLURM_NTASKS}"
echo "iface:  ib0"
date

if [[ ! -f build/tcp_incast_latency ]] || \
   [[ tcp_incast_latency.cpp -nt build/tcp_incast_latency ]] || \
   [[ CMakeLists.txt         -nt build/tcp_incast_latency ]]; then
    echo ">>> Building tcp_incast_latency ..."
    mkdir -p build
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_VERBOSE_MAKEFILE=OFF
    cmake --build build --target tcp_incast_latency -j 4
fi

mpirun -mca shmem_mmap_enable_nfs_warning 0 -mca orte_tmpdir_base /tmp \
  --map-by core --bind-to core \
  -np 192 ./build/tcp_incast_latency \
    --iface ib0 \
    -o "${SUMMARY_CSV}" \
    -d "${DETAIL_CSV}" \
    -t 0,96 \
    -p 64,1024,4096,16384,65536,262144,1048576,4194304 \
    -w 5 -i 20

echo ""
echo ">>> TCP Incast benchmark complete. Generating scaling plots..."
if [[ -f "${MPI_SUMMARY_CSV}" ]]; then
    "${PYTHON_BIN}" plot_heatmaps.py \
        --mpi-incast-csv "${MPI_SUMMARY_CSV}" \
        --tcp-incast-csv "${SUMMARY_CSV}"
else
    "${PYTHON_BIN}" plot_heatmaps.py \
        --tcp-incast-csv "${SUMMARY_CSV}"
fi

date
