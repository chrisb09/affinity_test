#!/bin/bash
# affinity_test/run_tcp_duplex_devel.sbatch
# =========================================
# TCP/IP (IPoIB) Full-Duplex rank-to-rank latency & bandwidth benchmark.
# 2 nodes, 96 ranks/node (192 total).
#
# Checkpointing: after 30 minutes the C++ binary writes progress to tcp_duplex_latency.state
# and auto-resubmits via sbatch --dependency=afterok:<job_id> until all pairs are complete.

#SBATCH -J tcp_duplex_latency
#SBATCH -N 2
#SBATCH --ntasks-per-node=96
#SBATCH --mem=238G
#SBATCH --time=01:00:00
#SBATCH -p devel
#SBATCH -o /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/affinity_test/logs/tcp_duplex_devel_%j.out

set -euo pipefail

WORKDIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/affinity_test"
cd "$WORKDIR"
mkdir -p "${WORKDIR}/logs"

source /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh

PYTHON_BIN="/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3"
SBATCH_SCRIPT="${WORKDIR}/run_tcp_duplex_devel.sbatch"
OUTPUT_CSV="tcp_192ranks_duplex_latency.csv"
STATE_FILE="tcp_duplex_latency.state"
SIMPLEX_CSV="tcp_192ranks_pair_latency.csv"

echo "=== TCP/IP Full-Duplex Pair Latency & Bandwidth Benchmark ==="
echo "Job:    ${SLURM_JOB_ID}"
echo "Nodes:  ${SLURM_JOB_NUM_NODES}, Ranks: ${SLURM_NTASKS}"
echo "iface:  ib0 (IPoIB)"
echo "State:  ${STATE_FILE}"
date

if [[ ! -f build/tcp_duplex_latency ]] || \
   [[ tcp_duplex_latency.cpp -nt build/tcp_duplex_latency ]] || \
   [[ CMakeLists.txt         -nt build/tcp_duplex_latency ]]; then
    echo ">>> Building tcp_duplex_latency ..."
    mkdir -p build
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_VERBOSE_MAKEFILE=OFF
    cmake --build build --target tcp_duplex_latency -j 4
fi

RESUME_FLAG=""
if [[ -f "${STATE_FILE}" ]]; then
    echo ">>> State file found — resuming."
    RESUME_FLAG="--resume"
else
    echo ">>> No state file — starting fresh."
fi

mpirun -mca shmem_mmap_enable_nfs_warning 0 -mca orte_tmpdir_base /tmp \
  --map-by core --bind-to core \
  -np 192 ./build/tcp_duplex_latency \
    --iface ib0 \
    -o "${OUTPUT_CSV}" \
    -s 64 -l 4194304 -w 10 -i 50 \
    --state-file "${STATE_FILE}" \
    --sbatch-script "${SBATCH_SCRIPT}" \
    --checkpoint-mins 30 \
    ${RESUME_FLAG}

if [[ ! -f "${STATE_FILE}" ]]; then
    echo ""
    echo ">>> All pairs complete. Generating heatmaps and simplex vs duplex comparison..."
    "${PYTHON_BIN}" plot_heatmaps.py \
        --tcp-csv "${SIMPLEX_CSV}" \
        --tcp-duplex-csv "${OUTPUT_CSV}"
    echo "Done."
else
    echo ">>> Checkpoint reached; continuation job submitted. Exiting."
fi

date
