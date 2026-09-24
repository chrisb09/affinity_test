#!/bin/bash
# affinity_test/run_cpu_duplex_devel.sbatch
# =========================================
# MPI Full-Duplex (simultaneous bidirectional) rank-to-rank latency & bandwidth benchmark.
# 2 nodes, 96 ranks/node (192 total).
#
# Checkpointing: after 30 minutes the C++ binary writes progress to cpu_duplex_latency.state
# and auto-resubmits via sbatch --dependency=afterok:<job_id> until all pairs are complete.

#SBATCH -J mpi_duplex_latency
#SBATCH -N 2
#SBATCH --ntasks-per-node=96
#SBATCH --mem=238G
#SBATCH --time=01:00:00
#SBATCH -p devel
#SBATCH -o /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/affinity_test/logs/cpu_duplex_devel_%j.out

set -euo pipefail

WORKDIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/affinity_test"
cd "$WORKDIR"
mkdir -p "${WORKDIR}/logs"

source /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh

PYTHON_BIN="/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3"
SBATCH_SCRIPT="${WORKDIR}/run_cpu_duplex_devel.sbatch"
OUTPUT_CSV="cpu_192ranks_duplex_latency.csv"
STATE_FILE="cpu_duplex_latency.state"
SIMPLEX_CSV="cpu_192ranks_pair_latency.csv"

echo "=== MPI Full-Duplex Pair Latency & Bandwidth Benchmark ==="
echo "Job:    ${SLURM_JOB_ID}"
echo "Nodes:  ${SLURM_JOB_NUM_NODES}, Ranks: ${SLURM_NTASKS}"
echo "State:  ${STATE_FILE}"
date

if [[ ! -f build/mpi_duplex_latency ]] || \
   [[ mpi_duplex_latency.cpp -nt build/mpi_duplex_latency ]] || \
   [[ CMakeLists.txt         -nt build/mpi_duplex_latency ]]; then
    echo ">>> Building mpi_duplex_latency ..."
    mkdir -p build
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_VERBOSE_MAKEFILE=OFF
    cmake --build build --target mpi_duplex_latency -j 4
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
  -np 192 ./build/mpi_duplex_latency \
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
        --cpu-csv "${SIMPLEX_CSV}" \
        --mpi-duplex-csv "${OUTPUT_CSV}"
    echo "Done."
else
    echo ">>> Checkpoint reached; continuation job submitted. Exiting."
fi

date
