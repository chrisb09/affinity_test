#!/bin/bash
# affinity_test/run_tcp_devel.sbatch
# =====================================
# TCP/IP (IPoIB) rank-to-rank latency & bandwidth benchmark.
# 2 nodes, 96 ranks/node (192 total), matching run_cpu_devel.sbatch.
#
# Checkpointing: after 30 minutes the C++ binary writes its progress to
# tcp_pair_latency.state and calls sbatch on this very script with
# --dependency=afterok:<current_job_id>, so jobs chain automatically until
# all 18,336 pairs are complete.  The CSV is appended across jobs; the state
# file is deleted when the final pair is done.
#
# Usage:
#   sbatch run_tcp_devel.sbatch            # first invocation
#   # subsequent invocations are submitted automatically by the binary

#SBATCH -J tcp_pair_latency
#SBATCH -N 2
#SBATCH --ntasks-per-node=96
#SBATCH --mem=238G
#SBATCH --time=01:00:00
#SBATCH -p devel
#SBATCH -o tcp_devel_%j.out

set -euo pipefail

WORKDIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/affinity_test"
cd "$WORKDIR"

source /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh

PYTHON_BIN="/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3"
SBATCH_SCRIPT="${WORKDIR}/run_tcp_devel.sbatch"
OUTPUT_CSV="tcp_192ranks_pair_latency.csv"
STATE_FILE="tcp_pair_latency.state"

echo "=== TCP/IP Pair Latency & Bandwidth Benchmark ==="
echo "Job:    ${SLURM_JOB_ID}"
echo "Nodes:  ${SLURM_JOB_NUM_NODES}, Ranks: ${SLURM_NTASKS}"
echo "iface:  ib0 (IPoIB)"
echo "State:  ${STATE_FILE}"
date

# Build if binary is missing or source is newer
if [[ ! -f build/tcp_pair_latency ]] || \
   [[ tcp_pair_latency.cpp -nt build/tcp_pair_latency ]] || \
   [[ CMakeLists.txt       -nt build/tcp_pair_latency ]]; then
    echo ">>> Building tcp_pair_latency ..."
    mkdir -p build
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_VERBOSE_MAKEFILE=OFF
    cmake --build build --target tcp_pair_latency -j 4
fi

# Decide whether to resume
RESUME_FLAG=""
if [[ -f "${STATE_FILE}" ]]; then
    echo ">>> State file found — resuming."
    RESUME_FLAG="--resume"
else
    echo ">>> No state file — starting fresh."
fi

mpirun -mca shmem_mmap_enable_nfs_warning 0 -mca orte_tmpdir_base /tmp \
  --map-by core --bind-to core \
  -np 192 ./build/tcp_pair_latency \
    --iface ib0 \
    -o "${OUTPUT_CSV}" \
    -s 64 -l 4194304 -w 10 -i 50 \
    --state-file  "${STATE_FILE}" \
    --sbatch-script "${SBATCH_SCRIPT}" \
    --checkpoint-mins 30 \
    ${RESUME_FLAG}

# If the state file is gone, all pairs are complete — run plots
if [[ ! -f "${STATE_FILE}" ]]; then
    echo ""
    echo ">>> All pairs complete. Generating heatmaps and locality summary..."
    "${PYTHON_BIN}" plot_heatmaps.py \
        --tcp-csv  "${OUTPUT_CSV}" \
        --tcp-prefix tcp_pair
    echo "Done."
else
    echo ">>> Checkpoint reached; continuation job submitted. Exiting."
fi

date
