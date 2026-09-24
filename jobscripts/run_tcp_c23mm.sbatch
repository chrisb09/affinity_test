#!/bin/bash
#SBATCH -J tcp_pair_c23mm
#SBATCH -N 2
#SBATCH --ntasks-per-node=96
#SBATCH --exclusive
#SBATCH --mem=238G
#SBATCH --time=08:00:00
#SBATCH -p c23mm
#SBATCH -A rwth2150
#SBATCH -o logs/tcp_c23mm_%j.out

set -euo pipefail

WORKDIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/affinity_test"
cd "$WORKDIR"
mkdir -p logs data/01_p2p_simplex

source /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh

PYTHON_BIN="/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3"
SBATCH_SCRIPT="${WORKDIR}/jobscripts/run_tcp_c23mm.sbatch"
OUTPUT_CSV="data/01_p2p_simplex/tcp_192ranks_pair_latency_c23mm.csv"
STATE_FILE="tcp_pair_latency_c23mm.state"

echo "================================================================="
echo "=== TCP/IP (IPoIB) Pair Latency & Bandwidth Benchmark (c23mm) ==="
echo "================================================================="
echo "Job:    ${SLURM_JOB_ID}"
echo "Nodes:  ${SLURM_JOB_NUM_NODES}, Ranks: ${SLURM_NTASKS}"
echo "iface:  ib0 (IPoIB)"
echo "Account: rwth2150, Partition: c23mm, Exclusive: YES"
echo "State:  ${STATE_FILE}"
date
echo ""

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
    --state-file "${STATE_FILE}" \
    --sbatch-script "${SBATCH_SCRIPT}" \
    --checkpoint-mins 450 \
    ${RESUME_FLAG}

if [[ ! -f "${STATE_FILE}" ]]; then
    echo ""
    echo ">>> All pairs complete. Generating heatmaps and locality summary..."
    "${PYTHON_BIN}" plot_heatmaps.py \
        --tcp-csv "${OUTPUT_CSV}" \
        --tcp-prefix data/01_p2p_simplex/tcp_pair_c23mm || true
    echo "Done."
else
    echo ">>> Checkpoint reached; continuation job submitted. Exiting."
fi

date
