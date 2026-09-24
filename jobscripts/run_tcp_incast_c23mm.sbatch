#!/bin/bash
#SBATCH -J tcp_incast_c23mm
#SBATCH -N 2
#SBATCH --ntasks-per-node=96
#SBATCH --exclusive
#SBATCH --mem=238G
#SBATCH --time=02:00:00
#SBATCH -p c23mm
#SBATCH -A rwth2150
#SBATCH -o logs/tcp_incast_c23mm_%j.out

set -euo pipefail

WORKDIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/affinity_test"
cd "$WORKDIR"
mkdir -p logs data/03_incast_flooding_payload_sweep

source /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh

PYTHON_BIN="/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3"
SUMMARY_CSV="data/03_incast_flooding_payload_sweep/tcp_incast_summary_c23mm.csv"
DETAIL_CSV="data/03_incast_flooding_payload_sweep/tcp_incast_details_c23mm.csv"
MPI_SUMMARY_CSV="data/03_incast_flooding_payload_sweep/mpi_incast_summary_c23mm.csv"

echo "================================================================="
echo "=== TCP/IP (IPoIB) Incast Flooding Benchmark (c23mm) ==="
echo "================================================================="
echo "Job:    ${SLURM_JOB_ID}"
echo "Nodes:  ${SLURM_JOB_NUM_NODES}, Ranks: ${SLURM_NTASKS}"
echo "iface:  ib0"
echo "Account: rwth2150, Partition: c23mm, Exclusive: YES"
date
echo ""

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
        --tcp-incast-csv "${SUMMARY_CSV}" || true
else
    "${PYTHON_BIN}" plot_heatmaps.py \
        --tcp-incast-csv "${SUMMARY_CSV}" || true
fi

date
