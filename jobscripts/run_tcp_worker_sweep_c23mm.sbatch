#!/bin/bash
#SBATCH -J tcp_worker_c23mm
#SBATCH -N 2
#SBATCH --ntasks-per-node=96
#SBATCH --exclusive
#SBATCH --mem=238G
#SBATCH --time=02:00:00
#SBATCH -p c23mm
#SBATCH -A rwth2150
#SBATCH -o logs/tcp_worker_sweep_c23mm_%j.out

set -euo pipefail

WORKDIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/affinity_test"
cd "$WORKDIR"
mkdir -p logs data/05_worker_count_scaling

source /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh

PYTHON_BIN="/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3"
MPI_SWEEP_SUM="data/05_worker_count_scaling/mpi_worker_sweep_summary_c23mm.csv"
TCP_SWEEP_SUM="data/05_worker_count_scaling/tcp_worker_sweep_summary_c23mm.csv"
TCP_SWEEP_DET="data/05_worker_count_scaling/tcp_worker_sweep_details_c23mm.csv"

echo "================================================================="
echo "=== TCP/IP (IPoIB) Active Worker Scaling Sweep (c23mm) ==="
echo "================================================================="
echo "Job:    ${SLURM_JOB_ID}"
echo "Nodes:  ${SLURM_JOB_NUM_NODES}, Ranks: ${SLURM_NTASKS}"
echo "iface:  ib0"
echo "Account: rwth2150, Partition: c23mm, Exclusive: YES"
date
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
        --output-prefix "worker_scaling_c23mm" || true
else
    "${PYTHON_BIN}" plot_coupling_schemes.py \
        --tcp-worker-sweep "${TCP_SWEEP_SUM}" \
        --output-prefix "worker_scaling_tcp_c23mm" || true
fi

echo "TCP worker scaling sweep completed successfully."
date
