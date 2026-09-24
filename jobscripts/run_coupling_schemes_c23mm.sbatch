#!/bin/bash
#SBATCH -J coupling_c23mm
#SBATCH -N 2
#SBATCH --ntasks-per-node=96
#SBATCH --exclusive
#SBATCH --mem=238G
#SBATCH --time=02:00:00
#SBATCH -p c23mm
#SBATCH -A rwth2150
#SBATCH -o logs/coupling_c23mm_%j.out

set -euo pipefail

WORKDIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/affinity_test"
cd "$WORKDIR"
mkdir -p logs data/04_coupling_schemes

source /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh

PYTHON_BIN="/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3"
MPI_SUM_CSV="data/04_coupling_schemes/mpi_coupling_summary_c23mm.csv"
MPI_DET_CSV="data/04_coupling_schemes/mpi_coupling_details_c23mm.csv"
TCP_SUM_CSV="data/04_coupling_schemes/tcp_coupling_summary_c23mm.csv"
TCP_DET_CSV="data/04_coupling_schemes/tcp_coupling_details_c23mm.csv"

echo "================================================================="
echo "=== HPC-ML Coupling Transport Schemes Suite (c23mm) ==="
echo "================================================================="
echo "Job:    ${SLURM_JOB_ID}"
echo "Nodes:  ${SLURM_JOB_NUM_NODES}, Ranks: ${SLURM_NTASKS}"
echo "Account: rwth2150, Partition: c23mm, Exclusive: YES"
date
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
    --output-prefix "coupling_schemes_c23mm" || true

echo "All coupling benchmarks completed successfully."
date
