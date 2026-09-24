#!/bin/bash
#SBATCH -J uds_c23mm
#SBATCH -N 1
#SBATCH --ntasks-per-node=96
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH -p c23mm
#SBATCH -A rwth2150
#SBATCH -o logs/uds_c23mm_%j.out

set -euo pipefail

WORKDIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/affinity_test"
cd "$WORKDIR"
mkdir -p logs data/uds_benchmarks /tmp/uds_${SLURM_JOB_ID}

# Load OpenMPI and compiler environment
source /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh

PYTHON_BIN="/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3"
SOCKET_DIR="/tmp/uds_${SLURM_JOB_ID}"

echo "================================================================="
echo "=== Unix Domain Socket (UDS) Multi-Tier Benchmark Suite (c23mm) ==="
echo "================================================================="
echo "Job ID: $SLURM_JOB_ID, Node: $(hostname)"
echo "Account: rwth2150, Partition: c23mm"
echo "Ranks: 96 (Single Node, Core-Bound, Exclusive)"
echo "Socket Dir: $SOCKET_DIR"
date
echo ""

# 1. Pairwise Simplex Latency & Bandwidth (4,560 pairs)
echo ">>> [1/4] Running UDS Pairwise Simplex Benchmark ..."
mpirun -mca shmem_mmap_enable_nfs_warning 0 -mca orte_tmpdir_base /tmp \
  --map-by core --bind-to core \
  -np 96 ./build/uds_pair_latency \
  -o data/uds_benchmarks/uds_96ranks_pair_latency_c23mm.csv \
  --socket-dir "$SOCKET_DIR" \
  -s 64 -l 4194304 -w 10 -i 50
date

# 2. Pairwise Full-Duplex Simultaneous Exchange (4,560 pairs)
echo ">>> [2/4] Running UDS Pairwise Full-Duplex Benchmark ..."
mpirun -mca shmem_mmap_enable_nfs_warning 0 -mca orte_tmpdir_base /tmp \
  --map-by core --bind-to core \
  -np 96 ./build/uds_duplex_latency \
  -o data/uds_benchmarks/uds_96ranks_duplex_latency_c23mm.csv \
  --socket-dir "$SOCKET_DIR" \
  -s 64 -l 4194304 -w 10 -i 50
date

# 3. Incast / Fan-in Flooding Payload Sweep (95 -> 1)
echo ">>> [3/4] Running UDS Incast Flooding Sweep ..."
mpirun -mca shmem_mmap_enable_nfs_warning 0 -mca orte_tmpdir_base /tmp \
  --map-by core --bind-to core \
  -np 96 ./build/uds_incast_latency \
  -o data/uds_benchmarks/uds_incast_summary_c23mm.csv \
  -d data/uds_benchmarks/uds_incast_details_c23mm.csv \
  -t 0 \
  -p 64,1024,4096,16384,65536,262144,1048576,4194304 \
  --socket-dir "$SOCKET_DIR" \
  -w 5 -i 20
date

# 4. HPC-ML Coupling Transport Schemes (95 Workers)
echo ">>> [4/4] Running UDS Coupling Schemes Suite ..."
mpirun -mca shmem_mmap_enable_nfs_warning 0 -mca orte_tmpdir_base /tmp \
  --map-by core --bind-to core \
  -np 96 ./build/uds_coupling_schemes \
  -o data/uds_benchmarks/uds_coupling_summary_c23mm.csv \
  -d data/uds_benchmarks/uds_coupling_details_c23mm.csv \
  -t 0 \
  -p 4194304 \
  --socket-dir "$SOCKET_DIR" \
  -w 3 -i 20
date

# Cleanup temporary sockets
rm -rf "$SOCKET_DIR"

# 5. Visualizations
echo ">>> [5/5] Generating Visualizations ..."
if [[ -f data/01_p2p_simplex/cpu_192ranks_pair_latency.csv && -f data/01_p2p_simplex/tcp_192ranks_pair_latency.csv ]]; then
  "$PYTHON_BIN" plot_heatmaps.py \
    --cpu-csv data/01_p2p_simplex/cpu_192ranks_pair_latency.csv \
    --tcp-csv data/01_p2p_simplex/tcp_192ranks_pair_latency.csv \
    --uds-csv data/uds_benchmarks/uds_96ranks_pair_latency_c23mm.csv \
    --uds-duplex-csv data/uds_benchmarks/uds_96ranks_duplex_latency_c23mm.csv \
    --mpi-incast-csv data/03_incast_flooding_payload_sweep/mpi_incast_summary.csv \
    --tcp-incast-csv data/03_incast_flooding_payload_sweep/tcp_incast_summary.csv \
    --uds-incast-csv data/uds_benchmarks/uds_incast_summary_c23mm.csv \
    --uds-prefix data/uds_benchmarks/uds_pair_c23mm || true
fi

echo "================================================================="
echo "=== UDS Benchmark Suite (c23mm) Finished Successfully ==="
echo "================================================================="
date
