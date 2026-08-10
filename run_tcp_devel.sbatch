#!/bin/bash
# affinity_test/run_tcp_devel.sbatch
# =====================================
# TCP/IP (IPoIB) rank-to-rank latency & bandwidth benchmark.
# Mirrors run_cpu_devel.sbatch exactly: 2 nodes, 96 ranks/node, core binding.
#
# Usage:
#   sbatch run_tcp_devel.sbatch
#
# Results written to:
#   affinity_test/tcp_192ranks_pair_latency.csv
#   affinity_test/tcp_pair_latency_heatmap.png
#   affinity_test/tcp_pair_bandwidth_heatmap.png
#   affinity_test/tcp_locality_summary.csv

#SBATCH -J tcp_pair_latency
#SBATCH -N 2
#SBATCH --ntasks-per-node=96
#SBATCH --mem=238G
#SBATCH --time=04:00:00
#SBATCH -p devel
#SBATCH -o tcp_devel_%j.out

set -euo pipefail

WORKDIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/affinity_test"
cd "$WORKDIR"

source /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh

PYTHON_BIN="/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3"

echo "=== Starting TCP/IP Pair Latency & Bandwidth Benchmark ==="
echo "Node count: $SLURM_JOB_NUM_NODES, Total ranks: $SLURM_NTASKS"
echo "Using mpirun from: $(which mpirun)"
echo "Interface: ib0  (IPoIB, the only non-loopback interface on CLAIX-23)"
date

# Build if binary is missing or source is newer
if [[ ! -f build/tcp_pair_latency ]] || \
   [[ tcp_pair_latency.cpp -nt build/tcp_pair_latency ]] || \
   [[ CMakeLists.txt -nt build/tcp_pair_latency ]]; then
    echo ">>> Building tcp_pair_latency ..."
    mkdir -p build
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
    cmake --build build --target tcp_pair_latency -j 4
fi

# Run with identical parameters to the MPI benchmark:
#   small payload = 64 B  (latency)
#   large payload = 4 MiB (bandwidth)
#   warmup = 10, iters = 50
mpirun -mca shmem_mmap_enable_nfs_warning 0 -mca orte_tmpdir_base /tmp \
  --map-by core --bind-to core \
  -np 192 ./build/tcp_pair_latency \
  --iface ib0 \
  -o tcp_192ranks_pair_latency.csv \
  -s 64 -l 4194304 -w 10 -i 50

echo ""
echo "TCP benchmark completed. Generating heatmaps and locality summary..."
"$PYTHON_BIN" plot_heatmaps.py \
    --tcp-csv tcp_192ranks_pair_latency.csv \
    --tcp-prefix tcp_pair

date
echo "Done."
