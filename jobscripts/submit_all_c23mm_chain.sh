#!/usr/bin/env bash
# submit_all_c23mm_chain.sh
# =========================
# Submits all 9 dual-node c23mm affinity & communication benchmarks in a clean serial dependency chain.
# Account: rwth2150
# Partition: c23mm
# Mode: --exclusive

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

echo "================================================================="
echo "=== Submitting c23mm Benchmark Suite Dependency Chain ==="
echo "================================================================="
echo "Account: rwth2150, Partition: c23mm, Exclusive: YES"
date
echo ""

# 1. MPI Simplex Pair Latency
JOB_CPU_SIMPLEX=$(sbatch jobscripts/run_cpu_c23mm.sbatch | awk '{print $NF}')
echo "Submitted [1/9] MPI Simplex Pair Matrix:      Job ${JOB_CPU_SIMPLEX}"

# 2. TCP Simplex Pair Latency (after MPI Simplex)
JOB_TCP_SIMPLEX=$(sbatch --dependency=afterok:${JOB_CPU_SIMPLEX} jobscripts/run_tcp_c23mm.sbatch | awk '{print $NF}')
echo "Submitted [2/9] TCP Simplex Pair Matrix:      Job ${JOB_TCP_SIMPLEX} (after ${JOB_CPU_SIMPLEX})"

# 3. MPI Duplex Pair Latency (after TCP Simplex)
JOB_CPU_DUPLEX=$(sbatch --dependency=afterok:${JOB_TCP_SIMPLEX} jobscripts/run_cpu_duplex_c23mm.sbatch | awk '{print $NF}')
echo "Submitted [3/9] MPI Duplex Pair Matrix:       Job ${JOB_CPU_DUPLEX} (after ${JOB_TCP_SIMPLEX})"

# 4. TCP Duplex Pair Latency (after MPI Duplex)
JOB_TCP_DUPLEX=$(sbatch --dependency=afterok:${JOB_CPU_DUPLEX} jobscripts/run_tcp_duplex_c23mm.sbatch | awk '{print $NF}')
echo "Submitted [4/9] TCP Duplex Pair Matrix:       Job ${JOB_TCP_DUPLEX} (after ${JOB_CPU_DUPLEX})"

# 5. MPI Incast Flooding (after TCP Duplex)
JOB_CPU_INCAST=$(sbatch --dependency=afterok:${JOB_TCP_DUPLEX} jobscripts/run_cpu_incast_c23mm.sbatch | awk '{print $NF}')
echo "Submitted [5/9] MPI Incast Flooding:          Job ${JOB_CPU_INCAST} (after ${JOB_TCP_DUPLEX})"

# 6. TCP Incast Flooding (after MPI Incast)
JOB_TCP_INCAST=$(sbatch --dependency=afterok:${JOB_CPU_INCAST} jobscripts/run_tcp_incast_c23mm.sbatch | awk '{print $NF}')
echo "Submitted [6/9] TCP Incast Flooding:          Job ${JOB_TCP_INCAST} (after ${JOB_CPU_INCAST})"

# 7. HPC-ML Coupling Transport Schemes (after TCP Incast)
JOB_COUPLING=$(sbatch --dependency=afterok:${JOB_TCP_INCAST} jobscripts/run_coupling_schemes_c23mm.sbatch | awk '{print $NF}')
echo "Submitted [7/9] Coupling Schemes Suite:       Job ${JOB_COUPLING} (after ${JOB_TCP_INCAST})"

# 8. MPI Worker Count Scaling Sweep (after Coupling)
JOB_MPI_WORKER=$(sbatch --dependency=afterok:${JOB_COUPLING} jobscripts/run_worker_sweep_c23mm.sbatch | awk '{print $NF}')
echo "Submitted [8/9] MPI Worker Scaling Sweep:     Job ${JOB_MPI_WORKER} (after ${JOB_COUPLING})"

# 9. TCP Worker Count Scaling Sweep (after MPI Worker)
JOB_TCP_WORKER=$(sbatch --dependency=afterok:${JOB_MPI_WORKER} jobscripts/run_tcp_worker_sweep_c23mm.sbatch | awk '{print $NF}')
echo "Submitted [9/9] TCP Worker Scaling Sweep:     Job ${JOB_TCP_WORKER} (after ${JOB_MPI_WORKER})"

echo ""
echo "All 9 c23mm benchmark jobs submitted successfully into dependency chain!"
echo "First active job: ${JOB_CPU_SIMPLEX}"
echo "Final job in chain: ${JOB_TCP_WORKER}"
