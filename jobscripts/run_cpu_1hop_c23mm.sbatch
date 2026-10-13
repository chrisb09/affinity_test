#!/bin/bash
#SBATCH -J cpu_1hop
#SBATCH -N 2
#SBATCH --ntasks-per-node=96
#SBATCH --exclusive
#SBATCH --nodelist=n23m[0364-0365]
#SBATCH --mem=238G
#SBATCH --time=02:30:00
#SBATCH -p c23mm
#SBATCH -A rwth2150
#SBATCH -o logs/cpu_1hop_%j.out

# The --nodelist pins the allocation to a single leaf switch (ibsw00 covers
# n23m[0001-0072]; this 20-node subset is the earliest-available c23mm block)
# so that the cross-node traffic is guaranteed to take the 1-switch-hop
# intra-leaf path. Slurm's --switches=1 is only advisory on CLAIX-2023 (the
# scheduler does not wait for a single-switch allocation), so an explicit
# single-switch nodelist is used instead.

set -euo pipefail

WORKDIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/affinity_test"
cd "$WORKDIR"
mkdir -p logs data/10_infiniband_topology

source /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh

MPI_OUT="data/10_infiniband_topology/cpu_1hop_mpi_pair_latency.csv"
TCP_OUT="data/10_infiniband_topology/cpu_1hop_tcp_pair_latency.csv"

echo "================================================================="
echo "=== CPU 1-hop (intra-leaf) MPI/TCP pair benchmark ==="
echo "================================================================="
echo "Job ID: $SLURM_JOB_ID, Nodes: $SLURM_JOB_NUM_NODES, Ranks: $SLURM_NTASKS"
echo "Account: rwth2150, Partition: c23ms, single-switch nodelist"
echo "NODELIST: $SLURM_JOB_NODELIST"
echo "Using mpirun from: $(which mpirun)"
date
echo ""

# Locate the (Level-0) leaf switch that contains a given node.
leaf_switch_of() {
  local node="$1"
  while IFS= read -r line; do
    case "$line" in
      SwitchName=*) ;;
      *) continue ;;
    esac
    case "$line" in
      *Level=0*) ;;
      *) continue ;;
    esac
    local sw expr hosts
    sw=$(printf '%s' "$line" | sed -E 's/^SwitchName=([^ ]+).*/\1/')
    expr=$(printf '%s' "$line" | grep -oE 'Nodes=[^ ]+' | sed 's/Nodes=//')
    hosts=$(scontrol show hostnames "$expr" 2>/dev/null || true)
    if printf '%s\n' "$hosts" | grep -qx "$node"; then
      printf '%s\n' "$sw"
      return 0
    fi
  done < <(scontrol show topology)
  return 1
}

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
SW0=$(leaf_switch_of "${NODES[0]}" || true)
SW1=$(leaf_switch_of "${NODES[1]}" || true)
echo "Node topology: ${NODES[0]} -> ${SW0:-?}, ${NODES[1]} -> ${SW1:-?}"

if [ -z "$SW0" ] || [ -z "$SW1" ] || [ "$SW0" != "$SW1" ]; then
  echo "ERROR: allocation does not sit on a single leaf switch (${SW0:-?} vs ${SW1:-?});"
  echo "       skipping benchmark to avoid mislabeling cross-switch data as 1-hop."
  exit 0
fi
echo "Confirmed single leaf switch: ${SW0} (1 switch hop)."
echo ""

# Record the allocation and per-node HCA inventory.
scontrol show topology > "data/10_infiniband_topology/slurm_topology_${SLURM_JOB_ID}.txt"
for n in "${NODES[@]}"; do
  echo "### ${n} (leaf ${SW0})"
  scontrol show node "$n"
  echo ""
done > "data/10_infiniband_topology/allocated_nodes_${SLURM_JOB_ID}.txt"
cat "data/10_infiniband_topology/allocated_nodes_${SLURM_JOB_ID}.txt"
"$WORKDIR/src/collect_ib_topology.sh" "data/10_infiniband_topology/hca_${SLURM_JOB_ID}"

echo ""
echo "=== MPI pair latency/goodput (1-hop) ==="
mpirun -mca shmem_mmap_enable_nfs_warning 0 -mca orte_tmpdir_base /tmp \
  --map-by core --bind-to core \
  -np 192 ./build/mpi_pair_latency \
  -o "${MPI_OUT}" -s 64 -l 4194304 -w 10 -i 50

echo ""
echo "=== TCP/IPoIB pair latency/goodput (1-hop) ==="
mpirun -mca shmem_mmap_enable_nfs_warning 0 -mca orte_tmpdir_base /tmp \
  --map-by core --bind-to core \
  -np 192 ./build/tcp_pair_latency \
  -o "${TCP_OUT}" -s 64 -l 4194304 -w 10 -i 50

echo ""
echo "=== Benchmark completed ==="
date
