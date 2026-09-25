#!/bin/bash
#SBATCH -J gpu_d2h_iso
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:25:00
#SBATCH -p c23g
#SBATCH -A thes2181
#SBATCH -o logs/gpu_d2h_iso_%j.out

# Isolate D2H destination placement from the H2D source and issuing CPU.
# GPU 0 runs only on CPU 0 with h_src pages fixed to NUMA 0; GPU 2 runs only
# on CPU 48 with h_src pages fixed to NUMA 4. For each, vary only h_dst over
# the four same-socket NUMA domains and compare D2H-only with interleaved H2D/D2H.

set -euo pipefail

WORKDIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/affinity_test"
cd "$WORKDIR"
source /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh

OUTDIR="data/08_gpu_d2h_isolation"
mkdir -p logs "$OUTDIR"
MASTER="$OUTDIR/gpu_d2h_isolation_${SLURM_JOB_ID}.csv"
TOPO="$OUTDIR/topology_${SLURM_JOB_ID}.txt"

{
  echo "Node: $(hostname)  Job: ${SLURM_JOB_ID}  Date: $(date)"
  echo "--- GPUs ---"
  nvidia-smi --query-gpu=index,uuid,pci.bus_id,compute_mode,power.limit,name --format=csv
  echo "--- GPU topology ---"
  nvidia-smi topo -m
  echo "--- NUMA distances ---"
  numactl --hardware
  echo "--- lstopo with I/O ---"
  lstopo --of console --whole-io 2>/dev/null || true
} > "$TOPO" 2>&1

run_one() {
  local gpu="$1" cpu="$2" src_node="$3" dst_node="$4" mode="$5" variant="$6"
  echo "=== gpu=$gpu cpu=$cpu src_pages=$src_node dst_pages=$dst_node mode=$mode variant=$variant ==="
  mpirun --bind-to none -np 1 \
    numactl --physcpubind="$cpu" \
    ./build/mpi_gpu_transfer \
      -d "$gpu" \
      -s 4096 \
      -l 67108864,268435456 \
      -w 5 -i 50 \
      --src-page-numa "$src_node" \
      --dst-page-numa "$dst_node" \
      --mode "$mode" \
      --variant "$variant" \
      --append -o "$MASTER"
}

# Reverse NUMA and mode order on the second independently allocated node.
if (( SLURM_JOB_ID % 2 == 0 )); then
  DOMAINS=(0 1 2 3)
  MODES=(d2h-only interleaved)
  GPU2_DOMAINS=(4 5 6 7)
else
  DOMAINS=(3 2 1 0)
  MODES=(interleaved d2h-only)
  GPU2_DOMAINS=(7 6 5 4)
fi

for dst in "${DOMAINS[@]}"; do
  for mode in "${MODES[@]}"; do
    run_one 0 0 0 "$dst" "$mode" "gpu0-cpu0-src0-dst${dst}-${mode}"
  done
done

for dst in "${GPU2_DOMAINS[@]}"; do
  for mode in "${MODES[@]}"; do
    run_one 2 48 4 "$dst" "$mode" "gpu2-cpu48-src4-dst${dst}-${mode}"
  done
done

echo "Completed D2H isolation job ${SLURM_JOB_ID} on $(hostname)"
echo "CSV: $MASTER"
echo "Topology: $TOPO"
wc -l "$MASTER"
date
