#!/bin/bash
#SBATCH -J gpu_d2h_diag
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH -p c23g
#SBATCH -A thes2181
#SBATCH -o logs/gpu_d2h_diag_%j.out

# Diagnostic rerun for the D2H same-socket asymmetry (GPU 0: NUMA 1 ~30 GiB/s
# vs NUMA 2 ~20 GiB/s; mirrored on GPU 2 with NUMA 5 vs 6).
#
# Decouples the issuing CPU from the destination-page placement:
#   * auto runs reproduce the original first-touch placement,
#   * cross runs force the host pages onto the neighbouring NUMA domain,
#   * alloc runs use cudaHostAlloc instead of first-touch + cudaHostRegister,
#   * a reversed pass rules out time-history (thermal/clock) artifacts.
# The binary records full per-page NUMA censuses, checks every CUDA call,
# and verifies the full-buffer checksum of each transfer pair.

set -euo pipefail

WORKDIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/affinity_test"
cd "$WORKDIR"
source /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh

OUTDIR="data/07_gpu_d2h_diagnostic"
mkdir -p logs "$OUTDIR"
MASTER="$OUTDIR/gpu_d2h_diagnostic_${SLURM_JOB_ID}.csv"
TOPO="$OUTDIR/topology_${SLURM_JOB_ID}.txt"

echo "================================================================="
echo "=== D2H asymmetry diagnostic | job ${SLURM_JOB_ID} ==="
echo "================================================================="
echo "Node: $(hostname)"
date

# ---------------------------------------------------------------------
# Topology evidence (PCIe ancestry + NUMA distances, not just endpoints)
# ---------------------------------------------------------------------
{
  echo "Node: $(hostname)  Job: ${SLURM_JOB_ID}  Date: $(date)"
  echo ""
  echo "--- nvidia-smi device summary ---"
  nvidia-smi --query-gpu=index,uuid,pci.bus_id,compute_mode,power.limit,name --format=csv
  echo ""
  echo "--- GPU interconnect & NUMA affinity topology matrix ---"
  nvidia-smi topo -m
  echo ""
  echo "--- Host NUMA hardware configuration ---"
  numactl --hardware
  echo ""
  echo "--- PCI device tree (lspci -Dtvnn) ---"
  lspci -Dtvnn 2>/dev/null || echo "lspci not available"
  echo ""
  echo "--- GPU PCIe ancestry with per-device numa_node (sysfs) ---"
  for busid in $(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader); do
    # normalize 00000000:1B:00.0 -> 0000:1b:00.0 (sysfs format)
    b="$(printf '%s' "$busid" | tr 'A-Z' 'a-z')"
    b="0000:${b#00000000:}"
    bdf="$b"
    dev="/sys/bus/pci/devices/$bdf"
    depth=0
    echo "Ancestry of $bdf:"
    while [ -e "$dev" ] && [ "$depth" -lt 16 ]; do
      printf '  %s numa=%s class=%s\n' "$bdf" "$(cat "$dev/numa_node" 2>/dev/null)" "$(cat "$dev/class" 2>/dev/null)"
      phys="$(readlink -f "$dev")"
      pbdf="$(basename "$(dirname "$phys")")"
      case "$pbdf" in
        pci*:*) break ;;
        *) dev="/sys/bus/pci/devices/$pbdf"; bdf="$pbdf" ;;
      esac
      depth=$((depth + 1))
    done
  done
  echo ""
  echo "--- lstopo (if available) ---"
  if command -v lstopo >/dev/null 2>&1; then
    lstopo --of console --whole-io 2>/dev/null || lstopo 2>/dev/null || echo "lstopo failed"
  else
    echo "lstopo not available"
  fi
  echo ""
  echo "--- GPU clocks/performance at start ---"
  (nvidia-smi -q -d PERFORMANCE,CLOCK 2>/dev/null | head -80) || true
} > "$TOPO" 2>&1
echo "Topology evidence written to $TOPO"

# ---------------------------------------------------------------------
# Single-rank measurement helper
#   run_one <gpu> <cpu-numa-node> <page-numa|-1> <reg-mode> <variant>
# ---------------------------------------------------------------------
run_one() {
  local gpu="$1" node="$2" pages="$3" reg="$4" variant="$5"
  echo ""
  echo "=== run: gpu=$gpu cpu-node=$node pages=${pages} reg=$reg variant=$variant ==="
  mpirun -np 1 \
    numactl --cpunodebind="$node" \
    ./build/mpi_gpu_transfer \
      -d "$gpu" \
      -s 4096 \
      -l 67108864,268435456 \
      -w 5 -i 50 \
      --page-numa "$pages" \
      --reg-mode "$reg" \
      --variant "$variant" \
      --append \
      -o "$MASTER"
}

echo "================================================================="
echo "=== Forward pass ==="
echo "================================================================="
# GPU 0 (root NUMA 0): all four same-socket domains, original placement
run_one 0 0 -1 0 "fwd-n0-auto"
run_one 0 1 -1 0 "fwd-n1-auto"
run_one 0 2 -1 0 "fwd-n2-auto"
run_one 0 3 -1 0 "fwd-n3-auto"
# Decisive cross tests: issuing CPU and destination pages decoupled
run_one 0 1  2 0 "fwd-n1-p2"
run_one 0 2  1 0 "fwd-n2-p1"
# Registration-mode variants for the decisive pair
run_one 0 1 -1 1 "fwd-n1-regP"
run_one 0 2 -1 1 "fwd-n2-regP"
run_one 0 1 -1 3 "fwd-n1-alloc"
run_one 0 2 -1 3 "fwd-n2-alloc"
# GPU 2 (root NUMA 4): mirrored same-socket domains
run_one 2 4 -1 0 "fwd-n4-auto"
run_one 2 5 -1 0 "fwd-n5-auto"
run_one 2 6 -1 0 "fwd-n6-auto"
run_one 2 7 -1 0 "fwd-n7-auto"
run_one 2 5  6 0 "fwd-n5-p6"
run_one 2 6  5 0 "fwd-n6-p5"
run_one 2 5 -1 3 "fwd-n5-alloc"
run_one 2 6 -1 3 "fwd-n6-alloc"

echo "================================================================="
echo "=== Reverse pass (auto placement, reversed order) ==="
echo "================================================================="
run_one 2 7 -1 0 "rev-n7-auto"
run_one 2 6 -1 0 "rev-n6-auto"
run_one 2 5 -1 0 "rev-n5-auto"
run_one 2 4 -1 0 "rev-n4-auto"
run_one 0 3 -1 0 "rev-n3-auto"
run_one 0 2 -1 0 "rev-n2-auto"
run_one 0 1 -1 0 "rev-n1-auto"
run_one 0 0 -1 0 "rev-n0-auto"

{
  echo "--- GPU clocks/performance at end ---"
  (nvidia-smi -q -d PERFORMANCE,CLOCK 2>/dev/null | head -80) || true
} >> "$TOPO"

echo ""
echo "================================================================="
echo "=== Diagnostic complete ==="
echo "Master CSV: $MASTER"
echo "Topology:   $TOPO"
wc -l "$MASTER"
date
echo "=== All tasks completed successfully! ==="
