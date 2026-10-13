#!/bin/bash
# Collect CLAIX-2023 InfiniBand fabric topology and HCA inventory.
#
# Captures the Slurm tree-topology switch layout, the per-node HCA hardware
# (controller, firmware, link rate) and the node-to-leaf-switch placement so
# that the switch-hop distance between any two nodes can be derived.
#
# Usage: collect_ib_topology.sh <output_dir>

set -euo pipefail

OUTDIR="${1:-data/10_infiniband_topology}"
mkdir -p "${OUTDIR}"

echo "=== Collecting InfiniBand topology into ${OUTDIR} ==="
date

# --- Cluster-wide Slurm tree topology ---------------------------------------
scontrol show topology > "${OUTDIR}/claix23_slurm_topology.txt"
echo "wrote ${OUTDIR}/claix23_slurm_topology.txt"

# --- Node placement for every node that appears in the topology -------------
scontrol show topology \
  | grep -oE '(Nodes|Switches)=[^ ]+' \
  | sed 's/^Nodes=//; s/^Switches=//' \
  > "${OUTDIR}/claix23_topo_members.txt"
echo "wrote ${OUTDIR}/claix23_topo_members.txt"

# --- Per-node HCA inventory -------------------------------------------------
# Prefer srun so that all allocated nodes are covered; fall back to local host.
if command -v srun >/dev/null 2>&1 && [ -n "${SLURM_JOB_ID:-}" ]; then
  srun --ntasks-per-node=1 bash -c '
    echo "### host=$(hostname)"
    echo "### uname=$(uname -r)"
    echo "### lspci_infiniband"
    lspci | grep -i infiniband || true
    echo "### ibstat"
    ibstat || true
    echo "### sysfs"
    for ca in /sys/class/infiniband/*; do
      [ -e "$ca" ] || continue
      echo "CA $(basename "$ca") fw_ver=$(cat "$ca/fw_ver" 2>/dev/null) hw_rev=$(cat "$ca/hw_rev" 2>/dev/null)"
      for p in "$ca"/ports/*; do
        [ -e "$p" ] || continue
        echo "  port $(basename "$p") rate=$(cat "$p/rate" 2>/dev/null) state=$(cat "$p/state" 2>/dev/null)"
      done
    done
  ' 2>&1 | tee "${OUTDIR}/hca_inventory.txt"
else
  {
    echo "### host=$(hostname)"
    echo "### uname=$(uname -r)"
    lspci | grep -i infiniband || true
    ibstat || true
  } 2>&1 | tee "${OUTDIR}/hca_inventory.txt"
fi

echo "wrote ${OUTDIR}/hca_inventory.txt"
echo "=== Done ==="
