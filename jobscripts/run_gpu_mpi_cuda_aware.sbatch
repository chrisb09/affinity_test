#!/bin/bash
#SBATCH -J gpu_mpi_cuda
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH -p c23g
#SBATCH -A thes2181
#SBATCH -o logs/gpu_mpi_cuda_%j.out

# CUDA-Aware MPI vs direct cudaMemcpyAsync GPU Host-to-Device and Device-to-Host sweep.
# Tests whether MPI-managed GPU transfers exhibit the same or different NUMA placement
# asymmetry as cudaMemcpyAsync on the same physical node.

set -euo pipefail

WORKDIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/affinity_test"
cd "$WORKDIR"
source /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh

OUTDIR="data/09_gpu_mpi_cuda_aware"
mkdir -p logs "$OUTDIR"
MASTER_MPI="$OUTDIR/gpu_mpi_cuda_aware_${SLURM_JOB_ID}.csv"
MASTER_REF="$OUTDIR/gpu_direct_cuda_ref_${SLURM_JOB_ID}.csv"
TOPO="$OUTDIR/topology_${SLURM_JOB_ID}.txt"
RANKFILE="/tmp/rf_${SLURM_JOB_ID}.txt"

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
  echo "--- Open MPI CUDA query ---"
  ompi_info --parsable --all | grep mpi_built_with_cuda_support || true
} > "$TOPO" 2>&1

echo "================================================================="
echo "=== Starting CUDA-Aware MPI & Direct CUDA Transfer Benchmark ==="
echo "================================================================="
echo "Node: $(hostname), Job: ${SLURM_JOB_ID}, Date: $(date)"

MYHOST=$(hostname -s)

run_mpi() {
  local gpu="$1" cpu0="$2" cpu1="$3" src_node="$4" dst_node="$5" mode="$6" transport="$7" variant="$8"
  cat << EOF > "$RANKFILE"
rank 0=${MYHOST} slot=${cpu0}
rank 1=${MYHOST} slot=${cpu1}
EOF

  local extra_flags=""
  if [ "$transport" = "ucx" ]; then
    export UCX_RNDV_THRESH=0
    extra_flags="-mca pml ucx"
  elif [ "$transport" = "smcuda" ]; then
    extra_flags="-mca pml ob1 -mca btl smcuda,self"
  fi

  echo "=== [MPI ${transport}] gpu=${gpu} cpu0=${cpu0} cpu1=${cpu1} src=${src_node} dst=${dst_node} mode=${mode} ==="
  mpirun -mca shmem_mmap_enable_nfs_warning 0 -mca orte_tmpdir_base /tmp \
    --rankfile "$RANKFILE" \
    $extra_flags \
    -np 2 ./build/mpi_cuda_aware_transfer \
      -d "$gpu" \
      -s 4096 \
      -l 67108864,268435456 \
      -w 5 -i 50 \
      --src-page-numa "$src_node" \
      --dst-page-numa "$dst_node" \
      --mode "$mode" \
      --variant "$variant" \
      --append -o "$MASTER_MPI"
}

run_cuda_ref() {
  local gpu="$1" cpu="$2" src_node="$3" dst_node="$4" mode="$5" variant="$6"
  echo "=== [Direct CUDA Ref] gpu=${gpu} cpu=${cpu} src=${src_node} dst=${dst_node} mode=${mode} ==="
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
      --append -o "$MASTER_REF"
}

# Sweep GPU 0 (Socket 0, Root NUMA 0, GPU Rank on CPU 1)
# 1. Isolated destination test (Host CPU fixed to 0, src_pages=0, dst_pages in 0..3)
for dst in 0 1 2 3; do
  for mode in d2h-only interleaved; do
    run_mpi 0 0 1 0 "$dst" "$mode" "ucx" "gpu0-iso-ucx-dst${dst}"
    run_mpi 0 0 1 0 "$dst" "$mode" "smcuda" "gpu0-iso-smcuda-dst${dst}"
    run_cuda_ref 0 0 0 "$dst" "$mode" "gpu0-iso-cuda-dst${dst}"
  done
done

# 2. Bound rank test (Host CPU local to buffer domain: CPU 0, 12, 24, 36)
for dst in 0 1 2 3; do
  cpu0=$(( dst * 12 ))
  for mode in d2h-only interleaved; do
    run_mpi 0 "$cpu0" 1 "$dst" "$dst" "$mode" "ucx" "gpu0-bound-ucx-numa${dst}"
    run_mpi 0 "$cpu0" 1 "$dst" "$dst" "$mode" "smcuda" "gpu0-bound-smcuda-numa${dst}"
    run_cuda_ref 0 "$cpu0" "$dst" "$dst" "$mode" "gpu0-bound-cuda-numa${dst}"
  done
done

# Sweep GPU 2 (Socket 1, Root NUMA 4, GPU Rank on CPU 49)
# 1. Isolated destination test (Host CPU fixed to 48, src_pages=4, dst_pages in 4..7)
for dst in 4 5 6 7; do
  for mode in d2h-only interleaved; do
    run_mpi 2 48 49 4 "$dst" "$mode" "ucx" "gpu2-iso-ucx-dst${dst}"
    run_mpi 2 48 49 4 "$dst" "$mode" "smcuda" "gpu2-iso-smcuda-dst${dst}"
    run_cuda_ref 2 48 4 "$dst" "$mode" "gpu2-iso-cuda-dst${dst}"
  done
done

# 2. Bound rank test (Host CPU local to buffer domain: CPU 48, 60, 72, 84)
for dst in 4 5 6 7; do
  cpu0=$(( dst * 12 ))
  for mode in d2h-only interleaved; do
    run_mpi 2 "$cpu0" 49 "$dst" "$dst" "$mode" "ucx" "gpu2-bound-ucx-numa${dst}"
    run_mpi 2 "$cpu0" 49 "$dst" "$dst" "$mode" "smcuda" "gpu2-bound-smcuda-numa${dst}"
    run_cuda_ref 2 "$cpu0" "$dst" "$dst" "$mode" "gpu2-bound-cuda-numa${dst}"
  done
done

rm -f "$RANKFILE"
echo "All benchmarks completed successfully on $(hostname)!"
date
