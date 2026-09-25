# D2H Same-Socket Asymmetry Diagnostic (jobs 4437262, 4437263)

Purpose: explain the D2H asymmetry observed in `01_p2p_simplex/gpu_c23g_transfer_latency.csv`
(GPU 0 at 256 MiB: D2H ~30.4 GiB/s from NUMA 1 vs ~19.7 GiB/s from NUMA 2; mirrored on
GPU 2 with NUMA 5 vs 6), i.e. decide whether the rate follows the issuing CPU, the
host-page placement, or neither.

Method: single-rank runs of `mpi_gpu_transfer` (see
`sourcecode/affinity_test/src/mpi_gpu_transfer.cu`) with the diagnostic options:
`--page-numa N` forces both host buffers onto NUMA node N (`numa_alloc_onnode`),
`--reg-mode` switches the pinning strategy, `--variant`/`--append` build one master
CSV. The binary records full per-page NUMA censuses of both buffers before and after
the timed loop, checks CUDA calls, and verifies full-buffer checksums. Payloads:
4 KiB latency + 64 MiB/256 MiB, 5 warm-up + 50 timed iterations, one bound rank per
case (`numactl --cpunodebind`), GPU 0 tested from NUMA 0-3 and GPU 2 from NUMA 4-7.

Runs: jobs 4437262 (n23g0009) and 4437263 (n23g0004) on 2026-09-24, both COMPLETED,
all 156 rows per job `OK`. A reversed pass repeats the auto cases in reverse order to
rule out time-history effects. `topology_*.txt` capture nvidia-smi topo, numactl
distances, the lspci PCI tree, lstopo (GPU root complexes), and GPU clocks.

## Result (256 MiB medians, GiB/s)

| GPU 0 | H2D | D2H |        | GPU 2 | H2D | D2H |
|---|---:|---:|---|---|---:|---:|
| CPU 0, pages 0 (local)     | 51.6 | 39.8 |  | CPU 4, pages 4 (local) | 51.6 | 39.8 |
| CPU 1, pages 1             | 44.4 | 40.1 |  | CPU 5, pages 5         | 42.7 | 40.0 |
| CPU 2, pages 2             | 38.9 | 28.7 |  | CPU 6, pages 6         | 40.0 | 26.9 |
| CPU 3, pages 3 (diagonal)  | 32.3 | 34.3 |  | CPU 7, pages 7 (diag.) | 31.6 | 35.3 |
| **CPU 1, pages 2** (cross) | 38.8 | **29.2** |  | CPU 5, pages 6 (cross) | 39.9 | **27.1** |
| **CPU 2, pages 1** (cross) | 44.3 | **40.7** |  | CPU 6, pages 5 (cross) | 42.7 | **40.6** |
| CPU 1, Portable-only reg.  | 44.4 | 40.1 |  | CPU 5, cudaHostAlloc   | 42.7 | 39.4 |
| CPU 2, Portable-only reg.  | 38.8 | 28.7 |  | CPU 6, cudaHostAlloc   | 40.0 | 27.0 |
| CPU 1, cudaHostAlloc       | 44.3 | 40.3 |  | CPU 7 diag., reversed pass | 31.6 | 35.3 |
| CPU 2, cudaHostAlloc       | 38.8 | 28.9 |  |                        |      |      |

64 MiB results show the same pattern. Page censuses confirm 65536/65536 pages on the
intended node before and after the transfers on every row; no CUDA errors; all
checksums verified. Both nodes agree within ~0.1 GiB/s.

## Limitation and Follow-Up

This first diagnostic does **not** isolate D2H destination placement. The `--page-numa`
option placed both `h_src` (the H2D source buffer) and `h_dst` (the D2H destination
buffer) on the same NUMA node. Each timed iteration also performs both H2D and D2H,
alternating which direction runs first. Therefore the cross-placement result
establishes that the combined copy setup responds to host-buffer placement, but it
does not by itself prove that changing only the D2H destination causes the D2H rate
change. In particular, it does not rule out interactions with the preceding H2D copy.

The full per-page census confirms that both buffers landed on the requested NUMA
node in this test; checksums and cross-node repetitions verify the run's data and
repeatability. Those controls do not remove the source/destination confound. The
registration variants show that the tested registration choices do not materially
change the combined result; they do not identify the bottleneck.

Jobs 4437262/4437263 showed a reproducible correlation with placement, but the
hardware cause remains undetermined. Do not attribute it to a particular IMC, PCIe
root, IIO/mesh route, DDIO, NIC, or GPU without further evidence.

The follow-up isolation job (`sourcecode/affinity_test/jobscripts/run_gpu_d2h_isolation.sbatch`)
fixes each process to one CPU core and fixes H2D source pages at the GPU's local
NUMA domain. It varies only the D2H destination pages and compares D2H-only timing
against the existing interleaved H2D/D2H sequence. D2H-only still performs one
untimed H2D initialization copy to populate the GPU source buffer. Results will be
archived under `data/affinity_test/08_gpu_d2h_isolation/` after the jobs complete.
