# Isolated D2H Destination-Placement Test (jobs 4446885, 4446886)

## Question

Does the GPU 0/NUMA 2 D2H slowdown remain when the H2D source placement and the
issuing CPU are held fixed, and does a timed H2D copy immediately before D2H
contribute to it?

## Method

Jobs 4446885 (`w23g0007`) and 4446886 (`w23g0011`) ran in the `c23g` partition on
two separate nodes. Each process was pinned to CPU 0 for GPU 0 or CPU 48 for GPU 2.
The H2D source buffer remained on the GPU's root NUMA domain (NUMA 0 for GPU 0,
NUMA 4 for GPU 2). Only the D2H destination buffer's NUMA domain varied: 0--3 for
GPU 0 and 4--7 for GPU 2.

For each destination placement the test measured two modes:

- `d2h-only`: after one untimed H2D setup copy to populate the GPU source, warm up
  and time D2H copies only.
- `interleaved`: time the existing H2D/D2H sequence, with the directions run
  sequentially on the same CUDA stream and alternating which direction goes first.

Both modes used 5 warm-up and 50 timed iterations at 64 MiB and 256 MiB. The
binary recorded full source and destination page-residency censuses before timing
and a destination census afterward. Each job produced 96 rows: 8 destination
placements x 2 modes x 2 directions x 3 payload sizes. `NOT_RUN (d2h-only)` on
the H2D result rows is expected; all measured D2H rows were `OK`.

The topology captures are `topology_4446885.txt` and `topology_4446886.txt`.
They confirm four NVIDIA H100 GPUs, eight 12-core NUMA domains, and GPU roots at
NUMA 0/2/4/6 on both nodes.

## Results

256 MiB D2H goodput, GiB/s (each cell is `d2h-only / interleaved`):

| GPU | Destination NUMA | Job 4446885, w23g0007 | Job 4446886, w23g0011 |
|---:|---:|---:|---:|
| 0 | 0 | 40.52 / 40.47 | 40.58 / 40.58 |
| 0 | 1 | 41.18 / 40.88 | 40.80 / 41.05 |
| 0 | 2 | 29.18 / 29.18 | 29.17 / 29.17 |
| 0 | 3 | 34.47 / 34.46 | 34.53 / 34.53 |
| 2 | 4 | 40.65 / 40.62 | 40.55 / 40.56 |
| 2 | 5 | 40.89 / 40.73 | 40.82 / 40.96 |
| 2 | 6 | 27.23 / 27.23 | 29.18 / 29.17 |
| 2 | 7 | 34.51 / 34.49 | 34.64 / 34.63 |

The 64 MiB results show the same placement ordering. Every source and destination
buffer had all 16,384 (64 MiB) or 65,536 (256 MiB) pages on its requested NUMA
node before the timed copies; the destination census was unchanged afterward.

## Interpretation and Limits

1. The slow GPU 0-to-NUMA 2 and GPU 2-to-NUMA 6 D2H results persist in
   `d2h-only` mode, with the source pages and issuing CPU fixed. This isolates the
   observed rate difference to the D2H destination-placement configuration in
   this test; the earlier test's jointly moved H2D source buffer is not needed to
   reproduce it.
2. D2H rates are nearly unchanged between `d2h-only` and `interleaved`. A timed
   H2D copy immediately before/after D2H therefore does not explain the observed
   difference at these sizes and placements.
3. This experiment does not identify the hardware mechanism. It does not prove a
   particular IMC, PCIe transaction path, cache/coherence behavior, DDIO effect,
   NIC/GPU contention, or physical mesh hop count is responsible. Those require
   counters or further controlled experiments.
4. These are single-process, per-placement measurements on two nodes. Absolute
   rates should not be substituted directly for the earlier full-node 96-rank
   campaign; the controlled result establishes the destination-placement effect
   and its repeatability across these two nodes.
