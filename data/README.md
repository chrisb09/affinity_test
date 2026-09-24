# Hardware Topology, Communication, and HPC-ML Transport Benchmarks

This directory contains benchmark data, measurement logs, and visualizations evaluating CPU core-to-core, host-to-device (GPU PCIe), and one-to-many controller communication across the CLAIX-23 cluster partition (Intel Xeon Sapphire Rapids 96 cores/node, NVIDIA GPUs, and InfiniBand fabric).

---

## Directory Structure

```
data/affinity_test/
├── 01_p2p_simplex/                         # Pairwise isolated 1-on-1 ping-pong latency & bandwidth (18,336 pairs)
├── 02_p2p_duplex/                          # Pairwise isolated 1-on-1 full-duplex simultaneous exchange (18,336 pairs)
├── 03_incast_flooding_payload_sweep/       # 191 -> 1 Fan-in flooding congestion sweep across payload sizes (64 B - 4 MiB)
├── 04_coupling_schemes/                    # HPC-ML coupling transport simulations (Gather-Scatter, Pipelined, Credits)
├── 05_worker_count_scaling/                # Active worker count scaling sweep (W = 1 to 191) comparing MPI vs TCP
└── 06_uds_benchmarks/                      # Single-node Unix Domain Socket (UDS) benchmarks (SmartSim colocated mode)
```

---

## 1. `01_p2p_simplex/` (Unidirectional / Ping-Pong Core & GPU Placement)

Measures uncontended point-to-point latency (64 B) and bandwidth (4 MiB) across all 18,336 rank pairs (192 ranks across 2 nodes) and GPU transfers per NUMA domain:

- **Same NUMA (Intra-L3):** MPI latency 0.24 µs, BW 10.12 GiB/s | TCP latency 9.52 µs, BW 7.00 GiB/s
- **Same Socket, Diff NUMA:** MPI latency 0.29 µs, BW 9.15 GiB/s | TCP latency 9.92 µs, BW 6.65 GiB/s
- **Cross-Socket (UPI Interconnect):** MPI latency 1.56 µs, BW 7.10 GiB/s | TCP latency 21.57 µs, BW 2.28 GiB/s
- **Cross-Node (InfiniBand / IPoIB):** MPI latency 7.55 µs, BW 5.57 GiB/s | TCP latency 23.98 µs, BW 1.92 GiB/s
- **GPU PCIe Transfers (H2D & D2H):**
  - **Local NUMA Domain:** H2D reaches 41.9–43.2 GiB/s (~45.0–46.4 GB/s); D2H reaches 33.2–34.5 GiB/s (~35.6–37.0 GB/s) on directly attached PCIe root complexes (GPUs 0, 1, 2, 3 attached to NUMA 0, 2, 4, 6 respectively).
  - **Same Socket, non-local NUMA:** H2D achieves 23.1–33.2 GiB/s; D2H achieves 19.6–31.5 GiB/s.
  - **Cross-Socket (UPI Interconnect):** H2D drops to 18.4–19.7 GiB/s (55% reduction); D2H drops to 8.9–11.5 GiB/s (72% reduction).
  - **Latency (4 KiB):** 12–14 µs across all placements.

---

## 2. `02_p2p_duplex/` (Simultaneous Bidirectional P2P Transfers)

Measures simultaneous bidirectional 4 MiB payload exchange between rank pairs to evaluate full-duplex interconnect capability:

- **MPI Duplex Scaling:**
  - Same NUMA: 22.36 GiB/s (2.21× scaling)
  - Same Socket: 21.14 GiB/s (2.31× scaling)
  - Cross-Socket: 21.51 GiB/s (3.03× scaling over serialized ping-pong due to non-blocking DMA overlap)
  - Cross-Node: 9.58 GiB/s (1.72× scaling over InfiniBand fabric)
- **TCP Duplex Scaling:**
  - Shows only 1.13× to 1.36× scaling due to software TCP stack CPU processing and socket lock contention.

---

## 3. `03_incast_flooding_payload_sweep/` (Fan-in Congestion Sweep)

191 senders simultaneously blast payloads into a single controller rank (testing both Node 0 Rank 0 and Node 1 Rank 96) sweeping payload sizes from 64 B to 4 MiB:

- **MPI Incast:**
  - Peak Ingest Rate: 11.01 GiB/s at 256 KiB payload (fitting inside L3 cache).
  - 4 MiB Payload ($191 \times 4\text{ MiB} \approx 764\text{ MiB}$ total): Ingest throughput settles at 7.27 GiB/s as main memory write buffers saturate.
- **TCP Incast:**
  - Saturated at 3.55 GiB/s due to kernel socket buffer locking across 191 concurrent connections.

---

## 4. `04_coupling_schemes/` (Coupling Transport Workflow Simulations)

Simulating realistic orchestrator/controller data exchange patterns (4 MiB in + 4 MiB out per worker, 191 workers, 0 controller compute time):

- **Incast (191 in):** MPI Makespan = 92.4 ms (16.14 GiB/s) | TCP Makespan = 215.1 ms (6.94 GiB/s)
- **Fan-Out (191 out):** MPI Makespan = 16.4 ms (90.81 GiB/s) | TCP Makespan = 120.4 ms (12.39 GiB/s)
  - *Fan-out is >5.5× faster than Incast* because memory reads suffer no write-lock contention.
- **Full-Duplex (191 in + 191 out):** MPI Makespan = 66.8 ms (22.35 GiB/s) | TCP Makespan = 313.2 ms (4.76 GiB/s)
- **Gather-Scatter (Bulk Sync):** MPI Makespan = 125.7 ms (11.87 GiB/s) | TCP Makespan = 305.3 ms (4.89 GiB/s)
- **Immediate Response (Pipelined):** MPI Makespan = 129.2 ms (Worker Turnaround p50 drops from 123.7 ms to 85.8 ms).
- **Serialized Reference:** MPI Makespan = 264.5 ms (2.1× slower than parallel).
- **Credit-Controlled Sweeps:** Fixed caps (1 to 64), Locality credits (1 local + 1 remote), and AIx ramping concurrency.

---

## 5. `05_worker_count_scaling/` (Active Worker Count Scaling Sweep)

Evaluating active concurrent worker count scaling across $W \in \{1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 191\}$ with 50 timed iterations each:

- **The Incast Contention Knee:** Throughput scales linearly up to $W = 4$ (peaking at 21.85 GiB/s), then plateaus and declines to ~7.00 GiB/s at $W = 191$ due to DRAM write buffer contention.
- **Fan-Out Scaling:** Egress bandwidth scales continuously, reaching 42.92 GiB/s at $W = 191$ (9.28× faster than TCP's 4.63 GiB/s).
- **Locality Inversion:** At high worker counts ($W = 191$), remote workers complete significantly faster than local workers (59.1 ms vs 126.9 ms) because InfiniBand PCIe DMA bypasses local CPU L3 ring/mesh coherency traffic.

---

## 6. `06_uds_benchmarks/` (Unix Domain Sockets / SmartSim Colocated Mode)

Evaluates node-local IPC performance over stream-oriented Unix Domain Sockets (`AF_UNIX`) across 96 core-bound ranks on a single node:

- **Pairwise Simplex Latency:**
  - **Same NUMA:** UDS latency is **3.97 µs** (2.4× faster than TCP's 9.52 µs; MPI is 0.24 µs).
  - **Same Socket, Diff NUMA:** UDS latency is **4.11 µs** (2.4× faster than TCP's 9.92 µs; MPI is 0.29 µs).
  - **Cross-Socket (UPI Link):** UDS latency is **7.49 µs** (2.9× faster than TCP's 21.57 µs; MPI is 1.56 µs).
- **Pairwise Simplex Goodput (4 MiB):**
  - **Same NUMA:** 7.52 GiB/s (exceeds TCP's 7.00 GiB/s).
  - **Same Socket, Diff NUMA:** 6.81 GiB/s (exceeds TCP's 6.65 GiB/s).
  - **Cross-Socket (UPI Link):** 3.36 GiB/s (1.47× faster than TCP's 2.28 GiB/s).
- **Incast Flooding (95 -> 1):**
  - UDS reaches **4.58 GiB/s** aggregate ingest throughput (29% higher than TCP's 3.55 GiB/s limit).
  - Tail latencies are substantially lower than TCP (e.g. at 4 KiB: UDS p50 is 1.6 ms vs TCP 4.0 ms).
