#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/*
 * affinity_test/mpi_coupling_schemes.cpp
 * ======================================
 * Transport Scheme Benchmark & Simulation Suite for HPC-ML Coupling (MPI).
 *
 * Evaluates both fundamental transport capabilities and real coupling transport schemes
 * across 192 ranks (1 controller + 191 workers) on 2 nodes.
 *
 * Timing & Synchronization Rules:
 *   - Only payload receives and ACK receives are preposted before the synchronization barrier.
 *   - No payload send is ever initiated before the start gate barrier.
 *   - The controller starts its makespan timer immediately after the barrier / before issuing its sends.
 *   - Workers start their latency timers immediately upon passing the start gate barrier.
 *   - For every output pattern, the worker sends a 4-byte ACK once its output buffer is fully received.
 *   - The controller makespan timer stops only after all payload operations AND all worker ACKs have completed.
 *
 * Patterns evaluated:
 *   1. 1_incast_191to1     : 191 -> 1 concurrent ingress with application-level ACKs.
 *   2. 2_fanout_1to191     : 1 -> 191 concurrent egress with worker ACKs.
 *   3. 3_full_duplex       : 191 <-> 1 concurrent bidirectional 4 MiB exchange.
 *   4. 4_gather_scatter    : Bulk synchronous coupling (gather all 191 inputs, then scatter all outputs).
 *   5. 5_immediate_response: Pipelined coupling (controller returns output for worker r as soon as input r arrives).
 *   6. 6_serialized_ref    : Zero-contention reference (1 worker transfer at a time: input then output).
 *   7. 7_credits_cap{1..64}: Credit-controlled concurrency sweep (C in {1, 2, 4, 8, 16, 32, 64}).
 *   8. 8_credits_locality  : 1 intra-node active worker + 1 remote-node active worker.
 *   9. 9_credits_aix       : Ramping concurrency (each completion releases 2 credits, up to cap).
 *
 * Payload: 4 MiB input + 4 MiB output per worker (configurable).
 * Controller computation time is strictly 0 to isolate pure communication scheduling.
 */

#include <mpi.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

static constexpr size_t DEFAULT_PAYLOAD = 4 * 1024 * 1024; // 4 MiB
static constexpr int    DEFAULT_WARMUP  = 3;
static constexpr int    DEFAULT_ITERS   = 20;
static constexpr double GiB             = 1024.0 * 1024.0 * 1024.0;

struct RankInfo {
    int rank{-1};
    char hostname[256]{0};
    int current_cpu{-1};
    std::string affinity_str;
};

static std::string get_affinity_string() {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    if (sched_getaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) return "unknown";
    std::ostringstream oss;
    bool first = true;
    int count = 0;
    for (int i = 0; i < CPU_SETSIZE; ++i) {
        if (CPU_ISSET(i, &cpuset)) {
            if (!first) oss << ",";
            oss << i;
            first = false;
            count++;
        }
    }
    return (count == CPU_SETSIZE) ? "unbound(all)" : oss.str();
}

static std::string get_locality_tier(const std::string& h_tgt, int cpu_tgt, const std::string& h_w, int cpu_w) {
    if (h_tgt != h_w) return "cross_node";
    if ((cpu_tgt / 12) == (cpu_w / 12)) return "same_numa";
    if ((cpu_tgt / 48) == (cpu_w / 48)) return "same_socket_diff_numa";
    return "cross_socket";
}

static double percentile_of(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p * (v.size() - 1);
    size_t lower = static_cast<size_t>(std::floor(idx));
    size_t upper = static_cast<size_t>(std::ceil(idx));
    if (lower == upper) return v[lower];
    return v[lower] + (idx - lower) * (v[upper] - v[lower]);
}

static double mean_of(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

static std::vector<int> parse_int_list(const std::string& str) {
    std::vector<int> res;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) res.push_back(std::stoi(token));
    }
    return res;
}

// ---------------------------------------------------------------------------
// Pattern 1: Incast (191 -> 1) with Application-level ACK
// ---------------------------------------------------------------------------
static void run_pattern_incast(
    int rank, int size, int target, size_t payload, int iters, int warmup,
    std::vector<char>& send_buf, std::vector<char>& recv_pool,
    std::vector<double>& out_makespan_ms, std::vector<double>& out_local_worker_lat_us)
{
    int num_workers = size - 1;
    std::vector<MPI_Request> recv_reqs(num_workers);
    uint32_t ack_val = 1;

    for (int it = 0; it < warmup + iters; ++it) {
        bool timed = (it >= warmup);

        if (rank == target) {
            int idx = 0;
            for (int src = 0; src < size; ++src) {
                if (src == target) continue;
                char* ptr = recv_pool.data() + static_cast<size_t>(idx) * payload;
                MPI_Irecv(ptr, static_cast<int>(payload), MPI_BYTE, src, 1000 + it, MPI_COMM_WORLD, &recv_reqs[idx]);
                idx++;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::high_resolution_clock::now();

        if (rank != target) {
            MPI_Request sreq;
            MPI_Isend(send_buf.data(), static_cast<int>(payload), MPI_BYTE, target, 1000 + it, MPI_COMM_WORLD, &sreq);
            MPI_Wait(&sreq, MPI_STATUS_IGNORE);

            uint32_t ack = 0;
            MPI_Recv(&ack, 1, MPI_UINT32_T, target, 2000 + it, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            auto t1 = std::chrono::high_resolution_clock::now();
            if (timed) {
                out_local_worker_lat_us[it - warmup] = std::chrono::duration<double, std::micro>(t1 - t0).count();
            }
        } else {
            for (int w = 0; w < num_workers; ++w) {
                int completed_idx = -1;
                MPI_Status status;
                MPI_Waitany(num_workers, recv_reqs.data(), &completed_idx, &status);
                int src_rank = status.MPI_SOURCE;
                MPI_Send(&ack_val, 1, MPI_UINT32_T, src_rank, 2000 + it, MPI_COMM_WORLD);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            if (timed) {
                out_makespan_ms[it - warmup] = std::chrono::duration<double, std::milli>(t1 - t0).count();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pattern 2: Fan-out (1 -> 191) with Worker ACKs
// ---------------------------------------------------------------------------
static void run_pattern_fanout(
    int rank, int size, int target, size_t payload, int iters, int warmup,
    std::vector<char>& send_buf, std::vector<char>& recv_pool,
    std::vector<double>& out_makespan_ms, std::vector<double>& out_local_worker_lat_us)
{
    int num_workers = size - 1;
    std::vector<MPI_Request> send_reqs(num_workers);
    std::vector<MPI_Request> ack_recv_reqs(num_workers);
    std::vector<uint32_t> ack_buf(num_workers, 0);
    uint32_t ack_val = 1;

    for (int it = 0; it < warmup + iters; ++it) {
        bool timed = (it >= warmup);
        MPI_Request w_recv_req = MPI_REQUEST_NULL;

        if (rank != target) {
            MPI_Irecv(recv_pool.data(), static_cast<int>(payload), MPI_BYTE, target, 3000 + it, MPI_COMM_WORLD, &w_recv_req);
        } else {
            int idx = 0;
            for (int dst = 0; dst < size; ++dst) {
                if (dst == target) continue;
                MPI_Irecv(&ack_buf[idx], 1, MPI_UINT32_T, dst, 4000 + it, MPI_COMM_WORLD, &ack_recv_reqs[idx]);
                idx++;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::high_resolution_clock::now();

        if (rank == target) {
            int idx = 0;
            for (int dst = 0; dst < size; ++dst) {
                if (dst == target) continue;
                MPI_Isend(send_buf.data(), static_cast<int>(payload), MPI_BYTE, dst, 3000 + it, MPI_COMM_WORLD, &send_reqs[idx]);
                idx++;
            }
            MPI_Waitall(num_workers, send_reqs.data(), MPI_STATUSES_IGNORE);
            MPI_Waitall(num_workers, ack_recv_reqs.data(), MPI_STATUSES_IGNORE);
            auto t1 = std::chrono::high_resolution_clock::now();
            if (timed) {
                out_makespan_ms[it - warmup] = std::chrono::duration<double, std::milli>(t1 - t0).count();
            }
        } else {
            MPI_Wait(&w_recv_req, MPI_STATUS_IGNORE);
            auto t1 = std::chrono::high_resolution_clock::now();
            MPI_Send(&ack_val, 1, MPI_UINT32_T, target, 4000 + it, MPI_COMM_WORLD);
            if (timed) {
                out_local_worker_lat_us[it - warmup] = std::chrono::duration<double, std::micro>(t1 - t0).count();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pattern 3: Full Duplex (191 <-> 1 Simultaneous Bidirectional)
// ---------------------------------------------------------------------------
static void run_pattern_full_duplex(
    int rank, int size, int target, size_t payload, int iters, int warmup,
    std::vector<char>& send_buf, std::vector<char>& recv_pool,
    std::vector<double>& out_makespan_ms, std::vector<double>& out_local_worker_lat_us)
{
    int num_workers = size - 1;
    std::vector<MPI_Request> tgt_recv_reqs(num_workers);
    std::vector<MPI_Request> tgt_send_reqs(num_workers);
    std::vector<MPI_Request> tgt_ack_reqs(num_workers);
    std::vector<uint32_t> ack_buf(num_workers, 0);
    uint32_t ack_val = 1;

    for (int it = 0; it < warmup + iters; ++it) {
        bool timed = (it >= warmup);
        MPI_Request w_recv_req = MPI_REQUEST_NULL;
        MPI_Request w_send_req = MPI_REQUEST_NULL;

        if (rank == target) {
            int idx = 0;
            for (int w = 0; w < size; ++w) {
                if (w == target) continue;
                char* rptr = recv_pool.data() + static_cast<size_t>(idx) * payload;
                MPI_Irecv(rptr, static_cast<int>(payload), MPI_BYTE, w, 5000 + it, MPI_COMM_WORLD, &tgt_recv_reqs[idx]);
                MPI_Irecv(&ack_buf[idx], 1, MPI_UINT32_T, w, 6500 + it, MPI_COMM_WORLD, &tgt_ack_reqs[idx]);
                idx++;
            }
        } else {
            MPI_Irecv(recv_pool.data(), static_cast<int>(payload), MPI_BYTE, target, 6000 + it, MPI_COMM_WORLD, &w_recv_req);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::high_resolution_clock::now();

        if (rank == target) {
            int idx = 0;
            for (int w = 0; w < size; ++w) {
                if (w == target) continue;
                MPI_Isend(send_buf.data(), static_cast<int>(payload), MPI_BYTE, w, 6000 + it, MPI_COMM_WORLD, &tgt_send_reqs[idx]);
                idx++;
            }
            MPI_Waitall(num_workers, tgt_recv_reqs.data(), MPI_STATUSES_IGNORE);
            MPI_Waitall(num_workers, tgt_send_reqs.data(), MPI_STATUSES_IGNORE);
            MPI_Waitall(num_workers, tgt_ack_reqs.data(), MPI_STATUSES_IGNORE);
            auto t1 = std::chrono::high_resolution_clock::now();
            if (timed) {
                out_makespan_ms[it - warmup] = std::chrono::duration<double, std::milli>(t1 - t0).count();
            }
        } else {
            MPI_Isend(send_buf.data(), static_cast<int>(payload), MPI_BYTE, target, 5000 + it, MPI_COMM_WORLD, &w_send_req);
            MPI_Wait(&w_recv_req, MPI_STATUS_IGNORE);
            MPI_Wait(&w_send_req, MPI_STATUS_IGNORE);
            auto t1 = std::chrono::high_resolution_clock::now();
            MPI_Send(&ack_val, 1, MPI_UINT32_T, target, 6500 + it, MPI_COMM_WORLD);
            if (timed) {
                out_local_worker_lat_us[it - warmup] = std::chrono::duration<double, std::micro>(t1 - t0).count();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pattern 4: Gather Then Scatter (Bulk Synchronous Coupling Scheme)
// ---------------------------------------------------------------------------
static void run_pattern_gather_scatter(
    int rank, int size, int target, size_t payload, int iters, int warmup,
    std::vector<char>& send_buf, std::vector<char>& recv_pool,
    std::vector<double>& out_makespan_ms, std::vector<double>& out_local_worker_lat_us)
{
    int num_workers = size - 1;
    std::vector<MPI_Request> tgt_recv_reqs(num_workers);
    std::vector<MPI_Request> tgt_send_reqs(num_workers);
    std::vector<MPI_Request> tgt_ack_reqs(num_workers);
    std::vector<uint32_t> ack_buf(num_workers, 0);
    uint32_t ack_val = 1;

    for (int it = 0; it < warmup + iters; ++it) {
        bool timed = (it >= warmup);
        MPI_Request w_recv_req = MPI_REQUEST_NULL;

        if (rank == target) {
            int idx = 0;
            for (int w = 0; w < size; ++w) {
                if (w == target) continue;
                char* rptr = recv_pool.data() + static_cast<size_t>(idx) * payload;
                MPI_Irecv(rptr, static_cast<int>(payload), MPI_BYTE, w, 7000 + it, MPI_COMM_WORLD, &tgt_recv_reqs[idx]);
                MPI_Irecv(&ack_buf[idx], 1, MPI_UINT32_T, w, 8500 + it, MPI_COMM_WORLD, &tgt_ack_reqs[idx]);
                idx++;
            }
        } else {
            MPI_Irecv(recv_pool.data(), static_cast<int>(payload), MPI_BYTE, target, 8000 + it, MPI_COMM_WORLD, &w_recv_req);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::high_resolution_clock::now();

        if (rank != target) {
            MPI_Send(send_buf.data(), static_cast<int>(payload), MPI_BYTE, target, 7000 + it, MPI_COMM_WORLD);
            MPI_Wait(&w_recv_req, MPI_STATUS_IGNORE);
            auto t1 = std::chrono::high_resolution_clock::now();
            MPI_Send(&ack_val, 1, MPI_UINT32_T, target, 8500 + it, MPI_COMM_WORLD);
            if (timed) {
                out_local_worker_lat_us[it - warmup] = std::chrono::duration<double, std::micro>(t1 - t0).count();
            }
        } else {
            MPI_Waitall(num_workers, tgt_recv_reqs.data(), MPI_STATUSES_IGNORE);

            int idx = 0;
            for (int w = 0; w < size; ++w) {
                if (w == target) continue;
                MPI_Isend(send_buf.data(), static_cast<int>(payload), MPI_BYTE, w, 8000 + it, MPI_COMM_WORLD, &tgt_send_reqs[idx]);
                idx++;
            }
            MPI_Waitall(num_workers, tgt_send_reqs.data(), MPI_STATUSES_IGNORE);
            MPI_Waitall(num_workers, tgt_ack_reqs.data(), MPI_STATUSES_IGNORE);
            auto t1 = std::chrono::high_resolution_clock::now();
            if (timed) {
                out_makespan_ms[it - warmup] = std::chrono::duration<double, std::milli>(t1 - t0).count();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pattern 5: Immediate Response (Pipelined per-worker Coupling Scheme)
// ---------------------------------------------------------------------------
static void run_pattern_immediate_response(
    int rank, int size, int target, size_t payload, int iters, int warmup,
    std::vector<char>& send_buf, std::vector<char>& recv_pool,
    std::vector<double>& out_makespan_ms, std::vector<double>& out_local_worker_lat_us)
{
    int num_workers = size - 1;
    std::vector<MPI_Request> tgt_recv_reqs(num_workers);
    std::vector<MPI_Request> tgt_send_reqs(num_workers, MPI_REQUEST_NULL);
    std::vector<MPI_Request> tgt_ack_reqs(num_workers);
    std::vector<uint32_t> ack_buf(num_workers, 0);
    uint32_t ack_val = 1;

    for (int it = 0; it < warmup + iters; ++it) {
        bool timed = (it >= warmup);
        MPI_Request w_recv_req = MPI_REQUEST_NULL;

        if (rank == target) {
            int idx = 0;
            for (int w = 0; w < size; ++w) {
                if (w == target) continue;
                char* rptr = recv_pool.data() + static_cast<size_t>(idx) * payload;
                MPI_Irecv(rptr, static_cast<int>(payload), MPI_BYTE, w, 9000 + it, MPI_COMM_WORLD, &tgt_recv_reqs[idx]);
                MPI_Irecv(&ack_buf[idx], 1, MPI_UINT32_T, w, 10500 + it, MPI_COMM_WORLD, &tgt_ack_reqs[idx]);
                idx++;
            }
        } else {
            MPI_Irecv(recv_pool.data(), static_cast<int>(payload), MPI_BYTE, target, 10000 + it, MPI_COMM_WORLD, &w_recv_req);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::high_resolution_clock::now();

        if (rank != target) {
            MPI_Send(send_buf.data(), static_cast<int>(payload), MPI_BYTE, target, 9000 + it, MPI_COMM_WORLD);
            MPI_Wait(&w_recv_req, MPI_STATUS_IGNORE);
            auto t1 = std::chrono::high_resolution_clock::now();
            MPI_Send(&ack_val, 1, MPI_UINT32_T, target, 10500 + it, MPI_COMM_WORLD);
            if (timed) {
                out_local_worker_lat_us[it - warmup] = std::chrono::duration<double, std::micro>(t1 - t0).count();
            }
        } else {
            for (int count = 0; count < num_workers; ++count) {
                int completed_idx = -1;
                MPI_Status status;
                MPI_Waitany(num_workers, tgt_recv_reqs.data(), &completed_idx, &status);
                int worker_rank = status.MPI_SOURCE;
                MPI_Isend(send_buf.data(), static_cast<int>(payload), MPI_BYTE, worker_rank, 10000 + it, MPI_COMM_WORLD, &tgt_send_reqs[completed_idx]);
            }
            MPI_Waitall(num_workers, tgt_send_reqs.data(), MPI_STATUSES_IGNORE);
            MPI_Waitall(num_workers, tgt_ack_reqs.data(), MPI_STATUSES_IGNORE);
            auto t1 = std::chrono::high_resolution_clock::now();
            if (timed) {
                out_makespan_ms[it - warmup] = std::chrono::duration<double, std::milli>(t1 - t0).count();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pattern 6: Serialized (Zero Contention Reference)
// ---------------------------------------------------------------------------
static void run_pattern_serialized(
    int rank, int size, int target, size_t payload, int iters, int warmup,
    std::vector<char>& send_buf, std::vector<char>& recv_pool,
    std::vector<double>& out_makespan_ms, std::vector<double>& out_local_worker_lat_us)
{
    uint32_t ack_val = 1;

    for (int it = 0; it < warmup + iters; ++it) {
        bool timed = (it >= warmup);

        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::high_resolution_clock::now();

        // 1. Worker inputs & outputs sequentially one at a time
        for (int w = 0; w < size; ++w) {
            if (w == target) continue;
            if (rank == w) {
                auto tw0 = std::chrono::high_resolution_clock::now();
                MPI_Send(send_buf.data(), static_cast<int>(payload), MPI_BYTE, target, 11000 + it, MPI_COMM_WORLD);
                MPI_Recv(recv_pool.data(), static_cast<int>(payload), MPI_BYTE, target, 12000 + it, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                auto tw1 = std::chrono::high_resolution_clock::now();
                MPI_Send(&ack_val, 1, MPI_UINT32_T, target, 12500 + it, MPI_COMM_WORLD);
                if (timed) {
                    out_local_worker_lat_us[it - warmup] = std::chrono::duration<double, std::micro>(tw1 - tw0).count();
                }
            } else if (rank == target) {
                MPI_Recv(recv_pool.data(), static_cast<int>(payload), MPI_BYTE, w, 11000 + it, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(send_buf.data(), static_cast<int>(payload), MPI_BYTE, w, 12000 + it, MPI_COMM_WORLD);
                uint32_t ack = 0;
                MPI_Recv(&ack, 1, MPI_UINT32_T, w, 12500 + it, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        if (rank == target) {
            auto t1 = std::chrono::high_resolution_clock::now();
            if (timed) {
                out_makespan_ms[it - warmup] = std::chrono::duration<double, std::milli>(t1 - t0).count();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pattern 7: Credit-Controlled Coupling (Fixed, Locality, or AIx Ramping)
// ---------------------------------------------------------------------------
enum class CreditMode { FIXED, LOCALITY, AIX_RAMPING };

static void run_pattern_credits(
    int rank, int size, int target, size_t payload, int iters, int warmup,
    CreditMode mode, int fixed_cap,
    const std::vector<RankInfo>& all_ranks,
    std::vector<char>& send_buf, std::vector<char>& recv_pool,
    std::vector<double>& out_makespan_ms, std::vector<double>& out_local_worker_lat_us)
{
    int num_workers = size - 1;
    uint32_t token = 1;
    uint32_t ack_val = 1;

    for (int it = 0; it < warmup + iters; ++it) {
        bool timed = (it >= warmup);

        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::high_resolution_clock::now();

        if (rank != target) {
            MPI_Recv(&token, 1, MPI_UINT32_T, target, 13000 + it, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            auto t_start_work = std::chrono::high_resolution_clock::now();

            MPI_Request rreq;
            MPI_Irecv(recv_pool.data(), static_cast<int>(payload), MPI_BYTE, target, 15000 + it, MPI_COMM_WORLD, &rreq);
            MPI_Send(send_buf.data(), static_cast<int>(payload), MPI_BYTE, target, 14000 + it, MPI_COMM_WORLD);
            MPI_Wait(&rreq, MPI_STATUS_IGNORE);
            auto t1 = std::chrono::high_resolution_clock::now();

            MPI_Send(&ack_val, 1, MPI_UINT32_T, target, 15500 + it, MPI_COMM_WORLD);

            if (timed) {
                out_local_worker_lat_us[it - warmup] = std::chrono::duration<double, std::micro>(t1 - t_start_work).count();
            }
        } else {
            std::queue<int> pending_workers;
            std::queue<int> local_queue, remote_queue;
            const std::string& tgt_host = all_ranks[target].hostname;

            for (int w = 0; w < size; ++w) {
                if (w == target) continue;
                if (mode == CreditMode::LOCALITY) {
                    if (all_ranks[w].hostname == tgt_host) local_queue.push(w);
                    else remote_queue.push(w);
                } else {
                    pending_workers.push(w);
                }
            }

            int in_flight = 0;
            int max_concurrency = (mode == CreditMode::FIXED) ? fixed_cap : ((mode == CreditMode::AIX_RAMPING) ? 191 : 2);
            int current_credit_limit = (mode == CreditMode::AIX_RAMPING) ? 1 : max_concurrency;

            auto grant_token = [&](int w) {
                MPI_Send(&token, 1, MPI_UINT32_T, w, 13000 + it, MPI_COMM_WORLD);
                in_flight++;
            };

            if (mode == CreditMode::LOCALITY) {
                if (!local_queue.empty()) { int w = local_queue.front(); local_queue.pop(); grant_token(w); }
                if (!remote_queue.empty()) { int w = remote_queue.front(); remote_queue.pop(); grant_token(w); }
            } else {
                while (!pending_workers.empty() && in_flight < current_credit_limit) {
                    int w = pending_workers.front(); pending_workers.pop();
                    grant_token(w);
                }
            }

            int completed_count = 0;
            while (completed_count < num_workers) {
                MPI_Status status;
                MPI_Recv(recv_pool.data(), static_cast<int>(payload), MPI_BYTE, MPI_ANY_SOURCE, 14000 + it, MPI_COMM_WORLD, &status);
                int worker_done = status.MPI_SOURCE;

                MPI_Send(send_buf.data(), static_cast<int>(payload), MPI_BYTE, worker_done, 15000 + it, MPI_COMM_WORLD);
                uint32_t ack = 0;
                MPI_Recv(&ack, 1, MPI_UINT32_T, worker_done, 15500 + it, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                completed_count++;
                in_flight--;

                if (mode == CreditMode::LOCALITY) {
                    if (all_ranks[worker_done].hostname == tgt_host && !local_queue.empty()) {
                        int next_w = local_queue.front(); local_queue.pop();
                        grant_token(next_w);
                    } else if (all_ranks[worker_done].hostname != tgt_host && !remote_queue.empty()) {
                        int next_w = remote_queue.front(); remote_queue.pop();
                        grant_token(next_w);
                    }
                } else if (mode == CreditMode::AIX_RAMPING) {
                    current_credit_limit = std::min(max_concurrency, current_credit_limit + 2);
                    while (!pending_workers.empty() && in_flight < current_credit_limit) {
                        int next_w = pending_workers.front(); pending_workers.pop();
                        grant_token(next_w);
                    }
                } else { // FIXED
                    while (!pending_workers.empty() && in_flight < current_credit_limit) {
                        int next_w = pending_workers.front(); pending_workers.pop();
                        grant_token(next_w);
                    }
                }
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            if (timed) {
                out_makespan_ms[it - warmup] = std::chrono::duration<double, std::milli>(t1 - t0).count();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string summary_file = "mpi_coupling_summary.csv";
    std::string detail_file  = "mpi_coupling_details.csv";
    std::string target_str   = (size >= 96) ? "0,96" : "0";
    size_t payload           = DEFAULT_PAYLOAD;
    int warmup               = DEFAULT_WARMUP;
    int iters                = DEFAULT_ITERS;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      ((arg == "-o" || arg == "--output")   && i + 1 < argc) summary_file = argv[++i];
        else if ((arg == "-d" || arg == "--detail")   && i + 1 < argc) detail_file  = argv[++i];
        else if ((arg == "-t" || arg == "--targets")  && i + 1 < argc) target_str   = argv[++i];
        else if ((arg == "-p" || arg == "--payload")  && i + 1 < argc) payload      = std::stoull(argv[++i]);
        else if ((arg == "-w" || arg == "--warmup")   && i + 1 < argc) warmup       = std::stoi(argv[++i]);
        else if ((arg == "-i" || arg == "--iters")    && i + 1 < argc) iters        = std::stoi(argv[++i]);
        else if (arg == "-h" || arg == "--help") {
            if (rank == 0) {
                std::cout << "Usage: " << argv[0] << " [options]\n"
                          << "  -o, --output <file>    Summary CSV (default: mpi_coupling_summary.csv)\n"
                          << "  -d, --detail <file>    Detail CSV (default: mpi_coupling_details.csv)\n"
                          << "  -t, --targets <list>   Target ranks (default: 0,96)\n"
                          << "  -p, --payload <bytes>  Payload size (default: 4194304 = 4 MiB)\n"
                          << "  -w, --warmup <count>   Warmup iters (default: 3)\n"
                          << "  -i, --iters <count>    Timed iters (default: 20)\n";
            }
            MPI_Finalize();
            return 0;
        }
    }

    std::vector<int> targets = parse_int_list(target_str);

    RankInfo local_info;
    local_info.rank = rank;
    gethostname(local_info.hostname, sizeof(local_info.hostname) - 1);
    local_info.current_cpu = sched_getcpu();
    local_info.affinity_str = get_affinity_string();

    std::vector<char> host_buf(size * 256, 0);
    MPI_Gather(local_info.hostname, 256, MPI_CHAR, host_buf.data(), 256, MPI_CHAR, 0, MPI_COMM_WORLD);

    std::vector<int> cpu_buf(size, 0);
    MPI_Gather(&local_info.current_cpu, 1, MPI_INT, cpu_buf.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    int aff_len = static_cast<int>(local_info.affinity_str.size());
    std::vector<int> aff_len_buf(size, 0);
    MPI_Gather(&aff_len, 1, MPI_INT, aff_len_buf.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> aff_disps(size, 0);
    int total_aff = 0;
    std::vector<char> aff_str_buf;
    if (rank == 0) {
        for (int r = 0; r < size; ++r) { aff_disps[r] = total_aff; total_aff += aff_len_buf[r]; }
        aff_str_buf.resize(total_aff);
    }
    MPI_Gatherv(local_info.affinity_str.data(), aff_len, MPI_CHAR,
                aff_str_buf.data(), aff_len_buf.data(), aff_disps.data(), MPI_CHAR,
                0, MPI_COMM_WORLD);

    std::vector<RankInfo> all_ranks(size);
    for (int r = 0; r < size; ++r) {
        all_ranks[r].rank = r;
        if (rank == 0) {
            std::string h(&host_buf[r * 256]);
            snprintf(all_ranks[r].hostname, sizeof(all_ranks[r].hostname), "%s", h.c_str());
            all_ranks[r].current_cpu = cpu_buf[r];
            all_ranks[r].affinity_str = std::string(&aff_str_buf[aff_disps[r]], aff_len_buf[r]);
        }
    }
    MPI_Bcast(host_buf.data(), size * 256, MPI_CHAR, 0, MPI_COMM_WORLD);
    for (int r = 0; r < size; ++r) {
        std::string h(&host_buf[r * 256]);
        snprintf(all_ranks[r].hostname, sizeof(all_ranks[r].hostname), "%s", h.c_str());
    }

    if (rank == 0) {
        std::cout << "=== MPI Transport Schemes & Coupling Simulation Suite ===\n"
                  << "Ranks: " << size << " (1 Controller + " << (size - 1) << " Workers)\n"
                  << "Payload: " << (payload / (1024 * 1024)) << " MiB in + " << (payload / (1024 * 1024)) << " MiB out\n"
                  << "Targets: " << target_str << "\n"
                  << "Warmup: " << warmup << ", Timed Iters: " << iters << "\n\n" << std::flush;
    }

    std::ofstream sum_csv, det_csv;
    if (rank == 0) {
        sum_csv.open(summary_file);
        sum_csv << "pattern,target_rank,target_host,target_cpu,target_affinity,workers_count,payload_bytes,iters,"
                << "makespan_min_ms,makespan_p25_ms,makespan_median_ms,makespan_mean_ms,makespan_p75_ms,makespan_p95_ms,makespan_max_ms,"
                << "ingress_bw_median_gibs,egress_bw_median_gibs,combined_bw_median_gibs,"
                << "worker_lat_min_us,worker_lat_p25_us,worker_lat_median_us,worker_lat_mean_us,worker_lat_p75_us,worker_lat_p95_us,worker_lat_p99_us,worker_lat_max_us\n";

        det_csv.open(detail_file);
        det_csv << "pattern,target_rank,target_host,worker_rank,worker_host,worker_cpu,worker_affinity,locality_tier,payload_bytes,iters,"
                << "lat_min_us,lat_p25_us,lat_median_us,lat_mean_us,lat_p75_us,lat_p95_us,lat_max_us\n";
    }

    std::vector<char> send_buf(payload, 'a');
    std::vector<char> recv_pool;
    if (std::find(targets.begin(), targets.end(), rank) != targets.end()) {
        recv_pool.resize(static_cast<size_t>(size - 1) * payload, 0);
    } else {
        recv_pool.resize(payload, 0);
    }

    auto evaluate_pattern = [&](const std::string& name, int target, auto runner_func) {
        if (rank == 0) {
            std::cout << ">>> Running [" << std::setw(22) << std::left << name << "] Targeting Rank " << target
                      << " (" << all_ranks[target].hostname << ") ... " << std::flush;
        }

        std::vector<double> makespan_ms(iters, 0.0);
        std::vector<double> local_worker_lat_us(iters, 0.0);

        runner_func(makespan_ms, local_worker_lat_us);

        std::vector<double> all_worker_lats;
        if (rank == 0) all_worker_lats.resize(size * iters);

        MPI_Gather(local_worker_lat_us.data(), iters, MPI_DOUBLE,
                   all_worker_lats.data(), iters, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);

        if (target != 0) {
            if (rank == target) {
                MPI_Send(makespan_ms.data(), iters, MPI_DOUBLE, 0, 9999, MPI_COMM_WORLD);
            } else if (rank == 0) {
                MPI_Recv(makespan_ms.data(), iters, MPI_DOUBLE, target, 9999, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        if (rank == 0) {
            std::vector<double> pooled_lats;
            pooled_lats.reserve((size - 1) * iters);

            for (int w = 0; w < size; ++w) {
                if (w == target) continue;
                std::vector<double> per_w_lats(iters);
                for (int it = 0; it < iters; ++it) {
                    double l = all_worker_lats[w * iters + it];
                    per_w_lats[it] = l;
                    pooled_lats.push_back(l);
                }

                std::string tier = get_locality_tier(all_ranks[target].hostname, all_ranks[target].current_cpu,
                                                     all_ranks[w].hostname, all_ranks[w].current_cpu);

                det_csv << name << "," << target << ",\"" << all_ranks[target].hostname << "\","
                        << w << ",\"" << all_ranks[w].hostname << "\","
                        << all_ranks[w].current_cpu << ",\"" << all_ranks[w].affinity_str << "\",\""
                        << tier << "\"," << payload << "," << iters << ","
                        << *std::min_element(per_w_lats.begin(), per_w_lats.end()) << ","
                        << percentile_of(per_w_lats, 0.25) << ","
                        << percentile_of(per_w_lats, 0.50) << ","
                        << mean_of(per_w_lats) << ","
                        << percentile_of(per_w_lats, 0.75) << ","
                        << percentile_of(per_w_lats, 0.95) << ","
                        << *std::max_element(per_w_lats.begin(), per_w_lats.end()) << "\n";
            }

            double med_makespan_sec = percentile_of(makespan_ms, 0.50) / 1000.0;
            double total_input_gb  = static_cast<double>(size - 1) * static_cast<double>(payload) / GiB;
            double total_output_gb = total_input_gb;

            double ingress_bw = 0.0;
            double egress_bw  = 0.0;
            double combined_bw = 0.0;

            if (name == "1_incast_191to1") {
                ingress_bw  = total_input_gb / med_makespan_sec;
                egress_bw   = 0.0;
                combined_bw = ingress_bw;
            } else if (name == "2_fanout_1to191") {
                ingress_bw  = 0.0;
                egress_bw   = total_output_gb / med_makespan_sec;
                combined_bw = egress_bw;
            } else {
                ingress_bw  = total_input_gb / med_makespan_sec;
                egress_bw   = total_output_gb / med_makespan_sec;
                combined_bw = (total_input_gb + total_output_gb) / med_makespan_sec;
            }

            sum_csv << name << "," << target << ",\"" << all_ranks[target].hostname << "\","
                    << all_ranks[target].current_cpu << ",\"" << all_ranks[target].affinity_str << "\","
                    << (size - 1) << "," << payload << "," << iters << ","
                    << *std::min_element(makespan_ms.begin(), makespan_ms.end()) << ","
                    << percentile_of(makespan_ms, 0.25) << ","
                    << percentile_of(makespan_ms, 0.50) << ","
                    << mean_of(makespan_ms) << ","
                    << percentile_of(makespan_ms, 0.75) << ","
                    << percentile_of(makespan_ms, 0.95) << ","
                    << *std::max_element(makespan_ms.begin(), makespan_ms.end()) << ","
                    << ingress_bw << "," << egress_bw << "," << combined_bw << ","
                    << *std::min_element(pooled_lats.begin(), pooled_lats.end()) << ","
                    << percentile_of(pooled_lats, 0.25) << ","
                    << percentile_of(pooled_lats, 0.50) << ","
                    << mean_of(pooled_lats) << ","
                    << percentile_of(pooled_lats, 0.75) << ","
                    << percentile_of(pooled_lats, 0.95) << ","
                    << percentile_of(pooled_lats, 0.99) << ","
                    << *std::max_element(pooled_lats.begin(), pooled_lats.end()) << "\n";

            sum_csv.flush();
            det_csv.flush();

            std::cout << "Makespan (p50): " << std::setw(6) << std::fixed << std::setprecision(1) << percentile_of(makespan_ms, 0.50) << " ms"
                      << " | Worker Lat (p50): " << std::setw(6) << std::fixed << std::setprecision(1) << (percentile_of(pooled_lats, 0.50) / 1000.0) << " ms"
                      << " | (p95): " << std::setw(6) << (percentile_of(pooled_lats, 0.95) / 1000.0) << " ms"
                      << " | BW: " << std::setw(6) << std::setprecision(2) << combined_bw << " GiB/s\n" << std::flush;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    };

    for (int target : targets) {
        if (target < 0 || target >= size) continue;

        // 1. Fundamental Transport Limits
        evaluate_pattern("1_incast_191to1", target, [&](auto& ms, auto& wl) {
            run_pattern_incast(rank, size, target, payload, iters, warmup, send_buf, recv_pool, ms, wl);
        });

        evaluate_pattern("2_fanout_1to191", target, [&](auto& ms, auto& wl) {
            run_pattern_fanout(rank, size, target, payload, iters, warmup, send_buf, recv_pool, ms, wl);
        });

        evaluate_pattern("3_full_duplex", target, [&](auto& ms, auto& wl) {
            run_pattern_full_duplex(rank, size, target, payload, iters, warmup, send_buf, recv_pool, ms, wl);
        });

        // 2. Coupling Transport Schemes
        evaluate_pattern("4_gather_scatter", target, [&](auto& ms, auto& wl) {
            run_pattern_gather_scatter(rank, size, target, payload, iters, warmup, send_buf, recv_pool, ms, wl);
        });

        evaluate_pattern("5_immediate_response", target, [&](auto& ms, auto& wl) {
            run_pattern_immediate_response(rank, size, target, payload, iters, warmup, send_buf, recv_pool, ms, wl);
        });

        evaluate_pattern("6_serialized_ref", target, [&](auto& ms, auto& wl) {
            run_pattern_serialized(rank, size, target, payload, iters, warmup, send_buf, recv_pool, ms, wl);
        });

        // 3. Credit Concurrency Sweeps
        for (int cap : {1, 2, 4, 8, 16, 32, 64}) {
            std::string pname = "7_credits_cap" + std::to_string(cap);
            evaluate_pattern(pname, target, [&](auto& ms, auto& wl) {
                run_pattern_credits(rank, size, target, payload, iters, warmup, CreditMode::FIXED, cap, all_ranks, send_buf, recv_pool, ms, wl);
            });
        }

        evaluate_pattern("8_credits_locality_1local_1remote", target, [&](auto& ms, auto& wl) {
            run_pattern_credits(rank, size, target, payload, iters, warmup, CreditMode::LOCALITY, 2, all_ranks, send_buf, recv_pool, ms, wl);
        });

        evaluate_pattern("9_credits_aix_ramping", target, [&](auto& ms, auto& wl) {
            run_pattern_credits(rank, size, target, payload, iters, warmup, CreditMode::AIX_RAMPING, 191, all_ranks, send_buf, recv_pool, ms, wl);
        });
    }

    if (rank == 0) {
        sum_csv.close();
        det_csv.close();
        std::cout << "\nAll coupling transport schemes complete. Summary: " << summary_file
                  << ", Details: " << detail_file << "\n";
    }

    MPI_Finalize();
    return 0;
}
