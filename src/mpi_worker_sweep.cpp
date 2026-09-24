#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/*
 * affinity_test/mpi_worker_sweep.cpp
 * ==================================
 * Active Worker Scaling Sweep (1 to 191 Workers) for HPC-ML Controller Communication.
 *
 * Evaluates throughput scaling, makespan scaling, and congestion onset as the number
 * of concurrent active workers W increases from 1 to 191.
 *
 * Balanced strided selection:
 *   Active workers are selected round-robin across Node 0 and Node 1 (50% local / 50% remote)
 *   and distributed across NUMA domains.
 *
 * Patterns evaluated across W in {1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 191}:
 *   1. incast             (W -> 1)
 *   2. fanout             (1 -> W)
 *   3. full_duplex        (W <-> 1)
 *   4. gather_scatter     (Bulk sync: gather all W inputs, scatter all W outputs)
 *   5. immediate_response (Pipelined per-worker)
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
#include <sstream>
#include <string>
#include <vector>

static constexpr size_t DEFAULT_PAYLOAD = 4 * 1024 * 1024; // 4 MiB
static constexpr int    DEFAULT_WARMUP  = 5;
static constexpr int    DEFAULT_ITERS   = 50;
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

static std::vector<int> select_balanced_workers(int target, int size, int W, const std::vector<RankInfo>& all_ranks) {
    std::vector<int> local_pool, remote_pool;
    const std::string& tgt_host = all_ranks[target].hostname;

    for (int r = 0; r < size; ++r) {
        if (r == target) continue;
        if (all_ranks[r].hostname == tgt_host) local_pool.push_back(r);
        else remote_pool.push_back(r);
    }

    std::vector<int> selected;
    selected.reserve(W);

    int n_local = 0, n_remote = 0;
    if (remote_pool.empty()) {
        n_local = std::min(W, static_cast<int>(local_pool.size()));
    } else {
        n_local = std::min(W / 2, static_cast<int>(local_pool.size()));
        n_remote = std::min(W - n_local, static_cast<int>(remote_pool.size()));
        // If one pool capped out, take remainder from other
        if (n_local + n_remote < W) {
            n_local = std::min(W - n_remote, static_cast<int>(local_pool.size()));
        }
    }

    // Strided selection across NUMA domains
    auto take_strided = [](const std::vector<int>& pool, int count) {
        std::vector<int> res;
        if (count <= 0 || pool.empty()) return res;
        double step = static_cast<double>(pool.size()) / count;
        for (int i = 0; i < count; ++i) {
            int idx = static_cast<int>(std::floor(i * step));
            res.push_back(pool[idx]);
        }
        return res;
    };

    auto sel_local  = take_strided(local_pool, n_local);
    auto sel_remote = take_strided(remote_pool, n_remote);

    selected.insert(selected.end(), sel_local.begin(), sel_local.end());
    selected.insert(selected.end(), sel_remote.begin(), sel_remote.end());
    std::sort(selected.begin(), selected.end());
    return selected;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string summary_file = "mpi_worker_sweep_summary.csv";
    std::string detail_file  = "mpi_worker_sweep_details.csv";
    int target               = 0;
    size_t payload           = DEFAULT_PAYLOAD;
    int warmup               = DEFAULT_WARMUP;
    int iters                = DEFAULT_ITERS;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      ((arg == "-o" || arg == "--output")  && i + 1 < argc) summary_file = argv[++i];
        else if ((arg == "-d" || arg == "--detail")  && i + 1 < argc) detail_file  = argv[++i];
        else if ((arg == "-t" || arg == "--target")  && i + 1 < argc) target       = std::stoi(argv[++i]);
        else if ((arg == "-p" || arg == "--payload") && i + 1 < argc) payload      = std::stoull(argv[++i]);
        else if ((arg == "-w" || arg == "--warmup")  && i + 1 < argc) warmup       = std::stoi(argv[++i]);
        else if ((arg == "-i" || arg == "--iters")   && i + 1 < argc) iters        = std::stoi(argv[++i]);
        else if (arg == "-h" || arg == "--help") {
            if (rank == 0) {
                std::cout << "Usage: " << argv[0] << " [options]\n"
                          << "  -o, --output <file>    Summary CSV (default: mpi_worker_sweep_summary.csv)\n"
                          << "  -d, --detail <file>    Detail CSV (default: mpi_worker_sweep_details.csv)\n"
                          << "  -t, --target <rank>    Controller target rank (default: 0)\n"
                          << "  -p, --payload <bytes>  Payload size (default: 4194304 = 4 MiB)\n"
                          << "  -w, --warmup <count>   Warmup iters (default: 5)\n"
                          << "  -i, --iters <count>    Timed iters (default: 50)\n";
            }
            MPI_Finalize();
            return 0;
        }
    }

    // Gather topology
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

    std::vector<int> sweep_counts;
    for (int c : {1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 191}) {
        if (c <= size - 1) sweep_counts.push_back(c);
    }
    if (std::find(sweep_counts.begin(), sweep_counts.end(), size - 1) == sweep_counts.end()) {
        sweep_counts.push_back(size - 1);
    }

    if (rank == 0) {
        std::cout << "=== MPI Active Worker Scaling Sweep Benchmark ===\n"
                  << "Ranks: " << size << " (Target Controller: Rank " << target << " on " << all_ranks[target].hostname << ")\n"
                  << "Payload: " << (payload / (1024 * 1024)) << " MiB in + " << (payload / (1024 * 1024)) << " MiB out\n"
                  << "Worker Counts Sweep: ";
        for (int c : sweep_counts) std::cout << c << " ";
        std::cout << "\nWarmup: " << warmup << ", Timed Iters: " << iters << "\n\n" << std::flush;
    }

    std::ofstream sum_csv, det_csv;
    if (rank == 0) {
        sum_csv.open(summary_file);
        sum_csv << "pattern,target_rank,target_host,target_cpu,active_workers,local_workers,remote_workers,payload_bytes,iters,"
                << "makespan_min_ms,makespan_p25_ms,makespan_median_ms,makespan_mean_ms,makespan_p75_ms,makespan_p95_ms,makespan_max_ms,"
                << "ingress_bw_median_gibs,egress_bw_median_gibs,combined_bw_median_gibs,"
                << "worker_lat_median_us,worker_lat_p95_us,worker_lat_max_us,"
                << "local_lat_median_us,local_lat_p95_us,remote_lat_median_us,remote_lat_p95_us\n";

        det_csv.open(detail_file);
        det_csv << "pattern,target_rank,active_workers,worker_rank,worker_host,is_remote,payload_bytes,iters,"
                << "lat_min_us,lat_p25_us,lat_median_us,lat_mean_us,lat_p75_us,lat_p95_us,lat_max_us\n";
    }

    std::vector<char> send_buf(payload, 's');
    std::vector<char> recv_pool(static_cast<size_t>(size - 1) * payload, 0);

    for (int W : sweep_counts) {
        std::vector<int> active_workers = select_balanced_workers(target, size, W, all_ranks);
        bool am_active_worker = (std::find(active_workers.begin(), active_workers.end(), rank) != active_workers.end());

        int n_local = 0, n_remote = 0;
        const std::string& tgt_host = all_ranks[target].hostname;
        for (int w : active_workers) {
            if (all_ranks[w].hostname == tgt_host) n_local++;
            else n_remote++;
        }

        if (rank == 0) {
            std::cout << "\n=======================================================\n"
                      << "Active Workers: " << std::setw(3) << W << " (" << n_local << " Local + " << n_remote << " Remote)\n"
                      << "=======================================================\n" << std::flush;
        }

        auto run_sweep_pattern = [&](const std::string& name, auto runner_func) {
            if (rank == 0) {
                std::cout << "  [" << std::setw(20) << std::left << name << "] ... " << std::flush;
            }

            std::vector<double> makespan_ms(iters, 0.0);
            std::vector<double> local_worker_lat_us(iters, 0.0);

            runner_func(makespan_ms, local_worker_lat_us, active_workers, am_active_worker);

            std::vector<double> all_worker_lats;
            if (rank == 0) all_worker_lats.resize(size * iters);

            MPI_Gather(local_worker_lat_us.data(), iters, MPI_DOUBLE,
                       all_worker_lats.data(), iters, MPI_DOUBLE,
                       0, MPI_COMM_WORLD);

            if (rank == 0) {
                std::vector<double> pooled_lats, local_lats, remote_lats;
                pooled_lats.reserve(W * iters);

                for (int w : active_workers) {
                    std::vector<double> per_w_lats(iters);
                    bool is_rem = (all_ranks[w].hostname != tgt_host);

                    for (int it = 0; it < iters; ++it) {
                        double l = all_worker_lats[w * iters + it];
                        per_w_lats[it] = l;
                        pooled_lats.push_back(l);
                        if (is_rem) remote_lats.push_back(l);
                        else local_lats.push_back(l);
                    }

                    det_csv << name << "," << target << "," << W << ","
                            << w << ",\"" << all_ranks[w].hostname << "\","
                            << (is_rem ? 1 : 0) << "," << payload << "," << iters << ","
                            << *std::min_element(per_w_lats.begin(), per_w_lats.end()) << ","
                            << percentile_of(per_w_lats, 0.25) << ","
                            << percentile_of(per_w_lats, 0.50) << ","
                            << mean_of(per_w_lats) << ","
                            << percentile_of(per_w_lats, 0.75) << ","
                            << percentile_of(per_w_lats, 0.95) << ","
                            << *std::max_element(per_w_lats.begin(), per_w_lats.end()) << "\n";
                }

                double med_makespan_sec = percentile_of(makespan_ms, 0.50) / 1000.0;
                double total_input_gb  = static_cast<double>(W) * static_cast<double>(payload) / GiB;
                double total_output_gb = total_input_gb;

                double ingress_bw = 0.0, egress_bw = 0.0, combined_bw = 0.0;
                if (name == "incast") {
                    ingress_bw = total_input_gb / med_makespan_sec;
                    combined_bw = ingress_bw;
                } else if (name == "fanout") {
                    egress_bw = total_output_gb / med_makespan_sec;
                    combined_bw = egress_bw;
                } else {
                    ingress_bw = total_input_gb / med_makespan_sec;
                    egress_bw  = total_output_gb / med_makespan_sec;
                    combined_bw = (total_input_gb + total_output_gb) / med_makespan_sec;
                }

                sum_csv << name << "," << target << ",\"" << all_ranks[target].hostname << "\","
                        << all_ranks[target].current_cpu << "," << W << "," << n_local << "," << n_remote << ","
                        << payload << "," << iters << ","
                        << *std::min_element(makespan_ms.begin(), makespan_ms.end()) << ","
                        << percentile_of(makespan_ms, 0.25) << ","
                        << percentile_of(makespan_ms, 0.50) << ","
                        << mean_of(makespan_ms) << ","
                        << percentile_of(makespan_ms, 0.75) << ","
                        << percentile_of(makespan_ms, 0.95) << ","
                        << *std::max_element(makespan_ms.begin(), makespan_ms.end()) << ","
                        << ingress_bw << "," << egress_bw << "," << combined_bw << ","
                        << percentile_of(pooled_lats, 0.50) << ","
                        << percentile_of(pooled_lats, 0.95) << ","
                        << *std::max_element(pooled_lats.begin(), pooled_lats.end()) << ","
                        << (local_lats.empty() ? 0.0 : percentile_of(local_lats, 0.50)) << ","
                        << (local_lats.empty() ? 0.0 : percentile_of(local_lats, 0.95)) << ","
                        << (remote_lats.empty() ? 0.0 : percentile_of(remote_lats, 0.50)) << ","
                        << (remote_lats.empty() ? 0.0 : percentile_of(remote_lats, 0.95)) << "\n";

                sum_csv.flush();
                det_csv.flush();

                std::cout << "Makespan (p50): " << std::setw(6) << std::fixed << std::setprecision(1) << percentile_of(makespan_ms, 0.50) << " ms"
                          << " | BW: " << std::setw(6) << std::setprecision(2) << combined_bw << " GiB/s"
                          << " | Worker Lat (p50): " << std::setw(6) << std::setprecision(1) << (percentile_of(pooled_lats, 0.50) / 1000.0) << " ms"
                          << " (Local: " << std::setw(5) << (local_lats.empty() ? 0.0 : percentile_of(local_lats, 0.50) / 1000.0)
                          << " ms, Remote: " << std::setw(5) << (remote_lats.empty() ? 0.0 : percentile_of(remote_lats, 0.50) / 1000.0) << " ms)\n"
                          << std::flush;
            }

            MPI_Barrier(MPI_COMM_WORLD);
        };

        // Pattern 1: Incast (W -> 1)
        run_sweep_pattern("incast", [&](auto& ms, auto& wl, const auto& act_w, bool am_act) {
            int nw = static_cast<int>(act_w.size());
            std::vector<MPI_Request> rreqs(nw);
            uint32_t ack_val = 1;

            for (int it = 0; it < warmup + iters; ++it) {
                bool timed = (it >= warmup);
                if (rank == target) {
                    for (int i = 0; i < nw; ++i) {
                        char* ptr = recv_pool.data() + static_cast<size_t>(i) * payload;
                        MPI_Irecv(ptr, static_cast<int>(payload), MPI_BYTE, act_w[i], 100 + it, MPI_COMM_WORLD, &rreqs[i]);
                    }
                }

                MPI_Barrier(MPI_COMM_WORLD);
                auto t0 = std::chrono::high_resolution_clock::now();

                if (am_act) {
                    MPI_Request sreq;
                    MPI_Isend(send_buf.data(), static_cast<int>(payload), MPI_BYTE, target, 100 + it, MPI_COMM_WORLD, &sreq);
                    MPI_Wait(&sreq, MPI_STATUS_IGNORE);
                    uint32_t ack = 0;
                    MPI_Recv(&ack, 1, MPI_UINT32_T, target, 200 + it, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    if (timed) wl[it - warmup] = std::chrono::duration<double, std::micro>(t1 - t0).count();
                } else if (rank == target) {
                    for (int i = 0; i < nw; ++i) {
                        int comp_idx = -1; MPI_Status st;
                        MPI_Waitany(nw, rreqs.data(), &comp_idx, &st);
                        MPI_Send(&ack_val, 1, MPI_UINT32_T, st.MPI_SOURCE, 200 + it, MPI_COMM_WORLD);
                    }
                    auto t1 = std::chrono::high_resolution_clock::now();
                    if (timed) ms[it - warmup] = std::chrono::duration<double, std::milli>(t1 - t0).count();
                }
            }
        });

        // Pattern 2: Fan-out (1 -> W)
        run_sweep_pattern("fanout", [&](auto& ms, auto& wl, const auto& act_w, bool am_act) {
            int nw = static_cast<int>(act_w.size());
            std::vector<MPI_Request> sreqs(nw), ack_reqs(nw);
            std::vector<uint32_t> ack_buf(nw, 0);
            uint32_t ack_val = 1;

            for (int it = 0; it < warmup + iters; ++it) {
                bool timed = (it >= warmup);
                MPI_Request w_rreq = MPI_REQUEST_NULL;
                if (am_act) {
                    MPI_Irecv(recv_pool.data(), static_cast<int>(payload), MPI_BYTE, target, 300 + it, MPI_COMM_WORLD, &w_rreq);
                } else if (rank == target) {
                    for (int i = 0; i < nw; ++i) {
                        MPI_Irecv(&ack_buf[i], 1, MPI_UINT32_T, act_w[i], 400 + it, MPI_COMM_WORLD, &ack_reqs[i]);
                    }
                }

                MPI_Barrier(MPI_COMM_WORLD);
                auto t0 = std::chrono::high_resolution_clock::now();

                if (rank == target) {
                    for (int i = 0; i < nw; ++i) {
                        MPI_Isend(send_buf.data(), static_cast<int>(payload), MPI_BYTE, act_w[i], 300 + it, MPI_COMM_WORLD, &sreqs[i]);
                    }
                    MPI_Waitall(nw, sreqs.data(), MPI_STATUSES_IGNORE);
                    MPI_Waitall(nw, ack_reqs.data(), MPI_STATUSES_IGNORE);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    if (timed) ms[it - warmup] = std::chrono::duration<double, std::milli>(t1 - t0).count();
                } else if (am_act) {
                    MPI_Wait(&w_rreq, MPI_STATUS_IGNORE);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    MPI_Send(&ack_val, 1, MPI_UINT32_T, target, 400 + it, MPI_COMM_WORLD);
                    if (timed) wl[it - warmup] = std::chrono::duration<double, std::micro>(t1 - t0).count();
                }
            }
        });

        // Pattern 3: Full Duplex (W <-> 1)
        run_sweep_pattern("full_duplex", [&](auto& ms, auto& wl, const auto& act_w, bool am_act) {
            int nw = static_cast<int>(act_w.size());
            std::vector<MPI_Request> rreqs(nw), sreqs(nw), ack_reqs(nw);
            std::vector<uint32_t> ack_buf(nw, 0);
            uint32_t ack_val = 1;

            for (int it = 0; it < warmup + iters; ++it) {
                bool timed = (it >= warmup);
                MPI_Request w_rreq = MPI_REQUEST_NULL, w_sreq = MPI_REQUEST_NULL;

                if (rank == target) {
                    for (int i = 0; i < nw; ++i) {
                        char* ptr = recv_pool.data() + static_cast<size_t>(i) * payload;
                        MPI_Irecv(ptr, static_cast<int>(payload), MPI_BYTE, act_w[i], 500 + it, MPI_COMM_WORLD, &rreqs[i]);
                        MPI_Irecv(&ack_buf[i], 1, MPI_UINT32_T, act_w[i], 650 + it, MPI_COMM_WORLD, &ack_reqs[i]);
                    }
                } else if (am_act) {
                    MPI_Irecv(recv_pool.data(), static_cast<int>(payload), MPI_BYTE, target, 600 + it, MPI_COMM_WORLD, &w_rreq);
                }

                MPI_Barrier(MPI_COMM_WORLD);
                auto t0 = std::chrono::high_resolution_clock::now();

                if (rank == target) {
                    for (int i = 0; i < nw; ++i) {
                        MPI_Isend(send_buf.data(), static_cast<int>(payload), MPI_BYTE, act_w[i], 600 + it, MPI_COMM_WORLD, &sreqs[i]);
                    }
                    MPI_Waitall(nw, rreqs.data(), MPI_STATUSES_IGNORE);
                    MPI_Waitall(nw, sreqs.data(), MPI_STATUSES_IGNORE);
                    MPI_Waitall(nw, ack_reqs.data(), MPI_STATUSES_IGNORE);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    if (timed) ms[it - warmup] = std::chrono::duration<double, std::milli>(t1 - t0).count();
                } else if (am_act) {
                    MPI_Isend(send_buf.data(), static_cast<int>(payload), MPI_BYTE, target, 500 + it, MPI_COMM_WORLD, &w_sreq);
                    MPI_Wait(&w_rreq, MPI_STATUS_IGNORE);
                    MPI_Wait(&w_sreq, MPI_STATUS_IGNORE);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    MPI_Send(&ack_val, 1, MPI_UINT32_T, target, 650 + it, MPI_COMM_WORLD);
                    if (timed) wl[it - warmup] = std::chrono::duration<double, std::micro>(t1 - t0).count();
                }
            }
        });

        // Pattern 4: Immediate Response (Pipelined)
        run_sweep_pattern("immediate_response", [&](auto& ms, auto& wl, const auto& act_w, bool am_act) {
            int nw = static_cast<int>(act_w.size());
            std::vector<MPI_Request> rreqs(nw), sreqs(nw, MPI_REQUEST_NULL), ack_reqs(nw);
            std::vector<uint32_t> ack_buf(nw, 0);
            uint32_t ack_val = 1;

            for (int it = 0; it < warmup + iters; ++it) {
                bool timed = (it >= warmup);
                MPI_Request w_rreq = MPI_REQUEST_NULL;

                if (rank == target) {
                    for (int i = 0; i < nw; ++i) {
                        char* ptr = recv_pool.data() + static_cast<size_t>(i) * payload;
                        MPI_Irecv(ptr, static_cast<int>(payload), MPI_BYTE, act_w[i], 900 + it, MPI_COMM_WORLD, &rreqs[i]);
                        MPI_Irecv(&ack_buf[i], 1, MPI_UINT32_T, act_w[i], 1050 + it, MPI_COMM_WORLD, &ack_reqs[i]);
                    }
                } else if (am_act) {
                    MPI_Irecv(recv_pool.data(), static_cast<int>(payload), MPI_BYTE, target, 1000 + it, MPI_COMM_WORLD, &w_rreq);
                }

                MPI_Barrier(MPI_COMM_WORLD);
                auto t0 = std::chrono::high_resolution_clock::now();

                if (rank == target) {
                    for (int count = 0; count < nw; ++count) {
                        int comp_idx = -1; MPI_Status st;
                        MPI_Waitany(nw, rreqs.data(), &comp_idx, &st);
                        int wrank = st.MPI_SOURCE;
                        MPI_Isend(send_buf.data(), static_cast<int>(payload), MPI_BYTE, wrank, 1000 + it, MPI_COMM_WORLD, &sreqs[comp_idx]);
                    }
                    MPI_Waitall(nw, sreqs.data(), MPI_STATUSES_IGNORE);
                    MPI_Waitall(nw, ack_reqs.data(), MPI_STATUSES_IGNORE);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    if (timed) ms[it - warmup] = std::chrono::duration<double, std::milli>(t1 - t0).count();
                } else if (am_act) {
                    MPI_Send(send_buf.data(), static_cast<int>(payload), MPI_BYTE, target, 900 + it, MPI_COMM_WORLD);
                    MPI_Wait(&w_rreq, MPI_STATUS_IGNORE);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    MPI_Send(&ack_val, 1, MPI_UINT32_T, target, 1050 + it, MPI_COMM_WORLD);
                    if (timed) wl[it - warmup] = std::chrono::duration<double, std::micro>(t1 - t0).count();
                }
            }
        });
    }

    if (rank == 0) {
        sum_csv.close();
        det_csv.close();
        std::cout << "\nWorker count sweep complete. Summary: " << summary_file << ", Details: " << detail_file << "\n";
    }

    MPI_Finalize();
    return 0;
}
