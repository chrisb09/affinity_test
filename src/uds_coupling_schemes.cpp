#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/*
 * affinity_test/src/uds_coupling_schemes.cpp
 * ==========================================
 * Transport Scheme Benchmark & Simulation Suite for HPC-ML Coupling (Unix Domain Sockets).
 *
 * Uses persistent UDS sockets across all benchmark rounds to avoid connection setup noise.
 * Evaluates transport patterns across 96 node-local cores (e.g. SmartSim colocated mode).
 */

#include <mpi.h>
#include <sched.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
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
#include <thread>
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

static bool send_all(int fd, const void* buf, size_t len) {
    const char* p = static_cast<const char*>(buf);
    while (len > 0) {
        ssize_t n = ::send(fd, p, len, MSG_NOSIGNAL);
        if (n <= 0) return false;
        p += n;
        len -= n;
    }
    return true;
}

static bool recv_all(int fd, void* buf, size_t len) {
    char* p = static_cast<char*>(buf);
    while (len > 0) {
        ssize_t n = ::recv(fd, p, len, MSG_WAITALL);
        if (n <= 0) return false;
        p += n;
        len -= n;
    }
    return true;
}

static void set_sock_buffers(int fd, int sz = 4 * 1024 * 1024) {
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sz, sizeof(sz));
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &sz, sizeof(sz));
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string summary_file = "uds_coupling_summary.csv";
    std::string detail_file  = "uds_coupling_details.csv";
    std::string target_str   = "0";
    std::string socket_dir   = "/tmp";
    size_t payload           = DEFAULT_PAYLOAD;
    int warmup               = DEFAULT_WARMUP;
    int iters                = DEFAULT_ITERS;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      ((arg == "-o" || arg == "--output")   && i + 1 < argc) summary_file = argv[++i];
        else if ((arg == "-d" || arg == "--detail")   && i + 1 < argc) detail_file  = argv[++i];
        else if ((arg == "-t" || arg == "--targets")  && i + 1 < argc) target_str   = argv[++i];
        else if (arg == "--socket-dir"                && i + 1 < argc) socket_dir   = argv[++i];
        else if ((arg == "-p" || arg == "--payload")  && i + 1 < argc) payload      = std::stoull(argv[++i]);
        else if ((arg == "-w" || arg == "--warmup")   && i + 1 < argc) warmup       = std::stoi(argv[++i]);
        else if ((arg == "-i" || arg == "--iters")    && i + 1 < argc) iters        = std::stoi(argv[++i]);
        else if (arg == "-h" || arg == "--help") {
            if (rank == 0) {
                std::cout << "Usage: " << argv[0] << " [options]\n"
                          << "  -o, --output <file>    Summary CSV (default: uds_coupling_summary.csv)\n"
                          << "  -d, --detail <file>    Detail CSV (default: uds_coupling_details.csv)\n"
                          << "  -t, --targets <list>   Target ranks (default: 0)\n"
                          << "  --socket-dir <dir>     UDS socket directory (default: /tmp)\n"
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
    MPI_Bcast(host_buf.data(), size * 256, MPI_CHAR, 0, MPI_COMM_WORLD);

    std::string host0(&host_buf[0]);
    for (int r = 0; r < size; ++r) {
        std::string hr(&host_buf[r * 256]);
        if (hr != host0) {
            if (rank == 0) {
                std::cerr << "FATAL: UDS coupling requires single-node allocation. Rank " << r
                          << " is on host '" << hr << "' != '" << host0 << "'\n";
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

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

    if (rank == 0) {
        std::cout << "=================================================================\n"
                  << "=== Unix Domain Socket (UDS) HPC-ML Coupling Schemes Suite ===\n"
                  << "=================================================================\n"
                  << "Node: " << host0 << ", Ranks: " << size << " (1 target, " << (size - 1) << " concurrent workers)\n"
                  << "Payload: " << (payload / (1024*1024)) << " MiB (in + out per worker)\n"
                  << "Warmup: " << warmup << ", Iters: " << iters << "\n"
                  << "Socket Dir: " << socket_dir << "\n"
                  << "Summary CSV: " << summary_file << "\n"
                  << "Detail CSV:  " << detail_file << "\n\n" << std::flush;

        std::ofstream sum_csv(summary_file, std::ios::trunc);
        sum_csv << "scheme_name,target_rank,target_host,target_cpu,target_affinity,workers_count,"
                << "payload_bytes,iters,warmup,iface,"
                << "makespan_min_ms,makespan_p25_ms,makespan_median_ms,makespan_mean_ms,makespan_p75_ms,makespan_max_ms,"
                << "ingress_bw_median_gibs,egress_bw_median_gibs,combined_bw_median_gibs,"
                << "worker_lat_p25_ms,worker_lat_median_ms,worker_lat_mean_ms,worker_lat_p75_ms,worker_lat_p95_ms,worker_lat_max_ms\n";
        sum_csv.close();

        std::ofstream det_csv(detail_file, std::ios::trunc);
        det_csv << "scheme_name,target_rank,target_host,target_cpu,worker_rank,worker_host,worker_cpu,worker_affinity,"
                << "ip,iface,locality_tier,payload_bytes,iters,"
                << "worker_lat_min_ms,worker_lat_p25_ms,worker_lat_median_ms,worker_lat_mean_ms,worker_lat_p75_ms,worker_lat_p95_ms,worker_lat_max_ms\n";
        det_csv.close();
    }

    std::vector<char> send_buf(payload, 'P');
    std::vector<char> recv_buf(payload, 0);
    std::vector<std::vector<char>> pool_in(size, std::vector<char>(payload, 0));
    std::vector<std::vector<char>> pool_out(size, std::vector<char>(payload, 'R'));

    for (int target : targets) {
        if (target < 0 || target >= size) continue;

        int srv_fd = -1;
        std::vector<int> client_fds(size, -1);
        std::ostringstream sp;
        sp << socket_dir << "/uds_cpl_t" << target << ".sock";
        std::string sock_path = sp.str();

        if (rank == target) {
            srv_fd = socket(AF_UNIX, SOCK_STREAM, 0);
            unlink(sock_path.c_str());

            struct sockaddr_un addr{};
            addr.sun_family = AF_UNIX;
            strncpy(addr.sun_path, sock_path.c_str(), sizeof(addr.sun_path) - 1);

            bind(srv_fd, (struct sockaddr*)&addr, sizeof(addr));
            listen(srv_fd, size + 16);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank != target) {
            struct sockaddr_un addr{};
            addr.sun_family = AF_UNIX;
            strncpy(addr.sun_path, sock_path.c_str(), sizeof(addr.sun_path) - 1);

            for (int attempt = 0; ; ++attempt) {
                client_fds[target] = socket(AF_UNIX, SOCK_STREAM, 0);
                set_sock_buffers(client_fds[target]);
                if (connect(client_fds[target], (struct sockaddr*)&addr, sizeof(addr)) == 0) {
                    send_all(client_fds[target], &rank, sizeof(rank));
                    break;
                }
                close(client_fds[target]);
                client_fds[target] = -1;
                usleep(2000);
            }
        } else {
            for (int count = 0; count < size - 1; ++count) {
                int cfd = accept(srv_fd, nullptr, nullptr);
                set_sock_buffers(cfd);
                int client_rank = -1;
                recv_all(cfd, &client_rank, sizeof(client_rank));
                client_fds[client_rank] = cfd;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        auto evaluate_uds_pattern = [&](
            const std::string& name,
            auto&& runner_fn)
        {
            std::vector<double> makespans(iters, 0.0);
            std::vector<double> worker_lats(iters, 0.0);

            for (int w = 0; w < warmup; ++w) {
                std::vector<double> dummy_ms(1, 0.0), dummy_wl(1, 0.0);
                runner_fn(dummy_ms, dummy_wl);
            }

            for (int it = 0; it < iters; ++it) {
                std::vector<double> single_ms(1, 0.0), single_wl(1, 0.0);
                runner_fn(single_ms, single_wl);
                if (rank == target) makespans[it] = single_ms[0];
                else worker_lats[it] = single_wl[0];
            }

            std::vector<double> all_worker_lats(size * iters, 0.0);
            MPI_Gather(worker_lats.data(), iters, MPI_DOUBLE, all_worker_lats.data(), iters, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            std::vector<double> all_target_makespans(iters, 0.0);
            if (rank == target) all_target_makespans = makespans;
            MPI_Bcast(all_target_makespans.data(), iters, MPI_DOUBLE, target, MPI_COMM_WORLD);

            if (rank == 0) {
                double total_in_bytes  = 0.0;
                double total_out_bytes = 0.0;
                if (name == "1_incast_191to1") {
                    total_in_bytes = (size - 1) * static_cast<double>(payload);
                } else if (name == "2_fanout_1to191") {
                    total_out_bytes = (size - 1) * static_cast<double>(payload);
                } else {
                    total_in_bytes  = (size - 1) * static_cast<double>(payload);
                    total_out_bytes = (size - 1) * static_cast<double>(payload);
                }

                std::vector<double> ing_bws(iters), eg_bws(iters), comb_bws(iters);
                for (int it = 0; it < iters; ++it) {
                    double sec = all_target_makespans[it] * 1e-3;
                    ing_bws[it]  = (total_in_bytes  / GiB) / sec;
                    eg_bws[it]   = (total_out_bytes / GiB) / sec;
                    comb_bws[it] = ((total_in_bytes + total_out_bytes) / GiB) / sec;
                }

                std::vector<double> pooled_lats;
                std::ofstream det_csv(detail_file, std::ios::app);

                for (int w = 0; w < size; ++w) {
                    if (w == target) continue;
                    std::vector<double> w_lats(iters);
                    for (int it = 0; it < iters; ++it) {
                        w_lats[it] = all_worker_lats[w * iters + it];
                        pooled_lats.push_back(w_lats[it]);
                    }
                    std::sort(w_lats.begin(), w_lats.end());

                    std::string loc = get_locality_tier(all_ranks[target].hostname, all_ranks[target].current_cpu,
                                                        all_ranks[w].hostname, all_ranks[w].current_cpu);

                    det_csv << name << "," << target << ",\"" << all_ranks[target].hostname << "\","
                            << all_ranks[target].current_cpu << ","
                            << w << ",\"" << all_ranks[w].hostname << "\","
                            << all_ranks[w].current_cpu << ",\"" << all_ranks[w].affinity_str << "\","
                            << "\"uds\",unix," << loc << "," << payload << "," << iters << ","
                            << w_lats.front() << ","
                            << percentile_of(w_lats, 0.25) << ","
                            << percentile_of(w_lats, 0.50) << ","
                            << mean_of(w_lats) << ","
                            << percentile_of(w_lats, 0.75) << ","
                            << percentile_of(w_lats, 0.95) << ","
                            << w_lats.back() << "\n";
                }
                det_csv.close();

                std::ofstream sum_csv(summary_file, std::ios::app);
                sum_csv << name << "," << target << ",\"" << all_ranks[target].hostname << "\","
                        << all_ranks[target].current_cpu << ",\"" << all_ranks[target].affinity_str << "\","
                        << (size - 1) << "," << payload << "," << iters << "," << warmup << ",unix,"
                        << percentile_of(all_target_makespans, 0.0)  << ","
                        << percentile_of(all_target_makespans, 0.25) << ","
                        << percentile_of(all_target_makespans, 0.50) << ","
                        << mean_of(all_target_makespans)             << ","
                        << percentile_of(all_target_makespans, 0.75) << ","
                        << percentile_of(all_target_makespans, 1.0)  << ","
                        << percentile_of(ing_bws, 0.50)              << ","
                        << percentile_of(eg_bws, 0.50)               << ","
                        << percentile_of(comb_bws, 0.50)             << ","
                        << percentile_of(pooled_lats, 0.25)          << ","
                        << percentile_of(pooled_lats, 0.50)          << ","
                        << mean_of(pooled_lats)                      << ","
                        << percentile_of(pooled_lats, 0.75)          << ","
                        << percentile_of(pooled_lats, 0.95)          << ","
                        << percentile_of(pooled_lats, 1.0)           << "\n";
                sum_csv.close();

                std::cout << "[Target " << target << " | " << std::setw(28) << std::left << name << "] "
                          << "Makespan (median): " << std::setw(6) << std::fixed << std::setprecision(1)
                          << percentile_of(all_target_makespans, 0.50) << " ms | "
                          << "Throughput: " << std::setw(6) << std::fixed << std::setprecision(2)
                          << (name == "1_incast_191to1" ? percentile_of(ing_bws, 0.50) :
                             (name == "2_fanout_1to191" ? percentile_of(eg_bws, 0.50) : percentile_of(comb_bws, 0.50)))
                          << " GiB/s | Worker p50: " << std::setw(6) << std::fixed << std::setprecision(1)
                          << percentile_of(pooled_lats, 0.50) << " ms\n" << std::flush;
            }
        };

        // 1. Incast
        evaluate_uds_pattern("1_incast_191to1", [&](auto& ms, auto& wl) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == target) {
                auto t0 = std::chrono::high_resolution_clock::now();
                std::vector<std::thread> pool;
                for (int w = 0; w < size; ++w) {
                    if (w == target) continue;
                    pool.emplace_back([&, w]() {
                        recv_all(client_fds[w], pool_in[w].data(), payload);
                        uint32_t ack = 0x1111;
                        send_all(client_fds[w], &ack, sizeof(ack));
                    });
                }
                for (auto& th : pool) th.join();
                auto t1 = std::chrono::high_resolution_clock::now();
                ms[0] = std::chrono::duration<double, std::milli>(t1 - t0).count();
            } else {
                auto t0 = std::chrono::high_resolution_clock::now();
                send_all(client_fds[target], send_buf.data(), payload);
                uint32_t ack = 0;
                recv_all(client_fds[target], &ack, sizeof(ack));
                auto t1 = std::chrono::high_resolution_clock::now();
                wl[0] = std::chrono::duration<double, std::milli>(t1 - t0).count();
            }
        });

        // 2. Fan-out
        evaluate_uds_pattern("2_fanout_1to191", [&](auto& ms, auto& wl) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == target) {
                auto t0 = std::chrono::high_resolution_clock::now();
                std::vector<std::thread> pool;
                for (int w = 0; w < size; ++w) {
                    if (w == target) continue;
                    pool.emplace_back([&, w]() {
                        send_all(client_fds[w], pool_out[w].data(), payload);
                        uint32_t ack = 0;
                        recv_all(client_fds[w], &ack, sizeof(ack));
                    });
                }
                for (auto& th : pool) th.join();
                auto t1 = std::chrono::high_resolution_clock::now();
                ms[0] = std::chrono::duration<double, std::milli>(t1 - t0).count();
            } else {
                auto t0 = std::chrono::high_resolution_clock::now();
                recv_all(client_fds[target], recv_buf.data(), payload);
                uint32_t ack = 0x2222;
                send_all(client_fds[target], &ack, sizeof(ack));
                auto t1 = std::chrono::high_resolution_clock::now();
                wl[0] = std::chrono::duration<double, std::milli>(t1 - t0).count();
            }
        });

        // 3. Full-duplex
        evaluate_uds_pattern("3_full_duplex", [&](auto& ms, auto& wl) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == target) {
                auto t0 = std::chrono::high_resolution_clock::now();
                std::vector<std::thread> pool;
                for (int w = 0; w < size; ++w) {
                    if (w == target) continue;
                    pool.emplace_back([&, w]() {
                        std::thread snd([&]() { send_all(client_fds[w], pool_out[w].data(), payload); });
                        recv_all(client_fds[w], pool_in[w].data(), payload);
                        snd.join();
                    });
                }
                for (auto& th : pool) th.join();
                auto t1 = std::chrono::high_resolution_clock::now();
                ms[0] = std::chrono::duration<double, std::milli>(t1 - t0).count();
            } else {
                auto t0 = std::chrono::high_resolution_clock::now();
                std::thread snd([&]() { send_all(client_fds[target], send_buf.data(), payload); });
                recv_all(client_fds[target], recv_buf.data(), payload);
                snd.join();
                auto t1 = std::chrono::high_resolution_clock::now();
                wl[0] = std::chrono::duration<double, std::milli>(t1 - t0).count();
            }
        });

        // 4. Gather-Scatter Bulk Sync
        evaluate_uds_pattern("4_gather_scatter_bulk_sync", [&](auto& ms, auto& wl) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == target) {
                auto t0 = std::chrono::high_resolution_clock::now();
                std::vector<std::thread> r_pool;
                for (int w = 0; w < size; ++w) {
                    if (w == target) continue;
                    r_pool.emplace_back([&, w]() { recv_all(client_fds[w], pool_in[w].data(), payload); });
                }
                for (auto& th : r_pool) th.join();

                std::vector<std::thread> s_pool;
                for (int w = 0; w < size; ++w) {
                    if (w == target) continue;
                    s_pool.emplace_back([&, w]() { send_all(client_fds[w], pool_out[w].data(), payload); });
                }
                for (auto& th : s_pool) th.join();
                auto t1 = std::chrono::high_resolution_clock::now();
                ms[0] = std::chrono::duration<double, std::milli>(t1 - t0).count();
            } else {
                auto t0 = std::chrono::high_resolution_clock::now();
                send_all(client_fds[target], send_buf.data(), payload);
                recv_all(client_fds[target], recv_buf.data(), payload);
                auto t1 = std::chrono::high_resolution_clock::now();
                wl[0] = std::chrono::duration<double, std::milli>(t1 - t0).count();
            }
        });

        // 5. Immediate Response (Pipelined)
        evaluate_uds_pattern("5_immediate_response", [&](auto& ms, auto& wl) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == target) {
                auto t0 = std::chrono::high_resolution_clock::now();
                std::vector<std::thread> pool;
                for (int w = 0; w < size; ++w) {
                    if (w == target) continue;
                    pool.emplace_back([&, w]() {
                        recv_all(client_fds[w], pool_in[w].data(), payload);
                        send_all(client_fds[w], pool_out[w].data(), payload);
                    });
                }
                for (auto& th : pool) th.join();
                auto t1 = std::chrono::high_resolution_clock::now();
                ms[0] = std::chrono::duration<double, std::milli>(t1 - t0).count();
            } else {
                auto t0 = std::chrono::high_resolution_clock::now();
                send_all(client_fds[target], send_buf.data(), payload);
                recv_all(client_fds[target], recv_buf.data(), payload);
                auto t1 = std::chrono::high_resolution_clock::now();
                wl[0] = std::chrono::duration<double, std::milli>(t1 - t0).count();
            }
        });

        // 6. Serialized Reference
        evaluate_uds_pattern("6_serialized_reference", [&](auto& ms, auto& wl) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == target) {
                auto t0 = std::chrono::high_resolution_clock::now();
                for (int w = 0; w < size; ++w) {
                    if (w == target) continue;
                    recv_all(client_fds[w], pool_in[w].data(), payload);
                    send_all(client_fds[w], pool_out[w].data(), payload);
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                ms[0] = std::chrono::duration<double, std::milli>(t1 - t0).count();
            } else {
                auto t0 = std::chrono::high_resolution_clock::now();
                send_all(client_fds[target], send_buf.data(), payload);
                recv_all(client_fds[target], recv_buf.data(), payload);
                auto t1 = std::chrono::high_resolution_clock::now();
                wl[0] = std::chrono::duration<double, std::milli>(t1 - t0).count();
            }
        });

        if (rank == target) {
            for (int s = 0; s < size; ++s) {
                if (client_fds[s] >= 0) close(client_fds[s]);
            }
            if (srv_fd >= 0) close(srv_fd);
            unlink(sock_path.c_str());
        } else {
            if (client_fds[target] >= 0) close(client_fds[target]);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        std::cout << "\n[Success] UDS Coupling Schemes Suite completed successfully.\n";
    }

    MPI_Finalize();
    return 0;
}
