#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/*
 * affinity_test/tcp_coupling_schemes.cpp
 * ======================================
 * Transport Scheme Benchmark & Simulation Suite for HPC-ML Coupling (TCP/IP over IPoIB).
 *
 * Uses persistent TCP sockets across all benchmark rounds to avoid connection setup noise.
 * Evaluates both fundamental transport limits and coupling transport schemes across 192 ranks.
 */

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <mpi.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sched.h>
#include <sys/socket.h>
#include <sys/types.h>
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
    std::string ip;
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

static std::string get_iface_ip(const std::string& iface_name, int rank) {
    struct ifaddrs* ifap = nullptr;
    if (getifaddrs(&ifap) != 0) {
        std::cerr << "[rank " << rank << "] getifaddrs failed: " << strerror(errno) << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::string preferred, fallback;
    for (struct ifaddrs* ifa = ifap; ifa; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_INET) continue;
        char buf[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr)->sin_addr, buf, sizeof(buf));
        std::string ip(buf);
        if (ip == "127.0.0.1") continue;
        if (std::string(ifa->ifa_name) == iface_name) { preferred = ip; break; }
        if (fallback.empty()) fallback = ip;
    }
    freeifaddrs(ifap);
    if (!preferred.empty()) return preferred;
    if (!fallback.empty()) return fallback;
    std::cerr << "[rank " << rank << "] FATAL: no non-loopback IPv4 found\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
    return "";
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

static void set_nodelay(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
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

    std::string summary_file = "tcp_coupling_summary.csv";
    std::string detail_file  = "tcp_coupling_details.csv";
    std::string target_str   = (size >= 96) ? "0,96" : "0";
    std::string iface        = "ib0";
    size_t payload           = DEFAULT_PAYLOAD;
    int warmup               = DEFAULT_WARMUP;
    int iters                = DEFAULT_ITERS;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      ((arg == "-o" || arg == "--output")   && i + 1 < argc) summary_file = argv[++i];
        else if ((arg == "-d" || arg == "--detail")   && i + 1 < argc) detail_file  = argv[++i];
        else if ((arg == "-t" || arg == "--targets")  && i + 1 < argc) target_str   = argv[++i];
        else if (arg == "--iface"                     && i + 1 < argc) iface        = argv[++i];
        else if ((arg == "-p" || arg == "--payload")  && i + 1 < argc) payload      = std::stoull(argv[++i]);
        else if ((arg == "-w" || arg == "--warmup")   && i + 1 < argc) warmup       = std::stoi(argv[++i]);
        else if ((arg == "-i" || arg == "--iters")    && i + 1 < argc) iters        = std::stoi(argv[++i]);
        else if (arg == "-h" || arg == "--help") {
            if (rank == 0) {
                std::cout << "Usage: " << argv[0] << " [options]\n"
                          << "  -o, --output <file>    Summary CSV (default: tcp_coupling_summary.csv)\n"
                          << "  -d, --detail <file>    Detail CSV (default: tcp_coupling_details.csv)\n"
                          << "  -t, --targets <list>   Target ranks (default: 0,96)\n"
                          << "  --iface <name>         Network interface (default: ib0)\n"
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
    local_info.ip = get_iface_ip(iface, rank);

    std::vector<char> host_buf(size * 256, 0);
    MPI_Gather(local_info.hostname, 256, MPI_CHAR, host_buf.data(), 256, MPI_CHAR, 0, MPI_COMM_WORLD);

    std::vector<int> cpu_buf(size, 0);
    MPI_Gather(&local_info.current_cpu, 1, MPI_INT, cpu_buf.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    constexpr int IP_LEN = 64;
    char ip_fixed[IP_LEN] = {};
    strncpy(ip_fixed, local_info.ip.c_str(), IP_LEN - 1);
    std::vector<char> ip_buf(size * IP_LEN, 0);
    MPI_Gather(ip_fixed, IP_LEN, MPI_CHAR, ip_buf.data(), IP_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(ip_buf.data(), size * IP_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);

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
        all_ranks[r].ip   = std::string(&ip_buf[r * IP_LEN]);
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
        std::cout << "=== TCP/IP (IPoIB) Transport Schemes & Coupling Simulation Suite ===\n"
                  << "Ranks: " << size << " (1 Controller + " << (size - 1) << " Workers)\n"
                  << "Interface: " << iface << "\n"
                  << "Payload: " << (payload / (1024 * 1024)) << " MiB in + " << (payload / (1024 * 1024)) << " MiB out\n"
                  << "Targets: " << target_str << "\n"
                  << "Warmup: " << warmup << ", Timed Iters: " << iters << "\n\n" << std::flush;
    }

    std::ofstream sum_csv, det_csv;
    if (rank == 0) {
        sum_csv.open(summary_file);
        sum_csv << "pattern,target_rank,target_host,target_cpu,target_affinity,target_ip,iface,workers_count,payload_bytes,iters,"
                << "makespan_min_ms,makespan_p25_ms,makespan_median_ms,makespan_mean_ms,makespan_p75_ms,makespan_p95_ms,makespan_max_ms,"
                << "ingress_bw_median_gibs,egress_bw_median_gibs,combined_bw_median_gibs,"
                << "worker_lat_min_us,worker_lat_p25_us,worker_lat_median_us,worker_lat_mean_us,worker_lat_p75_us,worker_lat_p95_us,worker_lat_p99_us,worker_lat_max_us\n";

        det_csv.open(detail_file);
        det_csv << "pattern,target_rank,target_host,worker_rank,worker_host,worker_cpu,worker_affinity,worker_ip,locality_tier,iface,payload_bytes,iters,"
                << "lat_min_us,lat_p25_us,lat_median_us,lat_mean_us,lat_p75_us,lat_p95_us,lat_max_us\n";
    }

    std::vector<char> send_buf(payload, 't');
    std::vector<char> recv_buf(payload, 0);

    for (int target : targets) {
        if (target < 0 || target >= size) continue;

        // Establish persistent connections for this target
        int target_port = 0;
        int srv_fd = -1;
        std::vector<int> client_fds(size, -1);
        int client_fd = -1;

        if (rank == target) {
            srv_fd = socket(AF_INET, SOCK_STREAM, 0);
            int opt = 1;
            setsockopt(srv_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
            struct sockaddr_in addr{};
            addr.sin_family      = AF_INET;
            addr.sin_port        = htons(0); // ephemeral
            addr.sin_addr.s_addr = INADDR_ANY;
            if (bind(srv_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
                std::cerr << "[Target " << rank << "] bind failed: " << strerror(errno) << "\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            struct sockaddr_in bound{};
            socklen_t blen = sizeof(bound);
            getsockname(srv_fd, (struct sockaddr*)&bound, &blen);
            target_port = ntohs(bound.sin_port);
            listen(srv_fd, size + 16);
        }

        MPI_Bcast(&target_port, 1, MPI_INT, target, MPI_COMM_WORLD);

        if (rank != target) {
            for (int attempt = 0; ; ++attempt) {
                client_fd = socket(AF_INET, SOCK_STREAM, 0);
                set_nodelay(client_fd);
                set_sock_buffers(client_fd);
                struct sockaddr_in addr{};
                addr.sin_family = AF_INET;
                addr.sin_port   = htons(target_port);
                inet_pton(AF_INET, all_ranks[target].ip.c_str(), &addr.sin_addr);
                if (connect(client_fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
                    send_all(client_fd, &rank, sizeof(rank));
                    break;
                }
                close(client_fd); client_fd = -1;
                if (attempt >= 200) { std::cerr << "[Worker " << rank << "] connect failed\n"; MPI_Abort(MPI_COMM_WORLD, 1); }
                usleep(5000);
            }
        } else {
            for (int count = 0; count < size - 1; ++count) {
                int cfd = accept(srv_fd, nullptr, nullptr);
                set_nodelay(cfd);
                set_sock_buffers(cfd);
                int client_rank = -1;
                recv_all(cfd, &client_rank, sizeof(client_rank));
                if (client_rank >= 0 && client_rank < size) {
                    client_fds[client_rank] = cfd;
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        auto evaluate_tcp_pattern = [&](const std::string& name, auto runner_func) {
            if (rank == 0) {
                std::cout << ">>> Running [TCP " << std::setw(22) << std::left << name << "] Targeting Rank " << target
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
                            << all_ranks[w].ip << "\",\"" << tier << "\",\"" << iface << "\","
                            << payload << "," << iters << ","
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
                        << all_ranks[target].current_cpu << ",\"" << all_ranks[target].affinity_str << "\",\""
                        << all_ranks[target].ip << "\",\"" << iface << "\","
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

                std::cout << "Makespan (p50): " << std::setw(7) << std::fixed << std::setprecision(1) << percentile_of(makespan_ms, 0.50) << " ms"
                          << " | Worker Lat (p50): " << std::setw(7) << std::fixed << std::setprecision(1) << (percentile_of(pooled_lats, 0.50) / 1000.0) << " ms"
                          << " | (p95): " << std::setw(7) << (percentile_of(pooled_lats, 0.95) / 1000.0) << " ms"
                          << " | Comb BW: " << std::setw(6) << std::setprecision(2) << combined_bw << " GiB/s\n" << std::flush;
            }

            MPI_Barrier(MPI_COMM_WORLD);
        };

        // Pattern 1: TCP Incast (191 -> 1)
        evaluate_tcp_pattern("1_incast_191to1", [&](auto& ms, auto& wl) {
            std::vector<std::vector<char>> tgt_recv_buf(size, std::vector<char>(payload, 0));
            uint32_t ack_val = 1;

            for (int it = 0; it < warmup + iters; ++it) {
                bool timed = (it >= warmup);
                MPI_Barrier(MPI_COMM_WORLD);
                auto t0 = std::chrono::high_resolution_clock::now();

                if (rank != target) {
                    send_all(client_fd, send_buf.data(), payload);
                    uint32_t ack = 0;
                    recv_all(client_fd, &ack, sizeof(ack));
                    auto t1 = std::chrono::high_resolution_clock::now();
                    if (timed) wl[it - warmup] = std::chrono::duration<double, std::micro>(t1 - t0).count();
                } else {
                    std::vector<std::thread> workers;
                    workers.reserve(size - 1);
                    for (int src = 0; src < size; ++src) {
                        if (src == target) continue;
                        workers.emplace_back([&, src]() {
                            recv_all(client_fds[src], tgt_recv_buf[src].data(), payload);
                            send_all(client_fds[src], &ack_val, sizeof(ack_val));
                        });
                    }
                    for (auto& t : workers) t.join();
                    auto t1 = std::chrono::high_resolution_clock::now();
                    if (timed) ms[it - warmup] = std::chrono::duration<double, std::milli>(t1 - t0).count();
                }
            }
        });

        // Pattern 2: TCP Fan-out (1 -> 191)
        evaluate_tcp_pattern("2_fanout_1to191", [&](auto& ms, auto& wl) {
            for (int it = 0; it < warmup + iters; ++it) {
                bool timed = (it >= warmup);
                MPI_Barrier(MPI_COMM_WORLD);
                auto t0 = std::chrono::high_resolution_clock::now();

                if (rank == target) {
                    std::vector<std::thread> workers;
                    workers.reserve(size - 1);
                    for (int dst = 0; dst < size; ++dst) {
                        if (dst == target) continue;
                        workers.emplace_back([&, dst]() {
                            send_all(client_fds[dst], send_buf.data(), payload);
                            uint32_t ack = 0;
                            recv_all(client_fds[dst], &ack, sizeof(ack));
                        });
                    }
                    for (auto& t : workers) t.join();
                    auto t1 = std::chrono::high_resolution_clock::now();
                    if (timed) ms[it - warmup] = std::chrono::duration<double, std::milli>(t1 - t0).count();
                } else {
                    recv_all(client_fd, recv_buf.data(), payload);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    uint32_t ack = 1;
                    send_all(client_fd, &ack, sizeof(ack));
                    if (timed) wl[it - warmup] = std::chrono::duration<double, std::micro>(t1 - t0).count();
                }
            }
        });

        // Pattern 3: TCP Full Duplex (191 <-> 1)
        evaluate_tcp_pattern("3_full_duplex", [&](auto& ms, auto& wl) {
            std::vector<std::vector<char>> tgt_recv_buf(size, std::vector<char>(payload, 0));
            for (int it = 0; it < warmup + iters; ++it) {
                bool timed = (it >= warmup);
                MPI_Barrier(MPI_COMM_WORLD);
                auto t0 = std::chrono::high_resolution_clock::now();

                if (rank != target) {
                    std::thread sender([&]() { send_all(client_fd, send_buf.data(), payload); });
                    recv_all(client_fd, recv_buf.data(), payload);
                    sender.join();
                    auto t1 = std::chrono::high_resolution_clock::now();
                    if (timed) wl[it - warmup] = std::chrono::duration<double, std::micro>(t1 - t0).count();
                } else {
                    std::vector<std::thread> workers;
                    workers.reserve(size - 1);
                    for (int w = 0; w < size; ++w) {
                        if (w == target) continue;
                        workers.emplace_back([&, w]() {
                            std::thread sender([&]() { send_all(client_fds[w], send_buf.data(), payload); });
                            recv_all(client_fds[w], tgt_recv_buf[w].data(), payload);
                            sender.join();
                        });
                    }
                    for (auto& t : workers) t.join();
                    auto t1 = std::chrono::high_resolution_clock::now();
                    if (timed) ms[it - warmup] = std::chrono::duration<double, std::milli>(t1 - t0).count();
                }
            }
        });

        // Pattern 4: TCP Gather Then Scatter
        evaluate_tcp_pattern("4_gather_scatter", [&](auto& ms, auto& wl) {
            std::vector<std::vector<char>> tgt_recv_buf(size, std::vector<char>(payload, 0));
            for (int it = 0; it < warmup + iters; ++it) {
                bool timed = (it >= warmup);
                MPI_Barrier(MPI_COMM_WORLD);
                auto t0 = std::chrono::high_resolution_clock::now();

                if (rank != target) {
                    send_all(client_fd, send_buf.data(), payload);
                    recv_all(client_fd, recv_buf.data(), payload);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    if (timed) wl[it - warmup] = std::chrono::duration<double, std::micro>(t1 - t0).count();
                } else {
                    // Phase 1: Drain all inputs
                    std::vector<std::thread> r_workers;
                    r_workers.reserve(size - 1);
                    for (int w = 0; w < size; ++w) {
                        if (w == target) continue;
                        r_workers.emplace_back([&, w]() { recv_all(client_fds[w], tgt_recv_buf[w].data(), payload); });
                    }
                    for (auto& t : r_workers) t.join();

                    // Phase 2: Send all outputs
                    std::vector<std::thread> s_workers;
                    s_workers.reserve(size - 1);
                    for (int w = 0; w < size; ++w) {
                        if (w == target) continue;
                        s_workers.emplace_back([&, w]() { send_all(client_fds[w], send_buf.data(), payload); });
                    }
                    for (auto& t : s_workers) t.join();

                    auto t1 = std::chrono::high_resolution_clock::now();
                    if (timed) ms[it - warmup] = std::chrono::duration<double, std::milli>(t1 - t0).count();
                }
            }
        });

        // Pattern 5: TCP Immediate Response
        evaluate_tcp_pattern("5_immediate_response", [&](auto& ms, auto& wl) {
            std::vector<std::vector<char>> tgt_recv_buf(size, std::vector<char>(payload, 0));
            for (int it = 0; it < warmup + iters; ++it) {
                bool timed = (it >= warmup);
                MPI_Barrier(MPI_COMM_WORLD);
                auto t0 = std::chrono::high_resolution_clock::now();

                if (rank != target) {
                    send_all(client_fd, send_buf.data(), payload);
                    recv_all(client_fd, recv_buf.data(), payload);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    if (timed) wl[it - warmup] = std::chrono::duration<double, std::micro>(t1 - t0).count();
                } else {
                    std::vector<std::thread> workers;
                    workers.reserve(size - 1);
                    for (int w = 0; w < size; ++w) {
                        if (w == target) continue;
                        workers.emplace_back([&, w]() {
                            recv_all(client_fds[w], tgt_recv_buf[w].data(), payload);
                            send_all(client_fds[w], send_buf.data(), payload);
                        });
                    }
                    for (auto& t : workers) t.join();
                    auto t1 = std::chrono::high_resolution_clock::now();
                    if (timed) ms[it - warmup] = std::chrono::duration<double, std::milli>(t1 - t0).count();
                }
            }
        });

        // Pattern 6: TCP Serialized Reference
        evaluate_tcp_pattern("6_serialized_ref", [&](auto& ms, auto& wl) {
            for (int it = 0; it < warmup + iters; ++it) {
                bool timed = (it >= warmup);
                MPI_Barrier(MPI_COMM_WORLD);
                auto t0 = std::chrono::high_resolution_clock::now();

                // 1. Inputs one by one
                for (int w = 0; w < size; ++w) {
                    if (w == target) continue;
                    if (rank == w) send_all(client_fd, send_buf.data(), payload);
                    else if (rank == target) recv_all(client_fds[w], recv_buf.data(), payload);
                }

                // 2. Outputs one by one
                for (int w = 0; w < size; ++w) {
                    if (w == target) continue;
                    if (rank == target) send_all(client_fds[w], send_buf.data(), payload);
                    else if (rank == w) {
                        recv_all(client_fd, recv_buf.data(), payload);
                        auto t1 = std::chrono::high_resolution_clock::now();
                        if (timed) wl[it - warmup] = std::chrono::duration<double, std::micro>(t1 - t0).count();
                    }
                }

                if (rank == target) {
                    auto t1 = std::chrono::high_resolution_clock::now();
                    if (timed) ms[it - warmup] = std::chrono::duration<double, std::milli>(t1 - t0).count();
                }
            }
        });

        // Cleanup connections
        if (rank == target) {
            for (int cfd : client_fds) if (cfd >= 0) close(cfd);
            if (srv_fd >= 0) close(srv_fd);
        } else {
            if (client_fd >= 0) close(client_fd);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        sum_csv.close();
        det_csv.close();
        std::cout << "\nTCP coupling transport schemes complete. Summary: " << summary_file
                  << ", Details: " << detail_file << "\n";
    }

    MPI_Finalize();
    return 0;
}
