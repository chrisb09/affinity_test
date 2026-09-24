#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/*
 * affinity_test/tcp_worker_sweep.cpp
 * ==================================
 * Active Worker Scaling Sweep (1 to 191 Workers) for TCP/IP over IPoIB (ib0).
 *
 * Uses persistent TCP sockets to isolate transport scaling across W in:
 * {1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 191}.
 *
 * Balanced strided selection:
 *   Active workers are selected round-robin across Node 0 and Node 1 (50% local / 50% remote)
 *   and distributed across NUMA domains.
 *
 * Patterns evaluated:
 *   1. incast             (W -> 1 with controller ACK)
 *   2. fanout             (1 -> W with worker ACK)
 *   3. full_duplex        (W <-> 1 simultaneous bidirectional)
 *   4. immediate_response (Pipelined per-worker)
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
#include <sstream>
#include <string>
#include <thread>
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
        if (n_local + n_remote < W) {
            n_local = std::min(W - n_remote, static_cast<int>(local_pool.size()));
        }
    }

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

    std::string summary_file = "tcp_worker_sweep_summary.csv";
    std::string detail_file  = "tcp_worker_sweep_details.csv";
    std::string iface        = "ib0";
    int target               = 0;
    size_t payload           = DEFAULT_PAYLOAD;
    int warmup               = DEFAULT_WARMUP;
    int iters                = DEFAULT_ITERS;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      ((arg == "-o" || arg == "--output")  && i + 1 < argc) summary_file = argv[++i];
        else if ((arg == "-d" || arg == "--detail")  && i + 1 < argc) detail_file  = argv[++i];
        else if (arg == "--iface"                    && i + 1 < argc) iface        = argv[++i];
        else if ((arg == "-t" || arg == "--target")  && i + 1 < argc) target       = std::stoi(argv[++i]);
        else if ((arg == "-p" || arg == "--payload") && i + 1 < argc) payload      = std::stoull(argv[++i]);
        else if ((arg == "-w" || arg == "--warmup")  && i + 1 < argc) warmup       = std::stoi(argv[++i]);
        else if ((arg == "-i" || arg == "--iters")   && i + 1 < argc) iters        = std::stoi(argv[++i]);
        else if (arg == "-h" || arg == "--help") {
            if (rank == 0) {
                std::cout << "Usage: " << argv[0] << " [options]\n"
                          << "  -o, --output <file>    Summary CSV (default: tcp_worker_sweep_summary.csv)\n"
                          << "  -d, --detail <file>    Detail CSV (default: tcp_worker_sweep_details.csv)\n"
                          << "  --iface <name>         Network interface (default: ib0)\n"
                          << "  -t, --target <rank>    Controller target rank (default: 0)\n"
                          << "  -p, --payload <bytes>  Payload size (default: 4194304 = 4 MiB)\n"
                          << "  -w, --warmup <count>   Warmup iters (default: 5)\n"
                          << "  -i, --iters <count>    Timed iters (default: 50)\n";
            }
            MPI_Finalize();
            return 0;
        }
    }

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

    // Establish persistent TCP connections to target rank
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

    std::vector<int> sweep_counts;
    for (int c : {1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 191}) {
        if (c <= size - 1) sweep_counts.push_back(c);
    }
    if (std::find(sweep_counts.begin(), sweep_counts.end(), size - 1) == sweep_counts.end()) {
        sweep_counts.push_back(size - 1);
    }

    if (rank == 0) {
        std::cout << "=== TCP/IP (IPoIB) Active Worker Scaling Sweep Benchmark ===\n"
                  << "Ranks: " << size << " (Target Controller: Rank " << target << " on " << all_ranks[target].hostname << ")\n"
                  << "Interface: " << iface << "\n"
                  << "Payload: " << (payload / (1024 * 1024)) << " MiB in + " << (payload / (1024 * 1024)) << " MiB out\n"
                  << "Worker Counts Sweep: ";
        for (int c : sweep_counts) std::cout << c << " ";
        std::cout << "\nWarmup: " << warmup << ", Timed Iters: " << iters << "\n\n" << std::flush;
    }

    std::ofstream sum_csv, det_csv;
    if (rank == 0) {
        sum_csv.open(summary_file);
        sum_csv << "pattern,target_rank,target_host,target_cpu,target_ip,iface,active_workers,local_workers,remote_workers,payload_bytes,iters,"
                << "makespan_min_ms,makespan_p25_ms,makespan_median_ms,makespan_mean_ms,makespan_p75_ms,makespan_p95_ms,makespan_max_ms,"
                << "ingress_bw_median_gibs,egress_bw_median_gibs,combined_bw_median_gibs,"
                << "worker_lat_median_us,worker_lat_p95_us,worker_lat_max_us,"
                << "local_lat_median_us,local_lat_p95_us,remote_lat_median_us,remote_lat_p95_us\n";

        det_csv.open(detail_file);
        det_csv << "pattern,target_rank,active_workers,worker_rank,worker_host,is_remote,payload_bytes,iters,"
                << "lat_min_us,lat_p25_us,lat_median_us,lat_mean_us,lat_p75_us,lat_p95_us,lat_max_us\n";
    }

    std::vector<char> send_buf(payload, 't');
    std::vector<char> recv_buf(payload, 0);
    std::vector<std::vector<char>> tgt_recv_buf(size, std::vector<char>(payload, 0));

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

        auto run_tcp_sweep_pattern = [&](const std::string& name, auto runner_func) {
            if (rank == 0) {
                std::cout << "  [TCP " << std::setw(18) << std::left << name << "] ... " << std::flush;
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
                        << all_ranks[target].current_cpu << ",\"" << all_ranks[target].ip << "\",\"" << iface << "\","
                        << W << "," << n_local << "," << n_remote << ","
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

        // Pattern 1: TCP Incast (W -> 1)
        run_tcp_sweep_pattern("incast", [&](auto& ms, auto& wl, const auto& act_w, bool am_act) {
            uint32_t ack_val = 1;
            for (int it = 0; it < warmup + iters; ++it) {
                bool timed = (it >= warmup);
                MPI_Barrier(MPI_COMM_WORLD);
                auto t0 = std::chrono::high_resolution_clock::now();

                if (am_act) {
                    send_all(client_fd, send_buf.data(), payload);
                    uint32_t ack = 0;
                    recv_all(client_fd, &ack, sizeof(ack));
                    auto t1 = std::chrono::high_resolution_clock::now();
                    if (timed) wl[it - warmup] = std::chrono::duration<double, std::micro>(t1 - t0).count();
                } else if (rank == target) {
                    std::vector<std::thread> workers;
                    workers.reserve(act_w.size());
                    for (int w : act_w) {
                        workers.emplace_back([&, w]() {
                            recv_all(client_fds[w], tgt_recv_buf[w].data(), payload);
                            send_all(client_fds[w], &ack_val, sizeof(ack_val));
                        });
                    }
                    for (auto& t : workers) t.join();
                    auto t1 = std::chrono::high_resolution_clock::now();
                    if (timed) ms[it - warmup] = std::chrono::duration<double, std::milli>(t1 - t0).count();
                }
            }
        });

        // Pattern 2: TCP Fan-out (1 -> W)
        run_tcp_sweep_pattern("fanout", [&](auto& ms, auto& wl, const auto& act_w, bool am_act) {
            for (int it = 0; it < warmup + iters; ++it) {
                bool timed = (it >= warmup);
                MPI_Barrier(MPI_COMM_WORLD);
                auto t0 = std::chrono::high_resolution_clock::now();

                if (rank == target) {
                    std::vector<std::thread> workers;
                    workers.reserve(act_w.size());
                    for (int w : act_w) {
                        workers.emplace_back([&, w]() {
                            send_all(client_fds[w], send_buf.data(), payload);
                            uint32_t ack = 0;
                            recv_all(client_fds[w], &ack, sizeof(ack));
                        });
                    }
                    for (auto& t : workers) t.join();
                    auto t1 = std::chrono::high_resolution_clock::now();
                    if (timed) ms[it - warmup] = std::chrono::duration<double, std::milli>(t1 - t0).count();
                } else if (am_act) {
                    recv_all(client_fd, recv_buf.data(), payload);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    uint32_t ack = 1;
                    send_all(client_fd, &ack, sizeof(ack));
                    if (timed) wl[it - warmup] = std::chrono::duration<double, std::micro>(t1 - t0).count();
                }
            }
        });

        // Pattern 3: TCP Full Duplex (W <-> 1)
        run_tcp_sweep_pattern("full_duplex", [&](auto& ms, auto& wl, const auto& act_w, bool am_act) {
            for (int it = 0; it < warmup + iters; ++it) {
                bool timed = (it >= warmup);
                MPI_Barrier(MPI_COMM_WORLD);
                auto t0 = std::chrono::high_resolution_clock::now();

                if (am_act) {
                    std::thread sender([&]() { send_all(client_fd, send_buf.data(), payload); });
                    recv_all(client_fd, recv_buf.data(), payload);
                    sender.join();
                    auto t1 = std::chrono::high_resolution_clock::now();
                    if (timed) wl[it - warmup] = std::chrono::duration<double, std::micro>(t1 - t0).count();
                } else if (rank == target) {
                    std::vector<std::thread> workers;
                    workers.reserve(act_w.size());
                    for (int w : act_w) {
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

        // Pattern 4: TCP Immediate Response
        run_tcp_sweep_pattern("immediate_response", [&](auto& ms, auto& wl, const auto& act_w, bool am_act) {
            for (int it = 0; it < warmup + iters; ++it) {
                bool timed = (it >= warmup);
                MPI_Barrier(MPI_COMM_WORLD);
                auto t0 = std::chrono::high_resolution_clock::now();

                if (am_act) {
                    send_all(client_fd, send_buf.data(), payload);
                    recv_all(client_fd, recv_buf.data(), payload);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    if (timed) wl[it - warmup] = std::chrono::duration<double, std::micro>(t1 - t0).count();
                } else if (rank == target) {
                    std::vector<std::thread> workers;
                    workers.reserve(act_w.size());
                    for (int w : act_w) {
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
    }

    // Cleanup
    if (rank == target) {
        for (int cfd : client_fds) if (cfd >= 0) close(cfd);
        if (srv_fd >= 0) close(srv_fd);
    } else {
        if (client_fd >= 0) close(client_fd);
    }

    if (rank == 0) {
        sum_csv.close();
        det_csv.close();
        std::cout << "\nTCP Worker count sweep complete. Summary: " << summary_file << ", Details: " << detail_file << "\n";
    }

    MPI_Finalize();
    return 0;
}
