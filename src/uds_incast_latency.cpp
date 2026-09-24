#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/*
 * affinity_test/src/uds_incast_latency.cpp
 * ========================================
 * Unix Domain Socket (UDS) Incast / Fan-in Flooding Benchmark.
 *
 * Simulates N-1 UDS client ranks simultaneously streaming data into a single
 * server / target rank over Unix domain stream sockets.
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
#include <sstream>
#include <string>
#include <thread>
#include <vector>

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

static std::vector<size_t> parse_size_list(const std::string& str) {
    std::vector<size_t> res;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) res.push_back(std::stoull(token));
    }
    return res;
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

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  -o, --output <file>         Summary CSV output (default: uds_incast_summary.csv)\n"
              << "  -d, --detail <file>         Detailed per-sender CSV output (default: uds_incast_details.csv)\n"
              << "  -t, --targets <list>        Comma-separated target ranks (default: 0)\n"
              << "  -p, --payloads <list>       Comma-separated payload sizes in bytes\n"
              << "                              (default: 64,1024,4096,16384,65536,262144,1048576,4194304)\n"
              << "  --socket-dir <dir>          UDS temporary socket directory (default: /tmp)\n"
              << "  -w, --warmup <count>        Warmup iterations per payload (default: 5)\n"
              << "  -i, --iters <count>         Measured iterations per payload (default: 20)\n"
              << "  -h, --help                  Show this help message\n";
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string summary_file = "uds_incast_summary.csv";
    std::string detail_file  = "uds_incast_details.csv";
    std::string target_str   = "0";
    std::string payload_str  = "64,1024,4096,16384,65536,262144,1048576,4194304";
    std::string socket_dir   = "/tmp";
    int warmup = 5;
    int iters  = 20;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      ((arg == "-o" || arg == "--output")   && i + 1 < argc) summary_file = argv[++i];
        else if ((arg == "-d" || arg == "--detail")   && i + 1 < argc) detail_file  = argv[++i];
        else if ((arg == "-t" || arg == "--targets")  && i + 1 < argc) target_str   = argv[++i];
        else if ((arg == "-p" || arg == "--payloads") && i + 1 < argc) payload_str  = argv[++i];
        else if (arg == "--socket-dir"                && i + 1 < argc) socket_dir   = argv[++i];
        else if ((arg == "-w" || arg == "--warmup")   && i + 1 < argc) warmup       = std::stoi(argv[++i]);
        else if ((arg == "-i" || arg == "--iters")    && i + 1 < argc) iters        = std::stoi(argv[++i]);
        else if (arg == "-h" || arg == "--help") {
            if (rank == 0) print_usage(argv[0]);
            MPI_Finalize();
            return 0;
        }
    }

    std::vector<int> targets = parse_int_list(target_str);
    std::vector<size_t> payloads = parse_size_list(payload_str);

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
                std::cerr << "FATAL: UDS incast requires single-node allocation. Rank " << r
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
                  << "=== Unix Domain Socket (UDS) Incast / Fan-in Flooding Benchmark ===\n"
                  << "=================================================================\n"
                  << "Node: " << host0 << ", Ranks: " << size << " (1 target, " << (size - 1) << " concurrent senders)\n"
                  << "Payloads: " << payload_str << " B\n"
                  << "Warmup: " << warmup << ", Iters: " << iters << "\n"
                  << "Socket Dir: " << socket_dir << "\n"
                  << "Summary CSV: " << summary_file << "\n"
                  << "Detail CSV:  " << detail_file << "\n\n" << std::flush;

        std::ofstream sum_csv(summary_file, std::ios::trunc);
        sum_csv << "target_rank,target_host,target_cpu,target_affinity,senders_count,"
                << "payload_bytes,iters,warmup,iface,"
                << "ingest_bw_min_gibs,ingest_bw_p25_gibs,ingest_bw_median_gibs,ingest_bw_mean_gibs,"
                << "ingest_bw_p75_gibs,ingest_bw_p95_gibs,ingest_bw_max_gibs,"
                << "sender_lat_min_us,sender_lat_p25_us,sender_lat_median_us,sender_lat_mean_us,"
                << "sender_lat_p75_us,sender_lat_p95_us,sender_lat_p99_us,sender_lat_max_us\n";
        sum_csv.close();

        std::ofstream det_csv(detail_file, std::ios::trunc);
        det_csv << "target_rank,target_host,target_cpu,sender_rank,sender_host,sender_cpu,sender_affinity,"
                << "ip,iface,payload_bytes,iters,"
                << "sender_lat_min_us,sender_lat_p25_us,sender_lat_median_us,sender_lat_mean_us,"
                << "sender_lat_p75_us,sender_lat_p95_us,sender_lat_max_us\n";
        det_csv.close();
    }

    size_t max_payload = *std::max_element(payloads.begin(), payloads.end());
    std::vector<char> send_buf(max_payload, 'X');
    std::vector<std::vector<char>> recv_pool(size, std::vector<char>(max_payload, 0));

    for (int target : targets) {
        if (target < 0 || target >= size) continue;

        int srv_fd = -1;
        std::vector<int> client_fds(size, -1);
        std::ostringstream sp;
        sp << socket_dir << "/uds_incast_t" << target << ".sock";
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
                    // Send my rank ID
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

        for (size_t payload : payloads) {
            // Warmup
            for (int w = 0; w < warmup; ++w) {
                MPI_Barrier(MPI_COMM_WORLD);
                if (rank == target) {
                    std::vector<std::thread> pool;
                    for (int s = 0; s < size; ++s) {
                        if (s == target) continue;
                        pool.emplace_back([&, s]() {
                            recv_all(client_fds[s], recv_pool[s].data(), payload);
                            uint32_t ack = 0xABCD1234;
                            send_all(client_fds[s], &ack, sizeof(ack));
                        });
                    }
                    for (auto& th : pool) th.join();
                } else {
                    send_all(client_fds[target], send_buf.data(), payload);
                    uint32_t ack = 0;
                    recv_all(client_fds[target], &ack, sizeof(ack));
                }
            }

            // Timed iterations
            std::vector<double> sender_lats(iters, 0.0);
            std::vector<double> target_makespans(iters, 0.0);

            for (int it = 0; it < iters; ++it) {
                MPI_Barrier(MPI_COMM_WORLD);

                if (rank == target) {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    std::vector<std::thread> pool;
                    for (int s = 0; s < size; ++s) {
                        if (s == target) continue;
                        pool.emplace_back([&, s]() {
                            recv_all(client_fds[s], recv_pool[s].data(), payload);
                            uint32_t ack = 0xABCD1234;
                            send_all(client_fds[s], &ack, sizeof(ack));
                        });
                    }
                    for (auto& th : pool) th.join();
                    auto t1 = std::chrono::high_resolution_clock::now();
                    target_makespans[it] = std::chrono::duration<double, std::micro>(t1 - t0).count();
                } else {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    send_all(client_fds[target], send_buf.data(), payload);
                    uint32_t ack = 0;
                    recv_all(client_fds[target], &ack, sizeof(ack));
                    auto t1 = std::chrono::high_resolution_clock::now();
                    sender_lats[it] = std::chrono::duration<double, std::micro>(t1 - t0).count();
                }
            }

            // Gather sender stats to rank 0
            std::vector<double> all_sender_lats(size * iters, 0.0);
            MPI_Gather(sender_lats.data(), iters, MPI_DOUBLE, all_sender_lats.data(), iters, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            std::vector<double> all_target_makespans(iters, 0.0);
            if (rank == target) {
                all_target_makespans = target_makespans;
            }
            MPI_Bcast(all_target_makespans.data(), iters, MPI_DOUBLE, target, MPI_COMM_WORLD);

            if (rank == 0) {
                constexpr double GiB = 1024.0 * 1024.0 * 1024.0;
                std::vector<double> ingest_bws(iters);
                for (int it = 0; it < iters; ++it) {
                    double sec = all_target_makespans[it] * 1e-6;
                    ingest_bws[it] = ((size - 1) * static_cast<double>(payload) / GiB) / sec;
                }

                std::vector<double> pooled_lats;
                std::ofstream det_csv(detail_file, std::ios::app);

                for (int s = 0; s < size; ++s) {
                    if (s == target) continue;
                    std::vector<double> s_lats(iters);
                    for (int it = 0; it < iters; ++it) {
                        s_lats[it] = all_sender_lats[s * iters + it];
                        pooled_lats.push_back(s_lats[it]);
                    }
                    std::sort(s_lats.begin(), s_lats.end());

                    det_csv << target << ",\"" << all_ranks[target].hostname << "\","
                            << all_ranks[target].current_cpu << ","
                            << s << ",\"" << all_ranks[s].hostname << "\","
                            << all_ranks[s].current_cpu << ",\"" << all_ranks[s].affinity_str << "\","
                            << "\"uds\",unix," << payload << "," << iters << ","
                            << s_lats.front() << ","
                            << percentile_of(s_lats, 0.25) << ","
                            << percentile_of(s_lats, 0.50) << ","
                            << mean_of(s_lats) << ","
                            << percentile_of(s_lats, 0.75) << ","
                            << percentile_of(s_lats, 0.95) << ","
                            << s_lats.back() << "\n";
                }
                det_csv.close();

                std::ofstream sum_csv(summary_file, std::ios::app);
                sum_csv << target << ",\"" << all_ranks[target].hostname << "\","
                        << all_ranks[target].current_cpu << ",\"" << all_ranks[target].affinity_str << "\","
                        << (size - 1) << "," << payload << "," << iters << "," << warmup << ",unix,"
                        << percentile_of(ingest_bws, 0.0)  << ","
                        << percentile_of(ingest_bws, 0.25) << ","
                        << percentile_of(ingest_bws, 0.50) << ","
                        << mean_of(ingest_bws)             << ","
                        << percentile_of(ingest_bws, 0.75) << ","
                        << percentile_of(ingest_bws, 0.95) << ","
                        << percentile_of(ingest_bws, 1.0)  << ","
                        << percentile_of(pooled_lats, 0.0)  << ","
                        << percentile_of(pooled_lats, 0.25) << ","
                        << percentile_of(pooled_lats, 0.50) << ","
                        << mean_of(pooled_lats)             << ","
                        << percentile_of(pooled_lats, 0.75) << ","
                        << percentile_of(pooled_lats, 0.95) << ","
                        << percentile_of(pooled_lats, 0.99) << ","
                        << percentile_of(pooled_lats, 1.0)  << "\n";
                sum_csv.close();

                std::cout << "[Target " << target << " | Payload " << std::setw(8) << payload << " B] "
                          << "Ingest BW (median): " << std::setw(6) << std::fixed << std::setprecision(2)
                          << percentile_of(ingest_bws, 0.50) << " GiB/s | "
                          << "Sender Latency (p50): " << std::setw(8) << std::fixed << std::setprecision(1)
                          << percentile_of(pooled_lats, 0.50) << " µs | (p95): "
                          << percentile_of(pooled_lats, 0.95) << " µs\n" << std::flush;
            }
        }

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
        std::cout << "\n[Success] UDS Incast Benchmark completed successfully.\n";
    }

    MPI_Finalize();
    return 0;
}
