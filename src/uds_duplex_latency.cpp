#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/*
 * affinity_test/src/uds_duplex_latency.cpp
 * ========================================
 * Unix Domain Socket (UDS) Full-Duplex (simultaneous bidirectional) Benchmark.
 *
 * Measures simultaneous bidirectional exchange between rank pairs over Unix domain stream sockets.
 */

#include <mpi.h>
#include <sched.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static constexpr size_t DEFAULT_SMALL       = 64;
static constexpr size_t DEFAULT_LARGE       = 4 * 1024 * 1024;
static constexpr int    DEFAULT_WARMUP      = 10;
static constexpr int    DEFAULT_ITERS       = 50;
static constexpr int    LISTEN_BACKLOG      = 4;

struct RankInfo {
    int rank{-1};
    char hostname[256]{0};
    int current_cpu{-1};
    std::string affinity_str;
};

static std::string get_affinity_string() {
    cpu_set_t cpuset; CPU_ZERO(&cpuset);
    if (sched_getaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) return "unknown";
    std::ostringstream oss; bool first = true; int count = 0;
    for (int i = 0; i < CPU_SETSIZE; ++i) {
        if (CPU_ISSET(i, &cpuset)) { if (!first) oss << ","; oss << i; first = false; ++count; }
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

static double median_of(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2 == 0) ? 0.5 * (v[n/2 - 1] + v[n/2]) : v[n/2];
}

static double mean_of(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  -o, --output <file>        CSV output (default: uds_duplex_latency.csv)\n"
              << "  -s, --small-bytes <bytes>  Latency payload (default: 64)\n"
              << "  -l, --large-bytes <bytes>  Bandwidth payload (default: 4194304)\n"
              << "  -w, --warmup <count>       Warmup iters per pair (default: 10)\n"
              << "  -i, --iters <count>        Timed iters per pair (default: 50)\n"
              << "  --socket-dir <dir>         UDS temporary socket directory (default: /tmp)\n"
              << "  -h, --help                 Show help\n";
}

struct PairStats {
    double connect_us     = 0;
    double lat_min_us     = 0, lat_mean_us = 0, lat_median_us = 0, lat_max_us = 0;
    double bw_median_gibs = 0, bw_mean_gibs = 0;
};

static PairStats measure_uds_duplex(
    int my_rank, int ri, int rj,
    const std::string& sock_path,
    size_t small_bytes, size_t large_bytes, int warmup, int iters,
    std::vector<char>& ss, std::vector<char>& rs,
    std::vector<char>& sl, std::vector<char>& rl)
{
    PairStats stats;
    int conn_fd = -1;

    auto t0_conn = std::chrono::high_resolution_clock::now();

    if (my_rank == ri) {
        int srv = socket(AF_UNIX, SOCK_STREAM, 0);
        if (srv < 0) {
            std::cerr << "[rank " << my_rank << "] socket(AF_UNIX) failed: " << strerror(errno) << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        unlink(sock_path.c_str());

        struct sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, sock_path.c_str(), sizeof(addr.sun_path) - 1);

        if (bind(srv, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            std::cerr << "[rank " << my_rank << "] bind(" << sock_path << ") failed: " << strerror(errno) << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        listen(srv, LISTEN_BACKLOG);

        int ready = 1;
        MPI_Send(&ready, 1, MPI_INT, rj, 42, MPI_COMM_WORLD);

        conn_fd = accept(srv, nullptr, nullptr);
        close(srv);
        set_sock_buffers(conn_fd);
    } else {
        int ready = 0;
        MPI_Recv(&ready, 1, MPI_INT, ri, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        struct sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, sock_path.c_str(), sizeof(addr.sun_path) - 1);

        for (int attempt = 0; ; ++attempt) {
            conn_fd = socket(AF_UNIX, SOCK_STREAM, 0);
            set_sock_buffers(conn_fd);
            if (connect(conn_fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) break;
            close(conn_fd); conn_fd = -1;
            if (attempt >= 100) {
                std::cerr << "[rank " << my_rank << "] connect(" << sock_path << ") failed: " << strerror(errno) << "\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            usleep(2000);
        }
    }

    stats.connect_us = std::chrono::duration<double, std::micro>(
        std::chrono::high_resolution_clock::now() - t0_conn).count();

    // 1. Small payload duplex exchange (Latency)
    for (int w = 0; w < warmup; ++w) {
        std::thread sender([&]() { send_all(conn_fd, ss.data(), small_bytes); });
        recv_all(conn_fd, rs.data(), small_bytes);
        sender.join();
    }

    std::vector<double> lats(iters);
    for (int it = 0; it < iters; ++it) {
        auto a = std::chrono::high_resolution_clock::now();
        std::thread sender([&]() { send_all(conn_fd, ss.data(), small_bytes); });
        recv_all(conn_fd, rs.data(), small_bytes);
        sender.join();
        auto b = std::chrono::high_resolution_clock::now();
        lats[it] = std::chrono::duration<double, std::micro>(b - a).count();
    }

    // 2. Large payload duplex bandwidth
    for (int w = 0; w < warmup; ++w) {
        std::thread sender([&]() { send_all(conn_fd, sl.data(), large_bytes); });
        recv_all(conn_fd, rl.data(), large_bytes);
        sender.join();
    }

    constexpr double GiB = 1024.0 * 1024.0 * 1024.0;
    std::vector<double> bws(iters);
    for (int it = 0; it < iters; ++it) {
        auto a = std::chrono::high_resolution_clock::now();
        std::thread sender([&]() { send_all(conn_fd, sl.data(), large_bytes); });
        recv_all(conn_fd, rl.data(), large_bytes);
        sender.join();
        auto b = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(b - a).count();

        double peer_elapsed_sec = 0.0;
        if (my_rank == ri) {
            MPI_Sendrecv(&elapsed_sec, 1, MPI_DOUBLE, rj, 501,
                         &peer_elapsed_sec, 1, MPI_DOUBLE, rj, 502,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            double max_sec = std::max(elapsed_sec, peer_elapsed_sec);
            bws[it] = (2.0 * static_cast<double>(large_bytes) / GiB) / max_sec;
        } else {
            MPI_Sendrecv(&elapsed_sec, 1, MPI_DOUBLE, ri, 502,
                         &peer_elapsed_sec, 1, MPI_DOUBLE, ri, 501,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    if (my_rank == ri) {
        std::sort(lats.begin(), lats.end());
        stats.lat_min_us     = lats.front();
        stats.lat_max_us     = lats.back();
        stats.lat_mean_us    = mean_of(lats);
        stats.lat_median_us  = median_of(lats);

        stats.bw_median_gibs = median_of(bws);
        stats.bw_mean_gibs   = mean_of(bws);
    }

    close(conn_fd);
    if (my_rank == ri) {
        unlink(sock_path.c_str());
    }
    return stats;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string output_file = "uds_duplex_latency.csv";
    std::string socket_dir  = "/tmp";
    size_t small_bytes      = DEFAULT_SMALL;
    size_t large_bytes      = DEFAULT_LARGE;
    int warmup              = DEFAULT_WARMUP;
    int iters               = DEFAULT_ITERS;

    for (int k = 1; k < argc; ++k) {
        std::string arg = argv[k];
        if      ((arg == "-o" || arg == "--output")      && k + 1 < argc) output_file = argv[++k];
        else if ((arg == "-s" || arg == "--small-bytes") && k + 1 < argc) small_bytes = std::stoull(argv[++k]);
        else if ((arg == "-l" || arg == "--large-bytes") && k + 1 < argc) large_bytes = std::stoull(argv[++k]);
        else if ((arg == "-w" || arg == "--warmup")      && k + 1 < argc) warmup      = std::stoi(argv[++k]);
        else if ((arg == "-i" || arg == "--iters")       && k + 1 < argc) iters       = std::stoi(argv[++k]);
        else if (arg == "--socket-dir"                   && k + 1 < argc) socket_dir  = argv[++k];
        else if (arg == "-h" || arg == "--help") {
            if (rank == 0) print_usage(argv[0]);
            MPI_Finalize();
            return 0;
        }
    }

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
                std::cerr << "FATAL: UDS benchmark requires single-node allocation. Rank " << r
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

    std::vector<char> ss(small_bytes, 'A'), rs(small_bytes, 0);
    std::vector<char> sl(large_bytes, 'B'), rl(large_bytes, 0);

    std::ofstream csv;
    if (rank == 0) {
        csv.open(output_file, std::ios::trunc);
        csv << "rank_i,host_i,cpu_i,affinity_i,ip_i,"
            << "rank_j,host_j,cpu_j,affinity_j,ip_j,"
            << "iface,small_bytes,large_bytes,warmup,iters,connect_us,"
            << "latency_min_us,latency_mean_us,latency_median_us,latency_max_us,"
            << "bandwidth_median_gibs,bandwidth_mean_gibs\n";

        std::cout << "=== Precision UDS Full-Duplex Latency & Bandwidth Benchmark ===\n"
                  << "Node: " << host0 << ", Ranks: " << size << " (" << (size * (size - 1) / 2) << " pairs)\n"
                  << "Small payload: " << small_bytes << " B, Large payload: " << (large_bytes / (1024*1024)) << " MiB\n"
                  << "Warmup: " << warmup << ", Iters: " << iters << "\n"
                  << "Socket Dir: " << socket_dir << "\n"
                  << "Output CSV: " << output_file << "\n\n" << std::flush;
    }

    int total_pairs = size * (size - 1) / 2;
    int pair_idx = 0;

    for (int i = 0; i < size; ++i) {
        for (int j = i + 1; j < size; ++j) {
            ++pair_idx;
            int active_pair[2] = {i, j};
            MPI_Bcast(active_pair, 2, MPI_INT, 0, MPI_COMM_WORLD);

            std::ostringstream sp;
            sp << socket_dir << "/uds_d_" << i << "_" << j << ".sock";
            std::string sock_path = sp.str();

            PairStats stats;
            if (rank == i || rank == j) {
                stats = measure_uds_duplex(rank, i, j, sock_path, small_bytes, large_bytes, warmup, iters, ss, rs, sl, rl);
            }

            if (rank == 0) {
                if (i != 0) {
                    MPI_Recv(&stats, sizeof(PairStats), MPI_BYTE, i, 999, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }

                csv << i << ",\"" << all_ranks[i].hostname << "\","
                    << all_ranks[i].current_cpu << ",\"" << all_ranks[i].affinity_str << "\",\"uds\","
                    << j << ",\"" << all_ranks[j].hostname << "\","
                    << all_ranks[j].current_cpu << ",\"" << all_ranks[j].affinity_str << "\",\"uds\","
                    << "unix," << small_bytes << "," << large_bytes << ","
                    << warmup << "," << iters << ","
                    << stats.connect_us << ","
                    << stats.lat_min_us << "," << stats.lat_mean_us << ","
                    << stats.lat_median_us << "," << stats.lat_max_us << ","
                    << stats.bw_median_gibs << "," << stats.bw_mean_gibs << "\n" << std::flush;

                if (pair_idx % 200 == 0 || pair_idx == total_pairs) {
                    std::cout << "Progress: [" << pair_idx << "/" << total_pairs << "] duplex pairs completed ("
                              << (100.0 * pair_idx / total_pairs) << "%)\n" << std::flush;
                }
            } else if (rank == i && i != 0) {
                MPI_Send(&stats, sizeof(PairStats), MPI_BYTE, 0, 999, MPI_COMM_WORLD);
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        csv.close();
        std::cout << "\n[Success] UDS Duplex Benchmark completed successfully. Saved to '" << output_file << "'.\n";
    }

    MPI_Finalize();
    return 0;
}
