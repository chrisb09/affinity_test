#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/*
 * affinity_test/tcp_pair_latency.cpp
 * ====================================
 * TCP/IP (IPoIB) rank-to-rank latency and bandwidth benchmark.
 *
 * Mirrors mpi_pair_latency.cpp exactly in measurement protocol:
 *   - Small payload (default 64 B)  ping-pong: reports half-RTT latency
 *     (min, mean, median, max) over 10 warmup + 50 timed iterations.
 *   - Large payload (default 4 MiB) ping-pong: reports bidirectional
 *     goodput (2 * large_bytes / RTT, GiB/s) over 10 warmup + 50 timed
 *     iterations.
 *
 * MPI is used only as the control plane:
 *   - Rank 0 gathers all endpoint IP addresses (ib0) and placement info.
 *   - Pair scheduling is serialised (same sequential i,j loop as MPI benchmark).
 *   - MPI_Bcast broadcasts the active pair; all other ranks wait at a barrier.
 *
 * Each active pair:
 *   - Rank i acts as TCP server (bind + listen + accept).
 *   - Rank j acts as TCP client (connect).
 *   - One TCP connection is opened per pair, reused for both phases, then closed.
 *   - TCP_NODELAY is set on both ends to suppress Nagle batching.
 *
 * CSV columns are a superset of mpi_pair_latency.csv, with added fields:
 *   ip_i, ip_j, connect_us, tcp_nodelay
 *
 * IP selection:
 *   Each rank discovers its ib0 address via getifaddrs(). If ib0 is absent,
 *   the benchmark falls back to any non-loopback IPv4 address and logs a
 *   warning. If no non-loopback address is found, it aborts.
 *
 * Port allocation:
 *   base_port + rank_i * max_ranks + rank_j (unique per ordered pair).
 *   Default base_port: 50000. This stays well above ephemeral-port range
 *   and avoids collisions across simultaneously active pairs (there is only
 *   ever one active pair at a time, but port reuse would require SO_REUSEADDR
 *   which we set anyway).
 *
 * Bandwidth unit note:
 *   Reported as GiB/s (binary), computed as
 *   (2 * large_bytes) / (1024^3) / elapsed_s — identical to mpi_pair_latency.
 *   The CSV column is named bandwidth_median_gibs / bandwidth_mean_gibs
 *   (not gbps) to avoid the existing ambiguity in the MPI CSV.
 *
 * Build: CMake adds target tcp_pair_latency, linked against MPI::MPI_CXX only.
 *        No CUDA dependency.
 */

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <mpi.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sched.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int    DEFAULT_BASE_PORT  = 50000;
static constexpr int    MAX_RANKS          = 4096;   // port-space ceiling
static constexpr size_t DEFAULT_SMALL      = 64;
static constexpr size_t DEFAULT_LARGE      = 4 * 1024 * 1024; // 4 MiB
static constexpr int    DEFAULT_WARMUP     = 10;
static constexpr int    DEFAULT_ITERS      = 50;
static constexpr int    LISTEN_BACKLOG     = 4;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string get_affinity_string() {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    if (sched_getaffinity(0, sizeof(cpu_set_t), &cpuset) != 0)
        return "unknown";
    std::ostringstream oss;
    bool first = true;
    int  count = 0;
    for (int i = 0; i < CPU_SETSIZE; ++i) {
        if (CPU_ISSET(i, &cpuset)) {
            if (!first) oss << ",";
            oss << i;
            first = false;
            ++count;
        }
    }
    if (count == CPU_SETSIZE) return "unbound(all)";
    return oss.str();
}

/* Returns the IPv4 address of interface iface_name (e.g. "ib0").
 * Falls back to any non-loopback IPv4 if iface not found; aborts if none. */
static std::string get_iface_ip(const std::string& iface_name, int rank) {
    struct ifaddrs* ifap = nullptr;
    if (getifaddrs(&ifap) != 0) {
        std::cerr << "[rank " << rank << "] getifaddrs failed: "
                  << strerror(errno) << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::string preferred;
    std::string fallback;

    for (struct ifaddrs* ifa = ifap; ifa != nullptr; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_INET) continue;
        char buf[INET_ADDRSTRLEN];
        auto* sin = reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr);
        inet_ntop(AF_INET, &sin->sin_addr, buf, sizeof(buf));
        std::string ip(buf);
        if (ip == "127.0.0.1") continue;                    // skip loopback
        if (std::string(ifa->ifa_name) == iface_name) {
            preferred = ip;
            break;
        }
        if (fallback.empty()) fallback = ip;
    }
    freeifaddrs(ifap);

    if (!preferred.empty()) return preferred;
    if (!fallback.empty()) {
        if (rank == 0)
            std::cerr << "[rank " << rank << "] WARNING: interface '"
                      << iface_name << "' not found; falling back to "
                      << fallback << "\n";
        return fallback;
    }
    std::cerr << "[rank " << rank << "] FATAL: no non-loopback IPv4 address found\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
    return "";
}

/* Fully reliable send: loops until all bytes are written. */
static bool send_all(int fd, const void* buf, size_t len) {
    const char* p = static_cast<const char*>(buf);
    while (len > 0) {
        ssize_t n = ::send(fd, p, len, MSG_NOSIGNAL);
        if (n <= 0) return false;
        p   += n;
        len -= static_cast<size_t>(n);
    }
    return true;
}

/* Fully reliable recv: loops until all bytes are read. */
static bool recv_all(int fd, void* buf, size_t len) {
    char* p = static_cast<char*>(buf);
    while (len > 0) {
        ssize_t n = ::recv(fd, p, len, MSG_WAITALL);
        if (n <= 0) return false;
        p   += n;
        len -= static_cast<size_t>(n);
    }
    return true;
}

/* Set TCP_NODELAY on fd. Returns 0 on success. */
static int set_nodelay(int fd) {
    int one = 1;
    return setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  -o, --output <file>        CSV output (default: tcp_pair_latency.csv)\n"
              << "  -s, --small-bytes <bytes>  Latency payload bytes (default: " << DEFAULT_SMALL << ")\n"
              << "  -l, --large-bytes <bytes>  Bandwidth payload bytes (default: 4194304 = 4 MiB)\n"
              << "  -w, --warmup <count>       Warmup iterations per pair (default: " << DEFAULT_WARMUP << ")\n"
              << "  -i, --iters <count>        Measured iterations per pair (default: " << DEFAULT_ITERS << ")\n"
              << "  -p, --base-port <port>     Base port for TCP listeners (default: " << DEFAULT_BASE_PORT << ")\n"
              << "  --iface <name>             Network interface (default: ib0)\n"
              << "  -h, --help                 Show this help\n";
}

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------
static double median_of(std::vector<double>& v) {
    size_t n = v.size();
    assert(n > 0);
    std::sort(v.begin(), v.end());
    return (n % 2 == 0) ? 0.5 * (v[n/2-1] + v[n/2]) : v[n/2];
}

static double mean_of(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

// ---------------------------------------------------------------------------
// Per-pair TCP measurement
// ---------------------------------------------------------------------------
struct PairStats {
    double connect_us       = 0.0;
    double lat_min_us       = 0.0;
    double lat_mean_us      = 0.0;
    double lat_median_us    = 0.0;
    double lat_max_us       = 0.0;
    double bw_median_gibs   = 0.0;
    double bw_mean_gibs     = 0.0;
};

/* rank_i is the server (listener), rank_j is the client (connector).
 * Called only by rank i and rank j; returns stats filled only for rank i. */
static PairStats measure_pair(
    int my_rank, int rank_i, int rank_j,
    const std::string& ip_i, int port,
    size_t small_bytes, size_t large_bytes,
    int warmup, int iters,
    std::vector<char>& sbuf_small, std::vector<char>& rbuf_small,
    std::vector<char>& sbuf_large, std::vector<char>& rbuf_large)
{
    PairStats stats;
    int conn_fd = -1;   // the connected socket used for measurements

    // ---- Connection setup ----
    auto t_conn0 = std::chrono::high_resolution_clock::now();

    if (my_rank == rank_i) {
        // Server: bind, listen, accept
        int srv = socket(AF_INET, SOCK_STREAM, 0);
        if (srv < 0) { perror("socket"); MPI_Abort(MPI_COMM_WORLD, 1); }
        int opt = 1;
        setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_port        = htons(static_cast<uint16_t>(port));
        addr.sin_addr.s_addr = INADDR_ANY;   // listen on all interfaces

        if (bind(srv, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
            std::cerr << "[rank " << my_rank << "] bind port " << port
                      << " failed: " << strerror(errno) << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        listen(srv, LISTEN_BACKLOG);

        conn_fd = accept(srv, nullptr, nullptr);
        if (conn_fd < 0) { perror("accept"); MPI_Abort(MPI_COMM_WORLD, 1); }
        close(srv);
        set_nodelay(conn_fd);
    } else {
        // Client: connect to rank_i's ib0 address
        int attempts = 0;
        while (true) {
            conn_fd = socket(AF_INET, SOCK_STREAM, 0);
            if (conn_fd < 0) { perror("socket"); MPI_Abort(MPI_COMM_WORLD, 1); }
            set_nodelay(conn_fd);

            struct sockaddr_in addr{};
            addr.sin_family = AF_INET;
            addr.sin_port   = htons(static_cast<uint16_t>(port));
            if (inet_pton(AF_INET, ip_i.c_str(), &addr.sin_addr) <= 0) {
                std::cerr << "[rank " << my_rank << "] inet_pton failed for "
                          << ip_i << "\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (connect(conn_fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) == 0)
                break;
            close(conn_fd);
            conn_fd = -1;
            if (++attempts > 100) {
                std::cerr << "[rank " << my_rank << "] connect to " << ip_i
                          << ":" << port << " failed after 100 attempts: "
                          << strerror(errno) << "\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            usleep(5000); // 5 ms back-off
        }
    }

    auto t_conn1 = std::chrono::high_resolution_clock::now();
    stats.connect_us = std::chrono::duration<double, std::micro>(t_conn1 - t_conn0).count();

    // ---- Phase 1: Small-payload latency (ping-pong, rank_i initiates) ----
    // Warmup
    for (int w = 0; w < warmup; ++w) {
        if (my_rank == rank_i) {
            send_all(conn_fd, sbuf_small.data(), small_bytes);
            recv_all(conn_fd, rbuf_small.data(), small_bytes);
        } else {
            recv_all(conn_fd, rbuf_small.data(), small_bytes);
            send_all(conn_fd, sbuf_small.data(), small_bytes);
        }
    }

    // Timed (only rank_i measures)
    std::vector<double> latencies_us(iters);
    for (int it = 0; it < iters; ++it) {
        if (my_rank == rank_i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            send_all(conn_fd, sbuf_small.data(), small_bytes);
            recv_all(conn_fd, rbuf_small.data(), small_bytes);
            auto t1 = std::chrono::high_resolution_clock::now();
            latencies_us[it] = std::chrono::duration<double, std::micro>(t1 - t0).count() / 2.0;
        } else {
            recv_all(conn_fd, rbuf_small.data(), small_bytes);
            send_all(conn_fd, sbuf_small.data(), small_bytes);
        }
    }

    if (my_rank == rank_i) {
        std::sort(latencies_us.begin(), latencies_us.end());
        stats.lat_min_us    = latencies_us.front();
        stats.lat_max_us    = latencies_us.back();
        stats.lat_mean_us   = mean_of(latencies_us);
        stats.lat_median_us = median_of(latencies_us);
    }

    // ---- Phase 2: Large-payload bandwidth (ping-pong, rank_i initiates) ----
    // Warmup
    for (int w = 0; w < warmup; ++w) {
        if (my_rank == rank_i) {
            send_all(conn_fd, sbuf_large.data(), large_bytes);
            recv_all(conn_fd, rbuf_large.data(), large_bytes);
        } else {
            recv_all(conn_fd, rbuf_large.data(), large_bytes);
            send_all(conn_fd, sbuf_large.data(), large_bytes);
        }
    }

    // Timed
    std::vector<double> bws_gibs(iters);
    constexpr double GiB = 1024.0 * 1024.0 * 1024.0;
    for (int it = 0; it < iters; ++it) {
        if (my_rank == rank_i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            send_all(conn_fd, sbuf_large.data(), large_bytes);
            recv_all(conn_fd, rbuf_large.data(), large_bytes);
            auto t1 = std::chrono::high_resolution_clock::now();
            double sec   = std::chrono::duration<double>(t1 - t0).count();
            bws_gibs[it] = (2.0 * static_cast<double>(large_bytes) / GiB) / sec;
        } else {
            recv_all(conn_fd, rbuf_large.data(), large_bytes);
            send_all(conn_fd, sbuf_large.data(), large_bytes);
        }
    }

    if (my_rank == rank_i) {
        stats.bw_median_gibs = median_of(bws_gibs);
        stats.bw_mean_gibs   = mean_of(bws_gibs);
    }

    close(conn_fd);
    return stats;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size > MAX_RANKS) {
        if (rank == 0)
            std::cerr << "FATAL: more ranks (" << size
                      << ") than MAX_RANKS (" << MAX_RANKS << ")\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // ---- Argument parsing ----
    std::string output_file = "tcp_pair_latency.csv";
    std::string iface       = "ib0";
    size_t small_bytes      = DEFAULT_SMALL;
    size_t large_bytes      = DEFAULT_LARGE;
    int    warmup           = DEFAULT_WARMUP;
    int    iters            = DEFAULT_ITERS;
    int    base_port        = DEFAULT_BASE_PORT;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-o" || arg == "--output")     && i+1 < argc) output_file  = argv[++i];
        else if ((arg == "-s" || arg == "--small-bytes") && i+1 < argc) small_bytes = std::stoull(argv[++i]);
        else if ((arg == "-l" || arg == "--large-bytes") && i+1 < argc) large_bytes = std::stoull(argv[++i]);
        else if ((arg == "-w" || arg == "--warmup")      && i+1 < argc) warmup      = std::stoi(argv[++i]);
        else if ((arg == "-i" || arg == "--iters")       && i+1 < argc) iters       = std::stoi(argv[++i]);
        else if ((arg == "-p" || arg == "--base-port")   && i+1 < argc) base_port   = std::stoi(argv[++i]);
        else if (arg == "--iface"                        && i+1 < argc) iface       = argv[++i];
        else if (arg == "-h" || arg == "--help") {
            if (rank == 0) print_usage(argv[0]);
            MPI_Finalize();
            return 0;
        }
    }

    // ---- Gather per-rank placement and IP info ----
    char hostname[256] = {0};
    gethostname(hostname, sizeof(hostname) - 1);
    int  current_cpu   = sched_getcpu();
    std::string affinity_str = get_affinity_string();
    std::string my_ip        = get_iface_ip(iface, rank);

    // Gather hostnames
    std::vector<char> host_buf(size * 256, 0);
    MPI_Gather(hostname, 256, MPI_CHAR, host_buf.data(), 256, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Gather CPUs
    std::vector<int> cpu_buf(size, 0);
    MPI_Gather(&current_cpu, 1, MPI_INT, cpu_buf.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Gather IPs (fixed 64 B per rank)
    constexpr int IP_LEN = 64;
    char ip_fixed[IP_LEN] = {0};
    strncpy(ip_fixed, my_ip.c_str(), IP_LEN - 1);
    std::vector<char> ip_buf(size * IP_LEN, 0);
    MPI_Gather(ip_fixed, IP_LEN, MPI_CHAR, ip_buf.data(), IP_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Broadcast the full IP table to all ranks (needed for port/connect logic)
    MPI_Bcast(ip_buf.data(), size * IP_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Gather affinity strings (variable length)
    int aff_len = static_cast<int>(affinity_str.size());
    std::vector<int> aff_len_buf(size, 0);
    MPI_Gather(&aff_len, 1, MPI_INT, aff_len_buf.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int>  aff_disps(size, 0);
    int total_aff = 0;
    std::vector<char> aff_str_buf;
    if (rank == 0) {
        for (int r = 0; r < size; ++r) { aff_disps[r] = total_aff; total_aff += aff_len_buf[r]; }
        aff_str_buf.resize(total_aff);
    }
    MPI_Gatherv(affinity_str.data(), aff_len, MPI_CHAR,
                aff_str_buf.data(), aff_len_buf.data(), aff_disps.data(), MPI_CHAR,
                0, MPI_COMM_WORLD);

    // Helper: get IP string for any rank
    auto rank_ip = [&](int r) -> std::string {
        return std::string(&ip_buf[r * IP_LEN]);
    };
    auto rank_host = [&](int r) -> std::string {
        if (rank == 0) return std::string(&host_buf[r * 256]);
        return "";
    };
    auto rank_aff = [&](int r) -> std::string {
        if (rank == 0) return std::string(&aff_str_buf[aff_disps[r]], aff_len_buf[r]);
        return "";
    };

    if (rank == 0) {
        std::cout << "=== TCP/IP (IPoIB) Pair Latency & Bandwidth Benchmark ===\n"
                  << "Ranks: " << size
                  << "  Interface: " << iface
                  << "  Small: " << small_bytes << " B"
                  << "  Large: " << (large_bytes / (1024*1024)) << " MiB\n"
                  << "Warmup: " << warmup << "  Iters: " << iters << "\n"
                  << "Starting pair-by-pair measurements (" << (size*(size-1)/2) << " pairs)...\n"
                  << std::flush;
    }

    // ---- Payload buffers ----
    std::vector<char> sbuf_small(small_bytes, 'a'), rbuf_small(small_bytes, 0);
    std::vector<char> sbuf_large(large_bytes, 'b'), rbuf_large(large_bytes, 0);

    // ---- CSV header (rank 0 only) ----
    std::ofstream csv;
    if (rank == 0) {
        csv.open(output_file);
        csv << "rank_i,host_i,cpu_i,affinity_i,ip_i"
            << ",rank_j,host_j,cpu_j,affinity_j,ip_j"
            << ",iface,small_bytes,large_bytes,warmup,iters"
            << ",connect_us"
            << ",latency_min_us,latency_mean_us,latency_median_us,latency_max_us"
            << ",bandwidth_median_gibs,bandwidth_mean_gibs\n";
    }

    // ---- Sequential pair loop ----
    // stats[0..6]: connect_us, lat_min, lat_mean, lat_median, lat_max,
    //              bw_median, bw_mean
    double stats[7] = {0.0};

    int active_pair[2] = {-1, -1};
    for (int i = 0; i < size; ++i) {
        for (int j = i + 1; j < size; ++j) {
            active_pair[0] = i;
            active_pair[1] = j;
            MPI_Bcast(active_pair, 2, MPI_INT, 0, MPI_COMM_WORLD);

            if (rank == i || rank == j) {
                int port = base_port + i * MAX_RANKS + j;
                PairStats ps = measure_pair(
                    rank, i, j,
                    rank_ip(i), port,
                    small_bytes, large_bytes,
                    warmup, iters,
                    sbuf_small, rbuf_small,
                    sbuf_large, rbuf_large);

                if (rank == i) {
                    stats[0] = ps.connect_us;
                    stats[1] = ps.lat_min_us;
                    stats[2] = ps.lat_mean_us;
                    stats[3] = ps.lat_median_us;
                    stats[4] = ps.lat_max_us;
                    stats[5] = ps.bw_median_gibs;
                    stats[6] = ps.bw_mean_gibs;
                    if (i != 0)
                        MPI_Send(stats, 7, MPI_DOUBLE, 0, 999, MPI_COMM_WORLD);
                }
            }

            if (rank == 0) {
                if (i != 0)
                    MPI_Recv(stats, 7, MPI_DOUBLE, i, 999, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                csv << i << ",\"" << rank_host(i) << "\","
                    << cpu_buf[i] << ",\"" << rank_aff(i) << "\","
                    << rank_ip(i)
                    << "," << j << ",\"" << rank_host(j) << "\","
                    << cpu_buf[j] << ",\"" << rank_aff(j) << "\","
                    << rank_ip(j)
                    << "," << iface
                    << "," << small_bytes << "," << large_bytes
                    << "," << warmup << "," << iters
                    << "," << stats[0]   // connect_us
                    << "," << stats[1]   // lat_min
                    << "," << stats[2]   // lat_mean
                    << "," << stats[3]   // lat_median
                    << "," << stats[4]   // lat_max
                    << "," << stats[5]   // bw_median
                    << "," << stats[6]   // bw_mean
                    << "\n";
            }

            // All ranks synchronise before moving to the next pair.
            // This ensures the server has closed its socket before the next
            // pair potentially reuses a port (SO_REUSEADDR should handle this,
            // but the barrier makes it deterministic).
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        csv.close();
        std::cout << "Done. Results written to " << output_file << "\n";
    }

    MPI_Finalize();
    return 0;
}
