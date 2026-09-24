#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/*
 * affinity_test/tcp_duplex_latency.cpp
 * ====================================
 * TCP/IP (IPoIB) full-duplex (simultaneous bidirectional) rank-to-rank latency and bandwidth benchmark.
 *
 * Measurement protocol:
 *   - Small payload (default 64 B)  duplex exchange: both ranks send and receive simultaneously.
 *   - Large payload (default 4 MiB) duplex bandwidth: concurrent sender and receiver threads per rank
 *     stream large_bytes bidirectionally without TCP window deadlock.
 *     Aggregate goodput = (2 * large_bytes) / max(elapsed_i, elapsed_j) in GiB/s.
 *
 * MPI is used only as the control plane (ephemeral port exchange, barrier, state gathering).
 *
 * Checkpointing / resumable execution:
 *   - Same state-file and auto-rescheduling protocol as tcp_pair_latency.cpp.
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
#include <cassert>
#include <chrono>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Constants & Defaults
// ---------------------------------------------------------------------------
static constexpr size_t DEFAULT_SMALL       = 64;
static constexpr size_t DEFAULT_LARGE       = 4 * 1024 * 1024;
static constexpr int    DEFAULT_WARMUP      = 10;
static constexpr int    DEFAULT_ITERS       = 50;
static constexpr int    LISTEN_BACKLOG      = 4;
static constexpr double CHECKPOINT_SECS     = 30.0 * 60.0;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
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
            ++count;
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
    if (!fallback.empty()) {
        if (rank == 0) std::cerr << "[rank " << rank << "] WARNING: '" << iface_name << "' not found; falling back to " << fallback << "\n";
        return fallback;
    }
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

static double median_of(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2 == 0) ? 0.5 * (v[n / 2 - 1] + v[n / 2]) : v[n / 2];
}

static double mean_of(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

static void write_state(const std::string& path, int next_i, int next_j) {
    std::ofstream f(path, std::ios::trunc);
    f << next_i << " " << next_j << "\n";
}

static bool read_state(const std::string& path, int& next_i, int& next_j) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    return bool(f >> next_i >> next_j);
}

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  -o, --output <file>        CSV output (default: tcp_duplex_latency.csv)\n"
              << "  -s, --small-bytes <bytes>  Latency payload (default: 64)\n"
              << "  -l, --large-bytes <bytes>  Bandwidth payload (default: 4194304)\n"
              << "  -w, --warmup <count>       Warmup iters per pair (default: 10)\n"
              << "  -i, --iters <count>        Timed iters per pair (default: 50)\n"
              << "  --iface <name>             Network interface (default: ib0)\n"
              << "  --state-file <file>        Checkpoint state file (default: tcp_duplex_latency.state)\n"
              << "  --resume                   Resume from state file if present\n"
              << "  --sbatch-script <file>     Sbatch script to re-submit for continuation\n"
              << "  --checkpoint-mins <m>      Resubmit after this many minutes (default: 30)\n"
              << "  -h, --help                 Show help\n";
}

// ---------------------------------------------------------------------------
// Per-Pair Duplex Measurement
// ---------------------------------------------------------------------------
struct PairStats {
    double connect_us     = 0;
    double lat_min_us     = 0, lat_mean_us = 0, lat_median_us = 0, lat_max_us = 0;
    double bw_median_gibs = 0, bw_mean_gibs = 0;
};

static PairStats measure_tcp_duplex(
    int my_rank, int ri, int rj,
    const std::string& ip_i,
    size_t small_bytes, size_t large_bytes, int warmup, int iters,
    std::vector<char>& ss, std::vector<char>& rs,
    std::vector<char>& sl, std::vector<char>& rl)
{
    PairStats stats;
    int conn_fd = -1;

    auto t0_conn = std::chrono::high_resolution_clock::now();

    if (my_rank == ri) {
        int srv = socket(AF_INET, SOCK_STREAM, 0);
        int opt = 1;
        setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        struct sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_port        = htons(0); // ephemeral
        addr.sin_addr.s_addr = INADDR_ANY;
        if (bind(srv, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            std::cerr << "[rank " << my_rank << "] bind failed: " << strerror(errno) << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        struct sockaddr_in bound{};
        socklen_t blen = sizeof(bound);
        if (getsockname(srv, (struct sockaddr*)&bound, &blen) < 0) {
            std::cerr << "[rank " << my_rank << "] getsockname failed: " << strerror(errno) << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int actual_port = ntohs(bound.sin_port);

        MPI_Send(&actual_port, 1, MPI_INT, rj, 42, MPI_COMM_WORLD);

        listen(srv, LISTEN_BACKLOG);
        conn_fd = accept(srv, nullptr, nullptr);
        close(srv);
        set_nodelay(conn_fd);
        set_sock_buffers(conn_fd);
    } else {
        int port = -1;
        MPI_Recv(&port, 1, MPI_INT, ri, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (port <= 0) {
            std::cerr << "[rank " << my_rank << "] invalid port from rank " << ri << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int attempt = 0; ; ++attempt) {
            conn_fd = socket(AF_INET, SOCK_STREAM, 0);
            set_nodelay(conn_fd);
            set_sock_buffers(conn_fd);
            struct sockaddr_in addr{};
            addr.sin_family = AF_INET;
            addr.sin_port   = htons(port);
            inet_pton(AF_INET, ip_i.c_str(), &addr.sin_addr);
            if (connect(conn_fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) break;
            close(conn_fd);
            conn_fd = -1;
            if (attempt >= 100) {
                std::cerr << "[rank " << my_rank << "] connect to " << ip_i << ":" << port << " failed\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            usleep(5000);
        }
    }

    stats.connect_us = std::chrono::duration<double, std::micro>(
        std::chrono::high_resolution_clock::now() - t0_conn).count();

    // --- Phase 1: Small payload duplex exchange ---
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

    // --- Phase 2: Large payload duplex bandwidth ---
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
    return stats;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string output_file   = "tcp_duplex_latency.csv";
    std::string state_file    = "tcp_duplex_latency.state";
    std::string iface         = "ib0";
    std::string sbatch_script = "";
    size_t small_bytes        = DEFAULT_SMALL;
    size_t large_bytes        = DEFAULT_LARGE;
    int    warmup             = DEFAULT_WARMUP;
    int    iters              = DEFAULT_ITERS;
    bool   do_resume          = false;
    double checkpoint_secs    = CHECKPOINT_SECS;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      ((a == "-o" || a == "--output")          && i + 1 < argc) output_file     = argv[++i];
        else if ((a == "-s" || a == "--small-bytes")     && i + 1 < argc) small_bytes     = std::stoull(argv[++i]);
        else if ((a == "-l" || a == "--large-bytes")     && i + 1 < argc) large_bytes     = std::stoull(argv[++i]);
        else if ((a == "-w" || a == "--warmup")          && i + 1 < argc) warmup          = std::stoi(argv[++i]);
        else if ((a == "-i" || a == "--iters")           && i + 1 < argc) iters           = std::stoi(argv[++i]);
        else if (a == "--iface"                          && i + 1 < argc) iface           = argv[++i];
        else if (a == "--state-file"                     && i + 1 < argc) state_file      = argv[++i];
        else if (a == "--sbatch-script"                  && i + 1 < argc) sbatch_script   = argv[++i];
        else if (a == "--checkpoint-mins"                && i + 1 < argc) checkpoint_secs = std::stod(argv[++i]) * 60.0;
        else if (a == "--resume")                                         do_resume       = true;
        else if (a == "-h" || a == "--help") { if (rank == 0) print_usage(argv[0]); MPI_Finalize(); return 0; }
    }

    int resume_i = 0, resume_j = 1;
    bool appending = false;
    if (do_resume && rank == 0) {
        if (read_state(state_file, resume_i, resume_j)) {
            std::cout << "Resuming from pair (" << resume_i << "," << resume_j << ")\n";
            appending = true;
        } else {
            std::cout << "No state file found; starting from beginning.\n";
        }
    }

    int resume_buf[3] = {resume_i, resume_j, appending ? 1 : 0};
    MPI_Bcast(resume_buf, 3, MPI_INT, 0, MPI_COMM_WORLD);
    resume_i  = resume_buf[0];
    resume_j  = resume_buf[1];
    appending = resume_buf[2] != 0;

    char hostname[256] = {}; gethostname(hostname, sizeof(hostname) - 1);
    int  current_cpu   = sched_getcpu();
    std::string affinity_str = get_affinity_string();
    std::string my_ip        = get_iface_ip(iface, rank);

    std::vector<char> host_buf(size * 256, 0);
    MPI_Gather(hostname, 256, MPI_CHAR, host_buf.data(), 256, MPI_CHAR, 0, MPI_COMM_WORLD);

    std::vector<int> cpu_buf(size, 0);
    MPI_Gather(&current_cpu, 1, MPI_INT, cpu_buf.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    constexpr int IP_LEN = 64;
    char ip_fixed[IP_LEN] = {}; strncpy(ip_fixed, my_ip.c_str(), IP_LEN - 1);
    std::vector<char> ip_buf(size * IP_LEN, 0);
    MPI_Gather(ip_fixed, IP_LEN, MPI_CHAR, ip_buf.data(), IP_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(ip_buf.data(), size * IP_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);

    int aff_len = static_cast<int>(affinity_str.size());
    std::vector<int> aff_len_buf(size, 0);
    MPI_Gather(&aff_len, 1, MPI_INT, aff_len_buf.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<int>  aff_disps(size, 0); int total_aff = 0;
    std::vector<char> aff_str_buf;
    if (rank == 0) {
        for (int r = 0; r < size; ++r) { aff_disps[r] = total_aff; total_aff += aff_len_buf[r]; }
        aff_str_buf.resize(total_aff);
    }
    MPI_Gatherv(affinity_str.data(), aff_len, MPI_CHAR,
                aff_str_buf.data(), aff_len_buf.data(), aff_disps.data(), MPI_CHAR,
                0, MPI_COMM_WORLD);

    auto rank_ip   = [&](int r) { return std::string(&ip_buf[r * IP_LEN]); };
    auto rank_host = [&](int r) { return rank == 0 ? std::string(&host_buf[r * 256]) : std::string(); };
    auto rank_aff  = [&](int r) { return rank == 0 ? std::string(&aff_str_buf[aff_disps[r]], aff_len_buf[r]) : std::string(); };

    int total_pairs = size * (size - 1) / 2;
    if (rank == 0) {
        std::cout << "=== TCP/IP (IPoIB) Full-Duplex Pair Latency & Bandwidth Benchmark ===\n"
                  << "Ranks=" << size << "  iface=" << iface
                  << "  small=" << small_bytes << " B  large=" << (large_bytes / (1024 * 1024)) << " MiB"
                  << "  warmup=" << warmup << "  iters=" << iters << "\n"
                  << "Total pairs=" << total_pairs
                  << "  resuming_from=(" << resume_i << "," << resume_j << ")\n"
                  << "Checkpoint after " << (checkpoint_secs / 60.0) << " min\n"
                  << std::flush;
    }

    std::vector<char> ss(small_bytes, 'a'), rs(small_bytes, 0);
    std::vector<char> sl(large_bytes, 'b'), rl(large_bytes, 0);

    std::ofstream csv;
    if (rank == 0) {
        csv.open(output_file, appending ? std::ios::app : std::ios::out);
        if (!appending) {
            csv << "rank_i,host_i,cpu_i,affinity_i,ip_i"
                << ",rank_j,host_j,cpu_j,affinity_j,ip_j"
                << ",iface,small_bytes,large_bytes,warmup,iters"
                << ",connect_us"
                << ",latency_min_us,latency_mean_us,latency_median_us,latency_max_us"
                << ",bandwidth_median_gibs,bandwidth_mean_gibs\n";
        }
    }

    auto job_start = std::chrono::steady_clock::now();
    double stats[7] = {};
    int active_pair[2] = {-1, -1};
    bool submitted_continuation = false;

    for (int i = 0; i < size && !submitted_continuation; ++i) {
        for (int j = i + 1; j < size && !submitted_continuation; ++j) {
            if (i < resume_i || (i == resume_i && j < resume_j)) {
                active_pair[0] = -1; active_pair[1] = -1;
                MPI_Bcast(active_pair, 2, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Barrier(MPI_COMM_WORLD);
                continue;
            }

            active_pair[0] = i; active_pair[1] = j;
            MPI_Bcast(active_pair, 2, MPI_INT, 0, MPI_COMM_WORLD);

            if (rank == i || rank == j) {
                PairStats ps = measure_tcp_duplex(rank, i, j, rank_ip(i),
                                                  small_bytes, large_bytes, warmup, iters,
                                                  ss, rs, sl, rl);
                if (rank == i) {
                    stats[0] = ps.connect_us; stats[1] = ps.lat_min_us; stats[2] = ps.lat_mean_us;
                    stats[3] = ps.lat_median_us; stats[4] = ps.lat_max_us;
                    stats[5] = ps.bw_median_gibs; stats[6] = ps.bw_mean_gibs;
                    if (i != 0) MPI_Send(stats, 7, MPI_DOUBLE, 0, 999, MPI_COMM_WORLD);
                }
            }

            if (rank == 0) {
                if (i != 0) MPI_Recv(stats, 7, MPI_DOUBLE, i, 999, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                csv << i << ",\"" << rank_host(i) << "\"," << cpu_buf[i] << ",\"" << rank_aff(i) << "\"," << rank_ip(i)
                    << "," << j << ",\"" << rank_host(j) << "\"," << cpu_buf[j] << ",\"" << rank_aff(j) << "\"," << rank_ip(j)
                    << "," << iface << "," << small_bytes << "," << large_bytes << "," << warmup << "," << iters
                    << "," << stats[0] << "," << stats[1] << "," << stats[2] << "," << stats[3] << "," << stats[4]
                    << "," << stats[5] << "," << stats[6] << "\n";
                csv.flush();
            }

            MPI_Barrier(MPI_COMM_WORLD);

            int should_stop = 0;
            if (rank == 0) {
                double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - job_start).count();

                int next_i = i, next_j = j + 1;
                if (next_j >= size) { next_i = i + 1; next_j = next_i + 1; }
                bool more_pairs = (next_i < size - 1) || (next_i == size - 1 && next_j < size);

                if (elapsed >= checkpoint_secs && more_pairs) {
                    write_state(state_file, next_i, next_j);
                    std::cout << "\n[checkpoint] " << (elapsed / 60.0) << " min elapsed after pair ("
                              << i << "," << j << "). Next=(" << next_i << "," << next_j << ").\n";

                    if (!sbatch_script.empty()) {
                        const char* jid_env = getenv("SLURM_JOB_ID");
                        std::string dep = jid_env ? std::string("--dependency=afterok:") + jid_env : "";
                        std::string cmd = "sbatch " + dep + " " + sbatch_script;
                        std::cout << "[checkpoint] Submitting: " << cmd << "\n" << std::flush;
                        int ret = system(cmd.c_str());
                        if (ret != 0) std::cerr << "[checkpoint] WARNING: sbatch returned " << ret << "\n";
                    }
                    should_stop = 1;
                } else if (!more_pairs) {
                    std::remove(state_file.c_str());
                }
            }

            MPI_Bcast(&should_stop, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (should_stop) submitted_continuation = true;
        }
    }

    if (rank == 0) {
        csv.close();
        if (submitted_continuation)
            std::cout << "Job exiting after checkpoint. Continuation job submitted.\n";
        else
            std::cout << "All " << total_pairs << " pairs complete. Results in " << output_file << "\n";
    }

    MPI_Finalize();
    return 0;
}
