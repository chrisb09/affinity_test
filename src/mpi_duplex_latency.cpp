#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/*
 * affinity_test/mpi_duplex_latency.cpp
 * ====================================
 * MPI full-duplex (simultaneous bidirectional) rank-to-rank latency and bandwidth benchmark.
 *
 * Measurement protocol:
 *   - Small payload (default 64 B)  duplex exchange: both ranks post MPI_Irecv + MPI_Isend
 *     simultaneously and wait on MPI_Waitall. Measures mutual exchange completion time (µs)
 *     over warmup + timed iterations.
 *   - Large payload (default 4 MiB) duplex bandwidth: both ranks stream large_bytes simultaneously.
 *     Aggregate bandwidth = (2 * large_bytes) / max(elapsed_i, elapsed_j) in GiB/s.
 *
 * Checkpointing / resumable execution:
 *   - Follows the same robust checkpointing protocol as tcp_pair_latency.cpp.
 *   - Rank 0 writes next (i, j) to state file and reschedules via Slurm dependency if elapsed
 *     wall time exceeds --checkpoint-mins.
 */

#include <mpi.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Constants & Defaults
// ---------------------------------------------------------------------------
static constexpr size_t DEFAULT_SMALL       = 64;
static constexpr size_t DEFAULT_LARGE       = 4 * 1024 * 1024; // 4 MiB
static constexpr int    DEFAULT_WARMUP      = 10;
static constexpr int    DEFAULT_ITERS       = 50;
static constexpr double CHECKPOINT_SECS     = 30.0 * 60.0;     // 30 minutes

struct RankInfo {
    int rank{-1};
    char hostname[256]{0};
    int current_cpu{-1};
    std::string affinity_str;
};

struct DuplexStats {
    double lat_min_us     = 0;
    double lat_mean_us    = 0;
    double lat_median_us  = 0;
    double lat_max_us     = 0;
    double bw_median_gibs = 0;
    double bw_mean_gibs   = 0;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static std::string get_affinity_string() {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    if (sched_getaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
        return "unknown";
    }

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
              << "Options:\n"
              << "  -o, --output <file>         CSV output filename (default: cpu_duplex_latency.csv)\n"
              << "  -s, --small-bytes <bytes>   Small payload for latency in bytes (default: 64)\n"
              << "  -l, --large-bytes <bytes>   Large payload for bandwidth in bytes (default: 4194304 = 4MiB)\n"
              << "  -w, --warmup <count>        Warmup iterations per pair (default: 10)\n"
              << "  -i, --iters <count>         Measured iterations per pair (default: 50)\n"
              << "  --state-file <file>         Checkpoint state file (default: cpu_duplex_latency.state)\n"
              << "  --resume                    Resume from state file if present\n"
              << "  --sbatch-script <file>      Sbatch script to re-submit for continuation\n"
              << "  --checkpoint-mins <m>       Resubmit after this many minutes (default: 30)\n"
              << "  -h, --help                  Show this help message\n";
}

// ---------------------------------------------------------------------------
// Per-Pair Duplex Measurement
// ---------------------------------------------------------------------------
static DuplexStats measure_mpi_duplex(
    int my_rank, int ri, int rj,
    size_t small_bytes, size_t large_bytes, int warmup, int iters,
    std::vector<char>& small_send, std::vector<char>& small_recv,
    std::vector<char>& large_send, std::vector<char>& large_recv)
{
    DuplexStats stats;
    int peer = (my_rank == ri) ? rj : ri;

    // --- Phase 1: Small Payload Duplex Exchange Latency ---
    MPI_Request reqs[2];

    for (int w = 0; w < warmup; ++w) {
        MPI_Irecv(small_recv.data(), static_cast<int>(small_bytes), MPI_BYTE, peer, 100, MPI_COMM_WORLD, &reqs[0]);
        MPI_Isend(small_send.data(), static_cast<int>(small_bytes), MPI_BYTE, peer, 100, MPI_COMM_WORLD, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
    }

    std::vector<double> latencies_us(iters);
    for (int it = 0; it < iters; ++it) {
        auto t0 = std::chrono::high_resolution_clock::now();
        MPI_Irecv(small_recv.data(), static_cast<int>(small_bytes), MPI_BYTE, peer, 101, MPI_COMM_WORLD, &reqs[0]);
        MPI_Isend(small_send.data(), static_cast<int>(small_bytes), MPI_BYTE, peer, 101, MPI_COMM_WORLD, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
        auto t1 = std::chrono::high_resolution_clock::now();
        latencies_us[it] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    // --- Phase 2: Large Payload Duplex Bandwidth ---
    for (int w = 0; w < warmup; ++w) {
        MPI_Irecv(large_recv.data(), static_cast<int>(large_bytes), MPI_BYTE, peer, 200, MPI_COMM_WORLD, &reqs[0]);
        MPI_Isend(large_send.data(), static_cast<int>(large_bytes), MPI_BYTE, peer, 200, MPI_COMM_WORLD, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
    }

    constexpr double GiB = 1024.0 * 1024.0 * 1024.0;
    std::vector<double> bws_gibs(iters);
    for (int it = 0; it < iters; ++it) {
        auto t0 = std::chrono::high_resolution_clock::now();
        MPI_Irecv(large_recv.data(), static_cast<int>(large_bytes), MPI_BYTE, peer, 201, MPI_COMM_WORLD, &reqs[0]);
        MPI_Isend(large_send.data(), static_cast<int>(large_bytes), MPI_BYTE, peer, 201, MPI_COMM_WORLD, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
        auto t1 = std::chrono::high_resolution_clock::now();

        double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
        double peer_elapsed_sec = 0.0;

        // Exchange local elapsed times to compute aggregate bidirectional bandwidth over max(t_i, t_j)
        if (my_rank == ri) {
            MPI_Sendrecv(&elapsed_sec, 1, MPI_DOUBLE, rj, 301,
                         &peer_elapsed_sec, 1, MPI_DOUBLE, rj, 302,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            double max_sec = std::max(elapsed_sec, peer_elapsed_sec);
            // 2 transfers of large_bytes in full duplex
            bws_gibs[it] = (2.0 * static_cast<double>(large_bytes) / GiB) / max_sec;
        } else {
            MPI_Sendrecv(&elapsed_sec, 1, MPI_DOUBLE, ri, 302,
                         &peer_elapsed_sec, 1, MPI_DOUBLE, ri, 301,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    if (my_rank == ri) {
        std::sort(latencies_us.begin(), latencies_us.end());
        stats.lat_min_us     = latencies_us.front();
        stats.lat_max_us     = latencies_us.back();
        stats.lat_mean_us    = mean_of(latencies_us);
        stats.lat_median_us  = median_of(latencies_us);

        stats.bw_median_gibs = median_of(bws_gibs);
        stats.bw_mean_gibs   = mean_of(bws_gibs);
    }

    return stats;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string output_file   = "cpu_duplex_latency.csv";
    std::string state_file    = "cpu_duplex_latency.state";
    std::string sbatch_script = "";
    size_t small_bytes        = DEFAULT_SMALL;
    size_t large_bytes        = DEFAULT_LARGE;
    int warmup                = DEFAULT_WARMUP;
    int iters                 = DEFAULT_ITERS;
    bool do_resume            = false;
    double checkpoint_secs    = CHECKPOINT_SECS;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      ((arg == "-o" || arg == "--output")          && i + 1 < argc) output_file     = argv[++i];
        else if ((arg == "-s" || arg == "--small-bytes")     && i + 1 < argc) small_bytes     = std::stoull(argv[++i]);
        else if ((arg == "-l" || arg == "--large-bytes")     && i + 1 < argc) large_bytes     = std::stoull(argv[++i]);
        else if ((arg == "-w" || arg == "--warmup")          && i + 1 < argc) warmup          = std::stoi(argv[++i]);
        else if ((arg == "-i" || arg == "--iters")           && i + 1 < argc) iters           = std::stoi(argv[++i]);
        else if (arg == "--state-file"                       && i + 1 < argc) state_file      = argv[++i];
        else if (arg == "--sbatch-script"                    && i + 1 < argc) sbatch_script   = argv[++i];
        else if (arg == "--checkpoint-mins"                  && i + 1 < argc) checkpoint_secs = std::stod(argv[++i]) * 60.0;
        else if (arg == "--resume")                                           do_resume       = true;
        else if (arg == "-h" || arg == "--help") {
            if (rank == 0) print_usage(argv[0]);
            MPI_Finalize();
            return 0;
        }
    }

    // Determine resume point
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

    // Gather topology and placement info
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
    int total_aff_len = 0;
    std::vector<char> aff_str_buf;
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            aff_disps[r] = total_aff_len;
            total_aff_len += aff_len_buf[r];
        }
        aff_str_buf.resize(total_aff_len);
    }
    MPI_Gatherv(local_info.affinity_str.data(), aff_len, MPI_CHAR,
                aff_str_buf.data(), aff_len_buf.data(), aff_disps.data(), MPI_CHAR,
                0, MPI_COMM_WORLD);

    auto rank_host = [&](int r) { return rank == 0 ? std::string(&host_buf[r * 256]) : std::string(); };
    auto rank_aff  = [&](int r) { return rank == 0 ? std::string(&aff_str_buf[aff_disps[r]], aff_len_buf[r]) : std::string(); };

    int total_pairs = size * (size - 1) / 2;
    if (rank == 0) {
        std::cout << "=== MPI Full-Duplex Pair Latency & Bandwidth Benchmark ===\n"
                  << "Ranks=" << size << "  small=" << small_bytes << " B  large=" << (large_bytes / (1024 * 1024)) << " MiB"
                  << "  warmup=" << warmup << "  iters=" << iters << "\n"
                  << "Total pairs=" << total_pairs
                  << "  resuming_from=(" << resume_i << "," << resume_j << ")\n"
                  << "Checkpoint after " << (checkpoint_secs / 60.0) << " min\n"
                  << std::flush;
    }

    std::vector<char> small_send(small_bytes, 'a');
    std::vector<char> small_recv(small_bytes, 0);
    std::vector<char> large_send(large_bytes, 'b');
    std::vector<char> large_recv(large_bytes, 0);

    std::ofstream csv;
    if (rank == 0) {
        csv.open(output_file, appending ? std::ios::app : std::ios::out);
        if (!appending) {
            csv << "rank_i,host_i,cpu_i,affinity_i,rank_j,host_j,cpu_j,affinity_j,"
                << "small_bytes,large_bytes,warmup,iters,"
                << "latency_min_us,latency_mean_us,latency_median_us,latency_max_us,"
                << "bandwidth_median_gibs,bandwidth_mean_gibs\n";
        }
    }

    auto job_start = std::chrono::steady_clock::now();
    int active_pair[2] = {-1, -1};
    double stats_buf[6] = {0.0};
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
                DuplexStats ds = measure_mpi_duplex(rank, i, j, small_bytes, large_bytes, warmup, iters,
                                                    small_send, small_recv, large_send, large_recv);
                if (rank == i) {
                    stats_buf[0] = ds.lat_min_us;
                    stats_buf[1] = ds.lat_mean_us;
                    stats_buf[2] = ds.lat_median_us;
                    stats_buf[3] = ds.lat_max_us;
                    stats_buf[4] = ds.bw_median_gibs;
                    stats_buf[5] = ds.bw_mean_gibs;
                    if (i != 0) {
                        MPI_Send(stats_buf, 6, MPI_DOUBLE, 0, 999, MPI_COMM_WORLD);
                    }
                }
            }

            if (rank == 0) {
                if (i != 0) {
                    MPI_Recv(stats_buf, 6, MPI_DOUBLE, i, 999, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                csv << i << ",\"" << rank_host(i) << "\"," << cpu_buf[i] << ",\"" << rank_aff(i) << "\","
                    << j << ",\"" << rank_host(j) << "\"," << cpu_buf[j] << ",\"" << rank_aff(j) << "\","
                    << small_bytes << "," << large_bytes << "," << warmup << "," << iters << ","
                    << stats_buf[0] << "," << stats_buf[1] << "," << stats_buf[2] << "," << stats_buf[3] << ","
                    << stats_buf[4] << "," << stats_buf[5] << "\n";
                csv.flush();
            }

            MPI_Barrier(MPI_COMM_WORLD);

            // Checkpoint check
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
            std::cout << "All " << total_pairs << " pairs complete. Results written to " << output_file << "\n";
    }

    MPI_Finalize();
    return 0;
}
