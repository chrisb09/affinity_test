#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/*
 * affinity_test/mpi_incast_latency.cpp
 * ====================================
 * MPI Incast / Fan-in Flooding Benchmark.
 *
 * Simulates N-1 ranks simultaneously flooding a single target receiver rank
 * (e.g. simulating SmartSim simulation ranks streaming data into an orchestrator/DB rank).
 *
 * Measurement protocol:
 *   - Sweeps multiple payload sizes (from 64 B to 4 MiB).
 *   - Target posts (N-1) MPI_Irecv calls.
 *   - Senders synchronize at MPI_Barrier and simultaneously issue MPI_Isend.
 *   - Target measures total time to receive all (N-1) transfers -> Ingest Throughput (GiB/s).
 *   - Senders measure individual transfer completion latency -> Tail Latency (p50, p95, p99, max).
 *   - Tests multiple target rank placements (e.g. Rank 0 on Node 0, Rank 96 on Node 1).
 */

#include <mpi.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
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
              << "  -o, --output <file>         Summary CSV output (default: mpi_incast_summary.csv)\n"
              << "  -d, --detail <file>         Detailed per-sender CSV output (default: mpi_incast_details.csv)\n"
              << "  -t, --targets <list>        Comma-separated target ranks (default: 0, or 0,96 if size>=96)\n"
              << "  -p, --payloads <list>       Comma-separated payload sizes in bytes\n"
              << "                              (default: 64,1024,4096,16384,65536,262144,1048576,4194304)\n"
              << "  -w, --warmup <count>        Warmup iterations per payload (default: 5)\n"
              << "  -i, --iters <count>         Measured iterations per payload (default: 20)\n"
              << "  -h, --help                  Show this help message\n";
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string summary_file = "mpi_incast_summary.csv";
    std::string detail_file  = "mpi_incast_details.csv";
    std::string target_str   = (size >= 96) ? "0,96" : "0";
    std::string payload_str  = "64,1024,4096,16384,65536,262144,1048576,4194304";
    int warmup = 5;
    int iters  = 20;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      ((arg == "-o" || arg == "--output")   && i + 1 < argc) summary_file = argv[++i];
        else if ((arg == "-d" || arg == "--detail")   && i + 1 < argc) detail_file  = argv[++i];
        else if ((arg == "-t" || arg == "--targets")  && i + 1 < argc) target_str   = argv[++i];
        else if ((arg == "-p" || arg == "--payloads") && i + 1 < argc) payload_str  = argv[++i];
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

    // Placement info
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
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            all_ranks[r].rank = r;
            std::string h(&host_buf[r * 256]);
            snprintf(all_ranks[r].hostname, sizeof(all_ranks[r].hostname), "%s", h.c_str());
            all_ranks[r].current_cpu = cpu_buf[r];
            all_ranks[r].affinity_str = std::string(&aff_str_buf[aff_disps[r]], aff_len_buf[r]);
        }
        std::cout << "=== MPI Incast / Flooding Congestion Benchmark ===\n"
                  << "Total Ranks: " << size << " (Senders: " << (size - 1) << " per test)\n"
                  << "Target Ranks: " << target_str << "\n"
                  << "Payloads: " << payload_str << "\n"
                  << "Warmup: " << warmup << ", Iters: " << iters << "\n\n" << std::flush;
    }

    std::ofstream sum_csv, det_csv;
    if (rank == 0) {
        sum_csv.open(summary_file);
        sum_csv << "target_rank,target_host,target_cpu,target_affinity,senders_count,payload_bytes,total_ingest_bytes,iters,"
                << "ingest_bw_min_gibs,ingest_bw_p25_gibs,ingest_bw_median_gibs,ingest_bw_mean_gibs,ingest_bw_p75_gibs,ingest_bw_p95_gibs,ingest_bw_max_gibs,"
                << "sender_lat_min_us,sender_lat_p25_us,sender_lat_median_us,sender_lat_mean_us,sender_lat_p75_us,sender_lat_p95_us,sender_lat_p99_us,sender_lat_max_us\n";

        det_csv.open(detail_file);
        det_csv << "target_rank,target_host,sender_rank,sender_host,sender_cpu,sender_affinity,payload_bytes,iters,"
                << "lat_min_us,lat_p25_us,lat_median_us,lat_mean_us,lat_p75_us,lat_p95_us,lat_max_us\n";
    }

    size_t max_payload = *std::max_element(payloads.begin(), payloads.end());

    // Sender buffer
    std::vector<char> send_buf(max_payload, 'x');

    // Receiver buffers (N-1 buffers of max_payload)
    std::vector<char> recv_pool;
    if (std::find(targets.begin(), targets.end(), rank) != targets.end()) {
        recv_pool.resize(static_cast<size_t>(size - 1) * max_payload, 0);
    }

    constexpr double GiB = 1024.0 * 1024.0 * 1024.0;

    for (int target : targets) {
        if (target < 0 || target >= size) continue;

        if (rank == 0) {
            std::cout << ">>> Running Incast Flood targeting Rank " << target
                      << " (" << all_ranks[target].hostname << ", CPU " << all_ranks[target].current_cpu << ") ...\n"
                      << std::flush;
        }

        for (size_t payload : payloads) {
            std::vector<MPI_Request> recv_reqs(size - 1);

            // Warmup iterations
            for (int w = 0; w < warmup; ++w) {
                if (rank == target) {
                    int req_idx = 0;
                    for (int src = 0; src < size; ++src) {
                        if (src == target) continue;
                        char* ptr = recv_pool.data() + static_cast<size_t>(req_idx) * max_payload;
                        MPI_Irecv(ptr, static_cast<int>(payload), MPI_BYTE, src, 100, MPI_COMM_WORLD, &recv_reqs[req_idx]);
                        req_idx++;
                    }
                }

                MPI_Barrier(MPI_COMM_WORLD);

                if (rank != target) {
                    MPI_Request sreq;
                    MPI_Isend(send_buf.data(), static_cast<int>(payload), MPI_BYTE, target, 100, MPI_COMM_WORLD, &sreq);
                    MPI_Wait(&sreq, MPI_STATUS_IGNORE);
                } else {
                    MPI_Waitall(size - 1, recv_reqs.data(), MPI_STATUSES_IGNORE);
                }
            }

            // Timed iterations
            std::vector<double> target_ingest_bws(iters);
            std::vector<double> local_sender_lats(iters, 0.0);

            for (int it = 0; it < iters; ++it) {
                if (rank == target) {
                    int req_idx = 0;
                    for (int src = 0; src < size; ++src) {
                        if (src == target) continue;
                        char* ptr = recv_pool.data() + static_cast<size_t>(req_idx) * max_payload;
                        MPI_Irecv(ptr, static_cast<int>(payload), MPI_BYTE, src, 200 + it, MPI_COMM_WORLD, &recv_reqs[req_idx]);
                        req_idx++;
                    }
                }

                MPI_Barrier(MPI_COMM_WORLD);

                auto t0 = std::chrono::high_resolution_clock::now();

                if (rank != target) {
                    MPI_Request sreq;
                    MPI_Isend(send_buf.data(), static_cast<int>(payload), MPI_BYTE, target, 200 + it, MPI_COMM_WORLD, &sreq);
                    MPI_Wait(&sreq, MPI_STATUS_IGNORE);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    local_sender_lats[it] = std::chrono::duration<double, std::micro>(t1 - t0).count();
                } else {
                    MPI_Waitall(size - 1, recv_reqs.data(), MPI_STATUSES_IGNORE);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
                    double total_bytes = static_cast<double>(size - 1) * static_cast<double>(payload);
                    target_ingest_bws[it] = (total_bytes / GiB) / elapsed_sec;
                }
            }

            // Gather sender latency data
            std::vector<double> all_sender_lats; // size * iters
            if (rank == 0) {
                all_sender_lats.resize(size * iters);
            }

            MPI_Gather(local_sender_lats.data(), iters, MPI_DOUBLE,
                       all_sender_lats.data(), iters, MPI_DOUBLE,
                       0, MPI_COMM_WORLD);

            // Send target bandwidth stats to rank 0 if target != 0
            if (target != 0) {
                if (rank == target) {
                    MPI_Send(target_ingest_bws.data(), iters, MPI_DOUBLE, 0, 777, MPI_COMM_WORLD);
                } else if (rank == 0) {
                    MPI_Recv(target_ingest_bws.data(), iters, MPI_DOUBLE, target, 777, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }

            if (rank == 0) {
                // Collect all sender latencies (excluding target)
                std::vector<double> pooled_sender_lats;
                pooled_sender_lats.reserve((size - 1) * iters);

                for (int src = 0; src < size; ++src) {
                    if (src == target) continue;
                    std::vector<double> per_sender_lats(iters);
                    for (int it = 0; it < iters; ++it) {
                        double lat = all_sender_lats[src * iters + it];
                        per_sender_lats[it] = lat;
                        pooled_sender_lats.push_back(lat);
                    }

                    // Write detail row
                    det_csv << target << ",\"" << all_ranks[target].hostname << "\","
                            << src << ",\"" << all_ranks[src].hostname << "\","
                            << all_ranks[src].current_cpu << ",\"" << all_ranks[src].affinity_str << "\","
                            << payload << "," << iters << ","
                            << *std::min_element(per_sender_lats.begin(), per_sender_lats.end()) << ","
                            << percentile_of(per_sender_lats, 0.25) << ","
                            << percentile_of(per_sender_lats, 0.50) << ","
                            << mean_of(per_sender_lats) << ","
                            << percentile_of(per_sender_lats, 0.75) << ","
                            << percentile_of(per_sender_lats, 0.95) << ","
                            << *std::max_element(per_sender_lats.begin(), per_sender_lats.end()) << "\n";
                }

                // Compute summary stats
                double total_ingest = static_cast<double>(size - 1) * static_cast<double>(payload);

                sum_csv << target << ",\"" << all_ranks[target].hostname << "\","
                        << all_ranks[target].current_cpu << ",\"" << all_ranks[target].affinity_str << "\","
                        << (size - 1) << "," << payload << "," << total_ingest << "," << iters << ","
                        << *std::min_element(target_ingest_bws.begin(), target_ingest_bws.end()) << ","
                        << percentile_of(target_ingest_bws, 0.25) << ","
                        << percentile_of(target_ingest_bws, 0.50) << ","
                        << mean_of(target_ingest_bws) << ","
                        << percentile_of(target_ingest_bws, 0.75) << ","
                        << percentile_of(target_ingest_bws, 0.95) << ","
                        << *std::max_element(target_ingest_bws.begin(), target_ingest_bws.end()) << ","
                        << *std::min_element(pooled_sender_lats.begin(), pooled_sender_lats.end()) << ","
                        << percentile_of(pooled_sender_lats, 0.25) << ","
                        << percentile_of(pooled_sender_lats, 0.50) << ","
                        << mean_of(pooled_sender_lats) << ","
                        << percentile_of(pooled_sender_lats, 0.75) << ","
                        << percentile_of(pooled_sender_lats, 0.95) << ","
                        << percentile_of(pooled_sender_lats, 0.99) << ","
                        << *std::max_element(pooled_sender_lats.begin(), pooled_sender_lats.end()) << "\n";

                sum_csv.flush();
                det_csv.flush();

                std::cout << "  Payload: " << std::setw(8) << payload << " B"
                          << " | Ingest BW (median): " << std::setw(7) << std::fixed << std::setprecision(2)
                          << percentile_of(target_ingest_bws, 0.50) << " GiB/s"
                          << " | Sender Lat (p50): " << std::setw(8) << std::fixed << std::setprecision(1)
                          << percentile_of(pooled_sender_lats, 0.50) << " us"
                          << " | (p95): " << std::setw(8) << percentile_of(pooled_sender_lats, 0.95) << " us"
                          << " | (max): " << std::setw(8) << *std::max_element(pooled_sender_lats.begin(), pooled_sender_lats.end()) << " us\n"
                          << std::flush;
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        sum_csv.close();
        det_csv.close();
        std::cout << "\nIncast benchmark complete. Summary saved to " << summary_file
                  << ", details saved to " << detail_file << "\n";
    }

    MPI_Finalize();
    return 0;
}
