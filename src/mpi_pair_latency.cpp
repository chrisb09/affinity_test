#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <sched.h>
#include <unistd.h>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
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
    if (count == CPU_SETSIZE) {
        return "unbound(all)";
    }
    return oss.str();
}

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  -o, --output <file>         CSV output filename (default: cpu_pair_latency.csv)\n"
              << "  -s, --small-bytes <bytes>   Small payload for latency in bytes (default: 64)\n"
              << "  -l, --large-bytes <bytes>   Large payload for bandwidth in bytes (default: 4194304 = 4MB)\n"
              << "  -w, --warmup <count>        Warmup iterations per pair (default: 10)\n"
              << "  -i, --iters <count>         Measured iterations per pair (default: 50)\n"
              << "  -h, --help                  Show this help message\n";
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string output_file = "cpu_pair_latency.csv";
    size_t small_bytes = 64;
    size_t large_bytes = 4 * 1024 * 1024; // 4 MB
    int warmup = 10;
    int iters = 50;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output_file = argv[++i];
        } else if ((arg == "-s" || arg == "--small-bytes") && i + 1 < argc) {
            small_bytes = std::stoull(argv[++i]);
        } else if ((arg == "-l" || arg == "--large-bytes") && i + 1 < argc) {
            large_bytes = std::stoull(argv[++i]);
        } else if ((arg == "-w" || arg == "--warmup") && i + 1 < argc) {
            warmup = std::stoi(argv[++i]);
        } else if ((arg == "-i" || arg == "--iters") && i + 1 < argc) {
            iters = std::stoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
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

    // Gather placement info
    std::vector<char> host_buf;
    std::vector<int> cpu_buf;
    std::vector<int> aff_len_buf;
    std::vector<char> aff_str_buf;

    int aff_len = static_cast<int>(local_info.affinity_str.size());

    if (rank == 0) {
        host_buf.resize(size * 256);
        cpu_buf.resize(size);
        aff_len_buf.resize(size);
    }

    MPI_Gather(local_info.hostname, 256, MPI_CHAR, host_buf.data(), 256, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_info.current_cpu, 1, MPI_INT, cpu_buf.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&aff_len, 1, MPI_INT, aff_len_buf.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> aff_disps;
    int total_aff_len = 0;
    if (rank == 0) {
        aff_disps.resize(size);
        for (int r = 0; r < size; ++r) {
            aff_disps[r] = total_aff_len;
            total_aff_len += aff_len_buf[r];
        }
        aff_str_buf.resize(total_aff_len);
    }

    MPI_Gatherv(local_info.affinity_str.data(), aff_len, MPI_CHAR,
                aff_str_buf.data(), aff_len_buf.data(), aff_disps.data(), MPI_CHAR,
                0, MPI_COMM_WORLD);

    std::vector<RankInfo> all_ranks;
    if (rank == 0) {
        all_ranks.resize(size);
        std::cout << "=== MPI Core-to-Core Latency & Bandwidth Benchmark ===\n";
        std::cout << "Ranks: " << size << ", Small payload: " << small_bytes << " B, Large payload: " << (large_bytes / (1024*1024)) << " MB\n";
        std::cout << "Warmup: " << warmup << ", Iters: " << iters << "\n";
        for (int r = 0; r < size; ++r) {
            all_ranks[r].rank = r;
            std::string h(&host_buf[r * 256]);
            snprintf(all_ranks[r].hostname, sizeof(all_ranks[r].hostname), "%s", h.c_str());
            all_ranks[r].current_cpu = cpu_buf[r];
            all_ranks[r].affinity_str = std::string(&aff_str_buf[aff_disps[r]], aff_len_buf[r]);
        }
        std::cout << "Starting pair-by-pair measurements...\n" << std::flush;
    }

    std::vector<char> small_send(small_bytes, 'a');
    std::vector<char> small_recv(small_bytes, 0);

    std::vector<char> large_send(large_bytes, 'b');
    std::vector<char> large_recv(large_bytes, 0);

    std::ofstream csv;
    if (rank == 0) {
        csv.open(output_file);
        csv << "rank_i,host_i,cpu_i,affinity_i,rank_j,host_j,cpu_j,affinity_j,small_bytes,large_bytes,warmup,iters,"
            << "latency_min_us,latency_mean_us,latency_median_us,latency_max_us,"
            << "bandwidth_median_gbps,bandwidth_mean_gbps\n";
    }

    int active_pair[2];
    for (int i = 0; i < size; ++i) {
        for (int j = i + 1; j < size; ++j) {
            active_pair[0] = i;
            active_pair[1] = j;

            MPI_Bcast(active_pair, 2, MPI_INT, 0, MPI_COMM_WORLD);

            double stats[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

            if (rank == i) {
                // 1. Small payload Latency test (Sender)
                for (int w = 0; w < warmup; ++w) {
                    MPI_Send(small_send.data(), small_bytes, MPI_BYTE, j, 0, MPI_COMM_WORLD);
                    MPI_Recv(small_recv.data(), small_bytes, MPI_BYTE, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }

                std::vector<double> latencies_us(iters);
                for (int it = 0; it < iters; ++it) {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    MPI_Send(small_send.data(), small_bytes, MPI_BYTE, j, 0, MPI_COMM_WORLD);
                    MPI_Recv(small_recv.data(), small_bytes, MPI_BYTE, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double rtt_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
                    latencies_us[it] = rtt_us / 2.0; // half RTT
                }

                std::sort(latencies_us.begin(), latencies_us.end());
                stats[0] = latencies_us.front(); // min
                double sum_lat = 0.0; for (double v : latencies_us) sum_lat += v;
                stats[1] = sum_lat / iters; // mean
                stats[2] = (iters % 2 == 0) ? 0.5 * (latencies_us[iters / 2 - 1] + latencies_us[iters / 2]) : latencies_us[iters / 2]; // median
                stats[3] = latencies_us.back(); // max

                // 2. Large payload Bandwidth test (Sender)
                for (int w = 0; w < warmup; ++w) {
                    MPI_Send(large_send.data(), large_bytes, MPI_BYTE, j, 1, MPI_COMM_WORLD);
                    MPI_Recv(large_recv.data(), large_bytes, MPI_BYTE, j, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }

                std::vector<double> bws_gbps(iters);
                for (int it = 0; it < iters; ++it) {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    MPI_Send(large_send.data(), large_bytes, MPI_BYTE, j, 1, MPI_COMM_WORLD);
                    MPI_Recv(large_recv.data(), large_bytes, MPI_BYTE, j, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double sec = std::chrono::duration<double>(t1 - t0).count();
                    // 2 transfers of large_bytes in round trip
                    bws_gbps[it] = (2.0 * static_cast<double>(large_bytes) / (1024.0 * 1024.0 * 1024.0)) / sec;
                }

                std::sort(bws_gbps.begin(), bws_gbps.end());
                stats[4] = (iters % 2 == 0) ? 0.5 * (bws_gbps[iters / 2 - 1] + bws_gbps[iters / 2]) : bws_gbps[iters / 2]; // median
                double sum_bw = 0.0; for (double v : bws_gbps) sum_bw += v;
                stats[5] = sum_bw / iters; // mean

                if (i != 0) {
                    MPI_Send(stats, 6, MPI_DOUBLE, 0, 999, MPI_COMM_WORLD);
                }
            } else if (rank == j) {
                // 1. Small payload Latency test (Receiver)
                for (int w = 0; w < warmup; ++w) {
                    MPI_Recv(small_recv.data(), small_bytes, MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(small_send.data(), small_bytes, MPI_BYTE, i, 0, MPI_COMM_WORLD);
                }

                for (int it = 0; it < iters; ++it) {
                    MPI_Recv(small_recv.data(), small_bytes, MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(small_send.data(), small_bytes, MPI_BYTE, i, 0, MPI_COMM_WORLD);
                }

                // 2. Large payload Bandwidth test (Receiver)
                for (int w = 0; w < warmup; ++w) {
                    MPI_Recv(large_recv.data(), large_bytes, MPI_BYTE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(large_send.data(), large_bytes, MPI_BYTE, i, 1, MPI_COMM_WORLD);
                }

                for (int it = 0; it < iters; ++it) {
                    MPI_Recv(large_recv.data(), large_bytes, MPI_BYTE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(large_send.data(), large_bytes, MPI_BYTE, i, 1, MPI_COMM_WORLD);
                }
            }

            if (rank == 0) {
                if (i != 0) {
                    MPI_Recv(stats, 6, MPI_DOUBLE, i, 999, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                csv << i << ",\"" << all_ranks[i].hostname << "\"," << all_ranks[i].current_cpu << ",\"" << all_ranks[i].affinity_str << "\","
                    << j << ",\"" << all_ranks[j].hostname << "\"," << all_ranks[j].current_cpu << ",\"" << all_ranks[j].affinity_str << "\","
                    << small_bytes << "," << large_bytes << "," << warmup << "," << iters << ","
                    << stats[0] << "," << stats[1] << "," << stats[2] << "," << stats[3] << ","
                    << stats[4] << "," << stats[5] << "\n";
            }
        }
    }

    if (rank == 0) {
        csv.close();
        std::cout << "Done! Results written to " << output_file << "\n";
    }

    MPI_Finalize();
    return 0;
}
