#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <sched.h>
#include <unistd.h>
#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct RankCpuInfo {
    int rank{-1};
    char hostname[256]{0};
    int current_cpu{-1};
    std::string affinity_str;
};

struct TransferResult {
    int rank{-1};
    int target_gpu{-1};
    char pci_bus_id[64]{"N/A"};
    char status[128]{"OK"};
    
    // Latencies (small payload) in microseconds
    double h2d_lat_min_us{0.0};
    double h2d_lat_median_us{0.0};
    double h2d_lat_mean_us{0.0};
    
    double d2h_lat_min_us{0.0};
    double d2h_lat_median_us{0.0};
    double d2h_lat_mean_us{0.0};

    // Bandwidth (large payload) in GB/s
    double h2d_bw_median_gbps{0.0};
    double d2h_bw_median_gbps{0.0};
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

static std::vector<int> parse_gpu_list(const std::string& str) {
    std::vector<int> gpus;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            gpus.push_back(std::stoi(token));
        }
    }
    return gpus;
}

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  -o, --output <file>         CSV output filename (default: gpu_transfer_latency.csv)\n"
              << "  -d, --devices <0,1,...>     Comma-separated CUDA device IDs to test (default: 0,1)\n"
              << "  -s, --small-size <bytes>    Small transfer payload size for latency (default: 4096)\n"
              << "  -l, --large-size <bytes>    Large transfer payload size for bandwidth (default: 67108864 = 64MB)\n"
              << "  -w, --warmup <count>        Warmup iterations (default: 10)\n"
              << "  -i, --iters <count>         Measured iterations per test (default: 50)\n"
              << "  -h, --help                  Show this help message\n";
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string output_file = "gpu_transfer_latency.csv";
    std::string device_str = "0,1";
    size_t small_size = 4096;
    size_t large_size = 64 * 1024 * 1024; // 64 MB
    int warmup = 10;
    int iters = 50;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output_file = argv[++i];
        } else if ((arg == "-d" || arg == "--devices") && i + 1 < argc) {
            device_str = argv[++i];
        } else if ((arg == "-s" || arg == "--small-size") && i + 1 < argc) {
            small_size = std::stoull(argv[++i]);
        } else if ((arg == "-l" || arg == "--large-size") && i + 1 < argc) {
            large_size = std::stoull(argv[++i]);
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

    std::vector<int> target_gpus = parse_gpu_list(device_str);

    RankCpuInfo local_info;
    local_info.rank = rank;
    gethostname(local_info.hostname, sizeof(local_info.hostname) - 1);
    local_info.current_cpu = sched_getcpu();
    local_info.affinity_str = get_affinity_string();

    // Gather cpu/rank info to rank 0
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

    std::vector<RankCpuInfo> all_ranks;
    if (rank == 0) {
        all_ranks.resize(size);
        std::cout << "=== MPI Rank-to-GPU Latency & Bandwidth Benchmark ===\n";
        std::cout << "Ranks: " << size << ", Small payload: " << small_size << " B, Large payload: " << (large_size / (1024*1024)) << " MB\n";
        std::cout << "Target GPUs: " << device_str << "\n";
        for (int r = 0; r < size; ++r) {
            all_ranks[r].rank = r;
            std::string h(&host_buf[r * 256]);
            snprintf(all_ranks[r].hostname, sizeof(all_ranks[r].hostname), "%s", h.c_str());
            all_ranks[r].current_cpu = cpu_buf[r];
            all_ranks[r].affinity_str = std::string(&aff_str_buf[aff_disps[r]], aff_len_buf[r]);
        }
        std::cout << "Starting GPU transfer measurements via token ring...\n" << std::flush;
    }

    std::vector<TransferResult> my_results;

    for (int gpu_id : target_gpus) {
        int gpu_ok = 1;
        char gpu_err_msg[128] = "OK";

        if (rank == 0) {
            cudaDeviceReset();
            cudaDeviceProp prop;
            cudaError_t err = cudaGetDeviceProperties(&prop, gpu_id);
            if (err == cudaSuccess && prop.computeMode != cudaComputeModeDefault) {
                gpu_ok = 0;
                snprintf(gpu_err_msg, sizeof(gpu_err_msg), "GPU in Exclusive/Prohibited compute mode (mode %d)", prop.computeMode);
            } else if (err == cudaSuccess) {
                err = cudaSetDevice(gpu_id);
                if (err == cudaSuccess) {
                    err = cudaFree(0);
                }
                if (err != cudaSuccess) {
                    gpu_ok = 0;
                    snprintf(gpu_err_msg, sizeof(gpu_err_msg), "cudaSetDevice/cudaFree failed: %s", cudaGetErrorString(err));
                    cudaGetLastError();
                } else {
                    cudaDeviceReset();
                }
            } else {
                gpu_ok = 0;
                snprintf(gpu_err_msg, sizeof(gpu_err_msg), "cudaGetDeviceProperties failed: %s", cudaGetErrorString(err));
                cudaGetLastError();
            }
        }

        MPI_Bcast(&gpu_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(gpu_err_msg, 128, MPI_CHAR, 0, MPI_COMM_WORLD);

        if (!gpu_ok) {
            if (rank == 0) {
                std::cout << "GPU " << gpu_id << " unavailable (" << gpu_err_msg << "), skipping...\n" << std::flush;
            }
            TransferResult res;
            res.rank = rank;
            res.target_gpu = gpu_id;
            snprintf(res.status, sizeof(res.status), "%s", gpu_err_msg);
            my_results.push_back(res);
            continue;
        }

        int token = 0;
        if (rank != 0) {
            MPI_Recv(&token, 1, MPI_INT, rank - 1, 777, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        TransferResult res;
        res.rank = rank;
        res.target_gpu = gpu_id;

        if (rank % 16 == 0) {
            std::cout << "[Rank " << rank << "] Testing GPU " << gpu_id << " (CPU " << local_info.current_cpu << ")...\n" << std::flush;
        }

        cudaDeviceReset();
        cudaError_t err = cudaSetDevice(gpu_id);
        if (err != cudaSuccess) {
            snprintf(res.status, sizeof(res.status), "cudaSetDevice failed: %s", cudaGetErrorString(err));
            cudaGetLastError();
        } else {
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, gpu_id) == cudaSuccess) {
                snprintf(res.pci_bus_id, sizeof(res.pci_bus_id), "%04x:%02x:%02x.0", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
            }

            void *h_small = nullptr, *h_large = nullptr;
            void *d_small = nullptr, *d_large = nullptr;

            cudaError_t err1 = cudaMallocHost(&h_small, small_size);
            cudaError_t err2 = cudaMallocHost(&h_large, large_size);
            cudaError_t err3 = cudaMalloc(&d_small, small_size);
            cudaError_t err4 = cudaMalloc(&d_large, large_size);

            if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess) {
                snprintf(res.status, sizeof(res.status), "Allocation failed");
                cudaGetLastError();
            } else {
                // Warmup
                for (int w = 0; w < warmup; ++w) {
                    cudaMemcpy(d_small, h_small, small_size, cudaMemcpyHostToDevice);
                    cudaMemcpy(h_small, d_small, small_size, cudaMemcpyDeviceToHost);
                }
                cudaDeviceSynchronize();

                // 1. Small H2D Latency
                std::vector<double> h2d_lats(iters);
                for (int it = 0; it < iters; ++it) {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    cudaMemcpy(d_small, h_small, small_size, cudaMemcpyHostToDevice);
                    cudaDeviceSynchronize();
                    auto t1 = std::chrono::high_resolution_clock::now();
                    h2d_lats[it] = std::chrono::duration<double, std::micro>(t1 - t0).count();
                }
                std::sort(h2d_lats.begin(), h2d_lats.end());
                res.h2d_lat_min_us = h2d_lats.front();
                res.h2d_lat_median_us = (iters % 2 == 0) ? 0.5 * (h2d_lats[iters/2 - 1] + h2d_lats[iters/2]) : h2d_lats[iters/2];
                double sum_h2d = 0; for (double v : h2d_lats) sum_h2d += v;
                res.h2d_lat_mean_us = sum_h2d / iters;

                // 2. Small D2H Latency
                std::vector<double> d2h_lats(iters);
                for (int it = 0; it < iters; ++it) {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    cudaMemcpy(h_small, d_small, small_size, cudaMemcpyDeviceToHost);
                    cudaDeviceSynchronize();
                    auto t1 = std::chrono::high_resolution_clock::now();
                    d2h_lats[it] = std::chrono::duration<double, std::micro>(t1 - t0).count();
                }
                std::sort(d2h_lats.begin(), d2h_lats.end());
                res.d2h_lat_min_us = d2h_lats.front();
                res.d2h_lat_median_us = (iters % 2 == 0) ? 0.5 * (d2h_lats[iters/2 - 1] + d2h_lats[iters/2]) : d2h_lats[iters/2];
                double sum_d2h = 0; for (double v : d2h_lats) sum_d2h += v;
                res.d2h_lat_mean_us = sum_d2h / iters;

                // 3. Large H2D Bandwidth
                std::vector<double> h2d_bws(iters);
                for (int it = 0; it < iters; ++it) {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    cudaMemcpy(d_large, h_large, large_size, cudaMemcpyHostToDevice);
                    cudaDeviceSynchronize();
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double sec = std::chrono::duration<double>(t1 - t0).count();
                    h2d_bws[it] = (static_cast<double>(large_size) / (1024.0 * 1024.0 * 1024.0)) / sec;
                }
                std::sort(h2d_bws.begin(), h2d_bws.end());
                res.h2d_bw_median_gbps = (iters % 2 == 0) ? 0.5 * (h2d_bws[iters/2 - 1] + h2d_bws[iters/2]) : h2d_bws[iters/2];

                // 4. Large D2H Bandwidth
                std::vector<double> d2h_bws(iters);
                for (int it = 0; it < iters; ++it) {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    cudaMemcpy(h_large, d_large, large_size, cudaMemcpyDeviceToHost);
                    cudaDeviceSynchronize();
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double sec = std::chrono::duration<double>(t1 - t0).count();
                    d2h_bws[it] = (static_cast<double>(large_size) / (1024.0 * 1024.0 * 1024.0)) / sec;
                }
                std::sort(d2h_bws.begin(), d2h_bws.end());
                res.d2h_bw_median_gbps = (iters % 2 == 0) ? 0.5 * (d2h_bws[iters/2 - 1] + d2h_bws[iters/2]) : d2h_bws[iters/2];
            }

            if (h_small) cudaFreeHost(h_small);
            if (h_large) cudaFreeHost(h_large);
            if (d_small) cudaFree(d_small);
            if (d_large) cudaFree(d_large);
            cudaDeviceReset();
        }

        my_results.push_back(res);

        if (rank < size - 1) {
            MPI_Send(&token, 1, MPI_INT, rank + 1, 777, MPI_COMM_WORLD);
        } else {
            MPI_Send(&token, 1, MPI_INT, 0, 777, MPI_COMM_WORLD);
        }
        if (rank == 0) {
            MPI_Recv(&token, 1, MPI_INT, size - 1, 777, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << "Completed GPU " << gpu_id << " token ring across all ranks.\n" << std::flush;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Gather all results to rank 0
    int num_items = static_cast<int>(my_results.size());
    std::vector<TransferResult> all_results;
    if (rank == 0) {
        all_results.resize(size * num_items);
    }

    MPI_Gather(my_results.data(), num_items * sizeof(TransferResult), MPI_BYTE,
               all_results.data(), num_items * sizeof(TransferResult), MPI_BYTE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::ofstream csv(output_file);
        csv << "rank,hostname,cpu,affinity,gpu_id,pci_bus_id,status,"
            << "small_bytes,large_bytes,iters,"
            << "h2d_lat_min_us,h2d_lat_median_us,h2d_lat_mean_us,"
            << "d2h_lat_min_us,d2h_lat_median_us,d2h_lat_mean_us,"
            << "h2d_bw_median_gbps,d2h_bw_median_gbps\n";

        for (int r = 0; r < size; ++r) {
            for (int i = 0; i < num_items; ++i) {
                const auto& res = all_results[r * num_items + i];
                csv << res.rank << ",\"" << all_ranks[res.rank].hostname << "\"," << all_ranks[res.rank].current_cpu << ",\"" << all_ranks[res.rank].affinity_str << "\","
                    << res.target_gpu << "," << res.pci_bus_id << ",\"" << res.status << "\","
                    << small_size << "," << large_size << "," << iters << ","
                    << res.h2d_lat_min_us << "," << res.h2d_lat_median_us << "," << res.h2d_lat_mean_us << ","
                    << res.d2h_lat_min_us << "," << res.d2h_lat_median_us << "," << res.d2h_lat_mean_us << ","
                    << res.h2d_bw_median_gbps << "," << res.d2h_bw_median_gbps << "\n";
            }
        }
        csv.close();
        std::cout << "Done! Results written to " << output_file << "\n";
    }

    MPI_Finalize();
    return 0;
}
