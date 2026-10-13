#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <sched.h>
#include <unistd.h>
#include <numa.h>
#include <numaif.h>
#include <mpi.h>
#include <mpi-ext.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct RankCpuInfo {
    int rank{-1};
    char hostname[256]{0};
    int current_cpu{-1};
    int cpu_numa_node{-1};
    std::string affinity_str;
};

struct Stats {
    double min{0.0};
    double p25{0.0};
    double median{0.0};
    double p75{0.0};
    double iqr{0.0};
    double max{0.0};
    double mean{0.0};
    double stddev{0.0};
};

struct DirectionTransferResult {
    int rank{-1};
    char hostname[256]{0};
    int cpu_id{-1};
    int cpu_numa_node{-1};
    char affinity_str[128]{0};

    int target_gpu{-1};
    char pci_bus_id[64]{"N/A"};
    int gpu_pci_numa_node{-1};
    char status[128]{"OK"};

    size_t payload_bytes{0};
    char payload_type[32]{"bandwidth"}; // "latency" or "bandwidth"
    char direction[16]{"H2D"};          // "H2D" or "D2H"
    int iters{0};
    int host_buf_numa_node{-1};

    int page_policy{-1};
    int src_page_policy{-2};
    int dst_page_policy{-2};
    int reg_mode{0};
    char variant[64]{0};
    char transfer_mode[24]{"interleaved"};
    char src_pages[192]{0};
    char dst_pages[192]{0};
    char dst_pages_after[192]{0};

    // Wall-clock latency (microseconds)
    double lat_min_us{0.0};
    double lat_p25_us{0.0};
    double lat_median_us{0.0};
    double lat_p75_us{0.0};
    double lat_iqr_us{0.0};
    double lat_max_us{0.0};
    double lat_mean_us{0.0};
    double lat_stddev_us{0.0};

    // Bandwidth in GiB/s
    double bw_gibs_min{0.0};
    double bw_gibs_p25{0.0};
    double bw_gibs_median{0.0};
    double bw_gibs_p75{0.0};
    double bw_gibs_iqr{0.0};
    double bw_gibs_max{0.0};
    double bw_gibs_mean{0.0};
    double bw_gibs_stddev{0.0};

    // Bandwidth in GB/s
    double bw_gbps_median{0.0};

    double gpu_dma_time_median_us{0.0};
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

static std::vector<size_t> parse_size_list(const std::string& str) {
    std::vector<size_t> sizes;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            sizes.push_back(std::stoull(token));
        }
    }
    return sizes;
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

static int get_gpu_pci_numa_node(int domain, int bus, int dev) {
    char path[256];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%04x:%02x:%02x.0/numa_node", domain, bus, dev);
    std::ifstream f(path);
    int node = -1;
    if (f >> node) {
        return node;
    }
    return -1;
}

static int get_page_numa_node(void* ptr) {
    int node = -1;
    if (get_mempolicy(&node, NULL, 0, ptr, MPOL_F_NODE | MPOL_F_ADDR) == 0) {
        return node;
    }
    return -1;
}

static std::string count_pages_by_numa(void* ptr, size_t size) {
    std::map<int, size_t> tally;
    const size_t page = 4096;
    const size_t pages = (size + page - 1) / page;
    for (size_t p = 0; p < pages; ++p) {
        int node = get_page_numa_node(static_cast<char*>(ptr) + p * page);
        tally[node] += 1;
    }
    std::ostringstream oss;
    bool first = true;
    for (const auto& kv : tally) {
        if (!first) oss << ",";
        oss << kv.first << ":" << kv.second;
        first = false;
    }
    return oss.str();
}

static unsigned long long buffer_checksum(const void* ptr, size_t size) {
    const unsigned char* bytes = static_cast<const unsigned char*>(ptr);
    unsigned long long h = 1469598103934665603ULL;
    size_t words = size / 8;
    for (size_t i = 0; i < words; ++i) {
        unsigned long long w;
        std::memcpy(&w, bytes + i * 8, 8);
        h ^= w;
        h *= 1099511628211ULL;
    }
    for (size_t i = words * 8; i < size; ++i) {
        h ^= static_cast<unsigned long long>(bytes[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

static Stats compute_stats(std::vector<double>& v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    s.min = v.front();
    s.max = v.back();

    double sum = 0.0;
    for (double x : v) sum += x;
    s.mean = sum / n;

    double sq_sum = 0.0;
    for (double x : v) sq_sum += (x - s.mean) * (x - s.mean);
    s.stddev = std::sqrt(sq_sum / n);

    auto percentile = [&](double p) {
        double idx = p * (n - 1);
        size_t lo = static_cast<size_t>(std::floor(idx));
        size_t hi = static_cast<size_t>(std::ceil(idx));
        if (lo == hi) return v[lo];
        return v[lo] + (idx - lo) * (v[hi] - v[lo]);
    };

    s.p25 = percentile(0.25);
    s.median = percentile(0.50);
    s.p75 = percentile(0.75);
    s.iqr = s.p75 - s.p25;
    return s;
}

struct HostAlloc {
    void* ptr{nullptr};
    size_t size{0};
    int reg_mode{0};
    bool from_numa_alloc{false};
    bool ok{false};
};

static HostAlloc allocate_pinned_host_memory(size_t size, int page_node, int reg_mode,
                                             bool initialize_full_payload) {
    HostAlloc ha;
    ha.size = size;
    ha.reg_mode = reg_mode;

    void* ptr = nullptr;
    if (reg_mode == 3) {
        if (cudaMallocHost(&ptr, size) != cudaSuccess) {
            return ha;
        }
    } else if (page_node >= 0) {
        ptr = numa_alloc_onnode(size, page_node);
        if (ptr) ha.from_numa_alloc = true;
    } else {
        if (posix_memalign(&ptr, 4096, size) != 0 || !ptr) {
            return ha;
        }
    }
    if (!ptr) {
        return ha;
    }

    volatile char* cp = static_cast<volatile char*>(ptr);
    for (size_t off = 0; off < size; off += 4096) {
        cp[off] = static_cast<char>((off ^ (off >> 8)) & 0x7F);
    }
    if (initialize_full_payload) {
        std::memset(ptr, 0xA5, size);
    }

    cudaError_t err = cudaSuccess;
    if (reg_mode == 0) {
        err = cudaHostRegister(ptr, size, cudaHostRegisterPortable | cudaHostRegisterMapped);
    } else if (reg_mode == 1) {
        err = cudaHostRegister(ptr, size, cudaHostRegisterPortable);
    } else if (reg_mode == 2) {
        err = cudaHostRegister(ptr, size, cudaHostRegisterMapped);
    }
    if (err != cudaSuccess) {
        if (reg_mode == 3) cudaFreeHost(ptr);
        else if (ha.from_numa_alloc) numa_free(ptr, size);
        else free(ptr);
        return ha;
    }

    ha.ptr = ptr;
    ha.ok = true;
    return ha;
}

static void free_pinned_host_memory(HostAlloc& ha) {
    if (!ha.ptr) return;
    if (ha.reg_mode != 3) {
        cudaHostUnregister(ha.ptr);
    }
    if (ha.reg_mode == 3) {
        cudaFreeHost(ha.ptr);
    } else if (ha.from_numa_alloc) {
        numa_free(ha.ptr, ha.size);
    } else {
        free(ha.ptr);
    }
    ha.ptr = nullptr;
}

static void print_usage(const char* prog) {
    std::cout << "Usage: mpirun -np 2 " << prog << " [options]\n"
              << "Options:\n"
              << "  -o, --output <file>         CSV output filename (default: mpi_cuda_aware_transfer.csv)\n"
              << "  -d, --devices <0,1,...>     Comma-separated CUDA device IDs to test (default: 0)\n"
              << "  -s, --small-sizes <b1,b2>   Small sizes for latency (default: 4096)\n"
              << "  -l, --large-sizes <b1,b2>   Large sizes for bandwidth (default: 67108864,268435456)\n"
              << "  -w, --warmup <count>        Warmup iterations (default: 5)\n"
              << "  -i, --iters <count>         Measured iterations per test (default: 50)\n"
              << "  -p, --page-numa <node>      Force host buffers onto NUMA <node> (-1 = auto first-touch, default)\n"
              << "      --src-page-numa <node> Independently place H2D source pages (-2 = use --page-numa)\n"
              << "      --dst-page-numa <node> Independently place D2H destination pages (-2 = use --page-numa)\n"
              << "      --mode <name>           interleaved (default), d2h-only, or h2d-only\n"
              << "  -r, --reg-mode <0-3>        Host pinning mode: 0=Register(Portable|Mapped), 1=Portable, 2=Mapped, 3=cudaHostAlloc\n"
              << "  -v, --variant <label>       Free-form run label written to the CSV (default: empty)\n"
              << "  -a, --append                Append to the output CSV instead of overwriting\n"
              << "  -h, --help                  Show this help message\n";
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            std::cerr << "Error: mpi_cuda_aware_transfer requires exactly 2 MPI ranks (rank 0 = CPU, rank 1 = GPU)\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::string output_file = "mpi_cuda_aware_transfer.csv";
    std::string device_str = "0";
    std::string small_sizes_str = "4096";
    std::string large_sizes_str = "67108864,268435456";
    int warmup = 5;
    int iters = 50;
    int page_numa = -1;
    int src_page_numa = -2;
    int dst_page_numa = -2;
    int reg_mode = 0;
    std::string transfer_mode = "interleaved";
    std::string variant_str;
    bool append_csv = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output_file = argv[++i];
        } else if ((arg == "-d" || arg == "--devices") && i + 1 < argc) {
            device_str = argv[++i];
        } else if ((arg == "-s" || arg == "--small-sizes") && i + 1 < argc) {
            small_sizes_str = argv[++i];
        } else if ((arg == "-l" || arg == "--large-sizes") && i + 1 < argc) {
            large_sizes_str = argv[++i];
        } else if ((arg == "-w" || arg == "--warmup") && i + 1 < argc) {
            warmup = std::stoi(argv[++i]);
        } else if ((arg == "-i" || arg == "--iters") && i + 1 < argc) {
            iters = std::stoi(argv[++i]);
        } else if ((arg == "-p" || arg == "--page-numa") && i + 1 < argc) {
            page_numa = std::stoi(argv[++i]);
        } else if (arg == "--src-page-numa" && i + 1 < argc) {
            src_page_numa = std::stoi(argv[++i]);
        } else if (arg == "--dst-page-numa" && i + 1 < argc) {
            dst_page_numa = std::stoi(argv[++i]);
        } else if ((arg == "-r" || arg == "--reg-mode") && i + 1 < argc) {
            reg_mode = std::stoi(argv[++i]);
        } else if (arg == "--mode" && i + 1 < argc) {
            transfer_mode = argv[++i];
        } else if ((arg == "-v" || arg == "--variant") && i + 1 < argc) {
            variant_str = argv[++i];
        } else if (arg == "-a" || arg == "--append") {
            append_csv = true;
        } else if (arg == "-h" || arg == "--help") {
            if (rank == 0) print_usage(argv[0]);
            MPI_Finalize();
            return 0;
        }
    }

    if (transfer_mode != "interleaved" && transfer_mode != "d2h-only" && transfer_mode != "h2d-only") {
        if (rank == 0) std::cerr << "Invalid --mode: " << transfer_mode << "\n";
        MPI_Finalize();
        return 2;
    }

    std::vector<int> target_gpus = parse_gpu_list(device_str);
    std::vector<size_t> small_sizes = parse_size_list(small_sizes_str);
    std::vector<size_t> large_sizes = parse_size_list(large_sizes_str);
    const int resolved_src_page = src_page_numa == -2 ? page_numa : src_page_numa;
    const int resolved_dst_page = dst_page_numa == -2 ? page_numa : dst_page_numa;

    RankCpuInfo local_info;
    local_info.rank = rank;
    gethostname(local_info.hostname, sizeof(local_info.hostname) - 1);
    local_info.current_cpu = sched_getcpu();
    local_info.cpu_numa_node = (numa_available() >= 0 && local_info.current_cpu >= 0)
                                ? numa_node_of_cpu(local_info.current_cpu)
                                : (local_info.current_cpu / 12);
    local_info.affinity_str = get_affinity_string();

    std::vector<char> host_buf(size * 256);
    std::vector<int> cpu_buf(size);
    std::vector<int> numa_buf(size);
    std::vector<int> aff_len_buf(size);
    int aff_len = static_cast<int>(local_info.affinity_str.size());

    MPI_Gather(local_info.hostname, 256, MPI_CHAR, host_buf.data(), 256, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_info.current_cpu, 1, MPI_INT, cpu_buf.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_info.cpu_numa_node, 1, MPI_INT, numa_buf.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&aff_len, 1, MPI_INT, aff_len_buf.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> aff_disps(size);
    int total_aff_len = 0;
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            aff_disps[r] = total_aff_len;
            total_aff_len += aff_len_buf[r];
        }
    }
    std::vector<char> aff_str_buf(total_aff_len > 0 ? total_aff_len : 1);
    MPI_Gatherv(local_info.affinity_str.data(), aff_len, MPI_CHAR,
                aff_str_buf.data(), aff_len_buf.data(), aff_disps.data(), MPI_CHAR,
                0, MPI_COMM_WORLD);

    std::vector<RankCpuInfo> all_ranks(size);
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            all_ranks[r].rank = r;
            std::string h(&host_buf[r * 256]);
            snprintf(all_ranks[r].hostname, sizeof(all_ranks[r].hostname), "%s", h.c_str());
            all_ranks[r].current_cpu = cpu_buf[r];
            all_ranks[r].cpu_numa_node = numa_buf[r];
            all_ranks[r].affinity_str = std::string(&aff_str_buf[aff_disps[r]], aff_len_buf[r]);
        }
        std::cout << "=== Precision CUDA-Aware MPI Host-to-Device & Device-to-Host Benchmark ===\n";
        std::cout << "Rank 0 (CPU): CPU " << all_ranks[0].current_cpu << " (NUMA " << all_ranks[0].cpu_numa_node << ")\n";
        std::cout << "Rank 1 (GPU): CPU " << all_ranks[1].current_cpu << " (NUMA " << all_ranks[1].cpu_numa_node << ")\n";
        std::cout << "Warmup: " << warmup << ", Iters: " << iters << ", Mode: " << transfer_mode << "\n";
        std::cout << "Target GPUs: " << device_str << "\n";
    }

    struct PayloadTask {
        size_t bytes;
        const char* type;
    };
    std::vector<PayloadTask> payloads;
    for (size_t s : small_sizes) payloads.push_back({s, "latency"});
    for (size_t s : large_sizes) payloads.push_back({s, "bandwidth"});

    std::vector<DirectionTransferResult> measured_results;

    for (int gpu_id : target_gpus) {
        char pci_bus_str[64] = "N/A";
        int gpu_pci_numa = -1;

        if (rank == 1) {
            cudaDeviceReset();
            cudaError_t err_dev = cudaSetDevice(gpu_id);
            if (err_dev == cudaSuccess) {
                cudaDeviceProp prop;
                if (cudaGetDeviceProperties(&prop, gpu_id) == cudaSuccess) {
                    snprintf(pci_bus_str, sizeof(pci_bus_str), "%04x:%02x:%02x.0", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
                    gpu_pci_numa = get_gpu_pci_numa_node(prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
                }
            }
        } else {
            // Ensure CUDA context is initialized on rank 0 so smcuda registration cache succeeds
            cudaSetDevice(gpu_id);
            cudaFree(0);
        }

        MPI_Bcast(pci_bus_str, 64, MPI_CHAR, 1, MPI_COMM_WORLD);
        MPI_Bcast(&gpu_pci_numa, 1, MPI_INT, 1, MPI_COMM_WORLD);

        for (const auto& task : payloads) {
            size_t size_bytes = task.bytes;

            DirectionTransferResult res_h2d;
            DirectionTransferResult res_d2h;

            auto init_res = [&](DirectionTransferResult& r, const char* dir) {
                r.rank = 0;
                snprintf(r.hostname, sizeof(r.hostname), "%s", all_ranks[0].hostname);
                r.cpu_id = all_ranks[0].current_cpu;
                r.cpu_numa_node = all_ranks[0].cpu_numa_node;
                snprintf(r.affinity_str, sizeof(r.affinity_str), "%s", all_ranks[0].affinity_str.c_str());
                r.target_gpu = gpu_id;
                snprintf(r.pci_bus_id, sizeof(r.pci_bus_id), "%s", pci_bus_str);
                r.gpu_pci_numa_node = gpu_pci_numa;
                r.payload_bytes = size_bytes;
                snprintf(r.payload_type, sizeof(r.payload_type), "%s", task.type);
                snprintf(r.direction, sizeof(r.direction), "%s", dir);
                r.iters = iters;
                r.page_policy = page_numa;
                r.src_page_policy = resolved_src_page;
                r.dst_page_policy = resolved_dst_page;
                r.reg_mode = reg_mode;
                snprintf(r.variant, sizeof(r.variant), "%s", variant_str.c_str());
                snprintf(r.transfer_mode, sizeof(r.transfer_mode), "%s", transfer_mode.c_str());
            };

            init_res(res_h2d, "H2D");
            init_res(res_d2h, "D2H");

            if (transfer_mode == "d2h-only") snprintf(res_h2d.status, sizeof(res_h2d.status), "NOT_RUN (d2h-only)");
            if (transfer_mode == "h2d-only") snprintf(res_d2h.status, sizeof(res_d2h.status), "NOT_RUN (h2d-only)");

            HostAlloc ha_src, ha_dst;
            void* d_src = nullptr;
            void* d_dst = nullptr;

            if (rank == 0) {
                ha_src = allocate_pinned_host_memory(size_bytes, resolved_src_page, reg_mode, true);
                ha_dst = allocate_pinned_host_memory(size_bytes, resolved_dst_page, reg_mode, false);
                res_h2d.host_buf_numa_node = get_page_numa_node(ha_src.ptr);
                res_d2h.host_buf_numa_node = get_page_numa_node(ha_dst.ptr);

                std::string src_census = count_pages_by_numa(ha_src.ptr, size_bytes);
                std::string dst_census = count_pages_by_numa(ha_dst.ptr, size_bytes);
                snprintf(res_h2d.src_pages, sizeof(res_h2d.src_pages), "%s", src_census.c_str());
                snprintf(res_h2d.dst_pages, sizeof(res_h2d.dst_pages), "%s", dst_census.c_str());
                snprintf(res_d2h.src_pages, sizeof(res_d2h.src_pages), "%s", src_census.c_str());
                snprintf(res_d2h.dst_pages, sizeof(res_d2h.dst_pages), "%s", dst_census.c_str());
            } else {
                cudaMalloc(&d_dst, size_bytes);
                cudaMalloc(&d_src, size_bytes);
            }

            // Sync setup: copy initial payload from host to GPU device source so D2H transfers are populated
            if (rank == 0) {
                MPI_Send(ha_src.ptr, static_cast<int>(size_bytes), MPI_BYTE, 1, 999, MPI_COMM_WORLD);
            } else {
                MPI_Recv(d_src, static_cast<int>(size_bytes), MPI_BYTE, 0, 999, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            MPI_Barrier(MPI_COMM_WORLD);

            uint32_t signal = 1;
            int count_bytes = static_cast<int>(size_bytes);

            auto run_h2d_once = [&]() -> double {
                if (rank == 0) {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    MPI_Send(ha_src.ptr, count_bytes, MPI_BYTE, 1, 100, MPI_COMM_WORLD);
                    MPI_Recv(&signal, 1, MPI_UINT32_T, 1, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    return std::chrono::duration<double, std::micro>(t1 - t0).count();
                } else {
                    MPI_Recv(d_dst, count_bytes, MPI_BYTE, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(&signal, 1, MPI_UINT32_T, 0, 101, MPI_COMM_WORLD);
                    return 0.0;
                }
            };

            auto run_d2h_once = [&]() -> double {
                if (rank == 0) {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    MPI_Send(&signal, 1, MPI_UINT32_T, 1, 201, MPI_COMM_WORLD);
                    MPI_Recv(ha_dst.ptr, count_bytes, MPI_BYTE, 1, 200, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    return std::chrono::duration<double, std::micro>(t1 - t0).count();
                } else {
                    MPI_Recv(&signal, 1, MPI_UINT32_T, 0, 201, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(d_src, count_bytes, MPI_BYTE, 0, 200, MPI_COMM_WORLD);
                    return 0.0;
                }
            };

            // Warmup iterations
            for (int w = 0; w < warmup; ++w) {
                if (transfer_mode == "interleaved") {
                    run_h2d_once();
                    run_d2h_once();
                } else if (transfer_mode == "d2h-only") {
                    run_d2h_once();
                } else if (transfer_mode == "h2d-only") {
                    run_h2d_once();
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);

            std::vector<double> h2d_lats(iters, 0.0);
            std::vector<double> h2d_bws(iters, 0.0);
            std::vector<double> d2h_lats(iters, 0.0);
            std::vector<double> d2h_bws(iters, 0.0);
            double size_gib = static_cast<double>(size_bytes) / (1024.0 * 1024.0 * 1024.0);

            for (int it = 0; it < iters; ++it) {
                if (transfer_mode == "d2h-only") {
                    double us = run_d2h_once();
                    if (rank == 0) {
                        d2h_lats[it] = us;
                        d2h_bws[it] = size_gib / (us * 1e-6);
                    }
                } else if (transfer_mode == "h2d-only") {
                    double us = run_h2d_once();
                    if (rank == 0) {
                        h2d_lats[it] = us;
                        h2d_bws[it] = size_gib / (us * 1e-6);
                    }
                } else {
                    // Interleaved counterbalancing
                    if (it % 2 == 0) {
                        double us_h = run_h2d_once();
                        double us_d = run_d2h_once();
                        if (rank == 0) {
                            h2d_lats[it] = us_h;
                            h2d_bws[it] = size_gib / (us_h * 1e-6);
                            d2h_lats[it] = us_d;
                            d2h_bws[it] = size_gib / (us_d * 1e-6);
                        }
                    } else {
                        double us_d = run_d2h_once();
                        double us_h = run_h2d_once();
                        if (rank == 0) {
                            d2h_lats[it] = us_d;
                            d2h_bws[it] = size_gib / (us_d * 1e-6);
                            h2d_lats[it] = us_h;
                            h2d_bws[it] = size_gib / (us_h * 1e-6);
                        }
                    }
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);

            if (rank == 0) {
                // Post-loop destination page census
                std::string dst_after = count_pages_by_numa(ha_dst.ptr, size_bytes);
                snprintf(res_h2d.dst_pages_after, sizeof(res_h2d.dst_pages_after), "%s", dst_after.c_str());
                snprintf(res_d2h.dst_pages_after, sizeof(res_d2h.dst_pages_after), "%s", dst_after.c_str());

                // Full buffer verification
                if (transfer_mode != "h2d-only") {
                    if (buffer_checksum(ha_src.ptr, size_bytes) != buffer_checksum(ha_dst.ptr, size_bytes)) {
                        snprintf(res_d2h.status, sizeof(res_d2h.status), "Data verification mismatch (full checksum)");
                    }
                }

                auto populate_stats = [&](DirectionTransferResult& r, std::vector<double>& lats, std::vector<double>& bws) {
                    Stats s_lat = compute_stats(lats);
                    r.lat_min_us = s_lat.min;
                    r.lat_p25_us = s_lat.p25;
                    r.lat_median_us = s_lat.median;
                    r.lat_p75_us = s_lat.p75;
                    r.lat_iqr_us = s_lat.iqr;
                    r.lat_max_us = s_lat.max;
                    r.lat_mean_us = s_lat.mean;
                    r.lat_stddev_us = s_lat.stddev;

                    Stats s_bw = compute_stats(bws);
                    r.bw_gibs_min = s_bw.min;
                    r.bw_gibs_p25 = s_bw.p25;
                    r.bw_gibs_median = s_bw.median;
                    r.bw_gibs_p75 = s_bw.p75;
                    r.bw_gibs_iqr = s_bw.iqr;
                    r.bw_gibs_max = s_bw.max;
                    r.bw_gibs_mean = s_bw.mean;
                    r.bw_gibs_stddev = s_bw.stddev;

                    r.bw_gbps_median = r.bw_gibs_median * (1024.0 * 1024.0 * 1024.0 / 1e9);
                    r.gpu_dma_time_median_us = r.lat_median_us;
                };

                populate_stats(res_h2d, h2d_lats, h2d_bws);
                populate_stats(res_d2h, d2h_lats, d2h_bws);

                measured_results.push_back(res_h2d);
                measured_results.push_back(res_d2h);

                free_pinned_host_memory(ha_src);
                free_pinned_host_memory(ha_dst);
            } else {
                if (d_dst) cudaFree(d_dst);
                if (d_src) cudaFree(d_src);
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        bool need_header = true;
        if (append_csv) {
            std::ifstream existing(output_file);
            if (existing && existing.peek() != std::ifstream::traits_type::eof()) {
                need_header = false;
            }
        }
        std::ofstream csv(output_file, append_csv ? std::ios::app : std::ios::trunc);
        if (need_header) {
            csv << "rank,hostname,cpu_id,cpu_affinity,cpu_numa_node,gpu_id,gpu_pci_bus_id,gpu_pci_numa_node,status,"
                << "payload_bytes,payload_type,direction,iters,host_buf_numa_node,"
                << "variant,page_policy,src_page_policy,dst_page_policy,reg_mode,transfer_mode,src_pages,dst_pages,dst_pages_after,"
                << "lat_min_us,lat_p25_us,lat_median_us,lat_p75_us,lat_iqr_us,lat_max_us,lat_mean_us,lat_stddev_us,"
                << "bw_gibs_min,bw_gibs_p25,bw_gibs_median,bw_gibs_p75,bw_gibs_iqr,bw_gibs_max,bw_gibs_mean,bw_gibs_stddev,"
                << "bw_gbps_median,gpu_dma_time_median_us\n";
        }

        for (const auto& r : measured_results) {
            csv << r.rank << ",\"" << r.hostname << "\","
                << r.cpu_id << ",\"" << r.affinity_str << "\","
                << r.cpu_numa_node << ","
                << r.target_gpu << ",\"" << r.pci_bus_id << "\"," << r.gpu_pci_numa_node << ",\""
                << r.status << "\","
                << r.payload_bytes << ",\"" << r.payload_type << "\",\"" << r.direction << "\","
                << r.iters << "," << r.host_buf_numa_node << ",\""
                << r.variant << "\"," << r.page_policy << "," << r.src_page_policy << ","
                << r.dst_page_policy << "," << r.reg_mode << ",\"" << r.transfer_mode << "\",\""
                << r.src_pages << "\",\"" << r.dst_pages << "\",\"" << r.dst_pages_after << "\","
                << r.lat_min_us << "," << r.lat_p25_us << "," << r.lat_median_us << ","
                << r.lat_p75_us << "," << r.lat_iqr_us << "," << r.lat_max_us << ","
                << r.lat_mean_us << "," << r.lat_stddev_us << ","
                << r.bw_gibs_min << "," << r.bw_gibs_p25 << "," << r.bw_gibs_median << ","
                << r.bw_gibs_p75 << "," << r.bw_gibs_iqr << "," << r.bw_gibs_max << ","
                << r.bw_gibs_mean << "," << r.bw_gibs_stddev << ","
                << r.bw_gbps_median << "," << r.gpu_dma_time_median_us << "\n";
        }
        csv.close();
        std::cout << "[Success] CUDA-aware MPI GPU transfer results written to '" << output_file << "'.\n";
    }

    MPI_Finalize();
    return 0;
}
