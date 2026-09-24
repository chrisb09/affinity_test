#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <sched.h>
#include <unistd.h>
#include <numa.h>
#include <numaif.h>
#include <mpi.h>
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

    // Diagnostic metadata (D2H asymmetry rerun campaign)
    int page_policy{-1};            // -1 = auto first-touch from bound CPU; >=0 = forced NUMA node
    int reg_mode{0};                // 0 = cudaHostRegister Portable|Mapped (original), 1 = Portable, 2 = Mapped, 3 = cudaHostAlloc
    char variant[64]{0};            // free-form run label from --variant
    char src_pages[192]{0};         // per-page NUMA census of h_src before the timed loop ("node:count,...")
    char dst_pages[192]{0};         // per-page NUMA census of h_dst before the timed loop
    char dst_pages_after[192]{0};   // per-page NUMA census of h_dst after the timed loop

    // Wall-clock latency (microseconds)
    double lat_min_us{0.0};
    double lat_p25_us{0.0};
    double lat_median_us{0.0};
    double lat_p75_us{0.0};
    double lat_iqr_us{0.0};
    double lat_max_us{0.0};
    double lat_mean_us{0.0};
    double lat_stddev_us{0.0};

    // Bandwidth in GiB/s (1024^3 bytes/sec)
    double bw_gibs_min{0.0};
    double bw_gibs_p25{0.0};
    double bw_gibs_median{0.0};
    double bw_gibs_p75{0.0};
    double bw_gibs_iqr{0.0};
    double bw_gibs_max{0.0};
    double bw_gibs_mean{0.0};
    double bw_gibs_stddev{0.0};

    // Bandwidth in GB/s (10^9 bytes/sec)
    double bw_gbps_median{0.0};

    // Pure GPU DMA Hardware time (from CUDA Events) in microseconds
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

// Full per-page NUMA residency census of a buffer; returns "node:count,node:count,...".
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

// FNV-1a checksum over the full buffer (8-byte words + tail bytes).
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

// Host allocation modes for the diagnostic rerun:
//   reg_mode 0: posix_memalign/numa_alloc_on_node + first-touch + cudaHostRegister(Portable|Mapped)  [original behavior]
//   reg_mode 1: same allocation + cudaHostRegister(Portable)
//   reg_mode 2: same allocation + cudaHostRegister(Mapped)
//   reg_mode 3: cudaMallocHost (cudaHostAllocDefault; driver-chosen placement, always pinned)
// page_node: -1 = auto first-touch from the bound CPU; >=0 = allocate on that NUMA node
//            (ignored for reg_mode 3, which the driver pins itself).
struct HostAlloc {
    void* ptr{nullptr};
    size_t size{0};
    int reg_mode{0};
    bool from_numa_alloc{false};
    bool ok{false};
};

static HostAlloc allocate_pinned_host_memory(size_t size, int page_node, int reg_mode) {
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

    // Strict first-touch write from calling bound CPU thread across every 4096-byte page
    volatile char* cp = static_cast<volatile char*>(ptr);
    for (size_t off = 0; off < size; off += 4096) {
        cp[off] = static_cast<char>((off ^ (off >> 8)) & 0x7F);
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

// Record a CUDA error into a status buffer without aborting the sweep.
static void cuda_note_error(cudaError_t err, char* status, const char* what) {
    if (err != cudaSuccess && status && status[0] == 'O') {
        snprintf(status, 128, "%s failed: %s", what, cudaGetErrorString(err));
    }
}

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  -o, --output <file>         CSV output filename (default: gpu_transfer_latency.csv)\n"
              << "  -d, --devices <0,1,...>     Comma-separated CUDA device IDs to test (default: 0,1,2,3)\n"
              << "  -s, --small-sizes <b1,b2>   Comma-separated small sizes for latency (default: 4096)\n"
              << "  -l, --large-sizes <b1,b2>   Comma-separated large sizes for bandwidth (default: 16777216,67108864,268435456)\n"
              << "  -w, --warmup <count>        Warmup iterations (default: 5)\n"
              << "  -i, --iters <count>         Measured iterations per test (default: 50)\n"
              << "  -p, --page-numa <node>      Force host buffer pages onto NUMA <node> (-1 = auto first-touch, default)\n"
              << "  -r, --reg-mode <0-3>        Host pinning mode: 0=Register(Portable|Mapped), 1=Portable, 2=Mapped, 3=cudaHostAlloc (default: 0)\n"
              << "  -v, --variant <label>       Free-form run label written to the CSV (default: empty)\n"
              << "  -a, --append                Append to the output CSV instead of overwriting (header written only for empty files)\n"
              << "  -h, --help                  Show this help message\n";
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string output_file = "gpu_transfer_latency.csv";
    std::string device_str = "0,1,2,3";
    std::string small_sizes_str = "4096";
    std::string large_sizes_str = "16777216,67108864,268435456";
    int warmup = 5;
    int iters = 50;
    int page_numa = -1;
    int reg_mode = 0;
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
        } else if ((arg == "-r" || arg == "--reg-mode") && i + 1 < argc) {
            reg_mode = std::stoi(argv[++i]);
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

    std::vector<int> target_gpus = parse_gpu_list(device_str);
    std::vector<size_t> small_sizes = parse_size_list(small_sizes_str);
    std::vector<size_t> large_sizes = parse_size_list(large_sizes_str);

    RankCpuInfo local_info;
    local_info.rank = rank;
    gethostname(local_info.hostname, sizeof(local_info.hostname) - 1);
    local_info.current_cpu = sched_getcpu();
    local_info.cpu_numa_node = (numa_available() >= 0 && local_info.current_cpu >= 0) 
                                ? numa_node_of_cpu(local_info.current_cpu) 
                                : (local_info.current_cpu / 12);
    local_info.affinity_str = get_affinity_string();

    // Gather rank metadata to rank 0
    std::vector<char> host_buf;
    std::vector<int> cpu_buf;
    std::vector<int> numa_buf;
    std::vector<int> aff_len_buf;
    std::vector<char> aff_str_buf;

    int aff_len = static_cast<int>(local_info.affinity_str.size());

    if (rank == 0) {
        host_buf.resize(size * 256);
        cpu_buf.resize(size);
        numa_buf.resize(size);
        aff_len_buf.resize(size);
    }

    MPI_Gather(local_info.hostname, 256, MPI_CHAR, host_buf.data(), 256, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_info.current_cpu, 1, MPI_INT, cpu_buf.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_info.cpu_numa_node, 1, MPI_INT, numa_buf.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
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
        std::cout << "=== Precision GPU Host-to-Device & Device-to-Host Benchmark ===\n";
        std::cout << "Ranks: " << size << ", Warmup: " << warmup << ", Iters: " << iters << "\n";
        std::cout << "Target GPUs: " << device_str << "\n";
        std::cout << "Small sizes: " << small_sizes_str << " B\n";
        std::cout << "Large sizes: " << large_sizes_str << " B\n";
        for (int r = 0; r < size; ++r) {
            all_ranks[r].rank = r;
            std::string h(&host_buf[r * 256]);
            snprintf(all_ranks[r].hostname, sizeof(all_ranks[r].hostname), "%s", h.c_str());
            all_ranks[r].current_cpu = cpu_buf[r];
            all_ranks[r].cpu_numa_node = numa_buf[r];
            all_ranks[r].affinity_str = std::string(&aff_str_buf[aff_disps[r]], aff_len_buf[r]);
        }
        std::cout << "Starting serialized token-ring measurements across all ranks...\n" << std::flush;
    }

    std::vector<DirectionTransferResult> my_results;

    // Combine payloads to test
    struct PayloadTask {
        size_t bytes;
        const char* type;
    };
    std::vector<PayloadTask> payloads;
    for (size_t s : small_sizes) payloads.push_back({s, "latency"});
    for (size_t s : large_sizes) payloads.push_back({s, "bandwidth"});

    for (int gpu_id : target_gpus) {
        int gpu_ok = 1;
        char gpu_err_msg[128] = "OK";
        int dev_count = 0;
        cudaGetDeviceCount(&dev_count);

        if (gpu_id >= dev_count) {
            gpu_ok = 0;
            snprintf(gpu_err_msg, sizeof(gpu_err_msg), "GPU %d out of bounds (count %d)", gpu_id, dev_count);
        }

        if (rank == 0 && gpu_ok) {
            cudaDeviceReset();
            cudaDeviceProp prop;
            cudaError_t err = cudaGetDeviceProperties(&prop, gpu_id);
            if (err != cudaSuccess) {
                gpu_ok = 0;
                snprintf(gpu_err_msg, sizeof(gpu_err_msg), "cudaGetDeviceProperties failed: %s", cudaGetErrorString(err));
            } else if (prop.computeMode != cudaComputeModeDefault) {
                gpu_ok = 0;
                snprintf(gpu_err_msg, sizeof(gpu_err_msg), "ComputeMode %d not default", prop.computeMode);
            }
        }

        MPI_Bcast(&gpu_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(gpu_err_msg, 128, MPI_CHAR, 0, MPI_COMM_WORLD);

        if (!gpu_ok) {
            if (rank == 0) {
                std::cout << "GPU " << gpu_id << " unavailable (" << gpu_err_msg << "), skipping.\n" << std::flush;
            }
            continue;
        }

        // Token ring serialization so only one MPI rank accesses the PCIe bus at a time
        int token = 0;
        if (rank != 0) {
            MPI_Recv(&token, 1, MPI_INT, rank - 1, 777, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (rank % 12 == 0) {
            std::cout << "[Rank " << rank << " | CPU " << local_info.current_cpu 
                      << " (NUMA " << local_info.cpu_numa_node << ")] Testing GPU " << gpu_id << "...\n" << std::flush;
        }

        cudaDeviceReset();
        cudaError_t err = cudaSetDevice(gpu_id);
        
        char pci_bus_str[64] = "N/A";
        int gpu_pci_numa = -1;

        if (err == cudaSuccess) {
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, gpu_id) == cudaSuccess) {
                snprintf(pci_bus_str, sizeof(pci_bus_str), "%04x:%02x:%02x.0", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
                gpu_pci_numa = get_gpu_pci_numa_node(prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
            }
        }

        for (const auto& task : payloads) {
            size_t size = task.bytes;

            DirectionTransferResult res_h2d;
            DirectionTransferResult res_d2h;

            auto init_res = [&](DirectionTransferResult& r, const char* dir) {
                r.rank = rank;
                snprintf(r.hostname, sizeof(r.hostname), "%s", local_info.hostname);
                r.cpu_id = local_info.current_cpu;
                r.cpu_numa_node = local_info.cpu_numa_node;
                snprintf(r.affinity_str, sizeof(r.affinity_str), "%s", local_info.affinity_str.c_str());
                r.target_gpu = gpu_id;
                snprintf(r.pci_bus_id, sizeof(r.pci_bus_id), "%s", pci_bus_str);
                r.gpu_pci_numa_node = gpu_pci_numa;
                r.payload_bytes = size;
                snprintf(r.payload_type, sizeof(r.payload_type), "%s", task.type);
                snprintf(r.direction, sizeof(r.direction), "%s", dir);
                r.iters = iters;
                r.page_policy = page_numa;
                r.reg_mode = reg_mode;
                snprintf(r.variant, sizeof(r.variant), "%s", variant_str.c_str());
            };

            init_res(res_h2d, "H2D");
            init_res(res_d2h, "D2H");

            if (err != cudaSuccess) {
                snprintf(res_h2d.status, sizeof(res_h2d.status), "cudaSetDevice failed: %s", cudaGetErrorString(err));
                snprintf(res_d2h.status, sizeof(res_d2h.status), "cudaSetDevice failed: %s", cudaGetErrorString(err));
                my_results.push_back(res_h2d);
                my_results.push_back(res_d2h);
                continue;
            }

            // Allocate distinct pinned host buffers and device buffers
            HostAlloc ha_src = allocate_pinned_host_memory(size, page_numa, reg_mode);
            HostAlloc ha_dst = allocate_pinned_host_memory(size, page_numa, reg_mode);
            void* h_src = ha_src.ptr;
            void* h_dst = ha_dst.ptr;
            void* d_dst = nullptr;
            void* d_src = nullptr;

            cudaError_t err_d1 = cudaMalloc(&d_dst, size);
            cudaError_t err_d2 = cudaMalloc(&d_src, size);

            if (!ha_src.ok || !ha_dst.ok || err_d1 != cudaSuccess || err_d2 != cudaSuccess) {
                snprintf(res_h2d.status, sizeof(res_h2d.status), "Memory allocation failed");
                snprintf(res_d2h.status, sizeof(res_d2h.status), "Memory allocation failed");
                free_pinned_host_memory(ha_src);
                free_pinned_host_memory(ha_dst);
                if (d_dst) cudaFree(d_dst);
                if (d_src) cudaFree(d_src);
                my_results.push_back(res_h2d);
                my_results.push_back(res_d2h);
                continue;
            }

            // Verify host buffer physical NUMA residency (first page + full per-page census)
            int h_src_numa = get_page_numa_node(h_src);
            int h_dst_numa = get_page_numa_node(h_dst);
            res_h2d.host_buf_numa_node = h_src_numa;
            res_d2h.host_buf_numa_node = h_dst_numa;
            {
                std::string src_census = count_pages_by_numa(h_src, size);
                std::string dst_census = count_pages_by_numa(h_dst, size);
                snprintf(res_h2d.src_pages, sizeof(res_h2d.src_pages), "%s", src_census.c_str());
                snprintf(res_h2d.dst_pages, sizeof(res_h2d.dst_pages), "%s", dst_census.c_str());
                snprintf(res_d2h.src_pages, sizeof(res_d2h.src_pages), "%s", src_census.c_str());
                snprintf(res_d2h.dst_pages, sizeof(res_d2h.dst_pages), "%s", dst_census.c_str());
            }

            // Populate d_src from h_src prior to timing so device source is valid
            cuda_note_error(cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice),
                            res_h2d.status, "initial H2D populate");
            cuda_note_error(cudaDeviceSynchronize(), res_h2d.status, "initial sync");

            // Setup CUDA Stream and Events for accurate hardware DMA timing
            cudaStream_t stream;
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
            cudaEvent_t ev_start, ev_stop;
            cudaEventCreate(&ev_start);
            cudaEventCreate(&ev_stop);

            // Warmup (Counterbalanced interleaved transfers)
            for (int w = 0; w < warmup; ++w) {
                cudaMemcpyAsync(d_dst, h_src, size, cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(h_dst, d_src, size, cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
            }
            {
                cudaError_t warm_err = cudaGetLastError();
                cuda_note_error(warm_err, res_h2d.status, "warmup CUDA error");
                cuda_note_error(warm_err, res_d2h.status, "warmup CUDA error");
            }

            // Timed Measurements
            std::vector<double> h2d_lats(iters);
            std::vector<double> h2d_dma_lats(iters);
            std::vector<double> h2d_bws_gibs(iters);
            std::vector<double> d2h_lats(iters);
            std::vector<double> d2h_dma_lats(iters);
            std::vector<double> d2h_bws_gibs(iters);

            double size_gib = static_cast<double>(size) / (1024.0 * 1024.0 * 1024.0);

            for (int it = 0; it < iters; ++it) {
                // Alternating / counterbalancing direction order based on iteration to eliminate ordering bias
                if (it % 2 == 0) {
                    // H2D first
                    cudaEventRecord(ev_start, stream);
                    auto t0_h2d = std::chrono::high_resolution_clock::now();
                    cudaMemcpyAsync(d_dst, h_src, size, cudaMemcpyHostToDevice, stream);
                    cudaEventRecord(ev_stop, stream);
                    cudaStreamSynchronize(stream);
                    auto t1_h2d = std::chrono::high_resolution_clock::now();

                    float dma_ms = 0.0f;
                    cudaEventElapsedTime(&dma_ms, ev_start, ev_stop);
                    double wall_us = std::chrono::duration<double, std::micro>(t1_h2d - t0_h2d).count();
                    h2d_lats[it] = wall_us;
                    h2d_dma_lats[it] = dma_ms * 1000.0;
                    h2d_bws_gibs[it] = size_gib / (wall_us * 1e-6);

                    // D2H second
                    cudaEventRecord(ev_start, stream);
                    auto t0_d2h = std::chrono::high_resolution_clock::now();
                    cudaMemcpyAsync(h_dst, d_src, size, cudaMemcpyDeviceToHost, stream);
                    cudaEventRecord(ev_stop, stream);
                    cudaStreamSynchronize(stream);
                    auto t1_d2h = std::chrono::high_resolution_clock::now();

                    dma_ms = 0.0f;
                    cudaEventElapsedTime(&dma_ms, ev_start, ev_stop);
                    wall_us = std::chrono::duration<double, std::micro>(t1_d2h - t0_d2h).count();
                    d2h_lats[it] = wall_us;
                    d2h_dma_lats[it] = dma_ms * 1000.0;
                    d2h_bws_gibs[it] = size_gib / (wall_us * 1e-6);
                } else {
                    // D2H first
                    cudaEventRecord(ev_start, stream);
                    auto t0_d2h = std::chrono::high_resolution_clock::now();
                    cudaMemcpyAsync(h_dst, d_src, size, cudaMemcpyDeviceToHost, stream);
                    cudaEventRecord(ev_stop, stream);
                    cudaStreamSynchronize(stream);
                    auto t1_d2h = std::chrono::high_resolution_clock::now();

                    float dma_ms = 0.0f;
                    cudaEventElapsedTime(&dma_ms, ev_start, ev_stop);
                    double wall_us = std::chrono::duration<double, std::micro>(t1_d2h - t0_d2h).count();
                    d2h_lats[it] = wall_us;
                    d2h_dma_lats[it] = dma_ms * 1000.0;
                    d2h_bws_gibs[it] = size_gib / (wall_us * 1e-6);

                    // H2D second
                    cudaEventRecord(ev_start, stream);
                    auto t0_h2d = std::chrono::high_resolution_clock::now();
                    cudaMemcpyAsync(d_dst, h_src, size, cudaMemcpyHostToDevice, stream);
                    cudaEventRecord(ev_stop, stream);
                    cudaStreamSynchronize(stream);
                    auto t1_h2d = std::chrono::high_resolution_clock::now();

                    dma_ms = 0.0f;
                    cudaEventElapsedTime(&dma_ms, ev_start, ev_stop);
                    wall_us = std::chrono::duration<double, std::micro>(t1_h2d - t0_h2d).count();
                    h2d_lats[it] = wall_us;
                    h2d_dma_lats[it] = dma_ms * 1000.0;
                    h2d_bws_gibs[it] = size_gib / (wall_us * 1e-6);
                }
            }

            // Sticky CUDA error check over the timed loop
            {
                cudaError_t loop_err = cudaGetLastError();
                cuda_note_error(loop_err, res_h2d.status, "timed loop CUDA error");
                cuda_note_error(loop_err, res_d2h.status, "timed loop CUDA error");
            }

            // Verify payload integrity outside timed loop (full-buffer checksum:
            // the timed D2H copies wrote d_src into h_dst, so both must match)
            if (buffer_checksum(h_src, size) != buffer_checksum(h_dst, size)) {
                snprintf(res_d2h.status, sizeof(res_d2h.status), "Data verification mismatch (full checksum)");
            }

            // Post-loop NUMA census of the D2H destination buffer (detects migration)
            {
                std::string dst_after = count_pages_by_numa(h_dst, size);
                snprintf(res_h2d.dst_pages_after, sizeof(res_h2d.dst_pages_after), "%s", dst_after.c_str());
                snprintf(res_d2h.dst_pages_after, sizeof(res_d2h.dst_pages_after), "%s", dst_after.c_str());
            }

            // Calculate comprehensive statistics
            auto populate_stats = [&](DirectionTransferResult& r, std::vector<double>& lats, std::vector<double>& bws, std::vector<double>& dmas) {
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

                // Standard GB/s (10^9 B/s) = GiB/s * (1024^3 / 1e9)
                r.bw_gbps_median = r.bw_gibs_median * (1024.0 * 1024.0 * 1024.0 / 1e9);

                Stats s_dma = compute_stats(dmas);
                r.gpu_dma_time_median_us = s_dma.median;
            };

            populate_stats(res_h2d, h2d_lats, h2d_bws_gibs, h2d_dma_lats);
            populate_stats(res_d2h, d2h_lats, d2h_bws_gibs, d2h_dma_lats);

            my_results.push_back(res_h2d);
            my_results.push_back(res_d2h);

            // Cleanup stream, events, and allocations
            cudaEventDestroy(ev_start);
            cudaEventDestroy(ev_stop);
            cudaStreamDestroy(stream);
            free_pinned_host_memory(ha_src);
            free_pinned_host_memory(ha_dst);
            cudaFree(d_dst);
            cudaFree(d_src);
        }

        cudaDeviceReset();

        // Pass token to next rank
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
    std::vector<DirectionTransferResult> all_results;
    if (rank == 0) {
        all_results.resize(size * num_items);
    }

    MPI_Gather(my_results.data(), num_items * sizeof(DirectionTransferResult), MPI_BYTE,
               all_results.data(), num_items * sizeof(DirectionTransferResult), MPI_BYTE,
               0, MPI_COMM_WORLD);

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
                << "variant,page_policy,reg_mode,src_pages,dst_pages,dst_pages_after,"
                << "lat_min_us,lat_p25_us,lat_median_us,lat_p75_us,lat_iqr_us,lat_max_us,lat_mean_us,lat_stddev_us,"
                << "bw_gibs_min,bw_gibs_p25,bw_gibs_median,bw_gibs_p75,bw_gibs_iqr,bw_gibs_max,bw_gibs_mean,bw_gibs_stddev,"
                << "bw_gbps_median,gpu_dma_time_median_us\n";
        }

        for (const auto& r : all_results) {
            csv << r.rank << ",\"" << r.hostname << "\","
                << r.cpu_id << ",\"" << r.affinity_str << "\","
                << r.cpu_numa_node << ","
                << r.target_gpu << "," << r.pci_bus_id << "," << r.gpu_pci_numa_node << ",\""
                << r.status << "\","
                << r.payload_bytes << ",\"" << r.payload_type << "\",\"" << r.direction << "\","
                << r.iters << "," << r.host_buf_numa_node << ",\""
                << r.variant << "\"," << r.page_policy << "," << r.reg_mode << ",\""
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
        std::cout << "[Success] Precision GPU transfer results written to '" << output_file << "'.\n";
    }

    MPI_Finalize();
    return 0;
}
