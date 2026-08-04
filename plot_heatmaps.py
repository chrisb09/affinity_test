#!/usr/bin/env python3
"""
Heatmap visualization script for CPU core-to-core latency & bandwidth and GPU rank transfer benchmarks.
Usage:
    python3 plot_heatmaps.py [--cpu-csv cpu_192ranks_pair_latency.csv] [--gpu-csv gpu_c23g_transfer_latency.csv]
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_cpu_pair_heatmaps(csv_path, output_prefix="cpu_pair"):
    """
    Plots NxN rank pair latency and bandwidth heatmaps for CPU benchmark.
    """
    if not os.path.exists(csv_path):
        print(f"[Warning] CPU CSV file not found: {csv_path}. Skipping CPU heatmaps.")
        return

    df = pd.read_csv(csv_path)
    ranks = sorted(list(set(df['rank_i']).union(set(df['rank_j']))))
    n = len(ranks)
    rank_map = {r: idx for idx, r in enumerate(ranks)}

    lat_mat = np.zeros((n, n), dtype=float)
    bw_mat = np.zeros((n, n), dtype=float)

    has_bw = 'bandwidth_median_gbps' in df.columns

    for _, row in df.iterrows():
        i = rank_map[int(row['rank_i'])]
        j = rank_map[int(row['rank_j'])]
        lat_val = float(row['latency_median_us'])
        lat_mat[i, j] = lat_val
        lat_mat[j, i] = lat_val
        if has_bw:
            bw_val = float(row['bandwidth_median_gbps'])
            bw_mat[i, j] = bw_val
            bw_mat[j, i] = bw_val

    np.fill_diagonal(lat_mat, np.nan)
    np.fill_diagonal(bw_mat, np.nan)

    # 1. Latency Heatmap
    plt.figure(figsize=(11, 9), dpi=300)
    cmap_lat = sns.color_palette("YlOrRd", as_cmap=True)
    cmap_lat.set_bad(color="gainsboro")

    ax1 = sns.heatmap(
        lat_mat,
        cmap=cmap_lat,
        cbar_kws={'label': 'Median One-Way Latency (µs)'},
        square=True,
        xticklabels=24 if n >= 96 else (12 if n >= 48 else True),
        yticklabels=24 if n >= 96 else (12 if n >= 48 else True),
    )

    if n == 192:
        ax1.axhline(96, color='blue', linewidth=2.5, linestyle='--')
        ax1.axvline(96, color='blue', linewidth=2.5, linestyle='--')
        ax1.text(48, -3, "Node 0 (Ranks 0-95)", color="blue", fontsize=11, ha="center", weight="bold")
        ax1.text(144, -3, "Node 1 (Ranks 96-191)", color="blue", fontsize=11, ha="center", weight="bold")
        ax1.text(-3, 48, "Node 0", color="blue", fontsize=11, va="center", ha="right", weight="bold", rotation=90)
        ax1.text(-3, 144, "Node 1", color="blue", fontsize=11, va="center", ha="right", weight="bold", rotation=90)

    plt.title(f"MPI Core-to-Core Latency Matrix ({n} Ranks)", fontsize=13, pad=20, weight="bold")
    plt.xlabel("MPI Rank", fontsize=11)
    plt.ylabel("MPI Rank", fontsize=11)

    lat_output = f"{output_prefix}_latency_heatmap.png"
    plt.tight_layout()
    plt.savefig(lat_output, dpi=300)
    plt.close()
    print(f"[Success] Saved CPU latency heatmap to '{lat_output}'.")

    # 2. Bandwidth Heatmap
    if has_bw:
        plt.figure(figsize=(11, 9), dpi=300)
        cmap_bw = sns.color_palette("viridis", as_cmap=True)
        cmap_bw.set_bad(color="gainsboro")

        ax2 = sns.heatmap(
            bw_mat,
            cmap=cmap_bw,
            cbar_kws={'label': 'Median Bandwidth (GB/s)'},
            square=True,
            xticklabels=24 if n >= 96 else (12 if n >= 48 else True),
            yticklabels=24 if n >= 96 else (12 if n >= 48 else True),
        )

        if n == 192:
            ax2.axhline(96, color='white', linewidth=2.5, linestyle='--')
            ax2.axvline(96, color='white', linewidth=2.5, linestyle='--')
            ax2.text(48, -3, "Node 0 (Ranks 0-95)", color="white", fontsize=11, ha="center", weight="bold")
            ax2.text(144, -3, "Node 1 (Ranks 96-191)", color="white", fontsize=11, ha="center", weight="bold")

            # Draw dashed lines for NUMA domains (every 12 ranks)
            for i in range(12, n, 12):
                if i != 96:
                    ax2.axhline(i, color='gray', linewidth=0.5, linestyle=':')
                    ax2.axvline(i, color='gray', linewidth=0.5, linestyle=':')

        plt.title(f"MPI Core-to-Core Bandwidth Matrix ({n} Ranks)", fontsize=13, pad=20, weight="bold")
        plt.xlabel("MPI Rank", fontsize=11)
        plt.ylabel("MPI Rank", fontsize=11)

        bw_output = f"{output_prefix}_bandwidth_heatmap.png"
        plt.tight_layout()
        plt.savefig(bw_output, dpi=300)
        plt.close()
        print(f"[Success] Saved CPU bandwidth heatmap to '{bw_output}'.")


def plot_gpu_transfer_heatmaps(csv_path, output_prefix="gpu_transfer"):
    """
    Plots highly readable NUMA-aggregated and full-rank GPU bandwidth & latency heatmaps.
    """
    if not os.path.exists(csv_path):
        print(f"[Warning] GPU CSV file not found: {csv_path}. Skipping GPU heatmaps.")
        return

    df = pd.read_csv(csv_path)

    # Compute NUMA node ID (12 cores per NUMA node on Sapphire Rapids)
    df['numa_node'] = df['cpu'] // 12

    # GPU Labels
    pci_map = {}
    for gid in sorted(df['gpu_id'].unique()):
        sub = df[df['gpu_id'] == gid]
        pci = sub['pci_bus_id'].dropna().iloc[0] if len(sub['pci_bus_id'].dropna()) > 0 else "N/A"
        pci_map[gid] = f"GPU {gid}\n({pci})"

    df['gpu_label'] = df['gpu_id'].map(pci_map)
    gpu_labels = [pci_map[gid] for gid in sorted(pci_map.keys())]

    # NUMA Domain Labels
    numa_labels = {
        0: "NUMA 0 (CPUs 0-11)\n[Socket 0 - Local GPU 0]",
        1: "NUMA 1 (CPUs 12-23)\n[Socket 0]",
        2: "NUMA 2 (CPUs 24-35)\n[Socket 0 - Local GPU 1]",
        3: "NUMA 3 (CPUs 36-47)\n[Socket 0]",
        4: "NUMA 4 (CPUs 48-59)\n[Socket 1]",
        5: "NUMA 5 (CPUs 60-71)\n[Socket 1]",
        6: "NUMA 6 (CPUs 72-83)\n[Socket 1 - Local GPU 2/3]",
        7: "NUMA 7 (CPUs 84-95)\n[Socket 1]"
    }
    df['numa_label'] = df['numa_node'].map(numa_labels)

    # Filter OK status rows for clean visualization
    ok_df = df[df['status'] == 'OK']

    # --- 1. NUMA Domain Aggregated Heatmap (8 NUMA Rows x 4 GPUs) ---
    numa_d2h = ok_df.groupby(['numa_label', 'gpu_label'])['d2h_bw_median_gbps'].mean().unstack()
    numa_h2d = ok_df.groupby(['numa_label', 'gpu_label'])['h2d_bw_median_gbps'].mean().unstack()
    numa_lat = ok_df.groupby(['numa_label', 'gpu_label'])['d2h_lat_median_us'].mean().unstack()

    # Sort rows by NUMA node order 0..7
    ordered_numa = [numa_labels[i] for i in range(8) if numa_labels[i] in numa_d2h.index]
    numa_d2h = numa_d2h.reindex(ordered_numa)
    numa_h2d = numa_h2d.reindex(ordered_numa)
    numa_lat = numa_lat.reindex(ordered_numa)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=300)
    cmap_bw = sns.color_palette("viridis", as_cmap=True)

    sns.heatmap(
        numa_d2h,
        annot=True,
        fmt=".1f",
        annot_kws={"size": 11, "weight": "bold"},
        cmap=cmap_bw,
        cbar_kws={'label': 'Bandwidth (GB/s)'},
        ax=axes[0],
        linewidths=1.0,
        linecolor="white"
    )
    axes[0].set_title("Device-to-Host (D2H) Bandwidth (GB/s)\nAggregated by NUMA Domain", fontsize=12, weight="bold")
    axes[0].set_xlabel("Target CUDA GPU", fontsize=10, weight="bold")
    axes[0].set_ylabel("NUMA Domain Placement", fontsize=10, weight="bold")

    sns.heatmap(
        numa_h2d,
        annot=True,
        fmt=".1f",
        annot_kws={"size": 11, "weight": "bold"},
        cmap=cmap_bw,
        cbar_kws={'label': 'Bandwidth (GB/s)'},
        ax=axes[1],
        linewidths=1.0,
        linecolor="white"
    )
    axes[1].set_title("Host-to-Device (H2D) Bandwidth (GB/s)\nAggregated by NUMA Domain", fontsize=12, weight="bold")
    axes[1].set_xlabel("Target CUDA GPU", fontsize=10, weight="bold")
    axes[1].set_ylabel("")

    plt.tight_layout()
    numa_bw_out = f"{output_prefix}_numa_bandwidth_heatmap.png"
    plt.savefig(numa_bw_out, dpi=300)
    plt.close()
    print(f"[Success] Saved NUMA aggregated GPU bandwidth heatmap to '{numa_bw_out}'.")

    # --- 2. Full 96-Rank Heatmap with NUMA Grid Separators ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 16), dpi=300)
    
    # Pivot per rank
    piv_d2h = ok_df.pivot(index='rank', columns='gpu_label', values='d2h_bw_median_gbps')
    piv_h2d = ok_df.pivot(index='rank', columns='gpu_label', values='h2d_bw_median_gbps')

    sns.heatmap(
        piv_d2h,
        cmap=cmap_bw,
        cbar_kws={'label': 'Bandwidth (GB/s)'},
        ax=axes[0],
        yticklabels=6
    )
    axes[0].set_title("D2H Bandwidth per MPI Rank (GB/s)", fontsize=12, weight="bold")
    axes[0].set_xlabel("Target CUDA GPU", fontsize=10, weight="bold")
    axes[0].set_ylabel("MPI Rank Index (0-95)", fontsize=10, weight="bold")

    sns.heatmap(
        piv_h2d,
        cmap=cmap_bw,
        cbar_kws={'label': 'Bandwidth (GB/s)'},
        ax=axes[1],
        yticklabels=6
    )
    axes[1].set_title("H2D Bandwidth per MPI Rank (GB/s)", fontsize=12, weight="bold")
    axes[1].set_xlabel("Target CUDA GPU", fontsize=10, weight="bold")
    axes[1].set_ylabel("")

    # Add NUMA domain horizontal line separators every 12 ranks
    for ax in axes:
        for r in range(12, 96, 12):
            ax.axhline(r, color="white", linewidth=1.5, linestyle="--")
        # Socket separator at rank 48
        ax.axhline(48, color="red", linewidth=2.5, linestyle="-")

    plt.tight_layout()
    rank_bw_out = f"{output_prefix}_rank_bandwidth_heatmap.png"
    plt.savefig(rank_bw_out, dpi=300)
    plt.close()
    print(f"[Success] Saved 96-rank GPU bandwidth heatmap to '{rank_bw_out}'.")


def main():
    parser = argparse.ArgumentParser(description="Generate heatmaps for MPI CPU pair latency/bandwidth & GPU transfers.")
    parser.add_argument("--cpu-csv", default="cpu_192ranks_pair_latency.csv", help="Path to CPU CSV file")
    parser.add_argument("--gpu-csv", default="gpu_c23g_transfer_latency.csv", help="Path to GPU CSV file")
    args = parser.parse_args()

    print(f"Generating heatmaps using:\n  CPU CSV: {args.cpu_csv}\n  GPU CSV: {args.gpu_csv}")
    plot_cpu_pair_heatmaps(args.cpu_csv)
    plot_gpu_transfer_heatmaps(args.gpu_csv)


if __name__ == "__main__":
    main()
