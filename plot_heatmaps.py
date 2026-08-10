#!/usr/bin/env python3
"""
Heatmap visualization script for CPU/TCP pair latency & bandwidth and GPU rank transfer benchmarks.

Usage:
    # MPI only
    python3 plot_heatmaps.py --cpu-csv cpu_192ranks_pair_latency.csv

    # TCP only
    python3 plot_heatmaps.py --tcp-csv tcp_192ranks_pair_latency.csv --tcp-prefix tcp_pair

    # Both (produces combined locality-tier comparison CSV)
    python3 plot_heatmaps.py --cpu-csv cpu_192ranks_pair_latency.csv \\
                             --tcp-csv tcp_192ranks_pair_latency.csv

    # GPU
    python3 plot_heatmaps.py --gpu-csv gpu_c23g_transfer_latency.csv

Locality tiers (derived from hostname and CPU id; 12 CPUs per NUMA domain on SPR):
    same_numa             same host, same NUMA domain   (CPUs 0-11, 12-23, …)
    same_socket_diff_numa same host, same socket        (CPUs 0-47 or 48-95)
    cross_socket          same host, different sockets
    cross_node            different hostnames
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


    plt.tight_layout()
    rank_bw_out = f"{output_prefix}_rank_bandwidth_heatmap.png"
    plt.savefig(rank_bw_out, dpi=300)
    plt.close()
    print(f"[Success] Saved 96-rank GPU bandwidth heatmap to '{rank_bw_out}'.")


# ---------------------------------------------------------------------------
# Locality tier assignment
# ---------------------------------------------------------------------------
# CLAIX-23 Sapphire Rapids: 96 physical cores per node, 12 per NUMA domain,
# 4 NUMA domains per socket (CPUs 0-47 = socket 0, CPUs 48-95 = socket 1).
CPUS_PER_NUMA   = 12
CPUS_PER_SOCKET = 48

def _locality_tier(host_i, cpu_i, host_j, cpu_j):
    """Return a locality-tier label for a rank pair."""
    if host_i != host_j:
        return "cross_node"
    numa_i = cpu_i // CPUS_PER_NUMA
    numa_j = cpu_j // CPUS_PER_NUMA
    sock_i = cpu_i // CPUS_PER_SOCKET
    sock_j = cpu_j // CPUS_PER_SOCKET
    if numa_i == numa_j:
        return "same_numa"
    if sock_i == sock_j:
        return "same_socket_diff_numa"
    return "cross_socket"

TIER_ORDER = ["same_numa", "same_socket_diff_numa", "cross_socket", "cross_node"]
TIER_LABELS = {
    "same_numa":             "Same NUMA",
    "same_socket_diff_numa": "Same socket,\ndiff NUMA",
    "cross_socket":          "Cross-socket",
    "cross_node":            "Cross-node\n(IPoIB)",
}


def _add_tier_column(df):
    """Add a 'locality_tier' column derived from host/cpu columns."""
    df = df.copy()
    df['locality_tier'] = [
        _locality_tier(str(row['host_i']), int(row['cpu_i']),
                       str(row['host_j']), int(row['cpu_j']))
        for _, row in df.iterrows()
    ]
    return df


# ---------------------------------------------------------------------------
# TCP heatmaps
# ---------------------------------------------------------------------------

def plot_tcp_pair_heatmaps(csv_path, output_prefix="tcp_pair"):
    """
    Plots NxN rank-pair latency and bandwidth heatmaps for the TCP benchmark.
    Identical layout to plot_cpu_pair_heatmaps; uses bandwidth_median_gibs
    (GiB/s, binary) rather than bandwidth_median_gbps.
    """
    if not os.path.exists(csv_path):
        print(f"[Warning] TCP CSV not found: {csv_path}. Skipping.")
        return

    df = pd.read_csv(csv_path)
    ranks = sorted(set(df['rank_i']).union(set(df['rank_j'])))
    n = len(ranks)
    rank_map = {r: idx for idx, r in enumerate(ranks)}

    lat_mat = np.full((n, n), np.nan)
    bw_mat  = np.full((n, n), np.nan)

    has_bw = 'bandwidth_median_gibs' in df.columns

    for _, row in df.iterrows():
        i = rank_map[int(row['rank_i'])]
        j = rank_map[int(row['rank_j'])]
        lat = float(row['latency_median_us'])
        lat_mat[i, j] = lat_mat[j, i] = lat
        if has_bw:
            bw = float(row['bandwidth_median_gibs'])
            bw_mat[i, j] = bw_mat[j, i] = bw

    np.fill_diagonal(lat_mat, np.nan)
    np.fill_diagonal(bw_mat,  np.nan)

    # Latency heatmap
    plt.figure(figsize=(11, 9), dpi=300)
    cmap_lat = sns.color_palette("YlOrRd", as_cmap=True)
    cmap_lat.set_bad(color="gainsboro")
    ax = sns.heatmap(
        lat_mat, cmap=cmap_lat,
        cbar_kws={'label': 'Median One-Way Latency (µs)'},
        square=True,
        xticklabels=24 if n >= 96 else (12 if n >= 48 else True),
        yticklabels=24 if n >= 96 else (12 if n >= 48 else True),
    )
    if n == 192:
        ax.axhline(96, color='blue', linewidth=2.5, linestyle='--')
        ax.axvline(96, color='blue', linewidth=2.5, linestyle='--')
        ax.text(48,  -3, "Node 0 (Ranks 0-95)",   color="blue", fontsize=11, ha="center", weight="bold")
        ax.text(144, -3, "Node 1 (Ranks 96-191)", color="blue", fontsize=11, ha="center", weight="bold")
        ax.text(-3, 48,  "Node 0", color="blue", fontsize=11, va="center", ha="right", weight="bold", rotation=90)
        ax.text(-3, 144, "Node 1", color="blue", fontsize=11, va="center", ha="right", weight="bold", rotation=90)
    plt.title(f"TCP/IP (IPoIB) Core-to-Core Latency Matrix ({n} Ranks)", fontsize=13, pad=20, weight="bold")
    plt.xlabel("MPI Rank"); plt.ylabel("MPI Rank")
    out = f"{output_prefix}_latency_heatmap.png"
    plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()
    print(f"[Success] Saved TCP latency heatmap to '{out}'.")

    # Bandwidth heatmap
    if has_bw:
        plt.figure(figsize=(11, 9), dpi=300)
        cmap_bw = sns.color_palette("viridis", as_cmap=True)
        cmap_bw.set_bad(color="gainsboro")
        ax2 = sns.heatmap(
            bw_mat, cmap=cmap_bw,
            cbar_kws={'label': 'Median Bidirectional Goodput (GiB/s)'},
            square=True,
            xticklabels=24 if n >= 96 else (12 if n >= 48 else True),
            yticklabels=24 if n >= 96 else (12 if n >= 48 else True),
        )
        if n == 192:
            ax2.axhline(96, color='white', linewidth=2.5, linestyle='--')
            ax2.axvline(96, color='white', linewidth=2.5, linestyle='--')
            ax2.text(48,  -3, "Node 0 (Ranks 0-95)",   color="white", fontsize=11, ha="center", weight="bold")
            ax2.text(144, -3, "Node 1 (Ranks 96-191)", color="white", fontsize=11, ha="center", weight="bold")
            for sep in range(12, n, 12):
                if sep != 96:
                    ax2.axhline(sep, color='gray', linewidth=0.5, linestyle=':')
                    ax2.axvline(sep, color='gray', linewidth=0.5, linestyle=':')
        plt.title(f"TCP/IP (IPoIB) Core-to-Core Bandwidth Matrix ({n} Ranks)", fontsize=13, pad=20, weight="bold")
        plt.xlabel("MPI Rank"); plt.ylabel("MPI Rank")
        out2 = f"{output_prefix}_bandwidth_heatmap.png"
        plt.tight_layout(); plt.savefig(out2, dpi=300); plt.close()
        print(f"[Success] Saved TCP bandwidth heatmap to '{out2}'.")


# ---------------------------------------------------------------------------
# Locality-tier summary
# ---------------------------------------------------------------------------

def summarize_locality(csv_path, output_prefix, transport="tcp"):
    """
    Groups pair results by locality tier and writes a summary CSV with
    pair count, mean, median, p25, p95, min, max for latency and bandwidth.
    Also produces a box-plot comparing the four tiers.
    """
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    df = _add_tier_column(df)

    has_bw = 'bandwidth_median_gibs' in df.columns or 'bandwidth_median_gbps' in df.columns
    bw_col = 'bandwidth_median_gibs' if 'bandwidth_median_gibs' in df.columns else 'bandwidth_median_gbps'
    bw_unit = 'GiB/s' if bw_col == 'bandwidth_median_gibs' else 'GiB/s (mislabelled gbps)'

    rows = []
    for tier in TIER_ORDER:
        sub = df[df['locality_tier'] == tier]
        if sub.empty:
            continue
        lat = sub['latency_median_us']
        row = {
            'transport':    transport,
            'locality_tier': tier,
            'pair_count':   len(sub),
            'lat_mean_us':  lat.mean(),
            'lat_median_us': lat.median(),
            'lat_p25_us':   lat.quantile(0.25),
            'lat_p75_us':   lat.quantile(0.75),
            'lat_p95_us':   lat.quantile(0.95),
            'lat_min_us':   lat.min(),
            'lat_max_us':   lat.max(),
        }
        if has_bw:
            bw = sub[bw_col]
            row.update({
                'bw_mean_gibs':   bw.mean(),
                'bw_median_gibs': bw.median(),
                'bw_p25_gibs':    bw.quantile(0.25),
                'bw_p75_gibs':    bw.quantile(0.75),
                'bw_p95_gibs':    bw.quantile(0.95),
                'bw_min_gibs':    bw.min(),
                'bw_max_gibs':    bw.max(),
            })
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_csv = f"{output_prefix}_locality_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[Success] Saved locality summary to '{summary_csv}'.")

    # Box plot — latency
    present_tiers = [t for t in TIER_ORDER if t in df['locality_tier'].values]
    data_lat = [df[df['locality_tier'] == t]['latency_median_us'].values for t in present_tiers]
    labels   = [TIER_LABELS[t] for t in present_tiers]

    fig, axes = plt.subplots(1, 2 if has_bw else 1,
                             figsize=(12 if has_bw else 6, 5), dpi=300)
    ax_lat = axes[0] if has_bw else axes

    ax_lat.boxplot(data_lat, labels=labels, patch_artist=True,
                   medianprops=dict(color='black', linewidth=2))
    ax_lat.set_title(f"{transport.upper()} Latency by Locality Tier", fontsize=12, weight="bold")
    ax_lat.set_ylabel("Median One-Way Latency (µs)")
    ax_lat.set_xlabel("Locality tier")
    ax_lat.grid(axis='y', linestyle=':', alpha=0.5)

    if has_bw:
        data_bw = [df[df['locality_tier'] == t][bw_col].values for t in present_tiers]
        ax_bw = axes[1]
        ax_bw.boxplot(data_bw, labels=labels, patch_artist=True,
                      medianprops=dict(color='black', linewidth=2))
        ax_bw.set_title(f"{transport.upper()} Bandwidth by Locality Tier", fontsize=12, weight="bold")
        ax_bw.set_ylabel(f"Median Bidirectional Goodput ({bw_unit})")
        ax_bw.set_xlabel("Locality tier")
        ax_bw.grid(axis='y', linestyle=':', alpha=0.5)

    plt.tight_layout()
    box_out = f"{output_prefix}_locality_boxplot.png"
    plt.savefig(box_out, dpi=300); plt.close()
    print(f"[Success] Saved locality box plot to '{box_out}'.")


# ---------------------------------------------------------------------------
# MPI vs TCP comparison (produced only when both CSVs are present)
# ---------------------------------------------------------------------------

def compare_locality(mpi_csv, tcp_csv, output_prefix="transport_comparison"):
    """
    Side-by-side bar chart comparing MPI and TCP median latency per locality tier.
    Also writes a merged locality summary CSV with both transports.
    """
    mpi_df = pd.read_csv(mpi_csv)
    tcp_df = pd.read_csv(tcp_csv)

    mpi_df = _add_tier_column(mpi_df)
    tcp_df = _add_tier_column(tcp_df)

    # Rename MPI bw column to match TCP for uniform handling
    if 'bandwidth_median_gbps' in mpi_df.columns:
        mpi_df = mpi_df.rename(columns={'bandwidth_median_gbps': 'bandwidth_median_gibs'})

    rows = []
    for transport, df in [("mpi", mpi_df), ("tcp", tcp_df)]:
        for tier in TIER_ORDER:
            sub = df[df['locality_tier'] == tier]
            if sub.empty:
                continue
            lat = sub['latency_median_us']
            row = {
                'transport': transport,
                'locality_tier': tier,
                'pair_count': len(sub),
                'lat_median_us': lat.median(),
                'lat_p25_us':   lat.quantile(0.25),
                'lat_p75_us':   lat.quantile(0.75),
            }
            if 'bandwidth_median_gibs' in df.columns:
                bw = sub['bandwidth_median_gibs']
                row['bw_median_gibs'] = bw.median()
            rows.append(row)

    cmp_df = pd.DataFrame(rows)
    cmp_csv = f"{output_prefix}_locality_summary.csv"
    cmp_df.to_csv(cmp_csv, index=False)
    print(f"[Success] Saved combined locality comparison to '{cmp_csv}'.")

    # Bar chart: latency per tier, MPI vs TCP
    present_tiers = [t for t in TIER_ORDER
                     if t in mpi_df['locality_tier'].values or t in tcp_df['locality_tier'].values]
    x = np.arange(len(present_tiers))
    width = 0.35

    mpi_medians = []
    tcp_medians = []
    mpi_p25, mpi_p75 = [], []
    tcp_p25, tcp_p75 = [], []
    for tier in present_tiers:
        for medians, p25s, p75s, df in [
            (mpi_medians, mpi_p25, mpi_p75, mpi_df),
            (tcp_medians, tcp_p25, tcp_p75, tcp_df),
        ]:
            sub = df[df['locality_tier'] == tier]['latency_median_us']
            if sub.empty:
                medians.append(0); p25s.append(0); p75s.append(0)
            else:
                medians.append(sub.median())
                p25s.append(sub.quantile(0.25))
                p75s.append(sub.quantile(0.75))

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    bars_mpi = ax.bar(x - width/2, mpi_medians, width, label='MPI (shared-mem / IB fabric)',
                      color='steelblue', alpha=0.85)
    bars_tcp = ax.bar(x + width/2, tcp_medians, width, label='TCP/IP (IPoIB)',
                      color='tomato', alpha=0.85)

    # IQR error bars
    def _iqr_err(medians, p25s, p75s):
        lo = [m - p for m, p in zip(medians, p25s)]
        hi = [p - m for m, p in zip(medians, p75s)]
        return [lo, hi]

    ax.errorbar(x - width/2, mpi_medians, yerr=_iqr_err(mpi_medians, mpi_p25, mpi_p75),
                fmt='none', color='black', capsize=4, linewidth=1.2)
    ax.errorbar(x + width/2, tcp_medians, yerr=_iqr_err(tcp_medians, tcp_p25, tcp_p75),
                fmt='none', color='black', capsize=4, linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels([TIER_LABELS[t] for t in present_tiers], fontsize=10)
    ax.set_ylabel("Median One-Way Latency (µs)", fontsize=11)
    ax.set_title("MPI vs TCP/IP (IPoIB) Latency by Locality Tier\n(bars = median of pair medians; error bars = IQR)",
                 fontsize=11, weight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    bar_out = f"{output_prefix}_latency_comparison.png"
    plt.savefig(bar_out, dpi=300); plt.close()
    print(f"[Success] Saved MPI vs TCP latency comparison to '{bar_out}'.")


def main():
    parser = argparse.ArgumentParser(description="Generate heatmaps for MPI/TCP CPU pair latency/bandwidth & GPU transfers.")
    parser.add_argument("--cpu-csv",    default="cpu_192ranks_pair_latency.csv", help="Path to MPI CPU CSV file")
    parser.add_argument("--tcp-csv",    default=None,  help="Path to TCP pair latency CSV file")
    parser.add_argument("--gpu-csv",    default="gpu_c23g_transfer_latency.csv", help="Path to GPU CSV file")
    parser.add_argument("--cpu-prefix", default="cpu_pair",  help="Output prefix for MPI CPU plots")
    parser.add_argument("--tcp-prefix", default="tcp_pair",  help="Output prefix for TCP plots")
    args = parser.parse_args()

    print(f"Generating heatmaps using:\n  CPU CSV: {args.cpu_csv}\n  TCP CSV: {args.tcp_csv or '(none)'}\n  GPU CSV: {args.gpu_csv}")

    if os.path.exists(args.cpu_csv):
        plot_cpu_pair_heatmaps(args.cpu_csv, output_prefix=args.cpu_prefix)
    else:
        print(f"[Warning] CPU CSV not found: {args.cpu_csv}. Skipping MPI heatmaps.")

    if args.tcp_csv:
        if os.path.exists(args.tcp_csv):
            plot_tcp_pair_heatmaps(args.tcp_csv, output_prefix=args.tcp_prefix)
            summarize_locality(args.tcp_csv, output_prefix=args.tcp_prefix, transport="tcp")
        else:
            print(f"[Warning] TCP CSV not found: {args.tcp_csv}. Skipping TCP heatmaps.")

    if os.path.exists(args.gpu_csv):
        plot_gpu_transfer_heatmaps(args.gpu_csv)
    else:
        print(f"[Warning] GPU CSV not found: {args.gpu_csv}. Skipping GPU heatmaps.")

    # If both MPI and TCP CSVs exist, produce a combined locality comparison
    if args.tcp_csv and os.path.exists(args.cpu_csv) and os.path.exists(args.tcp_csv):
        compare_locality(args.cpu_csv, args.tcp_csv)


if __name__ == "__main__":
    main()
