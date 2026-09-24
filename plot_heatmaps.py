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

    bw_col = 'bandwidth_median_gibs' if 'bandwidth_median_gibs' in df.columns else ('bandwidth_median_gbps' if 'bandwidth_median_gbps' in df.columns else None)
    has_bw = bw_col is not None

    for _, row in df.iterrows():
        i = rank_map[int(row['rank_i'])]
        j = rank_map[int(row['rank_j'])]
        lat_val = float(row['latency_median_us'])
        lat_mat[i, j] = lat_val
        lat_mat[j, i] = lat_val
        if has_bw:
            bw_val = float(row[bw_col])
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
    Plots highly readable NUMA-aggregated and full-rank GPU bandwidth & latency heatmaps
    using median-of-medians aggregation.
    """
    if not os.path.exists(csv_path):
        print(f"[Warning] GPU CSV file not found: {csv_path}. Skipping GPU heatmaps.")
        return

    df = pd.read_csv(csv_path)

    # Detect schema (new schema vs legacy schema)
    is_new_schema = 'direction' in df.columns and 'bw_gibs_median' in df.columns

    if is_new_schema:
        if 'cpu_numa_node' in df.columns:
            df['numa_node'] = df['cpu_numa_node']
        else:
            df['numa_node'] = df['cpu_id'] // 12
        rank_col = 'rank'
        bw_col = 'bw_gibs_median'
        lat_col = 'lat_median_us'
    else:
        df['numa_node'] = df['cpu'] // 12
        rank_col = 'rank'
        bw_col = 'd2h_bw_median_gbps'
        lat_col = 'd2h_lat_median_us'

    # Filter OK status rows for clean visualization
    ok_df = df[df['status'] == 'OK'].copy()

    # GPU Labels
    pci_map = {}
    pci_col = 'gpu_pci_bus_id' if 'gpu_pci_bus_id' in ok_df.columns else ('pci_bus_id' if 'pci_bus_id' in ok_df.columns else None)
    for gid in sorted(ok_df['gpu_id'].unique()):
        sub = ok_df[ok_df['gpu_id'] == gid]
        pci = sub[pci_col].dropna().iloc[0] if (pci_col and len(sub[pci_col].dropna()) > 0) else "N/A"
        pci_map[gid] = f"GPU {gid}\n({pci})"

    ok_df['gpu_label'] = ok_df['gpu_id'].map(pci_map)
    gpu_labels = [pci_map[gid] for gid in sorted(pci_map.keys())]

    # NUMA Domain Labels (Accurate CLAIX-23 c23g topology mapping)
    numa_labels = {
        0: "NUMA 0 (CPUs 0-11)\n[Socket 0 - Local GPU 0]",
        1: "NUMA 1 (CPUs 12-23)\n[Socket 0]",
        2: "NUMA 2 (CPUs 24-35)\n[Socket 0 - Local GPU 1]",
        3: "NUMA 3 (CPUs 36-47)\n[Socket 0]",
        4: "NUMA 4 (CPUs 48-59)\n[Socket 1 - Local GPU 2]",
        5: "NUMA 5 (CPUs 60-71)\n[Socket 1]",
        6: "NUMA 6 (CPUs 72-83)\n[Socket 1 - Local GPU 3]",
        7: "NUMA 7 (CPUs 84-95)\n[Socket 1]"
    }
    ok_df['numa_label'] = ok_df['numa_node'].map(numa_labels)
    ordered_numa = [numa_labels[i] for i in range(8) if numa_labels[i] in ok_df['numa_label'].values]

    # If new schema, handle large bandwidth payload filtering (e.g. 16MB / 64MB / max payload)
    if is_new_schema:
        # Get largest bandwidth payload available
        bw_df = ok_df[ok_df['payload_type'] == 'bandwidth']
        if not bw_df.empty:
            largest_payload = bw_df['payload_bytes'].max()
            bw_df = bw_df[bw_df['payload_bytes'] == largest_payload]
        else:
            bw_df = ok_df

        d2h_df = bw_df[bw_df['direction'] == 'D2H']
        h2d_df = bw_df[bw_df['direction'] == 'H2D']

        # Median of medians aggregation
        numa_d2h = d2h_df.groupby(['numa_label', 'gpu_label'])['bw_gibs_median'].median().unstack().reindex(ordered_numa)[gpu_labels]
        numa_h2d = h2d_df.groupby(['numa_label', 'gpu_label'])['bw_gibs_median'].median().unstack().reindex(ordered_numa)[gpu_labels]

        piv_d2h = d2h_df.pivot(index='rank', columns='gpu_label', values='bw_gibs_median')[gpu_labels]
        piv_h2d = h2d_df.pivot(index='rank', columns='gpu_label', values='bw_gibs_median')[gpu_labels]
        payload_label_str = f" ({largest_payload // (1024*1024)} MiB)" if not bw_df.empty else ""
    else:
        numa_d2h = ok_df.groupby(['numa_label', 'gpu_label'])['d2h_bw_median_gbps'].median().unstack().reindex(ordered_numa)[gpu_labels]
        numa_h2d = ok_df.groupby(['numa_label', 'gpu_label'])['h2d_bw_median_gbps'].median().unstack().reindex(ordered_numa)[gpu_labels]

        piv_d2h = ok_df.pivot(index='rank', columns='gpu_label', values='d2h_bw_median_gbps')[gpu_labels]
        piv_h2d = ok_df.pivot(index='rank', columns='gpu_label', values='h2d_bw_median_gbps')[gpu_labels]
        payload_label_str = ""

    # --- 1. NUMA Domain Aggregated Heatmap (8 NUMA Rows x 4 GPUs) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=300)
    cmap_bw = sns.color_palette("viridis", as_cmap=True)

    sns.heatmap(
        numa_d2h,
        annot=True,
        fmt=".1f",
        annot_kws={"size": 11, "weight": "bold"},
        cmap=cmap_bw,
        cbar_kws={'label': 'Bandwidth (GiB/s)'},
        ax=axes[0],
        linewidths=1.0,
        linecolor="white"
    )
    axes[0].set_title(f"Device-to-Host (D2H) Bandwidth (GiB/s){payload_label_str}\nMedian of Medians by NUMA Domain", fontsize=12, weight="bold")
    axes[0].set_xlabel("Target CUDA GPU", fontsize=10, weight="bold")
    axes[0].set_ylabel("NUMA Domain Placement", fontsize=10, weight="bold")

    sns.heatmap(
        numa_h2d,
        annot=True,
        fmt=".1f",
        annot_kws={"size": 11, "weight": "bold"},
        cmap=cmap_bw,
        cbar_kws={'label': 'Bandwidth (GiB/s)'},
        ax=axes[1],
        linewidths=1.0,
        linecolor="white"
    )
    axes[1].set_title(f"Host-to-Device (H2D) Bandwidth (GiB/s){payload_label_str}\nMedian of Medians by NUMA Domain", fontsize=12, weight="bold")
    axes[1].set_xlabel("Target CUDA GPU", fontsize=10, weight="bold")
    axes[1].set_ylabel("")

    plt.tight_layout()
    numa_bw_out = f"{output_prefix}_numa_bandwidth_heatmap.png"
    plt.savefig(numa_bw_out, dpi=300)
    plt.close()
    print(f"[Success] Saved NUMA aggregated GPU bandwidth heatmap to '{numa_bw_out}'.")

    # --- 2. Full 96-Rank Heatmap with NUMA Grid Separators ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 16), dpi=300)
    
    sns.heatmap(
        piv_d2h,
        cmap=cmap_bw,
        cbar_kws={'label': 'Bandwidth (GiB/s)'},
        ax=axes[0],
        yticklabels=6
    )
    axes[0].set_title(f"D2H Bandwidth per MPI Rank (GiB/s){payload_label_str}", fontsize=12, weight="bold")
    axes[0].set_xlabel("Target CUDA GPU", fontsize=10, weight="bold")
    axes[0].set_ylabel("MPI Rank Index (0-95)", fontsize=10, weight="bold")

    sns.heatmap(
        piv_h2d,
        cmap=cmap_bw,
        cbar_kws={'label': 'Bandwidth (GiB/s)'},
        ax=axes[1],
        yticklabels=6
    )
    axes[1].set_title(f"H2D Bandwidth per MPI Rank (GiB/s){payload_label_str}", fontsize=12, weight="bold")
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

    # --- Side-by-side bar charts: MPI vs TCP per locality tier ---
    present_tiers = [t for t in TIER_ORDER
                     if t in mpi_df['locality_tier'].values or t in tcp_df['locality_tier'].values]

    def _pair_stats(df, metric):
        """Return (medians, p25s, p75s) of `metric` grouped by locality tier."""
        medians, p25s, p75s = [], [], []
        for tier in present_tiers:
            sub = df[df['locality_tier'] == tier][metric]
            if sub.empty:
                medians.append(0); p25s.append(0); p75s.append(0)
            else:
                medians.append(sub.median())
                p25s.append(sub.quantile(0.25))
                p75s.append(sub.quantile(0.75))
        return medians, p25s, p75s

    def _iqr_err(medians, p25s, p75s):
        lo = [m - p for m, p in zip(medians, p25s)]
        hi = [p - m for m, p in zip(medians, p75s)]
        return [lo, hi]

    def _bar_chart(metric, ylabel, out_name, title):
        mpi_m, mpi_p25, mpi_p75 = _pair_stats(mpi_df, metric)
        tcp_m, tcp_p25, tcp_p75 = _pair_stats(tcp_df, metric)

        x = np.arange(len(present_tiers))
        width = 0.35

        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        ax.bar(x - width/2, mpi_m, width, label='MPI (shared-mem / IB fabric)',
               color='steelblue', alpha=0.85)
        ax.bar(x + width/2, tcp_m, width, label='TCP/IP (IPoIB)',
               color='tomato', alpha=0.85)

        ax.errorbar(x - width/2, mpi_m, yerr=_iqr_err(mpi_m, mpi_p25, mpi_p75),
                    fmt='none', color='black', capsize=4, linewidth=1.2)
        ax.errorbar(x + width/2, tcp_m, yerr=_iqr_err(tcp_m, tcp_p25, tcp_p75),
                    fmt='none', color='black', capsize=4, linewidth=1.2)

        ax.set_xticks(x)
        ax.set_xticklabels([TIER_LABELS[t] for t in present_tiers], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        plt.tight_layout()
        out = f"{output_prefix}_{out_name}.png"
        plt.savefig(out, dpi=300); plt.close()
        print(f"[Success] Saved MPI vs TCP comparison to '{out}'.")

    # 1. Latency per tier
    _bar_chart(
        metric='latency_median_us',
        ylabel="Median One-Way Latency (µs)",
        out_name="latency_comparison",
        title="MPI vs TCP/IP (IPoIB) Latency by Locality Tier\n"
              "(bars = median of pair medians; error bars = IQR)",
    )

    # 2. Datarate per tier (only if bandwidth data is present)
    if 'bandwidth_median_gibs' in mpi_df.columns and 'bandwidth_median_gibs' in tcp_df.columns:
        _bar_chart(
            metric='bandwidth_median_gibs',
            ylabel="Median Bidirectional Datarate (GiB/s)",
            out_name="datarate_comparison",
            title="MPI vs TCP/IP (IPoIB) Datarate by Locality Tier\n"
                  "(bars = median of pair medians; error bars = IQR)",
        )


# ---------------------------------------------------------------------------
# Simplex vs Duplex comparison
# ---------------------------------------------------------------------------

def compare_simplex_vs_duplex(simplex_csv, duplex_csv, transport="mpi", output_prefix=None):
    """
    Compares simplex (ping-pong) goodput vs duplex aggregate goodput per locality tier.
    Plots side-by-side bandwidth comparison and duplex efficiency ratio (BW_duplex / BW_simplex).
    """
    if not os.path.exists(simplex_csv) or not os.path.exists(duplex_csv):
        return

    if output_prefix is None:
        output_prefix = f"{transport}_simplex_vs_duplex"

    s_df = pd.read_csv(simplex_csv)
    d_df = pd.read_csv(duplex_csv)

    s_df = _add_tier_column(s_df)
    d_df = _add_tier_column(d_df)

    s_bw_col = 'bandwidth_median_gibs' if 'bandwidth_median_gibs' in s_df.columns else 'bandwidth_median_gbps'
    d_bw_col = 'bandwidth_median_gibs' if 'bandwidth_median_gibs' in d_df.columns else 'bandwidth_median_gbps'

    present_tiers = [t for t in TIER_ORDER if t in s_df['locality_tier'].values and t in d_df['locality_tier'].values]

    rows = []
    for tier in present_tiers:
        s_sub = s_df[s_df['locality_tier'] == tier]
        d_sub = d_df[d_df['locality_tier'] == tier]

        s_lat = s_sub['latency_median_us']
        d_lat = d_sub['latency_median_us']

        s_bw = s_sub[s_bw_col]
        d_bw = d_sub[d_bw_col]

        s_bw_med = s_bw.median()
        d_bw_med = d_bw.median()
        ratio = (d_bw_med / s_bw_med) if s_bw_med > 0 else 0.0

        rows.append({
            'transport': transport,
            'locality_tier': tier,
            'simplex_pairs': len(s_sub),
            'duplex_pairs': len(d_sub),
            'simplex_lat_median_us': s_lat.median(),
            'duplex_lat_median_us': d_lat.median(),
            'simplex_bw_median_gibs': s_bw_med,
            'duplex_bw_median_gibs': d_bw_med,
            'duplex_efficiency_ratio': ratio,
        })

    summary_df = pd.DataFrame(rows)
    summary_csv = f"{output_prefix}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[Success] Saved simplex vs duplex summary to '{summary_csv}'.")

    # 1. Side-by-side Bandwidth Bar Chart
    x = np.arange(len(present_tiers))
    width = 0.35

    s_bws = [r['simplex_bw_median_gibs'] for r in rows]
    d_bws = [r['duplex_bw_median_gibs'] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ax.bar(x - width / 2, s_bws, width, label='Simplex (Ping-Pong Goodput)', color='steelblue', alpha=0.85)
    ax.bar(x + width / 2, d_bws, width, label='Duplex (Aggregate Bidirectional BW)', color='forestgreen', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([TIER_LABELS[t] for t in present_tiers], fontsize=10)
    ax.set_ylabel("Median Bandwidth (GiB/s)", fontsize=11)
    ax.set_title(f"{transport.upper()} Simplex vs Full-Duplex Bandwidth by Locality Tier", fontsize=11, weight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    bw_out = f"{output_prefix}_bandwidth_comparison.png"
    plt.savefig(bw_out, dpi=300); plt.close()
    print(f"[Success] Saved simplex vs duplex bandwidth chart to '{bw_out}'.")

    # 2. Duplex Efficiency Ratio Chart (BW_duplex / BW_simplex)
    ratios = [r['duplex_efficiency_ratio'] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    bars = ax.bar(x, ratios, width=0.5, color='darkorange', alpha=0.85)
    ax.axhline(2.0, color='forestgreen', linestyle='--', linewidth=1.5, label='Ideal Full-Duplex (2.0x)')
    ax.axhline(1.0, color='crimson', linestyle=':', linewidth=1.5, label='Half-Duplex Ceiling (1.0x)')

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', weight='bold', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([TIER_LABELS[t] for t in present_tiers], fontsize=10)
    ax.set_ylabel("Duplex Ratio (BW_duplex / BW_simplex)", fontsize=11)
    ax.set_title(f"{transport.upper()} Duplex Scaling Ratio by Locality Tier", fontsize=11, weight="bold")
    ax.set_ylim(0, max(2.5, max(ratios) * 1.2 if ratios else 2.5))
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    ratio_out = f"{output_prefix}_efficiency_ratio.png"
    plt.savefig(ratio_out, dpi=300); plt.close()
    print(f"[Success] Saved duplex efficiency ratio chart to '{ratio_out}'.")


# ---------------------------------------------------------------------------
# Incast / Flooding Scaling Curves
# ---------------------------------------------------------------------------

def _format_payload_bytes(b):
    if b < 1024:
        return f"{b} B"
    elif b < 1024 * 1024:
        return f"{b // 1024} KiB"
    else:
        return f"{b // (1024 * 1024)} MiB"


def plot_uds_pair_heatmaps(csv_path, output_prefix="uds_pair"):
    """
    Plots 96x96 rank pair latency and bandwidth heatmaps for node-local UDS benchmark.
    """
    if not os.path.exists(csv_path):
        print(f"[Warning] UDS CSV file not found: {csv_path}. Skipping UDS heatmaps.")
        return

    df = pd.read_csv(csv_path)
    ranks = sorted(list(set(df['rank_i']).union(set(df['rank_j']))))
    n = len(ranks)
    rank_map = {r: idx for idx, r in enumerate(ranks)}

    lat_mat = np.zeros((n, n), dtype=float)
    bw_mat  = np.zeros((n, n), dtype=float)
    has_bw  = 'bandwidth_median_gibs' in df.columns or 'bandwidth_median_gbps' in df.columns
    bw_col  = 'bandwidth_median_gibs' if 'bandwidth_median_gibs' in df.columns else 'bandwidth_median_gbps'

    for _, row in df.iterrows():
        i = rank_map[int(row['rank_i'])]
        j = rank_map[int(row['rank_j'])]
        lat = float(row['latency_median_us'])
        lat_mat[i, j] = lat_mat[j, i] = lat
        if has_bw:
            bw = float(row[bw_col])
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
        xticklabels=12,
        yticklabels=12,
    )
    # Draw dashed lines for NUMA domains (every 12 ranks)
    for sep in range(12, n, 12):
        ax.axhline(sep, color='gray', linewidth=0.8, linestyle=':')
        ax.axvline(sep, color='gray', linewidth=0.8, linestyle=':')
    # Socket boundary at rank 48
    if n >= 48:
        ax.axhline(48, color='red', linewidth=2.0, linestyle='--')
        ax.axvline(48, color='red', linewidth=2.0, linestyle='--')

    plt.title(f"Unix Domain Socket (UDS) Core-to-Core Latency Matrix ({n} Ranks)", fontsize=13, pad=20, weight="bold")
    plt.xlabel("MPI Rank", fontsize=11); plt.ylabel("MPI Rank", fontsize=11)
    out = f"{output_prefix}_latency_heatmap.png"
    plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()
    print(f"[Success] Saved UDS latency heatmap to '{out}'.")

    # Bandwidth heatmap
    if has_bw:
        plt.figure(figsize=(11, 9), dpi=300)
        cmap_bw = sns.color_palette("viridis", as_cmap=True)
        cmap_bw.set_bad(color="gainsboro")
        ax2 = sns.heatmap(
            bw_mat, cmap=cmap_bw,
            cbar_kws={'label': 'Median Goodput (GiB/s)'},
            square=True,
            xticklabels=12,
            yticklabels=12,
        )
        for sep in range(12, n, 12):
            ax2.axhline(sep, color='gray', linewidth=0.8, linestyle=':')
            ax2.axvline(sep, color='gray', linewidth=0.8, linestyle=':')
        if n >= 48:
            ax2.axhline(48, color='red', linewidth=2.0, linestyle='--')
            ax2.axvline(48, color='red', linewidth=2.0, linestyle='--')

        plt.title(f"Unix Domain Socket (UDS) Core-to-Core Bandwidth Matrix ({n} Ranks)", fontsize=13, pad=20, weight="bold")
        plt.xlabel("MPI Rank", fontsize=11); plt.ylabel("MPI Rank", fontsize=11)
        out2 = f"{output_prefix}_bandwidth_heatmap.png"
        plt.tight_layout(); plt.savefig(out2, dpi=300); plt.close()
        print(f"[Success] Saved UDS bandwidth heatmap to '{out2}'.")


def compare_three_way_locality(mpi_csv, tcp_csv, uds_csv, output_prefix="transport_comparison_3way"):
    """
    Produces a 3-way side-by-side comparison across node-local tiers:
    MPI vs TCP vs UDS (Unix Domain Sockets).
    """
    local_tiers = ["same_numa", "same_socket_diff_numa", "cross_socket"]
    tier_labels = ["Same NUMA\n(Intra-L3)", "Same Socket,\nDiff NUMA", "Cross-Socket\n(UPI link)"]

    dfs = {}
    if mpi_csv and os.path.exists(mpi_csv):
        dfs['MPI'] = _add_tier_column(pd.read_csv(mpi_csv))
    if tcp_csv and os.path.exists(tcp_csv):
        dfs['TCP'] = _add_tier_column(pd.read_csv(tcp_csv))
    if uds_csv and os.path.exists(uds_csv):
        dfs['UDS'] = _add_tier_column(pd.read_csv(uds_csv))

    if len(dfs) < 2:
        return

    # Extract medians per local tier
    stats = {k: {'lat': [], 'bw': []} for k in dfs}
    for t in local_tiers:
        for trans, df in dfs.items():
            sub = df[df['locality_tier'] == t]
            if not sub.empty:
                lat_med = sub['latency_median_us'].median()
                bw_col = 'bandwidth_median_gibs' if 'bandwidth_median_gibs' in sub.columns else 'bandwidth_median_gbps'
                bw_med = sub[bw_col].median() if bw_col in sub.columns else 0.0
            else:
                lat_med, bw_med = np.nan, np.nan
            stats[trans]['lat'].append(lat_med)
            stats[trans]['bw'].append(bw_med)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    x = np.arange(len(local_tiers))
    width = 0.26
    colors = {'MPI': 'steelblue', 'TCP': 'tomato', 'UDS': 'forestgreen'}

    # 1. Latency (Log Scale)
    for idx, (trans, st) in enumerate(stats.items()):
        offset = (idx - (len(stats) - 1) / 2.0) * width
        bars = axes[0].bar(x + offset, st['lat'], width, label=trans, color=colors.get(trans, 'gray'), alpha=0.85)
        for b in bars:
            h = b.get_height()
            if not np.isnan(h) and h > 0:
                axes[0].text(b.get_x() + b.get_width()/2., h * 1.15, f"{h:.2f}µs", ha='center', va='bottom', fontsize=8, rotation=30)

    axes[0].set_yscale('log')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tier_labels, fontsize=10)
    axes[0].set_ylabel("One-Way Latency (µs, Log Scale)", fontsize=11)
    axes[0].set_title("Node-Local Latency: MPI vs TCP vs UDS", fontsize=12, weight="bold")
    axes[0].grid(True, which="both", linestyle=':', alpha=0.5)
    axes[0].legend(fontsize=10)

    # 2. Bandwidth
    for idx, (trans, st) in enumerate(stats.items()):
        offset = (idx - (len(stats) - 1) / 2.0) * width
        bars = axes[1].bar(x + offset, st['bw'], width, label=trans, color=colors.get(trans, 'gray'), alpha=0.85)
        for b in bars:
            h = b.get_height()
            if not np.isnan(h) and h > 0:
                axes[1].text(b.get_x() + b.get_width()/2., h + 0.15, f"{h:.2f}", ha='center', va='bottom', fontsize=8)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tier_labels, fontsize=10)
    axes[1].set_ylabel("Goodput (GiB/s)", fontsize=11)
    axes[1].set_title("Node-Local Bandwidth: MPI vs TCP vs UDS", fontsize=12, weight="bold")
    axes[1].grid(True, linestyle=':', alpha=0.5)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    out = f"{output_prefix}_comparison.png"
    plt.savefig(out, dpi=300); plt.close()
    print(f"[Success] Saved 3-way transport comparison to '{out}'.")


def plot_incast_results(mpi_sum_csv=None, tcp_sum_csv=None, uds_sum_csv=None, output_prefix="incast"):
    """
    Plots Incast / Fan-in Flooding curves:
      1. Aggregate Receiver Ingest Bandwidth vs Payload Size.
      2. Sender Tail Latency (p50, p95, max) vs Payload Size (log-log).
    """
    dfs = {}
    if mpi_sum_csv and os.path.exists(mpi_sum_csv):
        dfs['MPI'] = pd.read_csv(mpi_sum_csv)
    if tcp_sum_csv and os.path.exists(tcp_sum_csv):
        dfs['TCP'] = pd.read_csv(tcp_sum_csv)
    if uds_sum_csv and os.path.exists(uds_sum_csv):
        dfs['UDS'] = pd.read_csv(uds_sum_csv)

    if not dfs:
        return

    # 1. Ingest Bandwidth vs Payload Size
    plt.figure(figsize=(9, 5), dpi=300)
    markers = {'MPI': 'o', 'TCP': 's', 'UDS': '^'}
    colors  = {'MPI': 'steelblue', 'TCP': 'tomato', 'UDS': 'forestgreen'}

    for trans, df in dfs.items():
        targets = sorted(df['target_rank'].unique())
        for tgt in targets:
            sub = df[df['target_rank'] == tgt].sort_values('payload_bytes')
            label = f"{trans} (Target Rank {tgt})" if len(targets) > 1 else trans
            plt.plot(sub['payload_bytes'], sub['ingest_bw_median_gibs'],
                     marker=markers.get(trans, 'o'), label=label,
                     color=colors.get(trans, 'purple'), linewidth=2, markersize=7)

    plt.xscale('log', base=2)
    all_payloads = sorted(list(next(iter(dfs.values()))['payload_bytes'].unique()))
    plt.xticks(all_payloads, [_format_payload_bytes(p) for p in all_payloads], rotation=30)
    plt.xlabel("Message Payload Size per Sender", fontsize=11)
    plt.ylabel("Receiver Aggregate Ingest Bandwidth (GiB/s)", fontsize=11)
    plt.title("Incast Flooding: Receiver Ingest Bandwidth vs Payload Size", fontsize=12, weight="bold")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    bw_out = f"{output_prefix}_ingest_bandwidth_curve.png"
    plt.savefig(bw_out, dpi=300); plt.close()
    print(f"[Success] Saved Incast ingest bandwidth curve to '{bw_out}'.")

    # 2. Sender Tail Latency vs Payload Size (Log-Log)
    fig, axes = plt.subplots(1, len(dfs), figsize=(6.5 * len(dfs), 5), dpi=300, squeeze=False)
    for idx, (trans, df) in enumerate(dfs.items()):
        ax = axes[0, idx]
        tgt0 = df['target_rank'].iloc[0]
        sub = df[df['target_rank'] == tgt0].sort_values('payload_bytes')

        ax.plot(sub['payload_bytes'], sub['sender_lat_median_us'], 'o-', label='p50 (Median)', color='teal', linewidth=2)
        ax.plot(sub['payload_bytes'], sub['sender_lat_p95_us'], 's--', label='p95', color='darkorange', linewidth=1.8)
        ax.plot(sub['payload_bytes'], sub['sender_lat_max_us'], '^:', label='Max', color='crimson', linewidth=1.5)

        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks(all_payloads)
        ax.set_xticklabels([_format_payload_bytes(p) for p in all_payloads], rotation=30)
        ax.set_xlabel("Message Payload Size per Sender", fontsize=10)
        ax.set_ylabel("Sender Completion Latency (µs)", fontsize=10)
        ax.set_title(f"{trans} Incast Sender Latency Distribution\n(Target Rank {tgt0})", fontsize=11, weight="bold")
        ax.grid(True, which="both", linestyle=':', alpha=0.5)
        ax.legend(fontsize=9)

    plt.tight_layout()
    lat_out = f"{output_prefix}_tail_latency_curve.png"
    plt.savefig(lat_out, dpi=300); plt.close()
    print(f"[Success] Saved Incast tail latency curve to '{lat_out}'.")


def main():
    parser = argparse.ArgumentParser(description="Generate heatmaps and charts for MPI/TCP/UDS pair latency/bandwidth, duplex, incast, and GPU transfers.")
    parser.add_argument("--cpu-csv",        default="cpu_192ranks_pair_latency.csv", help="Path to MPI CPU simplex CSV file")
    parser.add_argument("--tcp-csv",        default=None,  help="Path to TCP pair simplex CSV file")
    parser.add_argument("--uds-csv",        default=None,  help="Path to UDS pair simplex CSV file")
    parser.add_argument("--gpu-csv",        default="gpu_c23g_transfer_latency.csv", help="Path to GPU CSV file")
    parser.add_argument("--mpi-duplex-csv", default=None,  help="Path to MPI CPU duplex CSV file")
    parser.add_argument("--tcp-duplex-csv", default=None,  help="Path to TCP CPU duplex CSV file")
    parser.add_argument("--uds-duplex-csv", default=None,  help="Path to UDS CPU duplex CSV file")
    parser.add_argument("--mpi-incast-csv", default=None,  help="Path to MPI incast summary CSV file")
    parser.add_argument("--tcp-incast-csv", default=None,  help="Path to TCP incast summary CSV file")
    parser.add_argument("--uds-incast-csv", default=None,  help="Path to UDS incast summary CSV file")
    parser.add_argument("--cpu-prefix",     default="cpu_pair",  help="Output prefix for MPI CPU plots")
    parser.add_argument("--tcp-prefix",     default="tcp_pair",  help="Output prefix for TCP plots")
    parser.add_argument("--uds-prefix",     default="uds_pair",  help="Output prefix for UDS plots")
    args = parser.parse_args()

    print(f"Generating heatmaps using:\n  CPU CSV: {args.cpu_csv}\n  TCP CSV: {args.tcp_csv or '(none)'}\n  UDS CSV: {args.uds_csv or '(none)'}\n  GPU CSV: {args.gpu_csv}")

    # Simplex MPI CPU
    if os.path.exists(args.cpu_csv):
        plot_cpu_pair_heatmaps(args.cpu_csv, output_prefix=args.cpu_prefix)
    else:
        print(f"[Warning] CPU CSV not found: {args.cpu_csv}. Skipping MPI heatmaps.")

    # Simplex TCP CPU
    if args.tcp_csv:
        if os.path.exists(args.tcp_csv):
            plot_tcp_pair_heatmaps(args.tcp_csv, output_prefix=args.tcp_prefix)
            summarize_locality(args.tcp_csv, output_prefix=args.tcp_prefix, transport="tcp")
        else:
            print(f"[Warning] TCP CSV not found: {args.tcp_csv}. Skipping TCP heatmaps.")

    # Simplex UDS CPU
    if args.uds_csv:
        if os.path.exists(args.uds_csv):
            plot_uds_pair_heatmaps(args.uds_csv, output_prefix=args.uds_prefix)
            summarize_locality(args.uds_csv, output_prefix=args.uds_prefix, transport="uds")
        else:
            print(f"[Warning] UDS CSV not found: {args.uds_csv}. Skipping UDS heatmaps.")

    # GPU
    if os.path.exists(args.gpu_csv):
        plot_gpu_transfer_heatmaps(args.gpu_csv)
    else:
        print(f"[Warning] GPU CSV not found: {args.gpu_csv}. Skipping GPU heatmaps.")

    # MPI vs TCP comparison
    if args.tcp_csv and os.path.exists(args.cpu_csv) and os.path.exists(args.tcp_csv):
        compare_locality(args.cpu_csv, args.tcp_csv)

    # MPI vs TCP vs UDS 3-way comparison
    if args.uds_csv and os.path.exists(args.uds_csv):
        compare_three_way_locality(args.cpu_csv, args.tcp_csv, args.uds_csv, output_prefix="transport_comparison_3way")

    # Duplex MPI
    if args.mpi_duplex_csv and os.path.exists(args.mpi_duplex_csv):
        plot_cpu_pair_heatmaps(args.mpi_duplex_csv, output_prefix="mpi_duplex")
        summarize_locality(args.mpi_duplex_csv, output_prefix="mpi_duplex", transport="mpi")
        if os.path.exists(args.cpu_csv):
            compare_simplex_vs_duplex(args.cpu_csv, args.mpi_duplex_csv, transport="mpi", output_prefix="mpi_duplex_vs_simplex")

    # Duplex TCP
    if args.tcp_duplex_csv and os.path.exists(args.tcp_duplex_csv):
        plot_tcp_pair_heatmaps(args.tcp_duplex_csv, output_prefix="tcp_duplex")
        summarize_locality(args.tcp_duplex_csv, output_prefix="tcp_duplex", transport="tcp")
        if args.tcp_csv and os.path.exists(args.tcp_csv):
            compare_simplex_vs_duplex(args.tcp_csv, args.tcp_duplex_csv, transport="tcp", output_prefix="tcp_duplex_vs_simplex")

    # Duplex UDS
    if args.uds_duplex_csv and os.path.exists(args.uds_duplex_csv):
        plot_uds_pair_heatmaps(args.uds_duplex_csv, output_prefix="uds_duplex")
        summarize_locality(args.uds_duplex_csv, output_prefix="uds_duplex", transport="uds")
        if args.uds_csv and os.path.exists(args.uds_csv):
            compare_simplex_vs_duplex(args.uds_csv, args.uds_duplex_csv, transport="uds", output_prefix="uds_duplex_vs_simplex")

    # Incast Flooding
    if (args.mpi_incast_csv and os.path.exists(args.mpi_incast_csv)) or \
       (args.tcp_incast_csv and os.path.exists(args.tcp_incast_csv)) or \
       (args.uds_incast_csv and os.path.exists(args.uds_incast_csv)):
        plot_incast_results(mpi_sum_csv=args.mpi_incast_csv,
                            tcp_sum_csv=args.tcp_incast_csv,
                            uds_sum_csv=args.uds_incast_csv,
                            output_prefix="incast_flooding")


if __name__ == "__main__":
    main()

