#!/usr/bin/env python3
"""
plot_coupling_schemes.py
========================
Visualizes HPC-ML Coupling Transport Schemes and Fundamental Limits.

Plots:
  1. Makespan Comparison (ms) across all transport schemes (MPI vs TCP).
  2. Aggregate Ingress/Egress Throughput (GiB/s) per scheme.
  3. Credit Concurrency Knee Curve (Makespan vs In-Flight Concurrency Limit).
  4. Worker Turnaround Latency by Locality Tier (Boxplot).
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

TIER_ORDER = ["same_numa", "same_socket_diff_numa", "cross_socket", "cross_node"]
TIER_LABELS = {
    "same_numa": "Same NUMA",
    "same_socket_diff_numa": "Same socket,\ndiff NUMA",
    "cross_socket": "Cross-socket",
    "cross_node": "Cross-node",
}

SCHEME_DISPLAY_NAMES = {
    "1_incast_191to1": "1. Incast (191->1)",
    "2_fanout_1to191": "2. Fan-out (1->191)",
    "3_full_duplex": "3. Full-Duplex (191<->1)",
    "4_gather_scatter": "4. Gather-Scatter",
    "5_immediate_response": "5. Immediate Response",
    "6_serialized_ref": "6. Serialized Ref",
    "7_credits_cap1": "Cap 1",
    "7_credits_cap2": "Cap 2",
    "7_credits_cap4": "Cap 4",
    "7_credits_cap8": "Cap 8",
    "7_credits_cap16": "Cap 16",
    "7_credits_cap32": "Cap 32",
    "7_credits_cap64": "Cap 64",
    "8_credits_locality_1local_1remote": "8. Locality Credits",
    "9_credits_aix_ramping": "9. AIx Ramping",
}


def plot_makespan_comparison(mpi_sum_df=None, tcp_sum_df=None, output_prefix="coupling"):
    """Side-by-side makespan comparison bar chart."""
    core_schemes = [
        "1_incast_191to1", "2_fanout_1to191", "3_full_duplex",
        "4_gather_scatter", "5_immediate_response", "6_serialized_ref",
        "8_credits_locality_1local_1remote", "9_credits_aix_ramping"
    ]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    x = np.arange(len(core_schemes))
    width = 0.35

    if mpi_sum_df is not None:
        sub_mpi = mpi_sum_df[mpi_sum_df['target_rank'] == 0]
        mpi_map = {row['pattern']: row['makespan_median_ms'] for _, row in sub_mpi.iterrows()}
        mpi_vals = [mpi_map.get(s, 0.0) for s in core_schemes]
        ax.bar(x - (width / 2 if tcp_sum_df is not None else 0), mpi_vals, width,
               label='MPI', color='steelblue', alpha=0.85)

    if tcp_sum_df is not None:
        sub_tcp = tcp_sum_df[tcp_sum_df['target_rank'] == 0]
        tcp_map = {row['pattern']: row['makespan_median_ms'] for _, row in sub_tcp.iterrows()}
        tcp_vals = [tcp_map.get(s, 0.0) for s in core_schemes]
        ax.bar(x + width / 2, tcp_vals, width,
               label='TCP/IP (IPoIB)', color='tomato', alpha=0.85)

    labels = [SCHEME_DISPLAY_NAMES.get(s, s) for s in core_schemes]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=10)
    ax.set_ylabel("Total Phase Makespan (ms)", fontsize=11)
    ax.set_title("HPC-ML Coupling Transport Schemes: End-to-End Makespan\n(4 MiB In + 4 MiB Out per Worker, 191 Workers)", fontsize=12, weight="bold")
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(fontsize=10)

    plt.tight_layout()
    out = f"{output_prefix}_makespan_comparison.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[Success] Saved makespan comparison chart to '{out}'.")


def plot_credit_concurrency_curve(mpi_sum_df=None, tcp_sum_df=None, output_prefix="coupling"):
    """Plots Makespan vs Concurrency Cap C showing the contention knee."""
    caps = [1, 2, 4, 8, 16, 32, 64]

    plt.figure(figsize=(9, 5), dpi=300)

    if mpi_sum_df is not None:
        sub = mpi_sum_df[mpi_sum_df['target_rank'] == 0]
        mpi_map = {row['pattern']: row['makespan_median_ms'] for _, row in sub.iterrows()}
        y_mpi = [mpi_map.get(f"7_credits_cap{c}", np.nan) for c in caps]
        plt.plot(caps, y_mpi, 'o-', label='MPI (Fixed Credits)', color='steelblue', linewidth=2, markersize=7)

        # Annotate AIx ramping baseline
        aix_ms = mpi_map.get("9_credits_aix_ramping", np.nan)
        if not np.isnan(aix_ms):
            plt.axhline(aix_ms, color='forestgreen', linestyle='--', label=f'MPI AIx Ramping ({aix_ms:.1f} ms)')

    plt.xscale('log', base=2)
    plt.xticks(caps, [str(c) for c in caps])
    plt.xlabel("Concurrent In-Flight Worker Exchanges (Credits)", fontsize=11)
    plt.ylabel("Total Phase Makespan (ms)", fontsize=11)
    plt.title("Credit-Controlled Coupling: Makespan vs Concurrency Cap\n(Finding the Optimal Contention / Pipeline Overlap Knee)", fontsize=12, weight="bold")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)

    plt.tight_layout()
    out = f"{output_prefix}_credits_knee_curve.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[Success] Saved credit knee curve to '{out}'.")


def plot_locality_boxplots(mpi_det_df=None, output_prefix="coupling"):
    """Boxplot of worker turnaround times by locality tier for immediate_response vs gather_scatter."""
    if mpi_det_df is None:
        return

    schemes_to_plot = ["4_gather_scatter", "5_immediate_response", "8_credits_locality_1local_1remote"]
    sub = mpi_det_df[(mpi_det_df['target_rank'] == 0) & (mpi_det_df['pattern'].isin(schemes_to_plot))]

    if sub.empty:
        return

    fig, axes = plt.subplots(1, len(schemes_to_plot), figsize=(5 * len(schemes_to_plot), 5), dpi=300, sharey=True)
    if len(schemes_to_plot) == 1:
        axes = [axes]

    for idx, sname in enumerate(schemes_to_plot):
        ax = axes[idx]
        s_sub = sub[sub['pattern'] == sname]
        data = [s_sub[s_sub['locality_tier'] == t]['lat_median_us'].values / 1000.0 for t in TIER_ORDER if t in s_sub['locality_tier'].values]
        labels = [TIER_LABELS[t] for t in TIER_ORDER if t in s_sub['locality_tier'].values]

        ax.boxplot(data, tick_labels=labels, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', color='steelblue'),
                   medianprops=dict(color='crimson', linewidth=2))
        ax.set_title(SCHEME_DISPLAY_NAMES.get(sname, sname), fontsize=11, weight="bold")
        ax.set_xlabel("Locality Tier", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Worker Turnaround Time (ms)", fontsize=11)
        ax.grid(axis='y', linestyle=':', alpha=0.5)

    plt.suptitle("Worker Turnaround Latency by Locality Tier", fontsize=13, weight="bold", y=1.02)
    plt.tight_layout()
    out = f"{output_prefix}_locality_turnaround_boxplots.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[Success] Saved locality turnaround boxplots to '{out}'.")


def plot_worker_scaling_curves(mpi_sweep_csv=None, tcp_sweep_csv=None, output_prefix="worker_scaling"):
    """
    Plots scaling curves as active worker count W increases from 1 to 191:
      1. Aggregate Throughput (GiB/s) vs Active Workers W (MPI vs TCP)
      2. Makespan (ms) vs Active Workers W
      3. Locality Breakdown (Local vs Remote) vs Active Workers W
    """
    dfs = {}
    if mpi_sweep_csv and os.path.exists(mpi_sweep_csv):
        dfs['MPI'] = pd.read_csv(mpi_sweep_csv)
    if tcp_sweep_csv and os.path.exists(tcp_sweep_csv):
        dfs['TCP'] = pd.read_csv(tcp_sweep_csv)

    if not dfs:
        return

    patterns = ['incast', 'fanout', 'full_duplex', 'immediate_response']
    colors = {
        'incast': 'steelblue',
        'fanout': 'forestgreen',
        'full_duplex': 'darkorange',
        'immediate_response': 'purple'
    }
    linestyles = {'MPI': '-', 'TCP': '--'}
    markers = {'MPI': 'o', 'TCP': 's'}

    # 1. Aggregate Bandwidth vs Active Workers W
    plt.figure(figsize=(11, 6), dpi=300)
    for trans, df in dfs.items():
        for pat in patterns:
            sub = df[df['pattern'] == pat].sort_values('active_workers')
            if sub.empty:
                continue
            bw_col = 'combined_bw_median_gibs' if pat in ['full_duplex', 'immediate_response'] else ('ingress_bw_median_gibs' if pat == 'incast' else 'egress_bw_median_gibs')
            plt.plot(sub['active_workers'], sub[bw_col],
                     linestyle=linestyles.get(trans, '-'),
                     marker=markers.get(trans, 'o'),
                     color=colors.get(pat, 'black'),
                     label=f"{trans} - {pat}",
                     linewidth=2, markersize=5)

    all_w = sorted(next(iter(dfs.values()))['active_workers'].unique())
    plt.xticks(all_w, [str(w) for w in all_w], rotation=30)
    plt.xlabel("Active Concurrent Workers (W)", fontsize=11)
    plt.ylabel("Controller Aggregate Bandwidth (GiB/s)", fontsize=11)
    plt.title("Controller Throughput Scaling: MPI vs TCP/IP (1 to 191 Workers)", fontsize=12, weight="bold")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    bw_out = f"{output_prefix}_throughput_curve.png"
    plt.savefig(bw_out, dpi=300); plt.close()
    print(f"[Success] Saved worker scaling throughput curve to '{bw_out}'.")

    # 2. Makespan vs Active Workers W
    plt.figure(figsize=(11, 6), dpi=300)
    for trans, df in dfs.items():
        for pat in patterns:
            sub = df[df['pattern'] == pat].sort_values('active_workers')
            if sub.empty:
                continue
            plt.plot(sub['active_workers'], sub['makespan_median_ms'],
                     linestyle=linestyles.get(trans, '-'),
                     marker=markers.get(trans, 'o'),
                     color=colors.get(pat, 'black'),
                     label=f"{trans} - {pat}",
                     linewidth=2, markersize=5)

    plt.xticks(all_w, [str(w) for w in all_w], rotation=30)
    plt.xlabel("Active Concurrent Workers (W)", fontsize=11)
    plt.ylabel("Total Phase Makespan (ms)", fontsize=11)
    plt.title("Phase Makespan Scaling: MPI vs TCP/IP (1 to 191 Workers)", fontsize=12, weight="bold")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    ms_out = f"{output_prefix}_makespan_curve.png"
    plt.savefig(ms_out, dpi=300); plt.close()
    print(f"[Success] Saved worker scaling makespan curve to '{ms_out}'.")

    # 3. Local vs Remote Worker Latency (Immediate Response)
    fig, axes = plt.subplots(1, len(dfs), figsize=(7 * len(dfs), 5), dpi=300, squeeze=False)
    for idx, (trans, df) in enumerate(dfs.items()):
        ax = axes[0, idx]
        sub_imm = df[df['pattern'] == 'immediate_response'].sort_values('active_workers')
        if not sub_imm.empty:
            ax.plot(sub_imm['active_workers'], sub_imm['local_lat_median_us'] / 1000.0, 'o-', color='royalblue', label='Local Workers (Node 0) - p50', linewidth=2)
            ax.plot(sub_imm['active_workers'], sub_imm['local_lat_p95_us'] / 1000.0, 'o--', color='royalblue', label='Local Workers (Node 0) - p95', linewidth=1.5, alpha=0.7)
            ax.plot(sub_imm['active_workers'], sub_imm['remote_lat_median_us'] / 1000.0, 's-', color='crimson', label='Remote Workers (Node 1) - p50', linewidth=2)
            ax.plot(sub_imm['active_workers'], sub_imm['remote_lat_p95_us'] / 1000.0, 's--', color='crimson', label='Remote Workers (Node 1) - p95', linewidth=1.5, alpha=0.7)

            ax.set_xticks(all_w)
            ax.set_xticklabels([str(w) for w in all_w], rotation=30)
            ax.set_xlabel("Active Concurrent Workers (W)", fontsize=10)
            ax.set_ylabel("Worker Turnaround Time (ms)", fontsize=10)
            ax.set_title(f"{trans} Locality Latency (Immediate Response)", fontsize=11, weight="bold")
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.legend(fontsize=9)

    plt.tight_layout()
    loc_out = f"{output_prefix}_locality_latency_curve.png"
    plt.savefig(loc_out, dpi=300); plt.close()
    print(f"[Success] Saved locality latency curve to '{loc_out}'.")


def main():
    parser = argparse.ArgumentParser(description="Plot coupling transport schemes results.")
    parser.add_argument("--mpi-summary", default="mpi_coupling_summary.csv", help="MPI summary CSV")
    parser.add_argument("--mpi-detail",  default="mpi_coupling_details.csv", help="MPI detail CSV")
    parser.add_argument("--tcp-summary", default="tcp_coupling_summary.csv", help="TCP summary CSV")
    parser.add_argument("--tcp-detail",  default="tcp_coupling_details.csv", help="TCP detail CSV")
    parser.add_argument("--mpi-worker-sweep", default=None, help="MPI worker count sweep summary CSV")
    parser.add_argument("--tcp-worker-sweep", default=None, help="TCP worker count sweep summary CSV")
    parser.add_argument("--output-prefix", default="coupling_schemes", help="Output prefix")
    args = parser.parse_args()

    mpi_sum = pd.read_csv(args.mpi_summary) if os.path.exists(args.mpi_summary) else None
    mpi_det = pd.read_csv(args.mpi_detail)  if os.path.exists(args.mpi_detail)  else None
    tcp_sum = pd.read_csv(args.tcp_summary) if os.path.exists(args.tcp_summary) else None
    tcp_det = pd.read_csv(args.tcp_detail)  if os.path.exists(args.tcp_detail)  else None

    if mpi_sum is not None or tcp_sum is not None:
        plot_makespan_comparison(mpi_sum, tcp_sum, output_prefix=args.output_prefix)
        plot_credit_concurrency_curve(mpi_sum, tcp_sum, output_prefix=args.output_prefix)

    if mpi_det is not None:
        plot_locality_boxplots(mpi_det, output_prefix=args.output_prefix)

    if (args.mpi_worker_sweep and os.path.exists(args.mpi_worker_sweep)) or (args.tcp_worker_sweep and os.path.exists(args.tcp_worker_sweep)):
        plot_worker_scaling_curves(args.mpi_worker_sweep, args.tcp_worker_sweep, output_prefix="worker_scaling")


if __name__ == "__main__":
    main()
