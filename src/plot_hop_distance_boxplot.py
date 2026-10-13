#!/usr/bin/env python3
"""Box plot of cross-node latency and goodput by switch-hop distance.

Reads the per-pair CSVs (via the same hop classification as
``analyze_hop_distance.py``) and draws a two-panel figure:

    (a) cross-node latency [us]            (b) cross-node goodput [GiB/s]

one box per (transport, hop-count) combination, so the full distribution -- not
just the median -- is visible. Boxes are grouped by transport (MPI, TCP/IPoIB)
with a 3-hop and a 1-hop box in each group.

The figure is styled with the thesis ``thesis_style`` module when reachable
(Latin Modern via the PGF backend, .pgf/.pdf/.png output); otherwise a plain
serif fallback is used.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path

import matplotlib

_THESIS_SC = Path("/rwthfs/rz/cluster/home/ro092286/Master-Thesis/gitlab/sourcecode")
_HAS_TS = False
if _THESIS_SC.is_dir():
    import sys

    sys.path.insert(0, str(_THESIS_SC))
    try:
        import thesis_style as ts  # type: ignore

        _HAS_TS = True
    except Exception:
        _HAS_TS = False

if not _HAS_TS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

TRANSPORTS = [("mpi", "MPI"), ("tcp", "TCP/IPoIB")]
HOPS = [3, 1]
HOP_COLORS = {3: "#c0504d", 1: "#4f81bd"}
HOP_LABELS = {3: "3 switch hops", 1: "1 switch hop"}


def expand_hostlist(spec: str) -> list[str]:
    hosts: list[str] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        m = re.match(r"^(.*?)\[(.*)\]$", item)
        if not m:
            hosts.append(item)
            continue
        prefix, ranges = m.group(1), m.group(2)
        for part in ranges.split(","):
            if "-" in part:
                start, end = part.split("-", 1)
                width = len(start)
                for n in range(int(start), int(end) + 1):
                    hosts.append(f"{prefix}{n:0{width}d}")
            else:
                hosts.append(f"{prefix}{part}")
    return hosts


def parse_topology(path: Path) -> dict[str, str]:
    host_to_switch: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("SwitchName=") or "Level=0" not in line:
            continue
        fields = dict(tok.split("=", 1) for tok in line.split() if "=" in tok)
        for host in expand_hostlist(fields.get("Nodes", "")):
            host_to_switch[host] = fields["SwitchName"]
    return host_to_switch


def short(host: str) -> str:
    return host.split(".")[0]


def collect_pairs(csv_path: Path, host_to_switch: dict[str, str],
                  hops_wanted: int) -> tuple[list[float], list[float]]:
    """Return (latency_us, goodput_gibs) for cross-node pairs at the given hop count."""
    lats: list[float] = []
    bws: list[float] = []
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        bw_key = "bandwidth_median_gibs" if "bandwidth_median_gibs" in cols else "bandwidth_median_gbps"
        for row in reader:
            hi, hj = short(row["host_i"]), short(row["host_j"])
            if hi == hj:
                continue  # intra-node pairs are not network paths
            si, sj = host_to_switch.get(hi), host_to_switch.get(hj)
            if si is None or sj is None:
                continue
            hops = 1 if si == sj else 3
            if hops != hops_wanted:
                continue
            lats.append(float(row["latency_median_us"]))
            bws.append(float(row[bw_key]))
    return lats, bws


def add_boxes(ax, data, metric, ylabel, title=None, logy=False):
    import numpy as np

    n_groups = len(TRANSPORTS)
    x = np.arange(n_groups)
    width = 0.5
    positions = {3: [], 1: []}
    for gi in range(n_groups):
        # Two boxes per group, offset left/right within the group slot.
        for idx, hops in enumerate(HOPS):
            positions[hops].append(x[gi] + (idx - 0.5) * width)

    for hops in HOPS:
        vals = [data[(base, hops)][metric] for base, _ in TRANSPORTS]
        bp = ax.boxplot(
            vals,
            positions=positions[hops],
            widths=width * 0.9,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.1),
            whiskerprops=dict(color="black", linewidth=0.8),
            capprops=dict(color="black", linewidth=0.8),
            boxprops=dict(color="black", linewidth=0.7),
        )
        for box in bp["boxes"]:
            box.set_facecolor(HOP_COLORS[hops])
            box.set_alpha(0.85)

        # Annotate the median so no separate table/bar chart is needed.
        for gi, values in enumerate(vals):
            if not values:
                continue
            median = statistics.median(values)
            ax.annotate(
                f"{median:.2f}",
                xy=(positions[hops][gi], median),
                xytext=(0, 2.5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=ts.ANNOT if _HAS_TS else 8.5,
                zorder=5,
                bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                          edgecolor="none", alpha=0.75),
            )

    # Legend proxies are collected by the caller for a single shared legend.
    import matplotlib.patches as mpatches

    handles = [mpatches.Patch(facecolor=HOP_COLORS[h], edgecolor="black",
                              alpha=0.85, label=HOP_LABELS[h]) for h in HOPS]
    ax._hop_handles = handles

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in TRANSPORTS])
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.set_axisbelow(True)
    ax.margins(y=0.12)
    ax.set_xlim(-0.6, n_groups - 0.4)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topology",
                    default="data/10_infiniband_topology/claix23_slurm_topology.txt")
    ap.add_argument("--csv", action="append", required=True,
                    help="transport=path, e.g. mpi=data/.../cpu_1hop_mpi_pair_latency.csv")
    ap.add_argument("--out",
                    default="data/10_infiniband_topology/hop_distance_boxplot")
    args = ap.parse_args()

    host_to_switch = parse_topology(Path(args.topology))

    # data[(transport, hops)] = {lat: [...], bw: [...]}
    data: dict[tuple[str, int], dict] = {}
    for spec in args.csv:
        transport, path = spec.split("=", 1)
        base = transport.split("_")[0]
        for hops in (1, 3):
            lats, bws = collect_pairs(Path(path), host_to_switch, hops)
            slot = data.setdefault((base, hops), {"lat": [], "bw": []})
            slot["lat"].extend(lats)
            slot["bw"].extend(bws)

    if _HAS_TS:
        ts.apply()
        width_in, height_in = ts.figure_size(1.0, aspect=0.42)
    else:
        plt.rcParams.update({"font.family": "serif", "axes.grid": True,
                             "grid.alpha": 0.3, "grid.linestyle": ":"})
        width_in, height_in = (5.9055, 2.48)

    fig, axes = plt.subplots(1, 2, figsize=(width_in, height_in))
    add_boxes(axes[0], data, "lat", ylabel="cross-node latency [\u00b5s]",
              title="(a) Latency")
    add_boxes(axes[1], data, "bw", ylabel="cross-node goodput [GiB/s]",
              title="(b) Goodput")
    fig.legend(axes[0]._hop_handles, [HOP_LABELS[3], HOP_LABELS[1]],
               loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = Path(args.out)
    if _HAS_TS:
        ts.save(fig, out)
        print(f"wrote {out}.pgf / .pdf / .png (thesis style)")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out) + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(str(out) + ".pdf")
        plt.close(fig)
        print(f"wrote {out}.png / .pdf (fallback style)")


if __name__ == "__main__":
    main()
