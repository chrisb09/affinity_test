#!/usr/bin/env python3
"""Analyze switch-hop distance from MPI/TCP pair-latency CSVs.

Derives the switch-hop distance between two ranks from the Slurm tree
topology (``scontrol show topology``) and aggregates the per-pair latency and
goodput by hop count. Supports both the MPI and the TCP/IPoIB pair CSVs.

The input CSVs use the column layout produced by ``mpi_pair_latency`` /
``tcp_pair_latency``:

    0 rank_i, 1 host_i, 2 cpu_i, 3 affinity_i,
    4 rank_j, 5 host_j, 6 cpu_j, 7 affinity_j,
    8 small_bytes, 9 large_bytes, 10 warmup, 11 iters,
    12 latency_min_us, 13 latency_mean_us, 14 latency_median_us, 15 latency_max_us,
    16 bandwidth_median_gbps, 17 bandwidth_mean_gbps
"""

import argparse
import csv
import statistics
import sys
from collections import defaultdict


def parse_topology(path):
    """Return {hostname: leaf_switch} for all Level-0 switches."""
    host_to_switch = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("SwitchName="):
                continue
            fields = dict(
                tok.split("=", 1) for tok in line.split() if "=" in tok
            )
            level = fields.get("Level", "0")
            if level != "0":
                continue
            switch = fields["SwitchName"]
            for host in expand_hostlist(fields.get("Nodes", "")):
                host_to_switch[host] = switch
    return host_to_switch


def expand_hostlist(spec):
    """Expand a Slurm hostlist expression into individual hostnames."""
    import re

    hosts = []
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


def short(host):
    return host.split(".")[0]


def hop_count(host_i, host_j, host_to_switch):
    si = host_to_switch.get(short(host_i))
    sj = host_to_switch.get(short(host_j))
    if si is None or sj is None:
        return None, si, sj
    if short(host_i) == short(host_j):
        return "intra_node", si, sj
    # Two-tier fat-tree: intra-leaf = 1 switch hop, inter-leaf = 3.
    return (1 if si == sj else 3), si, sj


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", action="append", required=True,
                    help="transport=path (repeatable), e.g. mpi=...csv")
    ap.add_argument("--topology", required=True,
                    help="scontrol show topology capture")
    ap.add_argument("--out", required=True, help="output summary CSV")
    args = ap.parse_args()

    host_to_switch = parse_topology(args.topology)

    # (transport, hop) -> list of (latency_median_us, bandwidth_median_gibs)
    buckets = defaultdict(list)
    unknown = 0
    for spec in args.csv:
        transport, path = spec.split("=", 1)
        with open(path) as fh:
            reader = csv.reader(fh)
            header = next(reader)
            cols = {name: idx for idx, name in enumerate(header)}

            def find(*names, default):
                for n in names:
                    if n in cols:
                        return cols[n]
                return default

            i_host = find("host_i", default=1)
            j_host = find("host_j", default=5)
            i_lat = find("latency_median_us", default=14)
            i_bw = find("bandwidth_median_gibs", "bandwidth_median_gbps",
                        default=16)
            # The header names differ slightly between tools; locate by name.
            for row in reader:
                if not row:
                    continue
                hi = row[i_host]
                hj = row[j_host]
                lat = float(row[i_lat])
                bw = float(row[i_bw])
                hops, _, _ = hop_count(hi, hj, host_to_switch)
                if hops is None:
                    unknown += 1
                    continue
                buckets[(transport, hops)].append((lat, bw))

    if unknown:
        print(f"warning: {unknown} pairs could not be mapped to a switch",
              file=sys.stderr)

    def hop_sort_key(k):
        transport, hops = k
        order = {"intra_node": 0, 1: 1, 3: 3}
        return (transport, order.get(hops, 99))

    with open(args.out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["transport", "switch_hops", "pair_count",
                         "lat_median_us", "lat_p25_us", "lat_p75_us",
                         "bw_median_gibs"])
        for key in sorted(buckets, key=hop_sort_key):
            transport, hops = key
            vals = buckets[key]
            lats = sorted(v[0] for v in vals)
            bws = sorted(v[1] for v in vals)

            def pct(xs, q):
                if not xs:
                    return float("nan")
                idx = q * (len(xs) - 1)
                lo, hi = int(idx), min(int(idx) + 1, len(xs) - 1)
                return xs[lo] + (xs[hi] - xs[lo]) * (idx - lo)

            writer.writerow([
                transport, hops, len(vals),
                f"{statistics.median(lats):.6f}",
                f"{pct(lats, 0.25):.6f}",
                f"{pct(lats, 0.75):.6f}",
                f"{statistics.median(bws):.6f}",
            ])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
