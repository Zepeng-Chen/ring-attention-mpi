#!/usr/bin/env python3
"""Build every measured figure of the report from the retrieved cluster CSVs.

Usage: MPLCONFIGDIR=results/matplotlib-cache python3 scripts/make_figures.py [name ...]
Each figure function returns a dict of the key numbers it plotted; they are
collected in report/generated/figure_numbers.json so that the text of the
report quotes exactly what the figures show.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plotstyle as ps
import results_io as io
from plotstyle import plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
GEN = ROOT / "report" / "generated"


def slope(xs, ys):
    """Least-squares slope of log y against log x."""
    lx, ly = np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float))
    return float(np.polyfit(lx, ly, 1)[0])


# ---------------------------------------------------------------------------
def fig_serial():
    rows = io.passed(io.load("e1-serial", "serial-n.csv"))
    if not rows:
        return {}
    ps.apply()
    fig, axes = plt.subplots(1, 3, figsize=(ps.FULL, 2.6), gridspec_kw={"wspace": 0.42})
    numbers = {}
    # (a) time against N
    ax = axes[0]
    by = io.summarise(rows, ["impl", "kernel", "N"])
    series = [("S0", "reference", ps.MUTED, "S0 naive, fp64", "v"),
              ("S0-fp32", "reference", ps.GRAY, "S0 naive, fp32", "D"),
              ("S1", "basic", ps.BLUE, "S1 blocked, scalar", "o"),
              ("S1", "tuned", ps.ORANGE, "S1 blocked, tuned", "s")]
    for impl, kernel, colour, label, marker in series:
        pts = sorted((int(k[2]), v["median"]) for k, v in by.items() if k[0] == impl and k[1] == kernel)
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker=marker, color=colour, label=label)
        big = [(x, y) for x, y in pts if x >= 4096]
        numbers[f"slope_N_{impl}_{kernel}"] = slope(*zip(*big)) if len(big) >= 2 else float("nan")
        numbers[f"time_{impl}_{kernel}"] = {str(x): y for x, y in pts}
    ps.log2_axis(ax, "x", [2**e for e in range(10, 18, 2)])
    ps.log10_axis(ax, "y")
    ax.set_xlabel("sequence length $N$"); ax.set_ylabel("run time (s)")
    ax.set_title("(a) Time vs. $N$  ($d=d_v=64$)")
    ax.legend(loc="lower right", fontsize=ps.TINY, handlelength=1.4)
    # (b) time against d
    ax = axes[1]
    rows_d = io.passed(io.load("e1-serial", "serial-d.csv"))
    by_d = io.summarise(rows_d, ["impl", "kernel", "d", "N"])
    for impl, kernel, colour, label, marker in series[1:]:
        pts = sorted((int(k[2]), v["median"] / (float(k[3]) ** 2) * 1e9) for k, v in by_d.items() if k[0] == impl and k[1] == kernel)
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker=marker, color=colour, label=label)
        numbers[f"slope_d_{impl}_{kernel}"] = slope(xs, ys)
    ps.log2_axis(ax, "x", [16, 32, 64, 128, 256]); ps.log10_axis(ax, "y")
    ax.set_xlabel("head dimension $d=d_v$"); ax.set_ylabel("time per query-key pair (ns)")
    ax.set_title("(b) Time vs. $d$")
    # (c) throughput against N: where caches end
    ax = axes[2]
    for impl, kernel, colour, label, marker in series[1:]:
        pts = sorted((int(k[2]), float(v["row"]["useful_flops"]) / v["median"] / 1e9) for k, v in by.items() if k[0] == impl and k[1] == kernel)
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker=marker, color=colour, label=label)
        numbers[f"gflops_{impl}_{kernel}"] = {str(x): y for x, y in pts}
    ps.log2_axis(ax, "x", [2**e for e in range(10, 18, 2)])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("sequence length $N$"); ax.set_ylabel("useful GFLOP/s")
    ax.set_title("(c) Throughput of one core")
    ps.save(fig, OUT / "serial_scaling")
    return numbers


def fig_tiles():
    rows = io.passed(io.load("e1-serial", "serial-tiles.csv"))
    if not rows:
        return {}
    ps.apply()
    by = io.summarise(rows, ["block_q", "block_k"])
    bqs = sorted({int(k[0]) for k in by}); bks = sorted({int(k[1]) for k in by})
    grid = np.full((len(bqs), len(bks)), np.nan)
    for (bq, bk), v in by.items():
        grid[bqs.index(int(bq)), bks.index(int(bk))] = float(v["row"]["useful_flops"]) / v["median"] / 1e9
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    image = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=np.nanmin(grid), vmax=np.nanmax(grid))
    for i in range(len(bqs)):
        for j in range(len(bks)):
            if not np.isnan(grid[i, j]):
                dark = grid[i, j] > (np.nanmin(grid) + np.nanmax(grid)) / 2
                ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=ps.SMALL, color="#ffffff" if dark else ps.INK)
    ax.set_xticks(range(len(bks))); ax.set_xticklabels([ps.human(b) for b in bks])
    ax.set_yticks(range(len(bqs))); ax.set_yticklabels(bqs)
    ax.set_xlabel("key tile $B_k$ (keys)"); ax.set_ylabel("query tile $B_q$ (rows)")
    ax.set_title("Useful GFLOP/s of the tuned kernel, one core")
    ax.grid(False)
    for side in ax.spines.values():
        side.set_visible(False)
    best = np.unravel_index(np.nanargmax(grid), grid.shape)
    ps.save(fig, OUT / "serial_tiles")
    return {"best_bq": bqs[best[0]], "best_bk": bks[best[1]], "best_gflops": float(np.nanmax(grid)),
            "worst_gflops": float(np.nanmin(grid))}


# ---------------------------------------------------------------------------
def fig_gpu_ladder():
    rows = io.passed(io.load("gpu-ladder", "ladder.csv"))
    if not rows:
        return {}
    ps.apply()
    by = io.summarise(rows, ["kernel", "N"])
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL, 2.7), gridspec_kw={"wspace": 0.3})
    numbers = {}
    serial = io.summarise(io.passed(io.load("e1-serial", "serial-n.csv")), ["impl", "kernel", "N"])
    cpu = sorted((int(k[2]), v["median"]) for k, v in serial.items() if k[0] == "S1" and k[1] == "tuned")
    for kernel in ("k0", "k1", "k2"):
        pts = sorted((int(k[1]), v["median"], float(v["row"]["useful_flops"])) for k, v in by.items() if k[0] == kernel)
        xs = [p[0] for p in pts]
        axes[0].plot(xs, [p[1] for p in pts], marker="o", color=ps.GPU_KERNEL[kernel], label=ps.GPU_LABEL[kernel])
        axes[1].plot(xs, [p[2] / p[1] / 1e9 for p in pts], marker="o", color=ps.GPU_KERNEL[kernel], label=ps.GPU_LABEL[kernel])
        numbers[f"gflops_{kernel}"] = {str(p[0]): p[2] / p[1] / 1e9 for p in pts}
        numbers[f"time_{kernel}"] = {str(p[0]): p[1] for p in pts}
    if cpu:
        xs, ys = zip(*cpu)
        axes[0].plot(xs, ys, marker="s", color=ps.AQUA, label="one CPU core (S1 tuned)")
        axes[1].plot(xs, [2 * 128 * x * x / y / 1e9 for x, y in cpu], marker="s", color=ps.AQUA, label="one CPU core (S1 tuned)")
    for ax in axes:
        ps.log2_axis(ax, "x", [2**e for e in range(11, 19, 2)]); ps.log10_axis(ax, "y")
        ax.set_xlabel("sequence length $N$")
    axes[0].set_ylabel("run time (s)"); axes[0].set_title("(a) Time on one A40")
    axes[1].set_ylabel("useful GFLOP/s"); axes[1].set_title("(b) Throughput")
    axes[0].legend(loc="upper left", fontsize=ps.TINY)
    ps.save(fig, OUT / "gpu_ladder")
    return numbers


# ---------------------------------------------------------------------------
# MPI scaling. T_S is the single-core tuned serial time (S1), the baseline of
# every speedup. M0 ("local") is the no-communication control at the same P.
def _strong_tables():
    return io.strong_table()


def fig_strong():
    data = _strong_tables()
    if not data:
        return {}
    serial, table = data
    ps.apply()
    sizes = sorted(serial)
    fig, axes = plt.subplots(1, 3, figsize=(ps.FULL, 2.65), gridspec_kw={"wspace": 0.38})
    cores = sorted({k[2] for k in table})
    # (a) speedup of the overlapped ring for every N
    ax = axes[0]
    ax.plot(cores, cores, color=ps.MUTED, lw=0.9)
    ax.annotate("ideal", (cores[-1], cores[-1]), xytext=(-4, 3), textcoords="offset points", ha="right", va="bottom", fontsize=ps.SMALL, color=ps.INK2)
    for n, colour in zip(sizes, ps.ramp(len(sizes))):
        pts = sorted((k[2], v["speedup"]) for k, v in table.items() if k[0] == "overlap" and k[1] == n)
        pts = [(1, serial[n] / table[("local", n, 1)]["time"])] + pts if ("local", n, 1) in table else pts
        ax.plot(*zip(*pts), marker="o", color=colour, label=f"$N$ = {ps.human(n)}")
    ps.log2_axis(ax, "x", [1, 4, 16, 72, 288]); ps.log2_axis(ax, "y", [1, 4, 16, 64, 256])
    ax.set_xlabel("cores (= MPI ranks)"); ax.set_ylabel("speedup $T_S/T_P$")
    ax.set_title("(a) Speedup, overlapped ring")
    ax.legend(loc="upper left", fontsize=ps.SMALL)
    # (b) efficiency of every variant at the largest N
    ax = axes[1]
    n = sizes[-1]
    for impl in ("local", "allgather", "ring", "overlap", "rma_get"):
        pts = sorted((k[2], v["efficiency"]) for k, v in table.items() if k[0] == impl and k[1] == n)
        if pts:
            ax.plot(*zip(*pts), marker=ps.VARIANT_MARKER[impl], color=ps.VARIANT[impl], label=ps.VARIANT_SHORT[impl],
                    zorder=3 if impl != "local" else 2)
    ps.log2_axis(ax, "x", [1, 4, 16, 72, 288])
    ax.set_ylim(0, 1.15); ax.set_xlabel("cores (= MPI ranks)"); ax.set_ylabel("parallel efficiency")
    ax.set_title(f"(b) Efficiency, $N$ = {ps.human(n)}")
    ax.legend(loc="lower left", fontsize=ps.TINY, handlelength=1.4)
    # (c) the same efficiency split into hardware and communication parts
    ax = axes[2]
    hw = sorted((k[2], v["efficiency"]) for k, v in table.items() if k[0] == "local" and k[1] == n)
    comm = sorted((k[2], table[("local", n, k[2])]["time"] / v["time"]) for k, v in table.items()
                  if k[0] == "overlap" and k[1] == n and ("local", n, k[2]) in table)
    ax.plot(*zip(*hw), marker="X", color=ps.GRAY, label="control M0: $E_0=T_S/(P\,T_{M0})$")
    ax.plot(*zip(*comm), marker="s", color=ps.ORANGE, label="M3 against M0: $E_{\\mathrm{comm}}=T_{M0}/T_{M3}$")
    ps.log2_axis(ax, "x", [1, 4, 16, 72, 288])
    ax.set_ylim(0, 1.15); ax.set_xlabel("cores (= MPI ranks)"); ax.set_ylabel("efficiency factor")
    ax.set_title("(c) The two factors of M3")
    ax.legend(loc="lower left", fontsize=ps.TINY, handlelength=1.4)
    ps.save(fig, OUT / "strong_scaling")
    numbers = {"serial": {str(k): v for k, v in serial.items()}}
    for (impl, n, ranks), v in table.items():
        numbers[f"{impl}|{n}|{ranks}"] = {k: v[k] for k in ("time", "speedup", "efficiency", "nodes", "wait_max", "wait_mean", "compute_max", "compute_mean")}
    return numbers


def model_parameters():
    """alpha (s per exchange), B (bytes/s) and F (flop/s per core, full node)
    measured independently of the scaling runs: the shared-memory microbenchmark
    and the no-communication control."""
    net = io.load("e5-model", "netbench-node.csv")
    hw = io.summarise(io.passed(io.load("e7-hardware", "hardware.csv")), ["ranks"])
    if not net or ("72",) not in hw:
        return None
    ping = {int(r["bytes"]): float(r["seconds"]) for r in net if r["mode"] == "pingpong" and r["same_node"] == "1"}
    small = ping[min(b for b in ping if b >= 1)]
    # one ring step exchanges K and V: two sends and two receives per rank
    alpha = 2 * small
    big = max(b for b in ping if b <= 2**20)
    bandwidth = big / ping[big]
    full = hw[("72",)]
    flops = float(full["row"]["useful_flops"]) / full["median"] / 72
    return {"alpha": alpha, "B": bandwidth, "F": flops, "message_bytes": big}


def model_efficiency(c, params, d=64, dv=64, heads=1, kv_heads=1):
    compute = 2 * (d + dv) * heads * c * c / params["F"]
    comm = params["alpha"] + (d + dv) * kv_heads * 4 * c / params["B"]
    return compute / (compute + comm)


def fitted_cstar(impls=("ring", "overlap"), cmin=16):
    """One-parameter fit E = 1 / (1 + c*/c) to the two-sided rings with thin ranks
    inside one node (least squares on E). Returns (c*, largest |E_fit - E_measured|, points)."""
    by = io.summarise(io.passed(io.load("e5-model", "model-thin.csv")), ["impl", "N", "ranks"])
    points = []
    for (impl, n, ranks), v in by.items():
        control = by.get(("local", n, ranks))
        c = int(n) / int(ranks)
        if impl in impls and control and c >= cmin:
            points.append((c, control["median"] / v["median"]))
    if len(points) < 3:
        return None
    cs = np.array([p[0] for p in points]); es = np.array([p[1] for p in points])
    grid = np.linspace(1, 200, 3981)
    errors = [np.sum((1 / (1 + g / cs) - es) ** 2) for g in grid]
    best = float(grid[int(np.argmin(errors))])
    return best, float(np.max(np.abs(1 / (1 + best / cs) - es))), points


def fig_collapse():
    """Communication efficiency T_M0/T_variant against rows per rank c, all runs."""
    sources = [("strong", None), ("e3-multi", "comm-small-c.csv"), ("e3-single", "weak-rows.csv"), ("e3-multi", "weak-rows.csv"),
               ("e3-single", "weak-work.csv"), ("e3-multi", "weak-work.csv"), ("e5-model", "model-thin.csv")]
    points = []
    for prefix, name in sources:
        rows = io.strong_rows() if name is None else io.passed(io.load(prefix, name))
        by = io.summarise(rows, ["impl", "N", "ranks"])
        for (impl, n, ranks), v in by.items():
            control = by.get(("local", n, ranks))
            if impl in ("local", "S1") or control is None or int(ranks) < 2:
                continue
            points.append((impl, int(n), int(ranks), int(n) / int(ranks), control["median"] / v["median"], int(v["row"]["nodes"])))
    if not points:
        return {}
    ps.apply()
    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    params = model_parameters()
    numbers = {"points": [list(p) for p in points]}
    cs = np.logspace(np.log2(6), np.log2(140000), 240, base=2)
    if params:
        ax.plot(cs, [model_efficiency(c, params) for c in cs], color=ps.INK2, lw=1.0, linestyle=(0, (4, 2)),
                label="model, parameters from\nthe two-rank microbenchmark", zorder=1)
        numbers["model"] = params
    fit = fitted_cstar()
    if fit:
        ax.plot(cs, 1 / (1 + fit[0] / cs), color=ps.INK, lw=1.2, label=f"same form, one constant fitted to the\none-node rings: $c^*$ = {fit[0]:.0f} rows", zorder=1)
        numbers["fit"] = {"cstar": fit[0], "maxdev": fit[1]}
    for impl in ("allgather", "ring", "overlap"):
        for multi, face in ((False, ps.SURFACE), (True, ps.VARIANT[impl])):
            pts = [(p[3], p[4]) for p in points if p[0] == impl and (p[5] > 1) == multi]
            if pts:
                ax.plot(*zip(*pts), linestyle="none", marker=ps.VARIANT_MARKER[impl], markersize=5, markerfacecolor=face,
                        markeredgecolor=ps.VARIANT[impl], markeredgewidth=1.0,
                        label=ps.VARIANT_SHORT[impl] + (" (several nodes)" if multi else " (one node)"), zorder=3)
    ps.log2_axis(ax, "x", [8, 32, 128, 512, 2048, 8192, 32768, 131072])
    ax.set_ylim(0, 1.25)
    ax.axhline(1.0, color=ps.AXIS, lw=0.8)
    ax.set_xlabel("rows per rank $c=N/P$"); ax.set_ylabel("$E_{\\mathrm{comm}}=T_{M0}/T$")
    ax.set_title("Speed relative to the control M0 against rows per rank")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=ps.TINY, handlelength=1.2)
    ps.save(fig, OUT / "collapse")
    return numbers


# ---------------------------------------------------------------------------
# Real model tensors (Qwen2.5-0.5B): what exact attention shows.
def _e11(n=32768):
    run = io.latest("real-e11", f"e11-N{n}-layer00.csv") or io.latest("real-data", f"e11-N{n}-layer00.csv")
    if not run:
        return None
    data = {}
    for layer in ("00", "11", "23"):
        rows = io.read(run / f"e11-N{n}-layer{layer}.csv")
        if rows:
            data[int(layer)] = rows
    return data


def fig_real_stats(n=32768):
    data = _e11(n)
    if not data:
        return {}
    ps.apply()
    windows = sorted(int(c[6:]) for c in next(iter(data.values()))[0] if c.startswith("mass_w"))
    windows = [w for w in windows if w < n]
    layers = sorted(data)
    colours = dict(zip(layers, ps.ramp(len(layers))))
    fig, axes = plt.subplots(1, 3, figsize=(ps.FULL, 2.6), gridspec_kw={"wspace": 0.4})
    numbers = {}
    panels = (("mass_w", "probability mass in the window", "(a) Locality"),
              ("err_w", "relative output error", "(b) Window only"),
              ("err_sw", "relative output error", "(c) Window + 4 sink tokens"))
    for ax, (column, ylabel, title) in zip(axes, panels):
        for layer in layers:
            # Queries in the second half of the sequence: every window is shorter than their context.
            late = [r for r in data[layer] if int(r["row"]) >= n // 2]
            med, lo, hi = [], [], []
            for w in windows:
                values = np.array([float(r[f"{column}{w}"]) for r in late])
                med.append(np.median(values)); lo.append(np.percentile(values, 25)); hi.append(np.percentile(values, 75))
                numbers[f"{column}|layer{layer}|{w}"] = {"median": float(med[-1]), "q25": float(lo[-1]), "q75": float(hi[-1]),
                                                        "p90": float(np.percentile(values, 90)), "mean": float(values.mean())}
            floor = 1e-4 if column != "mass_w" else 0
            ax.fill_between(windows, np.maximum(lo, floor), np.maximum(hi, floor), color=colours[layer], alpha=0.12, linewidth=0)
            ax.plot(windows, np.maximum(med, floor), marker="o", color=colours[layer], label=f"layer {layer}")
        ps.log2_axis(ax, "x", windows)
        ax.set_xlabel("window $w$ (tokens)"); ax.set_ylabel(ylabel); ax.set_title(title)
        if column == "mass_w":
            ax.set_ylim(0, 1.05)
        else:
            ps.log10_axis(ax, "y"); ax.set_ylim(1e-4, 3)
        ax.tick_params(axis="x", labelsize=7)
    axes[0].legend(loc="lower right", fontsize=ps.SMALL)
    ps.save(fig, OUT / "real_window")
    for layer in layers:
        late = [r for r in data[layer] if int(r["row"]) >= n // 2]
        for column in ("entropy", "mass_first", "mass_sink", "max_prob", "argmax_distance"):
            values = np.array([float(r[column]) for r in late])
            numbers[f"{column}|layer{layer}"] = {"median": float(np.median(values)), "mean": float(values.mean()),
                                                "q25": float(np.percentile(values, 25)), "q75": float(np.percentile(values, 75))}
    return numbers


def fig_real_heads(n=32768, window=1024):
    """Per-head view: how much of each head's output a window of `window` tokens loses."""
    data = _e11(n)
    if not data:
        return {}
    ps.apply()
    layers = sorted(data)
    heads = sorted({int(r["head"]) for r in data[layers[0]]})
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL, 1.9), gridspec_kw={"wspace": 0.16})
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#eef4fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    numbers = {}
    for ax, (column, title, fmt) in zip(axes, ((f"err_w{window}", f"(a) Median relative error of a {window:,}-token window", "{:.2f}"),
                                               ("mass_first", "(b) Median probability on the first token", "{:.2f}"))):
        grid = np.zeros((len(layers), len(heads)))
        for i, layer in enumerate(layers):
            for j, head in enumerate(heads):
                values = [float(r[column]) for r in data[layer] if int(r["head"]) == head and int(r["row"]) >= n // 2]
                grid[i, j] = np.median(values)
        # The colour saturates at 1 (an error of 100% or all the probability); numbers show the value.
        ax.imshow(np.minimum(grid, 1.0), cmap=cmap, aspect="auto", vmin=0, vmax=1.0)
        for i in range(len(layers)):
            for j in range(len(heads)):
                text = f"{grid[i, j]:.0f}" if grid[i, j] >= 9.5 else (f"{grid[i, j]:.1f}" if grid[i, j] >= 0.995 else fmt.format(grid[i, j])[1:])
                ax.text(j, i, text, ha="center", va="center", fontsize=ps.TINY,
                        color="#ffffff" if grid[i, j] > 0.55 else ps.INK)
        ax.set_xticks(range(len(heads))); ax.set_xticklabels(heads, fontsize=ps.TINY)
        ax.set_yticks(range(len(layers))); ax.set_yticklabels([f"layer {l}" for l in layers] if ax is axes[0] else [])
        ax.set_xlabel("query head"); ax.set_title(title, fontsize=ps.BASE)
        ax.grid(False)
        for side in ax.spines.values():
            side.set_visible(False)
        numbers[column] = grid.tolist()
    ps.save(fig, OUT / "real_heads")
    return numbers


# ---------------------------------------------------------------------------
def fig_hardware():
    """Per-core throughput of the no-communication control as a node fills up,
    and the clock frequency sampled during the same runs."""
    run = io.latest("e7-hardware", "hardware.csv")
    if not run:
        return {}
    ps.apply()
    packed = io.summarise(io.passed(io.read(run / "hardware.csv")), ["ranks"])
    spread = io.summarise(io.passed(io.read(run / "hardware-spread.csv")), ["ranks"])
    wide = io.summarise(io.passed(io.read(run / "hardware-w256.csv")), ["ranks"])

    def per_core(table):
        return sorted((int(k[0]), float(v["row"]["useful_flops"]) / v["median"] / int(k[0]) / 1e9) for k, v in table.items())

    def frequency(label):
        out = []
        for path in sorted(run.glob(f"freq-{label}-P*.txt")):
            values = [float(x) for x in io.read_text(path).split() if x.strip()]
            if values:
                # the busiest sustained level: upper quartile of the samples
                out.append((int(path.stem.split("-P")[1]), float(np.percentile(values, 75)) / 1000))
        return sorted(out)

    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL, 2.6), gridspec_kw={"wspace": 0.3})
    numbers = {}
    for ax, getter, ylabel, title in ((axes[0], per_core, "useful GFLOP/s per core", "(a) Throughput per core, no communication"),
                                      (axes[1], None, "clock of the busiest cores (GHz)", "(b) Sampled core frequency")):
        if getter:
            series = [("filling socket 0 first", per_core(packed), ps.BLUE, "o"), ("spread over both sockets", per_core(spread), ps.ORANGE, "s"),
                      ("256-bit vectors, packed", per_core(wide), ps.AQUA, "^")]
        else:
            series = [("filling socket 0 first", frequency("packed"), ps.BLUE, "o"), ("spread over both sockets", frequency("spread"), ps.ORANGE, "s")]
        for label, pts, colour, marker in series:
            if pts:
                ax.plot(*zip(*pts), marker=marker, color=colour, label=label)
                numbers[f"{title[:3]}|{label}"] = {str(x): y for x, y in pts}
        ps.log2_axis(ax, "x", [1, 2, 4, 9, 18, 36, 72])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("active cores on the node"); ax.set_ylabel(ylabel); ax.set_title(title)
    axes[0].legend(loc="lower left", fontsize=ps.SMALL)
    ps.save(fig, OUT / "hardware")
    return numbers


def fig_weak():
    rows_a = io.passed(io.load_all(io.MULTI, "weak-rows.csv"))
    rows_b = io.passed(io.load_all(io.MULTI, "weak-work.csv"))
    if not rows_a and not rows_b:
        return {}
    ps.apply()
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL, 2.6), gridspec_kw={"wspace": 0.3})
    numbers = {}
    # (a) rows per rank fixed: per-rank work grows like P, so time should grow like P
    ax = axes[0]
    by = io.summarise(rows_a, ["impl", "N", "ranks"])
    for c0, style in ((1024, "-"),):
        for impl in ("local", "ring", "overlap", "rma_get"):
            pts = sorted((int(k[2]), v["median"]) for k, v in by.items() if k[0] == impl and int(k[1]) == c0 * int(k[2]))
            if pts:
                ax.plot(*zip(*pts), marker=ps.VARIANT_MARKER[impl], color=ps.VARIANT[impl], label=ps.VARIANT_SHORT[impl])
                numbers[f"rows|{impl}|{c0}"] = {str(x): y for x, y in pts}
        ref = sorted((int(k[2]), v["median"]) for k, v in by.items() if k[0] == "local" and int(k[1]) == c0 * int(k[2]))
        if ref:
            ax.plot([p[0] for p in ref], [ref[0][1] * p[0] / ref[0][0] for p in ref], color=ps.MUTED, lw=0.9)
            ax.annotate("$\\propto P$", (ref[-1][0], ref[0][1] * ref[-1][0] / ref[0][0]), xytext=(-6, -10), textcoords="offset points",
                        ha="right", fontsize=ps.SMALL, color=ps.INK2)
    ps.log2_axis(ax, "x", [1, 4, 16, 72, 288]); ps.log10_axis(ax, "y")
    ax.set_xlabel("ranks $P$   ($N = 1024\\,P$)"); ax.set_ylabel("run time (s)")
    ax.set_title("(a) Constant rows per rank")
    ax.legend(loc="upper left", fontsize=ps.TINY)
    # (b) work per rank fixed: N grows like sqrt(P); ideal time is flat
    ax = axes[1]
    by = io.summarise(rows_b, ["impl", "N", "ranks"])
    n0 = 16384
    for impl in ("local", "ring", "overlap", "rma_get"):
        pts = sorted((int(k[2]), v["median"]) for k, v in by.items() if k[0] == impl and int(k[1]) ** 2 == n0 * n0 * int(k[2]))
        if pts:
            ax.plot(*zip(*pts), marker=ps.VARIANT_MARKER[impl], color=ps.VARIANT[impl], label=ps.VARIANT_SHORT[impl])
            numbers[f"work|{impl}|{n0}"] = {str(x): y for x, y in pts}
    ps.log2_axis(ax, "x", [1, 4, 16, 64, 256])
    ps.log_ticks(ax, "y", [1, 2, 5, 10]); ax.set_ylim(0.8, 14)
    ax.set_xlabel("ranks $P$   ($N = 16{,}384\\sqrt{P}$)"); ax.set_ylabel("run time (s)")
    ax.set_title("(b) Constant work per rank")
    ps.save(fig, OUT / "weak_scaling")
    return numbers


def fig_hybrid():
    rows = io.passed(io.load_all(io.MULTI, "hybrid.csv"))
    if not rows:
        return {}
    ps.apply()
    by = io.summarise(rows, ["impl", "ranks", "threads"])
    core_counts = sorted({int(k[1]) * int(k[2]) for k in by})
    fig, axes = plt.subplots(1, len(core_counts), figsize=(ps.HALF * len(core_counts), 2.5), squeeze=False, gridspec_kw={"wspace": 0.28})
    numbers = {}
    for ax, cores, tag in zip(axes[0], core_counts, "ab"):
        for impl in ("local", "overlap"):
            pts = sorted((int(k[2]), v["median"], v["low"], v["high"]) for k, v in by.items() if k[0] == impl and int(k[1]) * int(k[2]) == cores)
            if pts:
                xs = [p[0] for p in pts]
                ax.fill_between(xs, [p[2] for p in pts], [p[3] for p in pts], color=ps.VARIANT[impl], alpha=0.15, linewidth=0)
                ax.plot(xs, [p[1] for p in pts], marker=ps.VARIANT_MARKER[impl], color=ps.VARIANT[impl], label=ps.VARIANT_SHORT[impl])
                numbers[f"{impl}|{cores}"] = {str(p[0]): p[1] for p in pts}
        ps.log2_axis(ax, "x", [1, 2, 4, 9, 18, 36, 72])
        spans = [(v["low"], v["high"]) for k, v in by.items() if int(k[1]) * int(k[2]) == cores and k[0] in ("local", "overlap")]
        low, high = min(a for a, _ in spans), max(b for _, b in spans)
        ps.log_ticks(ax, "y", ps.nice_ticks(low, high, target=5), fmt="{:.2f}"); ax.set_ylim(low / 1.02, high * 1.02)
        nodes = cores // 72
        ax.set_xlabel("OpenMP threads per rank"); ax.set_ylabel("run time (s)")
        ax.set_title(f"({tag}) {cores} cores ({nodes} node{'s' if nodes > 1 else ''}): ranks = {cores} / threads")
    axes[0][0].legend(loc="lower right", fontsize=ps.SMALL)
    ps.save(fig, OUT / "hybrid")
    return numbers


def fig_netbench():
    rows = io.load("e3-multi", "netbench.csv")
    if not rows:
        return {}
    ps.apply()
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL, 2.6), gridspec_kw={"wspace": 0.3})
    numbers = {}
    series = [("pingpong", "1", "two-sided, same node", ps.BLUE, "o"), ("pingpong", "0", "two-sided, other node", ps.ORANGE, "s"),
              ("rget", "1", "one-sided Get, same node", ps.AQUA, "^"), ("rget", "0", "one-sided Get, other node", ps.MAGENTA, "v")]
    ax = axes[0]
    for mode, same, label, colour, marker in series:
        pts = sorted((int(r["bytes"]), float(r["seconds"])) for r in rows if r["mode"] == mode and r["same_node"] == same and int(r["bytes"]) > 0)
        if pts:
            ax.plot(*zip(*pts), marker=marker, color=colour, label=label, markersize=4.5)
            numbers[f"{mode}|{same}"] = {str(x): y for x, y in pts}
    ax.set_xscale("log", base=2); ps.log10_axis(ax, "y")
    ax.set_xticks([1, 2**10, 2**20, 2**24]); ax.set_xticklabels(["1 B", "1 KiB", "1 MiB", "16 MiB"])
    ax.set_xlabel("message size"); ax.set_ylabel("time per message (s)"); ax.set_title("(a) Latency and bandwidth")
    ax.legend(loc="upper left", fontsize=ps.TINY)
    # (b) does the transfer finish while the other side computes?
    ax = axes[1]
    busy = float(rows[0]["busy_seconds"])
    series = [("busy_sender_recv", "1", "receive from a busy sender, same node", ps.BLUE, "o"), ("busy_sender_recv", "0", "receive from a busy sender, other node", ps.ORANGE, "s"),
              ("busy_target_rget", "1", "Get from a busy target, same node", ps.AQUA, "^"), ("busy_target_rget", "0", "Get from a busy target, other node", ps.MAGENTA, "v")]
    for mode, same, label, colour, marker in series:
        pts = sorted((int(r["bytes"]), float(r["seconds"])) for r in rows if r["mode"] == mode and r["same_node"] == same)
        if pts:
            ax.plot(*zip(*pts), marker=marker, color=colour, label=label, markersize=4.5)
            numbers[f"{mode}|{same}"] = {str(x): y for x, y in pts}
    ax.axhline(busy, color=ps.MUTED, lw=0.9)
    ax.annotate(f"the peer computes for {busy:g} s without calling MPI", (2**16, busy), xytext=(0, -9), textcoords="offset points", fontsize=ps.SMALL, color=ps.INK2)
    ax.set_xscale("log", base=2); ps.log10_axis(ax, "y")
    ax.set_xticks([2**16, 2**20, 2**24]); ax.set_xticklabels(["64 KiB", "1 MiB", "16 MiB"])
    ax.set_ylim(2e-6, 0.6)
    ax.set_xlabel("message size"); ax.set_ylabel("time to completion (s)"); ax.set_title("(b) Progress while the peer is busy")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86), fontsize=ps.TINY)
    ps.save(fig, OUT / "netbench")
    return numbers


def fig_causal():
    rows = io.passed(io.load_all(io.MULTI, "causal.csv"))
    if not rows:
        return {}
    ps.apply()
    by = io.summarise(rows, ["impl", "mask", "partition", "ranks"])
    fig, axes = plt.subplots(1, 3, figsize=(ps.FULL, 2.6), gridspec_kw={"wspace": 0.42})
    numbers = {}
    # (a) per-rank compute time, eight ranks
    ax = axes[0]
    for part, marker in (("contiguous", "o"), ("zigzag", "s")):
        per_rank = {}
        for path in io.rank_files(io.MULTI, f"causal-overlap-{part}-P8.rank*.csv"):
            reps = io.read(path)
            per_rank[int(reps[0]["rank"])] = io.med(reps, "compute")
        if per_rank:
            xs = sorted(per_rank)
            ax.plot(xs, [per_rank[x] for x in xs], marker=marker, color=ps.PARTITION[part], label=part)
            numbers[f"per_rank|{part}"] = [per_rank[x] for x in xs]
    ax.set_xticks(range(8))
    ps.log_ticks(ax, "y", [0.5, 1, 2, 4, 8]); ax.set_ylim(0.4, 12)
    ax.set_xlabel("rank"); ax.set_ylabel("compute time (s)"); ax.set_title("(a) Work per rank, $P=8$")
    ax.legend(loc="lower right", fontsize=ps.SMALL)
    # (b) run time against P for the four combinations, relative to the unmasked run
    ax = axes[1]
    ranks = sorted({int(k[3]) for k in by})
    combos = [("overlap", "contiguous", ps.BLUE, "s", "M3, contiguous"), ("overlap", "zigzag", ps.ORANGE, "s", "M3, zigzag"),
              ("rma_get", "contiguous", ps.BLUE, "^", "M7, contiguous"), ("rma_get", "zigzag", ps.ORANGE, "^", "M7, zigzag")]
    for impl, part, colour, marker, label in combos:
        pts = []
        for P in ranks:
            full = by.get(("overlap", "none", "contiguous", str(P)))
            v = by.get((impl, "causal", part, str(P)))
            if full and v:
                pts.append((P, v["median"] / full["median"]))
                numbers[f"ratio|{impl}|{part}|{P}"] = v["median"] / full["median"]
                numbers[f"time|{impl}|{part}|{P}"] = v["median"]
        if pts:
            ax.plot(*zip(*pts), marker=marker, color=colour, label=label, linestyle="-" if impl == "overlap" else ":")
    ax.axhline(0.5, color=ps.MUTED, lw=0.9); ax.axhline(1.0, color=ps.MUTED, lw=0.9)
    ax.annotate("half the pairs", (ranks[0], 0.5), xytext=(0, -9), textcoords="offset points", fontsize=ps.SMALL, color=ps.INK2)
    ax.annotate("no mask", (ranks[0], 1.0), xytext=(0, 3), textcoords="offset points", fontsize=ps.SMALL, color=ps.INK2)
    ps.log2_axis(ax, "x", ranks)
    from matplotlib.ticker import FuncFormatter
    ax.set_yscale("log"); ax.set_ylim(0.3, 10); ax.set_yticks([0.5, 1, 2, 4, 8]); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("ranks $P$"); ax.set_ylabel("time relative to no mask"); ax.set_title("(b) Causal run time")
    ax.legend(loc="upper left", fontsize=ps.TINY, ncol=1, bbox_to_anchor=(0.0, 0.93))
    # (c) bytes received by the one-sided pull
    ax = axes[2]
    for part in ("contiguous", "zigzag"):
        pts = []
        for P in ranks:
            v = by.get(("rma_get", "causal", part, str(P)))
            ring = by.get(("overlap", "causal", part, str(P)))
            if v and ring:
                total_get = 0; total_ring = 0
                for path in io.rank_files(io.MULTI, f"causal-rma_get-{part}-P{P}.rank*.csv"):
                    total_get += float(io.read(path)[0]["bytes_recv"])
                for path in io.rank_files(io.MULTI, f"causal-overlap-{part}-P{P}.rank*.csv"):
                    total_ring += float(io.read(path)[0]["bytes_recv"])
                if total_ring:
                    pts.append((P, total_get / total_ring))
                    numbers[f"bytes_ratio|{part}|{P}"] = total_get / total_ring
        if pts:
            ax.plot(*zip(*pts), marker="^", color=ps.PARTITION[part], label=part)
    ps.log2_axis(ax, "x", ranks); ax.set_ylim(0, 1.15)
    ax.set_xlabel("ranks $P$"); ax.set_ylabel("bytes pulled / bytes of the ring"); ax.set_title("(c) Traffic of the one-sided pull")
    ax.legend(loc="lower right", fontsize=ps.SMALL)
    ps.save(fig, OUT / "causal")
    return numbers


# ---------------------------------------------------------------------------
def fig_gpu_ring():
    """Two GPUs: efficiency against rows per GPU for host-staged / CUDA-aware
    exchange, overlapped or blocking, with both GPUs in one node or one per node."""
    ps.apply()
    panels = [("gpu-ring-1node", "ring-one-node.csv", "(a) Two GPUs in one node"),
              ("gpu-ring-2node", "ring-two-nodes.csv", "(b) One GPU in each of two nodes")]
    available = [(prefix, name, title) for prefix, name, title in panels if io.load(prefix, name)]
    if not available:
        return {}
    fig, axes = plt.subplots(1, len(available), figsize=(ps.HALF * len(available), 2.6), squeeze=False, gridspec_kw={"wspace": 0.28})
    numbers = {}
    styles = [("aware", "overlap", ps.ORANGE, "s", "-", "CUDA-aware, overlapped"), ("aware", "blocking", ps.ORANGE, "o", ":", "CUDA-aware, blocking"),
              ("host", "overlap", ps.BLUE, "s", "-", "host-staged, overlapped"), ("host", "blocking", ps.BLUE, "o", ":", "host-staged, blocking")]
    for ax, (prefix, name, title) in zip(axes[0], available):
        rows = io.passed(io.load(prefix, name))
        by = io.summarise(rows, ["ranks", "comm", "schedule", "N"])
        single = {int(k[3]): v["median"] for k, v in by.items() if k[0] == "1"}
        for comm, schedule, colour, marker, line, label in styles:
            pts = sorted((int(k[3]) // 2, single[int(k[3])] / (2 * v["median"]), single[int(k[3])] / (2 * v["high"]),
                          single[int(k[3])] / (2 * v["low"])) for k, v in by.items()
                         if k[0] == "2" and k[1] == comm and k[2] == schedule and int(k[3]) in single)
            if pts:
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                ax.errorbar(xs, ys, yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]], marker=marker, color=colour,
                            linestyle=line, label=label, elinewidth=0.8, capsize=1.5)
                numbers[f"{name}|{comm}|{schedule}"] = {str(x): y for x, y in zip(xs, ys)}
        for (ranks, comm, schedule, n), v in by.items():
            reps = v["rows"]
            numbers[f"detail|{name}|{ranks}|{comm}|{schedule}|{n}"] = {
                "time": v["median"], "kernel": io.med(reps, "t_kernel_max"), "mpi": io.med(reps, "t_mpi_max"),
                "gpu_wait": io.med(reps, "t_gpu_wait_max"), "staging": io.med(reps, "t_staging_max")}
        ax.axhline(1.0, color=ps.AXIS, lw=0.8)
        ps.log2_axis(ax, "x", [2048, 8192, 32768, 65536])
        ax.set_ylim(0, 1.15)
        ax.set_xlabel("rows per GPU $c=N/2$"); ax.set_ylabel("efficiency $T_1/(2\\,T_2)$"); ax.set_title(title)
    axes[0][0].legend(loc="lower right", fontsize=ps.TINY)
    ps.save(fig, OUT / "gpu_ring")
    return numbers


def fig_small_c():
    """Run time of every variant relative to the control when ranks own few rows."""
    run = io.latest("e3-multi", "comm-small-c.csv")
    if not run:
        return {}
    ps.apply()
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL, 2.7), gridspec_kw={"wspace": 0.3})
    numbers = {}
    for ax, name, title, threads in ((axes[0], "comm-small-c.csv", "(a) 288 single-thread ranks", 1),
                                     (axes[1], "comm-fat.csv", "(b) 16 ranks $\\times$ 18 threads", 18)):
        rows = io.passed(io.read(run / name))
        by = io.summarise(rows, ["impl", "N", "ranks"])
        for impl in ("allgather", "iallgather", "ring", "overlap", "rma_get", "rma_put"):
            pts = []
            for (i, n, ranks), v in by.items():
                control = by.get(("local", n, ranks))
                if i == impl and control:
                    pts.append((int(n) / int(ranks), v["median"] / control["median"]))
                    numbers[f"{name}|{impl}|{n}"] = {"time": v["median"], "control": control["median"], "c": int(n) / int(ranks),
                                                      "wait_mean": io.med(v["rows"], "t_wait_mean")}
            if pts:
                pts.sort()
                ax.plot(*zip(*pts), marker=ps.VARIANT_MARKER[impl], color=ps.VARIANT[impl], label=ps.VARIANT_SHORT[impl])
        if threads == 18:
            # The repeat of the wide-rank configurations (suite fatcheck): one open marker per round.
            again = io.latest("e5-fatcheck", "fatcheck-r1.csv")
            shown = False
            for rnd in (1, 2):
                rep = io.summarise(io.passed(io.read(again / f"fatcheck-r{rnd}.csv")), ["impl", "N", "ranks"]) if again else {}
                for (i, n, ranks), v in rep.items():
                    control = rep.get(("local", n, ranks))
                    if i in ("ring", "overlap") and control:
                        ax.plot([int(n) / int(ranks)], [v["median"] / control["median"]], linestyle="none", marker=ps.VARIANT_MARKER[i],
                                markerfacecolor=ps.SURFACE, markeredgecolor=ps.VARIANT[i], markeredgewidth=1.1, markersize=6.5, zorder=4,
                                label=None if shown else "repeat, two rounds (open)")
                        shown = True
                        numbers[f"fatcheck|{i}|{n}|r{rnd}"] = {"time": v["median"], "control": control["median"], "c": int(n) / int(ranks),
                                                              "low": v["low"], "high": v["high"]}
                poll = io.summarise(io.passed(io.read(again / f"fatcheck-poll-r{rnd}.csv")), ["impl", "N", "ranks"]) if again else {}
                for (i, n, ranks), v in poll.items():
                    control = rep.get(("local", n, ranks))
                    if control:
                        numbers[f"fatcheck-poll|{i}|{n}|r{rnd}"] = {"time": v["median"], "control": control["median"], "c": int(n) / int(ranks)}
        ax.axhline(1.0, color=ps.AXIS, lw=0.8)
        ax.set_xscale("log", base=2)
        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: ps.human(v)))
        ax.set_yscale("log")
        top = max((float(np.max(np.asarray(line.get_ydata(), float))) for line in ax.get_lines()
                   if len(np.atleast_1d(line.get_ydata())) > 2), default=10)
        ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200, 500) if t <= top * 1.3]
        ax.set_yticks(ticks); ax.set_yticklabels([f"{t:g}" for t in ticks]); ax.minorticks_off()
        ax.set_ylim(0.8, top * 1.3)
        ax.set_xlabel("rows per rank $c$"); ax.set_ylabel("time / time of the control M0"); ax.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    extra = [(h, l) for h, l in zip(*axes[1].get_legend_handles_labels()) if l not in labels]
    handles += [h for h, _ in extra]; labels += [l for _, l in extra]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=ps.SMALL, handlelength=1.2, columnspacing=1.2)
    ps.save(fig, OUT / "small_c")
    # polling variants
    for name in ("comm-poll.csv", "comm-fat-poll.csv", "comm-order.csv", "comm-packed.csv", "comm-spread.csv"):
        rows = io.passed(io.read(run / name))
        for key, v in io.summarise(rows, ["impl", "N", "ranks", "poll_tiles", "fetch_order"]).items():
            numbers[f"{name}|" + "|".join(key)] = {"time": v["median"], "wait_mean": io.med(v["rows"], "t_wait_mean"),
                                                 "wait_max": io.med(v["rows"], "t_wait_max")}
    return numbers


def fig_gqa():
    rows = io.passed(io.load_all(io.MULTI, "gqa.csv"))
    real = io.passed(io.load("real-data", "e10-real.csv"))
    if not rows and not real:
        return {}
    numbers = {}
    for label, data in (("synthetic", rows), ("real", real)):
        for key, v in io.summarise(data, ["impl", "N", "ranks", "mha_control"]).items():
            numbers[f"{label}|" + "|".join(key)] = {"time": v["median"], "bytes": float(v["row"]["bytes_recv_max"]),
                                                   "wait_mean": io.med(v["rows"], "t_wait_mean"), "compute_max": io.med(v["rows"], "t_compute_max")}
    return numbers


def fig_memory():
    """Peak resident memory per rank (kernel high-water mark). The one-sided
    variant keeps the shards in a shared-memory window that is mapped by every
    rank of the node, so its resident size counts the window once per rank; its
    proportional set size (shared pages divided by the number of sharers) is
    shown as well."""
    rows = io.load("e7-memory", "memory.csv")
    if not rows:
        return {}
    ps.apply()
    by = io.summarise(rows, ["impl", "N"], value="peak_rss_mb_max")
    ranks = int(rows[0]["ranks"])
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    numbers = {}
    for impl in ("allgather", "rma_get", "ring", "overlap"):
        pts = sorted((int(k[1]), v["median"]) for k, v in by.items() if k[0] == impl)
        if pts:
            ax.plot(*zip(*pts), marker=ps.VARIANT_MARKER[impl], color=ps.VARIANT[impl], label=ps.VARIANT_SHORT[impl])
            numbers[impl] = {str(x): y for x, y in pts}
    shared = sorted((int(k[1]), float(v["row"]["pss_mb_total"]) / ranks) for k, v in by.items() if k[0] == "rma_get" and v["row"].get("pss_mb_total"))
    if shared:
        ax.plot(*zip(*shared), marker="^", color=ps.VARIANT["rma_get"], linestyle=":", markerfacecolor=ps.SURFACE,
                markeredgecolor=ps.VARIANT["rma_get"], label="M7, shared window counted once")
        numbers["rma_get_proportional"] = {str(x): y for x, y in shared}
    for impl in ("allgather", "ring", "overlap", "rma_get"):
        numbers[impl + "_pss_total"] = {k[1]: float(v["row"]["pss_mb_total"]) for k, v in by.items() if k[0] == impl and v["row"].get("pss_mb_total")}
    ps.log2_axis(ax, "x", sorted({int(k[1]) for k in by})); ps.log10_axis(ax, "y")
    ax.set_xlabel(f"sequence length $N$  ({ranks} ranks on one node)"); ax.set_ylabel("memory per rank (MiB)")
    ax.set_title("Peak resident memory of the largest rank")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=ps.TINY)
    ps.save(fig, OUT / "memory")
    return numbers


def fig_traces():
    """Per ring step of one rank: time blocked in MPI and time computing, three variants.
    Each value is the median over the three measured repetitions."""
    run = io.latest("e3-multi", "comm-trace.csv")
    if not run:
        return {}
    ps.apply()
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL, 2.5), gridspec_kw={"wspace": 0.3})
    numbers = {}
    for impl in ("ring", "overlap", "rma_get"):
        rows = io.read(run / "traces" / f"trace-{impl}.rank3.csv")
        if not rows:
            continue
        reps = sorted({r["rep"] for r in rows})
        steps = sorted({int(r["step"]) for r in rows})
        compute = {s: [] for s in steps}; wait = {s: [] for s in steps}
        totals = []
        for rep in reps:
            c_total = w_total = 0.0
            for r in rows:
                if r["rep"] != rep:
                    continue
                c = float(r["compute_end"]) - float(r["compute_begin"])
                w = max(0.0, float(r["wait_end"]) - float(r["wait_begin"]))
                compute[int(r["step"])].append(c); wait[int(r["step"])].append(w)
                c_total += c; w_total += w
            totals.append((c_total, w_total))
        med = lambda values: float(np.median(values))
        waits = [(s + 1, 1e3 * med(wait[s])) for s in steps if med(wait[s]) > 0]
        axes[0].plot(*zip(*waits), marker=ps.VARIANT_MARKER[impl], color=ps.VARIANT[impl], label=ps.VARIANT_SHORT[impl])
        axes[1].plot([s + 1 for s in steps], [1e3 * med(compute[s]) for s in steps], marker=ps.VARIANT_MARKER[impl],
                     color=ps.VARIANT[impl], label=ps.VARIANT_SHORT[impl])
        numbers[impl] = {"compute": med([t[0] for t in totals]), "wait": med([t[1] for t in totals]), "steps": len(steps),
                         "repetitions": len(reps), "wait_per_step_ms": {str(k): v for k, v in waits}}
    ps.log_ticks(axes[0], "y", [0.2, 0.5, 1, 2, 5, 10, 20]); ps.log_ticks(axes[1], "y", [5, 10, 20])
    axes[1].set_ylim(4, 25)
    for ax, title in zip(axes, ("(a) Blocked in MPI", "(b) Computing")):
        ax.set_xticks(range(1, 9)); ax.set_xlabel("ring step"); ax.set_ylabel("time in the step (ms)"); ax.set_title(title)
    axes[0].legend(loc="upper left", fontsize=ps.TINY)
    ps.save(fig, OUT / "traces")
    return numbers


FIGURES = {"serial": fig_serial, "tiles": fig_tiles, "gpu_ladder": fig_gpu_ladder, "gpu_ring": fig_gpu_ring,
           "small_c": fig_small_c, "gqa": fig_gqa, "memory": fig_memory, "traces": fig_traces,
           "strong": fig_strong, "collapse": fig_collapse, "hardware": fig_hardware, "weak": fig_weak,
           "hybrid": fig_hybrid, "netbench": fig_netbench, "causal": fig_causal,
           "real_stats": fig_real_stats, "real_heads": fig_real_heads}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True); GEN.mkdir(parents=True, exist_ok=True)
    strict = "--strict" in sys.argv
    wanted = [a for a in sys.argv[1:] if not a.startswith("--")] or list(FIGURES)
    store = GEN / "figure_numbers.json"
    numbers = json.loads(store.read_text()) if store.exists() else {}
    empty = []
    for name in wanted:
        io.trace_reset()
        result = FIGURES[name]()
        if result:
            numbers[name] = result
            io.record_provenance("figures", name)
            print(f"{name}: ok")
        else:
            numbers.pop(name, None)          # no numbers of an earlier build survive a figure without data
            empty.append(name)
            print(f"{name}: no data yet")
    store.write_text(json.dumps(numbers, indent=1, sort_keys=True) + "\n")
    if strict and empty:
        raise SystemExit(f"figures without data: {empty}")
    for name, width in ps.check_widths(OUT):
        print(f"WARNING: {name} is {width} in wide, more than the text width")
