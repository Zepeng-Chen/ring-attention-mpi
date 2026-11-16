#!/usr/bin/env python3
"""Every number quoted in the text of the report, computed from the result CSVs.

Writes report/generated/numbers.tex, which defines \\n{key} for each key below.
The prose never contains a measured value typed by hand, so rerunning an
experiment and this script updates the text consistently with the figures.
"""
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results_io as io

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "report" / "generated"
N = {}          # key -> already formatted LaTeX string
MISSING = []


def put(key, value, fmt="{:.2f}"):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        MISSING.append(key)
        return
    N[key] = fmt.format(value) if not isinstance(value, str) else value


def thousands(x):
    return f"{int(round(x)):,}".replace(",", "{,}")


def sig(x, digits=2):
    """Round to significant digits, plain notation."""
    if x == 0:
        return "0"
    d = digits - 1 - int(math.floor(math.log10(abs(x))))
    r = round(x, d)
    return f"{r:.{max(d, 0)}f}"


def hosts_by_rank(files):
    """Host of every rank, in rank order, from per-rank files (None unless ranks 0..P-1 are all there)."""
    host = {}
    for path in files:
        rows = io.read(path)
        if rows:
            host[int(rows[0]["rank"])] = rows[0]["host"]
    return [host[r] for r in range(len(host))] if host and sorted(host) == list(range(len(host))) else None


def ring_links_between_nodes(hosts):
    """Number of links rank r -> rank r+1 (cyclic) whose two ends are on different hosts."""
    return sum(hosts[r] != hosts[(r + 1) % len(hosts)] for r in range(len(hosts)))


def zigzag_pairs(n, ranks):
    """Query-key pairs that the causal mask allows, per rank, for the zigzag partition of attn::positions:
    2P chunks, the first n mod 2P of them one row longer; rank r owns chunks r and 2P-1-r."""
    chunks = 2 * ranks
    base, extra = divmod(n, chunks)
    def pairs(chunk):                      # rows start .. start+count-1; row i meets i+1 keys
        start, count = base * chunk + min(chunk, extra), base + (chunk < extra)
        return count * start + count * (count + 1) // 2
    return [pairs(r) + pairs(chunks - 1 - r) for r in range(ranks)]


def gflops(v):
    return float(v["row"]["useful_flops"]) / v["median"] / 1e9


def fit_slope(xs, ys):
    return float(np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float)), 1)[0])


# ---------------------------------------------------------------------------
def serial():
    rows = io.passed(io.load("e1-serial", "serial-n.csv"))
    if not rows:
        return
    by = io.summarise(rows, ["impl", "kernel", "N"])

    def series(impl, kernel):
        return sorted((int(k[2]), v) for k, v in by.items() if k[0] == impl and k[1] == kernel)

    for name, impl, kernel in (("tuned", "S1", "tuned"), ("basic", "S1", "basic"), ("naive", "S0-fp32", "reference")):
        pts = [(n, v["median"]) for n, v in series(impl, kernel) if n >= 8192]
        if len(pts) >= 2:
            put(f"serial.slope.{name}", fit_slope(*zip(*pts)))
        at = dict(series(impl, kernel)).get(16384)
        if at:
            put(f"serial.gflops.{name}", gflops(at), "{:.1f}")
            put(f"serial.t16k.{name}", at["median"])
    tuned = series("S1", "tuned")
    if tuned:
        nmax, vmax = tuned[-1]
        put("serial.Nmax.h", thousands(nmax))
        put("serial.tuned.tmax", vmax["median"], "{:.0f}")
        g = dict(tuned)
        if 16384 in g:
            b = dict(series("S1", "basic")).get(16384)
            if b:
                put("serial.speedup.tuned", b["median"] / g[16384]["median"], "{:.1f}")
    f64 = dict(series("S0", "reference")).get(16384)
    f32 = dict(series("S0-fp32", "reference")).get(16384)
    if f64 and f32:
        put("serial.fp64.ratio", f64["median"] / f32["median"], "{:.1f}")
    # head dimension
    rows_d = io.passed(io.load("e1-serial", "serial-d.csv"))
    by_d = io.summarise(rows_d, ["impl", "kernel", "d", "N"])
    for name, impl, kernel in (("tuned", "S1", "tuned"), ("basic", "S1", "basic"), ("naive", "S0-fp32", "reference")):
        pts = sorted((int(k[2]), v["median"] / float(k[3]) ** 2) for k, v in by_d.items() if k[0] == impl and k[1] == kernel)
        if len(pts) >= 2:
            put(f"serial.dslope.{name}", fit_slope(*zip(*pts)))
            put(f"serial.dratio.{name}", pts[-1][1] / pts[0][1], "{:.1f}")
    # tiles and vector width
    tiles = io.summarise(io.passed(io.load("e1-serial", "serial-tiles.csv")), ["block_q", "block_k"])
    if tiles:
        grid = {(int(k[0]), int(k[1])): gflops(v) for k, v in tiles.items()}
        (bq, bk), best = max(grid.items(), key=lambda kv: kv[1])
        put("tiles.best.bq", str(bq)); put("tiles.best.bk", str(bk)); put("tiles.best.gflops", best, "{:.1f}")
        put("tiles.best.kib", bk * 64 * 4 / 1024, "{:.0f}")
        nxt = grid.get((bq, 2 * bk))
        if nxt:
            put("tiles.cliff.pct", 100 * (1 - nxt / best), "{:.0f}")
        put("tiles.worst.gflops", min(grid.values()), "{:.1f}")
        # the tile sizes actually used by every other run (those of the tuned serial-n rows)
        used = next(((int(r["block_q"]), int(r["block_k"])) for r in rows if r["impl"] == "S1" and r["kernel"] == "tuned"), None)
        if used and used in grid:
            put("tiles.used.bq", str(used[0])); put("tiles.used.bk", str(used[1])); put("tiles.used.gflops", grid[used], "{:.1f}")
            put("tiles.used.kib", used[1] * 64 * 4 / 1024, "{:.0f}")
            put("tiles.used.below.pct", 100 * (1 - grid[used] / best), "{:.0f}")
            put("serial.gflops.w512", grid[used], "{:.1f}")
            twice = grid.get((used[0], 2 * used[1]))
            if twice:
                put("tiles.cliff.pct", 100 * (1 - twice / grid[used]), "{:.0f}")
            small = grid.get((8, used[1]))
            if small:
                put("tiles.bq8.gflops", small, "{:.0f}")
        w256 = io.summarise(io.passed(io.load("e1-serial", "serial-w256.csv")), ["block_k"])
        if used and w256 and (str(used[1]),) in w256:
            put("serial.gflops.w256", gflops(w256[(str(used[1]),)]), "{:.1f}")
            put("serial.w512.gain.pct", 100 * (grid[used] / gflops(w256[(str(used[1]),)]) - 1), "{:.0f}")
    ladder = io.summarise(io.passed(io.load("e1-serial", "serial-ladder-w256.csv")), ["impl", "kernel"])
    basic_rates = [gflops(v) for v in (dict(series("S1", "basic")).get(16384), ladder.get(("S1", "basic"))) if v]
    tuned16 = dict(series("S1", "tuned")).get(16384)
    if basic_rates and tuned16:
        put("serial.speedup.fair", gflops(tuned16) / max(basic_rates), "{:.1f}")
    for name, key in (("tuned", ("S1", "tuned")), ("basic", ("S1", "basic")), ("naive", ("S0-fp32", "reference"))):
        if key in ladder:
            put(f"serial.gflops.{name}.w256", gflops(ladder[key]), "{:.1f}")


def hardware():
    run = io.latest("e7-hardware", "hardware.csv")
    if not run:
        return
    packed = io.summarise(io.passed(io.read(run / "hardware.csv")), ["ranks"])

    def freq(label, P):
        path = run / f"freq-{label}-P{P}.txt"
        if not path.exists():
            return None
        values = [float(x) for x in io.read_text(path).split()]
        return float(np.percentile(values, 75)) / 1000 if values else None

    for P in (1, 9, 18, 36, 72):
        v = packed.get((str(P),))
        if v:
            put(f"hw.gflops.{P}", gflops(v) / P, "{:.1f}")
            f = freq("packed", P)
            if f:
                put(f"hw.freq.{P}", f, "{:.1f}")
    one, full = packed.get(("1",)), packed.get(("72",))
    if one and full:
        put("hw.eff.72", gflops(full) / 72 / gflops(one), "{:.2f}")
        f1, f72 = freq("packed", 1), freq("packed", 72)
        if f1 and f72:
            put("hw.freq.one", f1, "{:.1f}")
            put("hw.freq.ratio", f72 / f1, "{:.2f}")
            put("serial.peak", 2 * 16 * 2 * f1, "{:.0f}")
            put("serial.peakpct", 100 * gflops(one) / (2 * 16 * 2 * f1), "{:.0f}")
    spread = io.summarise(io.passed(io.read(run / "hardware-spread.csv")), ["ranks"])
    if spread.get(("36",)) and packed.get(("36",)):
        put("hw.spread.gain", gflops(spread[("36",)]) / gflops(packed[("36",)]), "{:.2f}")
        put("hw.spread.pct", 100 * (gflops(spread[("36",)]) / gflops(packed[("36",)]) - 1), "{:.0f}")
        f = freq("spread", 36)
        if f:
            put("hw.freq.spread36", f, "{:.1f}")
    alt = io.summarise(io.passed(io.read(run / "hardware-w256.csv")), ["ranks"])
    if alt.get(("1",)) and alt.get(("72",)):
        put("hw.w256.eff.72", gflops(alt[("72",)]) / 72 / gflops(alt[("1",)]))
        put("hw.w256.gflops.1", gflops(alt[("1",)]), "{:.1f}"); put("hw.w256.gflops.72", gflops(alt[("72",)]) / 72, "{:.1f}")
    for P in (1, 72):
        if alt.get((str(P),)) and packed.get((str(P),)):
            put(f"hw.w512.gain.{P}", 100 * (gflops(packed[(str(P),)]) / gflops(alt[(str(P),)]) - 1), "{:.0f}")


def ramp():
    rows = io.load("e1-extras", "ramp.csv")
    if not rows:
        return
    t = [float(r["t_total_max"]) for r in rows]
    slow, fast = statistics.median(t[:20]), statistics.median(t[-100:])
    put("ramp.ms.slow", slow * 1e3, "{:.0f}"); put("ramp.ms.fast", fast * 1e3, "{:.0f}")
    put("ramp.ratio", slow / fast, "{:.2f}")
    elapsed, step = 0.0, None
    for x in t:
        elapsed += x
        if x < (slow + fast) / 2:
            step = elapsed
            break
    if step:
        put("ramp.seconds", step, "{:.1f}")
    base = io.summarise(io.passed(io.load("e1-extras", "baseline.csv")), ["impl"])
    for impl in ("S1", "local", "ring"):
        if (impl,) in base:
            put(f"baseline.{impl}", base[(impl,)]["median"])
    times = [base[k]["median"] for k in (("S1",), ("local",), ("ring",)) if k in base]
    if len(times) >= 2:
        put("baseline.gap.pct", 100 * (max(times) / min(times) - 1), "{:.1f}")
    for impl in ("S1", "local", "ring"):
        if (impl,) in base:
            N[f"baseline.{impl}"] = f"{base[(impl,)]['median']:.3f}"


def seconds(t):
    if t >= 100:
        return f"{t:.0f}"
    if t >= 10:
        return f"{t:.1f}"
    if t >= 1:
        return f"{t:.2f}"
    return f"{t:.3f}"


def align():
    import make_tables as mt
    data = mt.align_data()
    if not data:
        return
    for a, v in data.items():
        put(f"align.T.{a}", v["time"]); put(f"align.gflops.{a}", v["gflops"], "{:.1f}")
        c = v["counters"]
        if c.get("cycles"):
            put(f"align.ipc.{a}", c["instructions"] / c["cycles"])
        if "ld_blocks.no_sr" in c and v["pairs"]:
            put(f"align.split.{a}", sig(1e3 * c["ld_blocks.no_sr"] / v["pairs"]))
            put(f"align.splitcount.{a}", mt.sci(c["ld_blocks.no_sr"]))
        if v["eff"]:
            put(f"align.Emin.{a}", min(v["eff"].values())); put(f"align.Emax.{a}", max(v["eff"].values()))
            worst = max(v["eff"].items(), key=lambda kv: kv[1])
            put(f"align.Emax.P.{a}", str(worst[0][1]))
    if 16 in data and 4096 in data:
        put("align.speedup", data[16]["time"] / data[4096]["time"])
        put("align.gain.pct", 100 * (data[16]["time"] / data[4096]["time"] - 1), "{:.0f}")
    if 64 in data and 4096 in data:
        put("align.page.vs.line.pct", 100 * abs(data[64]["time"] / data[4096]["time"] - 1), "{:.1f}")


def strong():
    import make_figures as mf
    data = mf._strong_tables()
    if not data:
        return
    serial_t, table = data
    sizes = sorted(serial_t)
    big, small = sizes[-1], sizes[0]
    put("strong.Nbig.h", thousands(big)); put("strong.Nsmall.h", thousands(small))
    put("strong.TS.big", seconds(serial_t[big]))
    cores = sorted({k[2] for k in table})
    put("strong.maxcores", str(cores[-1]))
    for P in (72, 288):
        v, m0 = table.get(("overlap", big, P)), table.get(("local", big, P))
        if v:
            put(f"strong.S.big.{P}", v["speedup"], "{:.0f}" if v["speedup"] >= 100 else "{:.1f}")
            put(f"strong.E.big.{P}", v["efficiency"]); put(f"strong.T.big.{P}", seconds(v["time"]))
        if v and m0:
            put(f"strong.Ehw.big.{P}", m0["serial"] / (P * m0["time"])); put(f"strong.Ecomm.big.{P}", m0["time"] / v["time"])
        for impl in ("allgather", "ring", "rma_get"):
            w = table.get((impl, big, P))
            if w:
                put(f"strong.E.{impl}.big.{P}", w["efficiency"])
    for n, tag in ((small, "small"), (sizes[len(sizes) // 2], "mid"), (big, "big")):
        for P in (72, 144, 288):
            v = table.get(("overlap", n, P))
            if v:
                put(f"strong.E.{tag}.{P}", v["efficiency"])
                put(f"strong.c.{tag}.{P}", thousands(n // P))
            m0 = table.get(("local", n, P))
            if m0:
                put(f"strong.E0.{tag}.{P}", m0["efficiency"])
            if v and m0:
                put(f"strong.Ecomm.{tag}.{P}", m0["time"] / v["time"])
            for impl in ("ring", "allgather", "rma_get"):
                w = table.get((impl, n, P))
                if w and m0:
                    put(f"strong.Ecomm.{impl}.{tag}.{P}", m0["time"] / w["time"])
                    put(f"strong.slow.{impl}.{tag}.{P}", w["time"] / m0["time"], "{:.0f}" if w["time"] / m0["time"] >= 10 else "{:.1f}")
    # efficiency at small P on one socket (same clock as the serial run)
    low = [v["efficiency"] for (impl, n, P), v in table.items() if P <= 8]
    if low:
        put("strong.Elow.min", min(low)); put("strong.Elow.max", max(low))
        put("strong.Elow.loss.pct", 100 * (1 - min(low)), "{:.0f}")
    # time inside the compute kernel: the ring's ranks against the control's (mean over ranks)
    ratios = []
    for (impl, n, P), v in table.items():
        m0 = table.get(("local", n, P))
        if impl in ("ring", "overlap") and m0 and P >= 2:
            ratios.append(v["compute_mean"] / m0["compute_mean"])
            if impl == "overlap" and n == big and P in (72, 288):
                put(f"strong.kernel.ratio.big.{P}", v["compute_mean"] / m0["compute_mean"], "{:.3f}")
    if ratios:
        put("strong.kernel.ratio.min", min(ratios), "{:.2f}"); put("strong.kernel.ratio.max", max(ratios), "{:.2f}")
        put("strong.kernel.ratio.n", str(len(ratios)))
    # the validation error against fp64 for the same problem under different rank counts and variants
    spread = []
    by_n = {}
    for r in io.strong_rows():
        if r["impl"] != "S1":
            by_n.setdefault(r["N"], []).append(float(r["max_abs_err"]))
    for values in by_n.values():
        spread.append(max(values) - min(values))
    if spread:
        import make_tables as mt
        N["val.rankspread"] = mt.sci(max(spread))
    # node-to-node spread: per-host median compute time of the control at the largest P
    per_host = {}
    for path in io.rank_files(["e2-strongx", "e2-strong"], f"strong-local-N{big}-P{cores[-1]}.rank*.csv"):
        reps = io.read(path)
        if reps:
            per_host.setdefault(reps[0]["host"], []).append(io.med(reps, "compute"))
    if len(per_host) > 1:
        host_time = {h: statistics.median(v) for h, v in per_host.items()}
        fast, slow = min(host_time.values()), max(host_time.values())
        put("nodes.fast", fast, "{:.4f}"); put("nodes.slow", slow, "{:.4f}"); put("nodes.spread.pct", 100 * (slow / fast - 1), "{:.1f}")
        put("nodes.count", str(len(per_host)))
        every = [x for v in per_host.values() for x in v]
        put("nodes.rank.spread.pct", 100 * (max(every) / min(every) - 1), "{:.1f}")
    # where the neighbours of a rank are: ranks are numbered node by node
    hosts = hosts_by_rank(io.rank_files(["e2-strongx", "e2-strong"], f"strong-local-N{big}-P{cores[-1]}.rank*.csv"))
    if hosts:
        P = len(hosts)
        put(f"topo.ring.cross.{P}", str(ring_links_between_nodes(hosts)))
        other = sum(a != b for a in hosts for b in hosts)         # ordered pairs of ranks on different nodes
        put(f"topo.pull.cross.pct.{P}", 100 * other / (P * (P - 1)), "{:.0f}")
    # what the timed phase leaves out: the set-up, which for the one-sided pull creates the MPI window
    setup = {}
    for r in io.strong_rows():
        if r["impl"] != "S1":
            setup.setdefault((r["impl"], int(r["ranks"])), []).append(float(r["t_setup_max"]))
    for P in (8, 72, 288):
        s = setup.get(("rma_get", P))
        if s:
            put(f"setup.rma_get.P{P}.min", min(s)); put(f"setup.rma_get.P{P}.max", max(s))
    rings = [x for (impl, _), v in setup.items() if impl in ("ring", "overlap") for x in v]
    if rings:
        put("setup.ring.max.ms", 1e3 * max(rings), "{:.1f}")
    for n, tag, P in ((small, "small", 72), (big, "big", 288)):
        pull = table.get(("rma_get", n, P))
        s = [float(r["t_setup_max"]) for r in io.strong_rows() if r["impl"] == "rma_get" and int(r["N"]) == n and int(r["ranks"]) == P]
        if pull and s:
            put(f"setup.rma_get.timed.{tag}.{P}", seconds(pull["time"]))
            put(f"setup.rma_get.{tag}.{P}", max(s))
            put(f"setup.rma_get.over.timed.{tag}.{P}", max(s) / pull["time"], "{:.0f}" if max(s) / pull["time"] >= 10 else "{:.1f}")

    both = [(io.serial_time(n, 1), io.serial_time(n, 10 ** 6)) for n in sizes]
    if all(a and b for a, b in both) and any(a != b for a, b in both):
        put("strong.serial.dev.pct", 100 * max(abs(b / a - 1) for a, b in both), "{:.1f}")
    anchor = io.strong_anchor()
    if anchor:
        put("strong.anchor.maxdev.pct", 100 * max(abs(b / a - 1) for a, b in anchor.values()), "{:.0f}")
        put("strong.anchor.meddev.pct", 100 * statistics.median(abs(b / a - 1) for a, b in anchor.values()), "{:.1f}")


def model():
    import make_figures as mf
    params = mf.model_parameters()
    if not params:
        return
    put("model.alpha.us", params["alpha"] * 1e6, "{:.1f}"); put("model.B.gbs", params["B"] / 1e9, "{:.0f}")
    put("model.F.gflops", params["F"] / 1e9, "{:.0f}")
    predicted = 4 * params["F"] / (2 * params["B"])
    put("model.cstar", predicted, "{:.0f}")
    put("model.calpha", math.sqrt(params["alpha"] * params["F"] / (2 * 128)), "{:.0f}")
    fit = mf.fitted_cstar()
    if fit:
        put("model.cstar.fit", fit[0], "{:.0f}"); put("model.fit.maxdev", fit[1])
        put("model.cstar.ratio", fit[0] / predicted, "{:.1f}")
        put("model.c95.fit", thousands(round(19 * fit[0], -1)))          # E = 0.95  <=>  c = 19 c*
        put("model.cmin.fit", "16"); put("model.cmax.fit", thousands(max(p[0] for p in fit[2])))
        put("model.fit.n", str(len(fit[2])))
        # Runs that took no part in the fit: single-thread ranks of the strong- and weak-scaling runs.
        heldout = {"one": [], "cross.overlap": [], "cross.ring": [], "cross.allgather": []}
        for prefix, name in (("strong", None), ("e3-single", "weak-rows.csv"), ("e3-multi", "weak-rows.csv"),
                             ("e3-single", "weak-work.csv"), ("e3-multi", "weak-work.csv"), ("e3-multi", "comm-small-c.csv")):
            rows = io.strong_rows() if name is None else io.passed(io.load(prefix, name))
            by = io.summarise(rows, ["impl", "N", "ranks"])
            for (impl, n, ranks), v in by.items():
                control = by.get(("local", n, ranks))
                if impl not in ("ring", "overlap", "allgather") or not control or int(ranks) < 2:
                    continue
                c = int(n) / int(ranks)
                if c < 16:
                    continue
                deviation = control["median"] / v["median"] - 1 / (1 + fit[0] / c)
                if int(v["row"]["nodes"]) == 1:
                    if impl != "allgather":
                        heldout["one"].append(deviation)
                else:
                    heldout[f"cross.{impl}"].append(deviation)
        for tag, values in heldout.items():
            if values:
                put(f"model.heldout.{tag}.n", str(len(values)))
                put(f"model.heldout.{tag}.maxdev", max(abs(x) for x in values))
                put(f"model.heldout.{tag}.below", -min(values))
    for impls, tag in ((("allgather",), "allgather"),):
        other = mf.fitted_cstar(impls)
        if other:
            put(f"model.cstar.fit.{tag}", other[0], "{:.0f}")
    # measured against predicted E_comm, thin ranks inside one node
    rows = io.passed(io.load("e5-model", "model-thin.csv"))
    by = io.summarise(rows, ["impl", "N", "ranks"])
    get_all = []
    for (impl, n, ranks), v in by.items():
        control = by.get(("local", n, ranks))
        if impl == "local" or not control:
            continue
        c = int(n) // int(ranks)
        measured = control["median"] / v["median"]
        put(f"collapse.E.c{c}.{impl}", measured)
        put(f"model.pred.c{c}", mf.model_efficiency(c, params))
        if impl == "rma_get" and c >= 16:
            get_all.append(measured)
    if get_all:
        put("collapse.get.min", min(get_all)); put("collapse.get.max", max(get_all))
    if "collapse.E.c8.ring" in N:
        N["collapse.E.c8"] = N["collapse.E.c8.ring"]
    put("model.thin.ranks", str(max(int(k[2]) for k in by)))
    # fat ranks: same rows per rank, kernel many times faster
    fat = io.summarise(io.passed(io.load("e5-model", "model-fat.csv")), ["impl", "N", "ranks", "threads"])
    for (impl, n, ranks, threads), v in fat.items():
        control = fat.get(("local", n, ranks, threads))
        if impl == "local" or not control:
            continue
        c = int(n) // int(ranks)
        put(f"fat.E.{impl}.T{threads}.c{c}", control["median"] / v["median"])
        put("fat.ranks", ranks); put("fat.threads", threads)
    if fit and fat:
        # what the thin-rank curve gives at the same rows per rank
        for c in (1152, 2304):
            put(f"fat.thin.c{c}", 1 / (1 + fit[0] / c))
            put(f"fat.thin.loss.pct.c{c}", 100 * (1 - 1 / (1 + fit[0] / c)), "{:.0f}")
            ring = fat.get(("ring", str(c * 4), "4", "18")); control = fat.get(("local", str(c * 4), "4", "18"))
            if ring and control:
                put(f"fat.loss.pct.c{c}", 100 * (1 - control["median"] / ring["median"]), "{:.0f}")


def net():
    rows = io.load("e3-multi", "netbench.csv")
    if not rows:
        return

    def ping(mode, same):
        return {int(r["bytes"]): float(r["seconds"]) for r in rows if r["mode"] == mode and r["same_node"] == same}

    for tag, same in (("intra", "1"), ("inter", "0")):
        pp = ping("pingpong", same)
        if not pp:
            continue
        small = pp[min(b for b in pp if b >= 1)]
        put(f"net.lat.{tag}.us", sig(small * 1e6))
        if 2 ** 20 in pp:
            put(f"net.bw.{tag}.gbs", sig(2 ** 20 / pp[2 ** 20] / 1e9))
        if 2 ** 24 in pp:
            put(f"net.bw16.{tag}.gbs", sig(2 ** 24 / pp[2 ** 24] / 1e9))
        get = ping("rget", same)
        if get and 2 ** 20 in get:
            put(f"net.get.bw.{tag}.gbs", sig(2 ** 20 / get[2 ** 20] / 1e9))
        busy = ping("busy_sender_recv", same)
        if busy and 2 ** 20 in busy:
            put(f"net.busy.{tag}.ms", sig(busy[2 ** 20] * 1e3))
        busy_get = ping("busy_target_rget", same)
        if busy_get and 2 ** 20 in busy_get:
            put(f"net.busyget.{tag}.ms", sig(busy_get[2 ** 20] * 1e3))
    if "net.busy.inter.ms" in N:
        N["net.busy.ms"] = N["net.busy.inter.ms"]
    put("net.busy.set.ms", float(rows[0]["busy_seconds"]) * 1e3, "{:.0f}")


def pct(x, digits=0):
    return f"{100 * x:.{digits}f}"


def weak():
    rows = io.summarise(io.passed(io.load_all(io.MULTI, "weak-rows.csv")), ["impl", "N", "ranks"])
    for c0 in (256, 1024):
        def at(impl, P):
            return rows.get((impl, str(c0 * P), str(P)))
        one = at("local", 1)
        if not one:
            continue
        put(f"weak.rows{c0}.T1", seconds(one["median"]))
        for P in (72, 288):
            for impl in ("local", "ring", "overlap", "rma_get"):
                v = at(impl, P)
                if v:
                    # per-rank work grows like P, so the ideal time is P * T(1)
                    put(f"weak.rows{c0}.E.{impl}.{P}", one["median"] * P / v["median"])
                    # scaled speedup: one core would need P^2 T(1) for the P-times longer sequence
                    put(f"weak.rows{c0}.S.{impl}.{P}", one["median"] * P * P / v["median"], "{:.0f}")
                    put(f"weak.rows{c0}.T.{impl}.{P}", seconds(v["median"]))
    work = io.summarise(io.passed(io.load_all(io.MULTI, "weak-work.csv")), ["impl", "N", "ranks"])
    for n0 in (8192, 16384):
        one = work.get(("local", str(n0), "1"))
        if not one:
            continue
        put(f"weak.work{n0}.T1", seconds(one["median"]))
        for P, factor in ((16, 4), (64, 8), (256, 16)):
            for impl in ("local", "ring", "overlap", "rma_get"):
                v = work.get((impl, str(n0 * factor), str(P)))
                if v:
                    put(f"weak.work{n0}.T.{impl}.{P}", seconds(v["median"]))
                    put(f"weak.work{n0}.E.{impl}.{P}", one["median"] / v["median"])
                    put(f"weak.work{n0}.S.{impl}.{P}", P * one["median"] / v["median"], "{:.0f}")


def hybrid():
    by = io.summarise(io.passed(io.load_all(io.MULTI, "hybrid.csv")), ["impl", "ranks", "threads"])
    if not by:
        return
    n = int(next(iter(by.values()))["row"]["N"])
    put("hybrid.N.h", thousands(n))
    ts_of = lambda cores: io.serial_time(n, cores)
    for cores in (72, 288):
        splits = {int(k[2]): v for k, v in by.items() if k[0] == "overlap" and int(k[1]) * int(k[2]) == cores}
        control = {int(k[2]): v for k, v in by.items() if k[0] == "local" and int(k[1]) * int(k[2]) == cores}
        if not splits:
            continue
        best = min(splits, key=lambda th: splits[th]["median"])
        worst = max(splits, key=lambda th: splits[th]["median"])
        put(f"hybrid.best.threads.{cores}", str(best)); put(f"hybrid.best.ranks.{cores}", str(cores // best))
        put(f"hybrid.c.best.{cores}", thousands(n // (cores // best)))
        put(f"hybrid.best.time.{cores}", seconds(splits[best]["median"]))
        put(f"hybrid.worst.threads.{cores}", str(worst)); put(f"hybrid.worst.time.{cores}", seconds(splits[worst]["median"]))
        put(f"hybrid.range.pct.{cores}", 100 * (splits[worst]["median"] / splits[best]["median"] - 1), "{:.0f}")
        if 1 in splits:
            put(f"hybrid.thin.time.{cores}", seconds(splits[1]["median"]))
            put(f"hybrid.gain.pct.{cores}", 100 * (splits[1]["median"] / splits[best]["median"] - 1), "{:.0f}")
        for th, v in splits.items():
            put(f"hybrid.time.{cores}.T{th}", seconds(v["median"]))
            if th in control:
                put(f"hybrid.ecomm.{cores}.T{th}", control[th]["median"] / v["median"])
                put(f"hybrid.control.{cores}.T{th}", seconds(control[th]["median"]))
        put(f"hybrid.repspread.pct.{cores}", 100 * max((v["high"] - v["low"]) / v["median"] for v in splits.values()), "{:.0f}")
        put(f"hybrid.repspread.best.pct.{cores}", 100 * (splits[best]["high"] - splits[best]["low"]) / splits[best]["median"], "{:.1f}")
        ts = ts_of(cores)
        if ts:
            put(f"hybrid.speedup.{cores}", ts / splits[best]["median"], "{:.0f}")
            put(f"hybrid.eff.{cores}", ts / splits[best]["median"] / cores)


def causal():
    by = io.summarise(io.passed(io.load_all(io.MULTI, "causal.csv")), ["impl", "mask", "partition", "ranks"])
    if not by:
        return
    for P in sorted({int(k[3]) for k in by}):
        full = by.get(("overlap", "none", "contiguous", str(P)))
        if not full:
            continue
        for impl in ("overlap", "rma_get", "local"):
            for part in ("contiguous", "zigzag"):
                v = by.get((impl, "causal", part, str(P)))
                if v:
                    put(f"causal.ratio.{impl}.{part}.{P}", v["median"] / full["median"])
                    put(f"causal.T.{impl}.{part}.{P}", seconds(v["median"]))
        for part in ("contiguous", "zigzag"):
            get = sum(float(io.read(f)[0]["bytes_recv"]) for f in io.rank_files(io.MULTI, f"causal-rma_get-{part}-P{P}.rank*.csv"))
            ring = sum(float(io.read(f)[0]["bytes_recv"]) for f in io.rank_files(io.MULTI, f"causal-overlap-{part}-P{P}.rank*.csv"))
            if ring:
                put(f"causal.bytes.{part}.{P}", get / ring)
    for P in sorted({int(k[3]) for k in by}):
        for impl in ("overlap", "rma_get"):
            a, b = by.get((impl, "causal", "contiguous", str(P))), by.get((impl, "causal", "zigzag", str(P)))
            if a and b:
                put(f"causal.zigzag.gain.{impl}.{P}", a["median"] / b["median"])
                put(f"causal.zigzag.cost.pct.{impl}.{P}", 100 * (b["median"] / a["median"] - 1), "{:.0f}")
        ring, pull = by.get(("overlap", "causal", "zigzag", str(P))), by.get(("rma_get", "causal", "contiguous", str(P)))
        if ring and pull:
            put(f"causal.pull.over.ring.{P}", pull["median"] / ring["median"], "{:.1f}" if pull["median"] / ring["median"] >= 3 else "{:.2f}")
    # Balance of the zigzag partition: pairs counted by every rank of the ring, against the closed form.
    for P in sorted({int(k[3]) for k in by}):
        counted = {}
        for f in io.rank_files(io.MULTI, f"causal-overlap-zigzag-P{P}.rank*.csv"):
            rows = io.read(f)
            counted[int(rows[0]["rank"])] = int(rows[0]["useful_pairs"])
        if len(counted) != P:
            continue
        ring = by.get(("overlap", "causal", "zigzag", str(P)))
        if not ring:
            continue
        n = int(ring["row"]["N"])
        if [counted[r] for r in range(P)] != zigzag_pairs(n, P):
            raise SystemExit(f"causal, zigzag, P={P}: the recorded pair counts differ from the closed form of the partition")
        mean = sum(counted.values()) / P
        deviation = max(abs(x / mean - 1) for x in counted.values())
        if (n % (2 * P) == 0) != (deviation == 0):
            raise SystemExit(f"causal, zigzag, P={P}: the balance is exact if and only if 2P divides N; the records say otherwise")
        put(f"causal.pairs.maxdev.pct.zigzag.{P}", 100 * deviation, "{:.2f}")
    small = zigzag_pairs(37, 5)            # a unit-test shape that 2P does not divide (closed form, no measurement)
    put("causal.pairs.maxdev.pct.zigzag.N37.P5", 100 * max(abs(x * 5 / sum(small) - 1) for x in small), "{:.0f}")
    for part in ("contiguous", "zigzag"):
        per_rank = [io.med(io.read(f), "compute") for f in io.rank_files(io.MULTI, f"causal-overlap-{part}-P8.rank*.csv")]
        if per_rank:
            put(f"causal.rank.min.{part}", seconds(min(per_rank))); put(f"causal.rank.max.{part}", seconds(max(per_rank)))
            put(f"causal.rank.spread.{part}", max(per_rank) / max(min(per_rank), 1e-9), "{:.1f}")


def memory():
    rows = io.load("e7-memory", "memory.csv")
    if not rows:
        return
    by = io.summarise(rows, ["impl", "N"], value="peak_rss_mb_max")
    ranks = int(rows[0]["ranks"])
    sizes = sorted({int(k[1]) for k in by})
    big = sizes[-1]
    put("mem.N.h", thousands(big)); put("mem.ranks", str(ranks))
    for impl in ("allgather", "ring", "overlap", "rma_get"):
        v = by.get((impl, str(big)))
        if v:
            put(f"mem.{impl}.mb", v["median"], "{:.0f}")
            if v["row"].get("pss_mb_total"):
                put(f"mem.{impl}.pss.mb", float(v["row"]["pss_mb_total"]) / ranks, "{:.0f}")
    a, r = by.get(("allgather", str(big))), by.get(("ring", str(big)))
    if a and r:
        put("mem.ratio", a["median"] / r["median"], "{:.1f}")
        put("mem.node.allgather.gb", a["median"] * ranks / 1024, "{:.0f}"); put("mem.node.ring.gb", r["median"] * ranks / 1024, "{:.1f}")
    # K and V of the whole sequence, d = dv = 64, in MiB
    put("mem.kv.mb", big * 128 * 4 / 2 ** 20, "{:.0f}")
    put("mem.shard.mb", big / ranks * 128 * 4 / 2 ** 20, "{:.1f}")


def gqa():
    by = io.summarise(io.passed(io.load_all(io.MULTI, "gqa.csv")), ["impl", "N", "ranks", "mha_control"])
    for (impl, n, ranks, control), v in by.items():
        if control != "0":
            continue
        mha = by.get((impl, n, ranks, "1"))
        if not mha:
            continue
        tag = f"{impl}.N{n}.P{ranks}"
        put(f"gqa.T.{tag}", seconds(v["median"])); put(f"gqa.Tmha.{tag}", seconds(mha["median"]))
        put(f"gqa.gain.pct.{tag}", 100 * (mha["median"] / v["median"] - 1), "{:.0f}")
        if impl == "overlap" and float(v["row"]["bytes_recv_max"]) > 0:
            put(f"gqa.bytes.ratio.N{n}.P{ranks}", float(mha["row"]["bytes_recv_max"]) / float(v["row"]["bytes_recv_max"]), "{:.2f}")
            put(f"gqa.mib.N{n}.P{ranks}", float(v["row"]["bytes_recv_max"]) / 2 ** 20, "{:.1f}")
            put(f"gqa.mib.mha.N{n}.P{ranks}", float(mha["row"]["bytes_recv_max"]) / 2 ** 20, "{:.1f}")
    real = io.summarise(io.passed(io.load("real-data", "e10-real.csv")), ["impl", "ranks", "mha_control"])
    for (impl, ranks, control), v in real.items():
        mha = real.get((impl, ranks, "1"))
        if control == "0" and mha:
            put(f"gqa.real.T.{impl}.P{ranks}", seconds(v["median"])); put(f"gqa.real.Tmha.{impl}.P{ranks}", seconds(mha["median"]))
            put(f"gqa.real.gain.pct.{impl}.P{ranks}", 100 * (mha["median"] / v["median"] - 1), "{:.0f}")


def smallc():
    run = io.latest("e3-multi", "comm-small-c.csv")
    if not run:
        return
    for name, tag in (("comm-small-c.csv", "thin"), ("comm-fat.csv", "fat")):
        by = io.summarise(io.passed(io.read(run / name)), ["impl", "N", "ranks"])
        cs = sorted({int(k[1]) // int(k[2]) for k in by})
        for (impl, n, ranks), v in by.items():
            control = by.get(("local", n, ranks))
            if impl == "local" or not control:
                continue
            c = int(n) // int(ranks)
            if c in (cs[0], cs[-1]) or c in (32, 128, 1152):
                put(f"smallc.{tag}.{impl}.c{c}", v["median"] / control["median"], "{:.1f}" if v["median"] / control["median"] >= 3 else "{:.2f}")
        put(f"smallc.{tag}.cmin", str(cs[0])); put(f"smallc.{tag}.cmax", thousands(cs[-1]))
        if tag == "thin":
            put("smallc.ranks", str(max(int(k[2]) for k in by)))
            # set-up of the one-sided push (its window), outside the timed phase
            push = [float(r["t_setup_max"]) for r in io.read(run / name) if r["impl"] == "rma_put"]
            if push:
                put("setup.rma_put.P288", statistics.median(push))
    order = {}
    for name in ("staggered", "ascending"):      # results_io.read decodes the driver's enum value
        reps = [r for r in io.passed(io.read(run / "comm-order.csv")) if r["fetch_order"] == name]
        if reps and not all(f"--fetch-order {name}" in r["command"] for r in reps):
            raise SystemExit("comm-order.csv: decoded fetch order disagrees with the recorded command line")
        if reps:
            order[name] = statistics.median(float(r["t_total_max"]) for r in reps)
            put(f"order.T.{name}", seconds(order[name]))
    if len(order) == 2:
        put("order.ratio", order["ascending"] / order["staggered"], "{:.1f}")
    poll = io.summarise(io.passed(io.read(run / "comm-poll.csv")), ["impl", "N", "poll_tiles"])
    base = io.summarise(io.passed(io.read(run / "comm-small-c.csv")), ["impl", "N"])
    gains = []
    for (impl, n, tiles), v in poll.items():
        b = base.get((impl, n))
        if b:
            gains.append((impl, int(n), int(tiles), b["median"] / v["median"]))
    for impl in ("overlap", "rma_get"):
        g = [x[3] for x in gains if x[0] == impl]
        if g:
            put(f"poll.best.{impl}", max(g), "{:.2f}"); put(f"poll.worst.{impl}", min(g), "{:.2f}")
    fat = io.summarise(io.passed(io.read(run / "comm-fat.csv")), ["impl", "N"])
    fat_poll = io.summarise(io.passed(io.read(run / "comm-fat-poll.csv")), ["impl", "N"])
    for (impl, n), v in fat_poll.items():
        control = fat.get(("local", n))
        ranks = int(v["row"]["ranks"])
        if control:
            put(f"smallc.fat.{impl}poll.c{int(n) // ranks}", v["median"] / control["median"], "{:.2f}")
        if (impl, n) in fat:
            put(f"pollfat.gain.{impl}.c{int(n) // ranks}", fat[(impl, n)]["median"] / v["median"])
    again = io.latest("e5-fatcheck", "fatcheck-r1.csv")
    if again:
        ratios = {}
        for rnd in (1, 2):
            rep = io.summarise(io.passed(io.read(again / f"fatcheck-r{rnd}.csv")), ["impl", "N", "ranks"])
            polled = io.summarise(io.passed(io.read(again / f"fatcheck-poll-r{rnd}.csv")), ["impl", "N", "ranks"])
            for (impl, n, ranks), v in rep.items():
                control = rep.get(("local", n, ranks))
                if impl != "local" and control:
                    c = int(n) // int(ranks)
                    put(f"fatcheck.{impl}.c{c}.r{rnd}", v["median"] / control["median"])
                    ratios.setdefault((impl, c), []).append(v["median"] / control["median"])
                    put("fatcheck.ranks", ranks); put("fatcheck.threads", v["row"]["threads"])
            for (impl, n, ranks), v in polled.items():
                control = rep.get(("local", n, ranks))
                if control:
                    put(f"fatcheck.{impl}poll.c{int(n) // int(ranks)}.r{rnd}", v["median"] / control["median"])
        for (impl, c), values in ratios.items():
            put(f"fatcheck.{impl}.c{c}.min", min(values)); put(f"fatcheck.{impl}.c{c}.max", max(values))
        put("fatcheck.rounds", "2"); put("fatcheck.reps", str(3 * 2))
    packed = io.summarise(io.passed(io.read(run / "comm-packed.csv")), ["impl"])
    spread = io.summarise(io.passed(io.read(run / "comm-spread.csv")), ["impl"])
    for impl in ("local", "overlap"):
        if (impl,) in packed and (impl,) in spread:
            put(f"placement.packed.{impl}", seconds(packed[(impl,)]["median"])); put(f"placement.spread.{impl}", seconds(spread[(impl,)]["median"]))
            put(f"placement.gain.pct.{impl}", 100 * (packed[(impl,)]["median"] / spread[(impl,)]["median"] - 1), "{:.0f}")


def traces():
    run = io.latest("e3-multi", "comm-trace.csv")
    if not run:
        return
    # Numbering of the ranks under the per-node mapping rule that the placement run also used.
    layouts = {impl: hosts_by_rank(sorted((run / "traces").glob(f"ranks-{impl}.rank*.csv"))) for impl in ("ring", "overlap", "rma_get")}
    crossing = {(len(h), ring_links_between_nodes(h)) for h in layouts.values() if h}
    if len(crossing) == 1:
        links, between = crossing.pop()
        put("topo.trace.links", str(links)); put("topo.trace.cross", str(between))
    for impl in ("ring", "overlap", "rma_get"):
        rows = io.read(run / "traces" / f"trace-{impl}.rank3.csv")
        if not rows:
            continue
        totals = []
        for rep in sorted({r["rep"] for r in rows}):
            steps = [r for r in rows if r["rep"] == rep]
            totals.append((sum(float(s["compute_end"]) - float(s["compute_begin"]) for s in steps),
                           sum(max(0.0, float(s["wait_end"]) - float(s["wait_begin"])) for s in steps), len(steps)))
        compute, wait = statistics.median(t[0] for t in totals), statistics.median(t[1] for t in totals)
        put(f"trace.compute.{impl}", seconds(compute)); put(f"trace.wait.ms.{impl}", wait * 1e3, "{:.0f}" if wait >= 0.01 else "{:.1f}")
        put(f"trace.wait.pct.{impl}", 100 * wait / (compute + wait), "{:.0f}")
        put(f"trace.wait.step.ms.{impl}", wait * 1e3 / max(totals[0][2] - 1, 1), "{:.1f}")
        put("trace.steps", str(totals[0][2]))
    total = io.summarise(io.passed(io.read(run / "comm-trace.csv")), ["impl"])
    for (impl,), v in total.items():
        put(f"trace.total.{impl}", seconds(v["median"]))
    if ("ring",) in total and ("overlap",) in total:
        put("trace.total.gain.pct", 100 * (total[("ring",)]["median"] / total[("overlap",)]["median"] - 1), "{:.0f}")


def gpu():
    by = io.summarise(io.passed(io.load("gpu-ladder", "ladder.csv")), ["kernel", "N"])
    if not by:
        return
    rate = lambda v: float(v["row"]["useful_flops"]) / v["median"]
    for kernel in ("k0", "k1", "k2"):
        rates = {int(k[1]): rate(v) for k, v in by.items() if k[0] == kernel}
        if not rates:
            continue
        unit, fmt = (1e9, "{:.0f}") if kernel == "k0" else (1e12, "{:.2f}")
        put(f"gpu.{kernel}.min", min(rates.values()) / unit, fmt); put(f"gpu.{kernel}.max", max(rates.values()) / unit, fmt)
        put(f"gpu.{kernel}.Nmax.h", thousands(max(rates)))
    k2 = {int(k[1]): v for k, v in by.items() if k[0] == "k2"}
    k1 = {int(k[1]): v for k, v in by.items() if k[0] == "k1"}
    k0 = {int(k[1]): v for k, v in by.items() if k[0] == "k0"}
    big = max(k2)
    put("gpu.k2.Tmax", seconds(k2[big]["median"])); put("gpu.k2.pairs", f"{big * big / 1e10:.1f}")
    # against the vendor's single-precision peak of the A40 (37.4 TFLOP/s, datasheet)
    put("gpu.peakpct", 100 * max(rate(v) for v in k2.values()) / 37.4e12, "{:.0f}")
    common = max(set(k0) & set(k2))
    put("gpu.N.common.h", thousands(common)); put("gpu.k1.over.k0", rate(k1[common]) / rate(k0[common]), "{:.0f}")
    put("gpu.k1.Nmin.h", thousands(min(k1))); put("gpu.k2.Nmin.h", thousands(min(k2)))
    put("gpu.k2.over.k0", rate(k2[common]) / rate(k0[common]), "{:.0f}")
    put("gpu.k2.over.k1.small", k1[min(k1)]["median"] / k2[min(k1)]["median"], "{:.2f}")
    put("gpu.k2.over.k1.big", k1[max(k1)]["median"] / k2[max(k1)]["median"], "{:.2f}")
    # against one CPU core and a full node at the same N
    serial_rows = io.summarise(io.passed(io.load("e1-serial", "serial-n.csv")), ["impl", "kernel", "N"])
    for n in (32768, 131072):
        s = serial_rows.get(("S1", "tuned", str(n)))
        if s and n in k2:
            put(f"gpu.vs.core.{n}", s["median"] / k2[n]["median"], "{:.0f}")
    strong_by = io.summarise(io.strong_rows(), ["impl", "N", "ranks"])
    node = strong_by.get(("overlap", "131072", "72"))
    if node and 131072 in k2:
        put("gpu.vs.node", node["median"] / k2[131072]["median"], "{:.2f}")
        put("gpu.node.T", seconds(node["median"])); put("gpu.k2.T131k", seconds(k2[131072]["median"]))
    one_core = serial_rows.get(("S1", "tuned", "131072"))
    if node and one_core and 131072 in k2:
        # the GPU expressed in cores of a fully loaded node: 72 * T_node / T_gpu
        put("gpu.equiv.cores", 72 * node["median"] / k2[131072]["median"], "{:.0f}")
    full = strong_by.get(("overlap", "131072", "288"))
    if full and 131072 in k2:
        put("gpu.vs.288", k2[131072]["median"] / full["median"], "{:.1f}")
    causal_rows = io.summarise(io.passed(io.load("gpu-ladder", "ladder-causal.csv")), ["kernel", "N"])
    for (kernel, n), v in causal_rows.items():
        full_run = by.get((kernel, n))
        if full_run:
            put(f"gpu.causal.{kernel}.N{n}", v["median"] / full_run["median"])
            put(f"gpu.causal.T.{kernel}.N{n}", seconds(v["median"]))
    real = io.summarise(io.passed(io.load("gpu-ladder", "real.csv")), ["N", "purpose"])
    r = real.get(("32768", "final-real-layer11"))
    if r:
        put("gpu.real.T", seconds(r["median"])); put("gpu.real.layers", seconds(24 * r["median"]))
    # what the fitted one-node communication curve predicts for GPU ranks (c* scales with the compute rate)
    import make_figures as mf
    fit, params = mf.fitted_cstar(), mf.model_parameters()
    if fit and params:
        ratio = max(rate(v) for v in k2.values()) / params["F"]
        put("gpu.vs.loadedcore", ratio, "{:.0f}")
        put("gpu.cstar.pred", thousands(round(fit[0] * ratio, -2)))
        put("gpu.c95.pred", thousands(round(19 * fit[0] * ratio, -3)))
    # MPI+CUDA ring with two ranks sharing one GPU (functional check)
    check = io.load("gpu-ringcheck", "ringcheck.csv")
    if check:
        import make_tables as mt
        configs = {(r["kernel"], r["comm"], r["schedule"], r["mask"], r["partition"], r["N"], r["H"]) for r in check}
        put("gpucheck.configs", str(len(configs))); put("gpucheck.reps", str(len(check)))
        put("gpucheck.failed", str(sum(r["validation"] != "passed" for r in check)))
        N["gpucheck.maxerr"] = mt.sci(max(float(r["max_abs_err"]) for r in check))
        put("gpucheck.ranks", check[0]["ranks"]); put("gpucheck.gpus", check[0]["gpus"])
    # two GPUs
    for prefix, name, tag in (("gpu-ring-1node", "ring-one-node.csv", "one"), ("gpu-ring-2node", "ring-two-nodes.csv", "two")):
        ring = io.summarise(io.passed(io.load(prefix, name)), ["ranks", "comm", "schedule", "N"])
        single = {int(k[3]): v["median"] for k, v in ring.items() if k[0] == "1"}
        for (ranks, comm, schedule, n), v in ring.items():
            if ranks == "2" and int(n) in single:
                put(f"gpu2.{tag}.E.{comm}.{schedule}.N{n}", single[int(n)] / (2 * v["median"]))
        if not single:
            continue
        put(f"gpu2.{tag}.Nmin.h", thousands(min(single))); put(f"gpu2.{tag}.Nmax.h", thousands(max(single)))
        rows = io.load(prefix, name) + io.load(prefix, name.replace("ring-", "ring-causal-"))
        import make_tables as mt
        put(f"gpu2.{tag}.reps", str(len(rows))); put(f"gpu2.{tag}.failed", str(sum(r["validation"] != "passed" for r in rows)))
        put(f"gpu2.{tag}.configs", str(len({(r["ranks"], r["comm"], r["schedule"], r["N"], r["mask"], r["partition"]) for r in rows})))
        put(f"gpu2.{tag}.valrows", rows[0]["val_rows"])
        N[f"gpu2.{tag}.maxerr"] = mt.sci(max(float(r["max_abs_err"]) for r in rows))
        for (ranks, comm, schedule, n), v in ring.items():
            if ranks != "2":
                continue
            one = ring.get(("1", "host", "overlap", n))
            put(f"gpu2.{tag}.S.{comm}.{schedule}.N{n}", single[int(n)] / v["median"])
            put(f"gpu2.{tag}.T.{comm}.{schedule}.N{n}", seconds(v["median"]))
            put(f"gpu2.{tag}.mpi.ms.{comm}.{schedule}.N{n}", 1e3 * io.med(v["rows"], "t_mpi_max"), "{:.0f}" if io.med(v["rows"], "t_mpi_max") >= 0.01 else "{:.1f}")
            put(f"gpu2.{tag}.kernel.ms.{comm}.{schedule}.N{n}", 1e3 * io.med(v["rows"], "t_kernel_max"), "{:.1f}")
            if one:
                # the same ratio for the time inside the kernels alone
                put(f"gpu2.{tag}.Ek.{comm}.{schedule}.N{n}", io.med(one["rows"], "t_kernel_max") / (2 * io.med(v["rows"], "t_kernel_max")))
        for n, t in single.items():
            put(f"gpu2.{tag}.T1.N{n}", seconds(t))
        best = [single[int(k[3])] / (2 * v["median"]) for k, v in ring.items() if k[0] == "2" and k[2] == "overlap" and int(k[3]) >= 16384]
        if best:
            put(f"gpu2.{tag}.E.overlap.min", min(best)); put(f"gpu2.{tag}.E.overlap.max", max(best))
        causal2 = io.summarise(io.passed(io.load(prefix, name.replace("ring-", "ring-causal-"))), ["partition"])
        if ("contiguous",) in causal2 and ("zigzag",) in causal2:
            put(f"gpu2.{tag}.causal.T.contiguous", seconds(causal2[("contiguous",)]["median"]))
            put(f"gpu2.{tag}.causal.T.zigzag", seconds(causal2[("zigzag",)]["median"]))
            put(f"gpu2.{tag}.causal.gain", causal2[("contiguous",)]["median"] / causal2[("zigzag",)]["median"])
            unmasked = ring.get(("2", "aware", "overlap", causal2[("zigzag",)]["row"]["N"]))
            if unmasked:
                put(f"gpu2.{tag}.causal.ratio.contiguous", causal2[("contiguous",)]["median"] / unmasked["median"])
                put(f"gpu2.{tag}.causal.ratio.zigzag", causal2[("zigzag",)]["median"] / unmasked["median"])
    slow = [1 / float(N[k]) for k in N if k.startswith("gpu2.two.S.") and k.endswith(".N4096")]
    if slow:
        put("gpu2.two.slow.min.N4096", min(slow), "{:.0f}"); put("gpu2.two.slow.max.N4096", max(slow), "{:.0f}")
    if fit and params and "gpu2.one.E.host.overlap.N4096" in N:
        cstar = fit[0] * ratio
        for c in (2048, 65536):
            put(f"gpu.pred.E.c{c}", 1 / (1 + cstar / c))
    # GPU unit tests (every row against fp64)
    unit = io.load("gpu-validate", "gpu-validation.csv")
    if unit:
        import make_tables as mt
        put("gpuunit.records", str(len(unit))); put("gpuunit.failed", str(sum(r["validation"] != "passed" for r in unit)))
        put("gpuunit.configs", str(len({(r["kernel"], r["N"], r["Nq"], r["mask"], r["partition"], r["alpha"], r["H"]) for r in unit})))
        N["gpuunit.maxerr"] = mt.sci(max(float(r["max_abs_err"]) for r in unit))


def real():
    rows = io.load("real-data", "v7.csv")
    if rows:
        # all differences relative to the largest component of the model's full output (as in the table)
        cpu = [float(r["max_abs_err"]) / float(r["oref_abs_max"]) for r in rows]
        torch = [float(r["oref_max_abs_err"]) / float(r["oref_abs_max"]) for r in rows if float(r.get("oref_abs_max") or 0) > 0]
        import make_tables as mt
        put("real.err.fp64.max", mt.sci(max(cpu))); put("real.err.torch.max", mt.sci(max(torch)) if torch else None)
        put("real.configs", str(len({r["purpose"] for r in rows})))
        put("real.failed", str(sum(r["validation"] != "passed" for r in rows)))
        scales = [float(r["oref_abs_max"]) for r in rows]
        put("real.scale.min", sig(min(scales))); put("real.scale.max", sig(max(scales)))
        # the automatic check: difference to fp64 at most 1e-3 times the largest reference component of the sampled rows
        put("real.tol.margin", min(float(r["atol"]) / float(r["max_abs_err"]) for r in rows), "{:.0f}")
        put("real.valrows", rows[0]["val_rows"])
    variants = io.load("real-data", "v7-variants.csv")
    if variants:
        put("real.variants.runs", str(len({(r["impl"], r["partition"]) for r in variants})))
        put("real.variants.failed", str(sum(r["validation"] != "passed" for r in variants)))
        put("real.variants.err.max", __import__("make_tables").sci(max(float(r["max_abs_err"]) / float(r["ref_abs_max"]) for r in variants)))
    store = GEN / "figure_numbers.json"
    if not store.exists():
        return
    stats = json.loads(store.read_text()).get("real_stats", {})
    for key, v in stats.items():
        parts = key.split("|")
        if parts[0] in ("mass_w", "err_w", "err_sw") and len(parts) == 3 and parts[2] in ("1024", "4096", "16384"):
            x = v["median"]
            name = f"real.{parts[0]}.{parts[1]}.{parts[2]}"
            put(name, sig(100 * x, 2) if x < 1 else f"{x:.1f}")
        elif parts[0] == "entropy":
            put(f"real.entropy.{parts[1]}", v["median"], "{:.1f}")
            put(f"real.entropy.eff.{parts[1]}", math.exp(v["median"]), "{:.0f}")
        elif parts[0] == "mass_sink":
            put(f"real.sink.{parts[1]}", sig(100 * v["median"], 2))
        if parts[0] == "mass_w" and len(parts) == 3 and parts[2] in ("1024", "4096"):
            put(f"real.beyond.{parts[1]}.{parts[2]}", 100 * (1 - v["median"]), "{:.0f}")
        if parts[0] == "err_w" and len(parts) == 3 and "p90" in v:
            put(f"real.err_w.p90.{parts[1]}.{parts[2]}", v["p90"], "{:.0f}")
    heads = json.loads(store.read_text()).get("real_heads", {})
    if heads:
        err = np.array(heads["err_w1024"]); first = np.array(heads["mass_first"])
        exact = err[0] < 0.01                              # layer 0: heads reproduced by a 1,024-token window
        put("real.heads.exact.layer0", str(int(exact.sum())))
        put("real.heads.rest.min.layer0", 100 * err[0][~exact].min(), "{:.0f}"); put("real.heads.rest.max.layer0", 100 * err[0][~exact].max(), "{:.0f}")
        sink = first[1] > 0.4                              # layer 11: heads that look at the first token
        put("real.sinkheads.count", str(int(sink.sum())))
        put("real.sinkheads.min", 100 * first[1][sink].min(), "{:.0f}"); put("real.sinkheads.max", 100 * first[1][sink].max(), "{:.0f}")
        put("real.sinkheads.first", str(int(np.argmax(sink)))); put("real.sinkheads.last", str(int(len(sink) - 1 - np.argmax(sink[::-1]))))
        put("real.otherheads.max", 100 * first[1][~sink].max(), "{:.0f}")


def errors():
    """Error study: the tuned kernel against fp64 on every row, plain and stressed.
    Rows that exceeded the tolerance are part of the result and are not filtered."""
    import make_tables as mt
    rows = io.load("e1-serial", "serial-error.csv")
    if not rows:
        return
    err = io.summarise(rows, ["N", "block_k", "alpha"], value="max_abs_err")
    plain = [v["high"] for k, v in err.items() if float(k[2]) == 1]
    if plain:
        N["err.plain.max"] = mt.sci(max(plain)); N["err.plain.min"] = mt.sci(min(plain))
    sizes = sorted({int(k[0]) for k in err})
    put("err.rows.max", thousands(sizes[-1]))
    alphas = sorted({float(k[2]) for k in err if float(k[2]) != 1})
    if alphas:
        put("err.alpha", f"{alphas[0]:g}")
    for n in sizes:
        stress = {int(k[1]): v["high"] for k, v in err.items() if int(k[0]) == n and float(k[2]) != 1}
        if stress:
            N[f"err.stress.N{n}"] = mt.sci(max(stress.values()))
            put(f"err.stress.bkspread.pct.N{n}", 100 * (max(stress.values()) / min(stress.values()) - 1), "{:.1f}")
    failed = [r for r in rows if r["validation"] == "failed"]
    put("err.failed.reps", str(len(failed)))
    put("err.failed.configs", str(len({(r["N"], r["block_k"], r["alpha"]) for r in failed})))
    if failed:
        put("err.failed.Nmin.h", thousands(min(int(r["N"]) for r in failed)))
        N["err.failed.max"] = mt.sci(max(float(r["max_abs_err"]) for r in failed))
    bks = sorted({int(k[1]) for k in err})
    put("err.bk.list", ", ".join(thousands(b) for b in bks))


def clock():
    """The control M0 under perf stat: clock and work per cycle (suite clock)."""
    table = io.clock_table()
    if not table:
        return
    one = table.get("packed-P1")
    for label, v in table.items():
        tag = label.replace("packed-P", "").replace("spread-P", "spread")
        put(f"clock.ghz.{tag}", v["ghz"]); put(f"clock.cpp.{tag}", v["cycles_per_pair"], "{:.1f}"); put(f"clock.ipc.{tag}", v["ipc"])
        if one:
            f_ratio, w_ratio = v["ghz"] / one["ghz"], one["cycles_per_pair"] / v["cycles_per_pair"]
            timed = one["time"] / (v["ranks"] * v["time"])
            put(f"clock.fratio.{tag}", f_ratio); put(f"clock.wratio.{tag}", w_ratio)
            put(f"clock.product.{tag}", f_ratio * w_ratio); put(f"clock.timed.{tag}", timed)
            put(f"clock.wloss.pct.{tag}", 100 * (1 - w_ratio), "{:.0f}"); put(f"clock.floss.pct.{tag}", 100 * (1 - f_ratio), "{:.0f}")
    if one:
        put("clock.N.h", thousands(one["N"]))
        gaps = [abs(v["ghz"] / one["ghz"] * one["cycles_per_pair"] / v["cycles_per_pair"] / (one["time"] / (v["ranks"] * v["time"])) - 1)
                for v in table.values()]
        put("clock.product.gap.pct", 100 * max(gaps), "{:.1f}")
        put("clock.seconds.min", min(v["cpu_seconds"] / v["ranks"] for v in table.values()), "{:.0f}")
        put("clock.seconds.max", max(v["cpu_seconds"] / v["ranks"] for v in table.values()), "{:.0f}")
    full = table.get("packed-P72")
    if one and full:
        import make_figures as mf
        data = mf._strong_tables()
        if data:
            serial_t, strong = data
            sizes = sorted(serial_t)
            for n, tag in ((sizes[0], "small"), (sizes[len(sizes) // 2], "mid"), (sizes[-1], "big")):
                m0 = strong.get(("local", n, 72))
                if m0:
                    # efficiency of the control in the strong-scaling run divided by the clock factor measured here
                    put(f"clock.implied.w.{tag}", m0["efficiency"] / (full["ghz"] / one["ghz"]))
    a, b = table.get("packed-P36"), table.get("spread-P36")
    if a and b:
        put("clock.spread.wratio", a["cycles_per_pair"] / b["cycles_per_pair"])
        put("clock.spread.gain.pct", 100 * (a["time"] / b["time"] - 1), "{:.0f}")


def ringx():
    """One ring step without computation (suite ringx) against the model and the measured rings."""
    import make_figures as mf
    exchange = io.ring_exchange()
    params = mf.model_parameters()
    if not exchange or not params:
        return
    ranks = sorted({p for p, _ in exchange})
    most, least = ranks[-1], ranks[0]
    put("ringx.ranks.max", str(most)); put("ringx.ranks.min", str(least)); put("ringx.ranks.list", ", ".join(str(r) for r in ranks))
    for (p, c), t in exchange.items():
        if c in (16, 128, 512, 4096):
            put(f"ringx.us.P{p}.c{c}", t * 1e6, "{:.1f}" if t * 1e6 < 100 else "{:.0f}")
    for c in (16, 128, 512, 4096):
        if (most, c) in exchange and (least, c) in exchange:
            put(f"ringx.crowd.c{c}", exchange[(most, c)] / exchange[(least, c)], "{:.1f}")
    # the same exchange predicted from the two-rank ping-pong parameters of the model
    for c in (128, 512):
        predicted = params["alpha"] + 512 * c / params["B"]
        put(f"ringx.model.us.c{c}", predicted * 1e6, "{:.1f}")
        if (most, c) in exchange:
            put(f"ringx.over.model.c{c}", exchange[(most, c)] / predicted, "{:.1f}")
        if (least, c) in exchange:
            put(f"ringx.pair.over.model.c{c}", exchange[(least, c)] / predicted, "{:.1f}")
    # effective constants of the exchange of all ranks: least squares of time against rows, 16 <= c <= 512
    pts = sorted((c, t) for (p, c), t in exchange.items() if p == most and 16 <= c <= 512)
    if len(pts) >= 3:
        slope, intercept = np.polyfit([c for c, _ in pts], [t for _, t in pts], 1)
        bandwidth = 512 / slope
        put("ringx.B.gbs", bandwidth / 1e9, "{:.1f}"); put("ringx.alpha.us", max(intercept, 0) * 1e6, "{:.1f}")
        put("ringx.cstar", 4 * params["F"] / (2 * bandwidth), "{:.0f}")
        put("ringx.B.ratio", params["B"] / bandwidth, "{:.1f}")
    # efficiency that the measured exchange alone would give, against the measured rings (thin ranks, one node)
    by = io.summarise(io.passed(io.load("e5-model", "model-thin.csv")), ["impl", "N", "ranks"])
    gaps = []
    for (impl, n, p), v in by.items():
        control = by.get(("local", n, p))
        c = int(n) // int(p)
        if impl != "ring" or not control or (most, c) not in exchange or c < 16:
            continue
        step = control["median"] / int(p)                 # one block of the control
        predicted = step / (step + exchange[(most, c)])
        measured = control["median"] / v["median"]
        put(f"ringx.pred.E.c{c}", predicted); put(f"ringx.meas.E.c{c}", measured)
        gaps.append(measured - predicted)
        # share of the measured loss that the exchange alone accounts for
        if measured < 1:
            put(f"ringx.share.pct.c{c}", 100 * (1 - predicted) / (1 - measured), "{:.0f}")
    if gaps:
        put("ringx.pred.maxdev", max(abs(g) for g in gaps)); put("ringx.pred.n", str(len(gaps)))
        put("ringx.pred.bias", sum(gaps) / len(gaps), "{:+.2f}")


def unit():
    """Unit tests of the submitted source, run on Vanda (scripts/hpc/jobs/unit_tests.pbs)."""
    import re
    import make_tables as mt
    run = io.latest("unit-tests", "unit-tests.csv")
    if not run:
        return
    outcome = io.read(run / "unit-tests.csv")
    put("unit.steps", str(len(outcome))); put("unit.failed", str(sum(r["result"] != "passed" for r in outcome)))
    core = [float(x) for x in re.findall(r"max_abs=([0-9.eE+-]+) ", io.read_text(run / "core-tests.log"))]
    put("unit.core.cases", str(len(core)))
    worst, groups, ranks = max(core), 0, []
    for path in sorted(run.glob("mpi-tests-P*.log")):
        text = io.read_text(path)
        errors = [float(x) for x in re.findall(r"max_abs=([0-9.eE+-]+),", text)]
        groups = max(groups, len(errors)); ranks.append(int(re.search(r"P(\d+)", path.name).group(1)))
        if max(errors) > worst:
            worst = max(errors)
            put("unit.maxerr.case", re.findall(r"^(\w+),.*max_abs=%s," % re.escape(re.findall(r"max_abs=([0-9.eE+-]+),", text)[errors.index(max(errors))]), text, re.M)[0].replace("_", " "))
    N["unit.maxerr"] = mt.sci(worst)
    # the cases with dominant boundary keys print what a wrong boundary key would change, and what they require
    sensitivity = [(float(a), float(b)) for path in sorted(run.glob("mpi-tests-P*.log"))
                   for a, b in re.findall(r",sensitivity,P=\d+,smallest_change=([0-9.eE+-]+),required=([0-9.eE+-]+)", io.read_text(path))]
    if sensitivity:
        smallest = min(a for a, b in sensitivity)
        if smallest < max(b for a, b in sensitivity):
            raise SystemExit("unit tests: a boundary-key case reports a sensitivity below the one it requires")
        put("unit.boundary.min", sig(smallest)); put("unit.boundary.factor", thousands(float(sig(smallest)) / 1e-5))
        put("unit.boundary.cases", str(len(sensitivity)))
    put("unit.margin", 1e-5 / worst, "{:.0f}")
    put("unit.mpi.groups", str(groups)); put("unit.mpi.pmax", str(max(ranks))); put("unit.mpi.pcount", str(len(ranks)))
    env = io.read_text(run / "environment.txt")
    put("unit.tag", re.search(r"tag=(\w+)", env).group(1))


def audit():
    """Figures of the data audit (scripts/audit_results.py), quoted in the set-up and the appendix."""
    path = ROOT / "results" / "audit.json"
    if not path.exists():
        return
    a = json.loads(path.read_text())
    put("audit.runs", str(a["selected_run_directories"])); put("audit.excluded", str(a["excluded_run_directories"]))
    put("audit.validated", thousands(a["repetitions_validated"])); put("audit.failed", str(a["repetitions_failed"]))
    for tag, key in (("long", "at_least_50ms"), ("short", "below_50ms"), ("gpu.long", "gpu_at_least_50ms"), ("gpu.short", "gpu_below_50ms")):
        r = a["repeatability"][key]
        if r:
            put(f"rep.{tag}.n", str(r["configurations"])); put(f"rep.{tag}.median.pct", 100 * r["median"], "{:.1f}")
            put(f"rep.{tag}.p90.pct", 100 * r["p90"], "{:.0f}" if 100 * r["p90"] >= 10 else "{:.1f}")
            put(f"rep.{tag}.above10.pct", 100 * r["share_above_10_percent"], "{:.0f}")
    files = sum(len(e["files"]) for e in a["runs"].values())
    put("audit.files", str(files))
    driver = a["sources"].get("cpu", {}).get("files", {}).get("apps/attn_bench.cpp", {}).get("changed_after_measurement")
    if driver:
        put("audit.driver.commands", thousands(sum(r["commands"] for r in driver["records_checked"].values())))
        put("audit.driver.runs", str(len(driver["records_checked"])))
    checked = a["sources"].get("analysis", {}).get("files", {}).get("apps/attn_analyze.cpp", {}).get("changed_after_measurement")
    if checked:
        records = list(checked["records_checked"].values())[0]
        put("audit.e11.records", thousands(records["records"]))
        put("audit.e11.minmass", __import__("make_tables").sci(records["smallest_window_probability"]))
    external = a.get("external_reference")
    if external and external["records"]:
        put("audit.external.records", str(external["records"]))
        put("audit.external.max", __import__("make_tables").sci(external["largest_difference_over_scale"]))
    gpu_driver = a["sources"].get("gpu", {}).get("files", {}).get("apps/attn_gpu_bench.cpp", {}).get("changed_after_measurement")
    if gpu_driver:
        versions = gpu_driver["measured_versions"].values()
        put("audit.gpudriver.versions", str(len(versions)))
        put("audit.gpudriver.restored", str(sum("restored_byte_for_byte_by_undoing" in v for v in versions)))
        put("audit.gpudriver.commands", str(sum(r["commands"] for r in gpu_driver["records_checked"].values())))


SECTIONS = [serial, hardware, ramp, align, strong, model, net, weak, hybrid, causal, memory, gqa, smallc, traces, gpu, real, errors, clock, ringx, unit, audit]


def write():
    lines = ["% generated by scripts/make_numbers.py -- do not edit",
             r"\makeatletter",
             r"\newcommand{\n}[1]{\@ifundefined{num@#1}{\textbf{??#1??}}{\@nameuse{num@#1}}}"]
    for key in sorted(N):
        lines.append(rf"\@namedef{{num@{key}}}{{{N[key]}}}")
    lines.append(r"\makeatother")
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n")
    (GEN / "numbers.json").write_text(json.dumps(N, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    origin = {}
    for section in SECTIONS:
        before = set(N)
        io.trace_reset()
        section()
        files = io.provenance()
        if section.__name__ == "real":           # also uses the statistics that the real-data figures computed
            previous = json.loads(io.PROVENANCE.read_text()).get("figures", {}) if io.PROVENANCE.exists() else {}
            files = sorted(set(files) | set(previous.get("real_stats", [])) | set(previous.get("real_heads", [])))
        io.record_provenance("number_groups", section.__name__, files)
        for key in set(N) - before:
            origin[key] = section.__name__
    write()
    (GEN / "number_origin.json").write_text(json.dumps(origin, indent=1, sort_keys=True) + "\n")
    print(f"{len(N)} numbers written; sections without data yet: {sorted(set(k.split('.')[0] for k in MISSING))}")
    if "--strict" in sys.argv and MISSING:
        raise SystemExit(f"numbers without data: {sorted(MISSING)[:20]}")
    if "-v" in sys.argv:
        for key in sorted(N):
            print(f"  {key} = {N[key]}")
