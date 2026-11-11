#!/usr/bin/env python3
"""LaTeX tables of the report, generated from the retrieved cluster CSVs.

Writes report/generated/*.tex. Prose is written by hand; only tables whose
every cell is a measurement are generated here, so that the report cannot
drift from the data.
"""
import re
import statistics
from collections import OrderedDict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results_io as io

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "report" / "generated"


def sci(x, digits=1):
    """1.23e-06 -> 1.2\\times10^{-6} (math mode)."""
    if x == 0:
        return "0"
    mantissa, exponent = f"{x:.{digits}e}".split("e")
    return f"{mantissa}\\times10^{{{int(exponent)}}}"


def seconds(t):
    if t >= 100:
        return f"{t:.0f}"
    if t >= 10:
        return f"{t:.1f}"
    if t >= 1:
        return f"{t:.2f}"
    return f"{t:.3f}"


def table_strong():
    data = io.strong_table()
    if not data:
        return False
    serial, table = data
    sizes = sorted(serial)
    ranks = sorted({k[2] for k in table})
    lines = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{4.2pt}",
             r"\begin{tabular}{@{}rr" + "rrr" * len(sizes) + r"@{}}", r"\toprule",
             " & & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{$N={n:,}$}}".replace(",", "{,}") for n in sizes) + r"\\",
             "".join(rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(sizes))),
             r"cores & nodes & " + " & ".join([r"$T$ (s) & $S$ & $E$"] * len(sizes)) + r"\\", r"\midrule",
             r"1 (serial S1) & 1 & " + " & ".join(f"{seconds(serial[n])} & 1.0 & 1.00" for n in sizes) + r"\\"]
    for P in ranks:
        if P == 1:
            continue
        cells, nodes = [], "?"
        for n in sizes:
            v = table.get(("overlap", n, P))
            if v:
                cells.append(f"{seconds(v['time'])} & {v['speedup']:.1f} & {v['efficiency']:.2f}")
                nodes = v["nodes"]
            else:
                cells.append("-- & -- & --")
        lines.append(f"{P} & {nodes} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Strong scaling of the overlapped ring (M3), one thread per rank. $T$: median of three repetitions of the "
              r"slowest rank's time; $S=T_S/T$ with $T_S$ the single-core tuned serial time measured in the same job on the "
              r"first node; $E=S/\text{cores}$.}",
              r"\label{tab:strong}", r"\end{table}"]
    (GEN / "table_strong.tex").write_text("\n".join(lines) + "\n")
    return True


def table_v7():
    rows = io.load("real-data", "v7.csv")
    gpu = io.passed(io.load("gpu-ladder", "real.csv"))
    if not rows:
        return False
    by = io.summarise(rows, ["purpose"])
    gpu_by = io.summarise(gpu, ["N", "purpose"])
    lines = [r"\begin{table}[t]\centering\small", r"\begin{tabular}{@{}rrrrrrr@{}}", r"\toprule",
             r"$N$ & layer & $\max|O|$ & CPU ring vs.\ fp64 & CPU ring vs.\ PyTorch & GPU K2 vs.\ fp64 & status\\", r"\midrule"]
    for (purpose,), v in by.items():
        n, layer = purpose.split("-")[1][1:], purpose.split("-")[2][5:]
        r = v["row"]
        # One scale for the whole row: the largest component of the model's own output over all rows and heads.
        scale = float(r["oref_abs_max"])
        g = gpu_by.get((n, f"final-real-layer{layer}"))
        gpu_err = f"${sci(max(float(x['max_abs_err']) for x in g['rows']) / scale)}$" if g else "--"
        ok = all(x["validation"] == "passed" for x in v["rows"]) and (not g or all(x["validation"] == "passed" for x in g["rows"]))
        lines.append(f"{int(n):,} & {int(layer)} & {scale:.2f} & ${sci(max(float(x['max_abs_err']) for x in v['rows']) / scale)}$ & "
                     f"${sci(max(float(x['oref_max_abs_err']) for x in v['rows']) / scale)}$ & {gpu_err} & {'pass' if ok else 'FAIL'}\\\\".replace(",", "{,}"))
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Real tensors of Qwen2.5-0.5B (14 query heads, 2 key/value heads, causal mask). Every entry is a largest "
              r"absolute difference divided by $\max|O|$, the largest component of the model's own attention output over all rows "
              r"and heads. ``fp64'': against my double-precision reference on 512 sampled rows (CPU) or 64 sampled rows (GPU); "
              r"``PyTorch'': against the model's output on all rows. CPU: 8 ranks $\times$ 9 threads, overlapped ring, zigzag "
              r"partition. ``pass'' refers to the automatic check of the runs: difference to fp64 at most $10^{-3}$ times the largest "
              r"reference component of the sampled rows.}", r"\label{tab:v7}", r"\end{table}"]
    (GEN / "table_v7.tex").write_text("\n".join(lines) + "\n")
    return True


def perf_counters(path):
    """Counter values of one `perf stat -o` file: name -> count."""
    out = {}
    if not path.exists():
        return out
    for line in io.read_text(path).splitlines():
        m = re.match(r"\s*([\d,]+)\s+([A-Za-z_.0-9]+?)(?::u)?\s", line)
        if m:
            out[m.group(2).lower()] = int(m.group(1).replace(",", ""))
    return out


def align_data():
    """Buffer-alignment comparison: per alignment the serial time, the counters of a
    fixed amount of work and the spread of parallel efficiency over the variants."""
    run = io.latest("e0-align", "align.csv")
    if not run:
        return None
    by = io.summarise(io.passed(io.read(run / "align.csv")), ["purpose", "impl", "ranks"])
    perf_rows = io.summarise(io.read(run / "align-perf.csv"), ["purpose"])
    data = {}
    for align in (16, 64, 4096):
        serial = by.get((f"align-{align}", "S1", "1"))
        if not serial:
            continue
        eff = {}
        for (purpose, impl, ranks), v in by.items():
            if purpose == f"align-{align}" and impl != "S1":
                eff[(impl, int(ranks))] = serial["median"] / v["median"] / int(ranks)
        counters = perf_counters(run / f"perf-a{align}.txt")
        fixed = perf_rows.get((f"align-perf-{align}",))
        # one warm-up and three measured repetitions of N^2 pairs each
        pairs = 4 * float(fixed["row"]["N"]) ** 2 if fixed else None
        data[align] = {"time": serial["median"], "gflops": float(serial["row"]["useful_flops"]) / serial["median"] / 1e9,
                       "eff": eff, "counters": counters, "pairs": pairs}
    return data


def table_align():
    data = align_data()
    if not data:
        return False
    lines = [r"\begin{table}[t]\centering\small", r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Alignment of every buffer & $T_S$ (s) & GFLOP/s & instructions & blocked split loads & efficiency of M0, M2, M3\\",
             r" & & & per cycle & per $10^3$ pairs & at $P=1,2,4,8$ (min--max)\\", r"\midrule"]
    names = {16: r"16 bytes (\code{malloc})", 64: "64 bytes (cache line)", 4096: "4096 bytes (page), used"}
    for align, v in data.items():
        c = v["counters"]
        ipc = f"{c['instructions'] / c['cycles']:.2f}" if c.get("cycles") else "--"
        blocked = "--"
        if "ld_blocks.no_sr" in c and v["pairs"]:
            x = 1e3 * c["ld_blocks.no_sr"] / v["pairs"]
            blocked = f"{x:.0f}" if x >= 10 else (f"{x:.1f}" if x >= 0.1 else f"${sci(x)}$")
        eff = v["eff"].values()
        spread = f"{min(eff):.2f}--{max(eff):.2f}" if eff else "--"
        lines.append(f"{names[align]} & {v['time']:.2f} & {v['gflops']:.1f} & {ipc} & {blocked} & {spread}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{The same source built with three alignments of its tensors and scratch buffers ($N=32{,}768$, one core for "
              r"$T_S$). Counters: \code{perf stat} over one warm-up and three repetitions at $N=16{,}384$; a ``blocked split load'' "
              r"is a load that crosses a cache-line boundary and has to wait for the processor's split-access resources "
              r"(\code{ld\_blocks.no\_sr}). Efficiency: $T_S/(P\,T)$ over the variants M0, M2 and M3 and $P\in\{1,2,4,8\}$, all on "
              r"one socket at the same clock, where the ideal value is one.}",
              r"\label{tab:align}", r"\end{table}"]
    (GEN / "table_align.tex").write_text("\n".join(lines) + "\n")
    return True


JOBS = [  # (run prefix, what the job measured)
    ("e0-align", "buffer alignment and hardware counters (\\cref{tab:align})"),
    ("e1-serial", "serial scaling, kernel ladder, tile sizes, vector width, error (\\cref{sec:serial})"),
    ("e1-extras", "clock ramp; S1 against one rank of M0 and M2"),
    ("e2-strong", "strong scaling up to 72 cores"),
    ("e2-strongx", "strong scaling, 72 to 288 cores"),
    ("e3-single", "weak scaling, ranks $\\times$ threads, causal mask, GQA up to 72 cores"),
    ("e3-multi", "the same at 288 cores; microbenchmark, few rows per rank, traces"),
    ("e5-model", "few rows per rank inside one node (model test)"),
    ("e7-hardware", "no-communication control and clock frequency"),
    ("e7-memory", "peak memory"),
    ("real-data", "real tensors: validation against PyTorch; GQA"),
    ("real-e11", "real tensors: attention statistics"),
    ("gpu-ladder", "one GPU: kernels K0--K2, causal mask, real tensors"),
    ("gpu-ringcheck", "MPI+CUDA ring, two ranks sharing one GPU (correctness)"),
    ("gpu-ring-1node", "two GPUs in one node"),
    ("gpu-ring-2node", "two GPUs in two nodes"),
]


def table_jobs():
    """One line per cluster job behind the report, from the record each job writes."""
    import datetime
    parse = lambda s: datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
    jobs = OrderedDict()
    for prefix, what in JOBS:
        run = io.latest(prefix)
        if not run or not (run / "environment.txt").exists():
            continue
        env = (run / "environment.txt").read_text(errors="replace")
        job = re.search(r"job=(\d+)", env)
        key = job.group(1) if job else run.name
        if key in jobs:                      # several experiments ran in the same job
            jobs[key]["what"].append(what)
            continue
        field = lambda name: (re.search(rf"Resource_List\.{name} = (\S+)", env) or [None, "?"])[1]
        cpus, gpus, nodes = field("ncpus"), field("ngpus"), field("nodect")
        end = (run / "finished-utc.txt").read_text().strip() if (run / "finished-utc.txt").exists() else None
        minutes = "--"
        if end:
            # the scheduler creates job.id at submission; the job log's first lines are written at start
            starts = [parse(env.splitlines()[0].strip())]
            logs = sorted(run.glob("*.o*"))
            first = re.search(r"=== suite \S+ start (\d\d):(\d\d):(\d\d)", logs[0].read_text(errors="replace")) if logs else None
            if first:
                t0 = parse(end).replace(hour=int(first.group(1)), minute=int(first.group(2)), second=int(first.group(3)))
                if t0 > parse(end):
                    t0 -= datetime.timedelta(days=1)
                starts.append(t0)
            minutes = f"{(parse(end) - min(starts)).total_seconds() / 60:.0f}"
        resources = f"{cpus} cores" + (f", {gpus} GPU{'s' if gpus != '1' else ''}" if gpus not in ("0", "?") else "")
        jobs[key] = {"nodes": nodes, "resources": resources, "minutes": minutes, "what": [what]}
    lines = [r"\begin{table}[h]\centering\small", r"\begin{tabularx}{\textwidth}{@{}l r l r Y@{}}", r"\toprule",
             r"PBS job & nodes & resources & minutes & measured\\", r"\midrule"]
    for key, j in jobs.items():
        lines.append(f"{key} & {j['nodes']} & {j['resources']} & {j['minutes']} & {'; '.join(j['what'])}\\\\")
    lines += [r"\bottomrule", r"\end{tabularx}",
              r"\caption{The Vanda jobs behind the results of this report, from the record that each job writes when it starts "
              r"(job identifier, allocated nodes, modules, compiler) and when it ends. Minutes: from the first experiment of the "
              r"job to its end. Cores are those allocated to the job; a GPU job is allocated 36 cores per GPU.}",
              r"\label{tab:jobs}", r"\end{table}"]
    (GEN / "table_jobs.tex").write_text("\n".join(lines) + "\n")
    return len(jobs)


def validation_summary():
    """Count validated repetitions over every formal run directory."""
    total = failed = 0
    worst = 0.0
    files = 0
    chosen = io.selected_runs() or {}
    for run in sorted({io.RUNS / name for name in chosen.values()}):
        if not (run / "exit-status.txt").exists():
            continue
        for path in sorted(run.glob("*.csv")):
            rows = io.read(path)
            if not rows or "validation" not in rows[0]:
                continue
            files += 1
            for r in rows:
                if r["validation"] == "not_requested":
                    continue
                total += 1
                failed += r["validation"] != "passed"
                if "tolerance_scale" not in r and float(r.get("atol", 1e-5) or 1e-5) <= 1e-5 + 1e-12:
                    worst = max(worst, float(r["max_abs_err"]))
    text = (f"% generated by scripts/make_tables.py\n\\newcommand{{\\ValidatedReps}}{{{total:,}}}\n"
            f"\\newcommand{{\\ValidatedFailed}}{{{failed}}}\n\\newcommand{{\\ValidatedFiles}}{{{files}}}\n"
            f"\\newcommand{{\\ValidatedWorst}}{{{sci(worst)}}}\n").replace(",", "{,}")
    (GEN / "validation_numbers.tex").write_text(text)
    return total, failed, worst, files


def traced(name, function):
    """Run one table generator and record the result files it read."""
    io.trace_reset()
    result = function()
    if result:
        io.record_provenance("tables", name)
    return result


if __name__ == "__main__":
    GEN.mkdir(parents=True, exist_ok=True)
    print("strong table:", traced("table_strong", table_strong))
    print("v7 table:", traced("table_v7", table_v7))
    print("alignment table:", traced("table_align", table_align))
    print("validation (repetitions, failed, worst abs error on synthetic, files):", traced("validation_numbers", validation_summary))
