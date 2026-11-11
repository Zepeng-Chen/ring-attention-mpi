"""Read benchmark CSVs retrieved from the cluster and reduce repetitions.

Every configuration was run with one warm-up and three measured repetitions.
The reported time is the median over repetitions of the per-repetition maximum
over ranks (the CSV already stores the rank maximum of each repetition).
"""
import csv
import json
import os
import statistics
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "hpc" / "vanda"
SELECTION = ROOT / "results" / "selected_runs.json"

# Every file handed to the figure, table and number scripts is noted here, so that
# each output can name the records it was computed from (see provenance()).
_TRACE = []


def trace_reset():
    del _TRACE[:]


def provenance():
    """Run-relative paths of the result files read since the last trace_reset()."""
    return sorted(set(_TRACE))


def _note(path):
    try:
        _TRACE.append(str(Path(path).resolve().relative_to(RUNS)))
    except ValueError:
        _TRACE.append(str(path))


PROVENANCE = ROOT / "report" / "generated" / "provenance.json"


def record_provenance(kind, name, files=None):
    """Store which result files an output (a figure, a table, a group of numbers) was computed from."""
    data = json.loads(PROVENANCE.read_text()) if PROVENANCE.exists() else {}
    data.setdefault(kind, {})[name] = provenance() if files is None else sorted(set(files))
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


def selected_runs():
    """experiment key -> run directory name, from results/selected_runs.json.
    The selection is explicit: no script picks "the newest run" by itself."""
    return json.loads(SELECTION.read_text())["runs"] if SELECTION.exists() else None


def preview():
    """True in a draft session that looks at a job which is still running. The audit
    and the report build refuse to run in this mode."""
    return bool(os.environ.get("ATTN_PREVIEW"))


def latest(prefix, required=None):
    """Run directory selected for the experiment `prefix` (None if it has no data
    or lacks the file `required`)."""
    chosen = selected_runs()
    if chosen is None or preview():
        return _newest(prefix, required)          # development only: no selection file, or previewing a running job
    name = chosen.get(prefix)
    if not name:
        return None
    path = RUNS / name
    if not path.is_dir() or (required and not (path / required).exists()):
        return None
    return path


def _newest(prefix, required=None):
    """Development helper: newest finished, not superseded run whose name starts with `prefix`."""
    candidates = sorted(p for p in RUNS.glob(prefix + "-2*") if p.is_dir() and not (p / "SUPERSEDED").exists())
    for path in reversed(candidates):
        status = path / "exit-status.txt"
        finished = status.exists() and status.read_text().strip() == "0"
        if not finished and not (os.environ.get("ATTN_PREVIEW") and not status.exists()):
            continue
        if required and not (path / required).exists():
            continue
        return path
    return None


# The benchmark driver writes the fetch order of the one-sided pull as the value of
# its enumeration (include/attn/distributed.hpp). It is decoded here, once, so that
# no script compares against the raw code; an unknown value stops the build.
FETCH_ORDER = {"0": "staggered", "1": "ascending", "2": "reverse",
               "staggered": "staggered", "ascending": "ascending", "reverse": "reverse"}


def read(path):
    if path is None or not Path(path).exists():
        return []
    import io
    _note(path)
    text = Path(path).read_bytes().replace(b"\0", b"").decode()
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    for row in rows:
        if "fetch_order" in row:
            if row["fetch_order"] not in FETCH_ORDER:
                raise ValueError(f"{path}: unknown fetch_order {row['fetch_order']!r}")
            row["fetch_order"] = FETCH_ORDER[row["fetch_order"]]
    return rows


def read_text(path):
    """Text of a non-CSV record (sampled frequencies, counter output), traced like read()."""
    _note(path)
    return Path(path).read_text(errors="replace")


def load(prefix, name):
    run = latest(prefix, name)
    return read(run / name) if run else []


def load_all(prefixes, name):
    """Rows of `name` from the latest run of each prefix, concatenated."""
    rows = []
    for prefix in prefixes:
        rows += load(prefix, name)
    return rows


# Configurations that fit on one node are measured in a one-node job and larger
# ones in a four-node job whose first node is the same machine (credits are
# charged per allocated core). SINGLE/MULTI name the two jobs of such a pair.
MULTI = ["e3-single", "e3-multi"]


def strong_rows():
    """Strong-scaling rows: up to one node from e2-strong, beyond from e2-strongx.
    Every row gets a "job" field ("one" or "multi"); a speedup is always formed
    with the serial time measured in the same job."""
    one = passed(load("e2-strong", "strong.csv"))
    if not one:
        return []
    limit = max(int(r["ranks"]) for r in one)
    for r in one:
        r["job"] = "one"
    many = [r for r in passed(load("e2-strongx", "strong.csv")) if int(r["ranks"]) > limit or r["impl"] == "S1"]
    for r in many:
        r["job"] = "multi"
    return one + many


def strong_table():
    """(serial, table): serial[N] is the one-node job's S1 time; table[(impl, N, ranks)]
    holds time, speedup and efficiency, each formed with the serial time of the job
    that measured the row."""
    rows = strong_rows()
    if not rows:
        return None
    by = summarise(rows, ["impl", "N", "ranks", "job"])
    own = {(k[3], int(k[1])): v["median"] for k, v in by.items() if k[0] == "S1"}
    serial = {n: t for (job, n), t in own.items() if job == "one"}
    table = {}
    for (impl, n, ranks, job), v in by.items():
        if impl == "S1":
            continue
        n, ranks = int(n), int(ranks)
        if n not in serial:
            continue
        reps = v["rows"]
        if (job, n) not in own:
            raise ValueError(f"strong scaling: no serial baseline for N={n} in the {job}-node job; "
                             "a speedup is only formed with the serial time of the same job")
        base = own[(job, n)]
        table[(impl, n, ranks)] = {
            "time": v["median"], "low": v["low"], "high": v["high"], "speedup": base / v["median"],
            "efficiency": base / v["median"] / ranks, "nodes": int(v["row"]["nodes"]), "serial": base, "job": job,
            "wait_max": med(reps, "t_wait_max"), "wait_mean": med(reps, "t_wait_mean"),
            "compute_max": med(reps, "t_compute_max"), "compute_mean": med(reps, "t_compute_mean"),
        }
    return serial, table


def serial_time(n, cores=1):
    """S1 time for sequence length n from the job that measures `cores` cores."""
    rows = [r for r in strong_rows() if r["impl"] == "S1" and int(r["N"]) == int(n)]
    one = [float(r["t_total_max"]) for r in rows if r["job"] == "one"]
    many = [float(r["t_total_max"]) for r in rows if r["job"] == "multi"]
    limit = max([int(r["ranks"]) for r in strong_rows() if r["job"] == "one"], default=0)
    pick = many if (cores > limit and many) else one
    return statistics.median(pick) if pick else None


def strong_anchor():
    """(impl, N, ranks) -> (time in the one-node job, time in the four-node job) for the
    configurations measured in both; checks that the two jobs are consistent."""
    one = summarise(passed(load("e2-strong", "strong.csv")), ["impl", "N", "ranks"])
    many = summarise(passed(load("e2-strongx", "strong.csv")), ["impl", "N", "ranks"])
    return {k: (one[k]["median"], many[k]["median"]) for k in one if k in many}


def rank_files(prefixes, pattern):
    """Per-rank CSV files matching `pattern` in the first of the selected runs that has any."""
    for prefix in prefixes:
        run = latest(prefix)
        if run:
            files = sorted((run / "ranks").glob(pattern))
            if files:
                return files
    return []


def group(rows, keys):
    """OrderedDict: key tuple -> list of rows (repetitions)."""
    result = OrderedDict()
    for row in rows:
        result.setdefault(tuple(row[k] for k in keys), []).append(row)
    return result


def summarise(rows, keys, value="t_total_max"):
    """key -> dict(median, low, high, n, row) for one numeric column.
    That every timed configuration has its complete set of valid repetitions is a
    rule of scripts/audit_results.py, which runs before anything is summarised."""
    out = OrderedDict()
    for key, reps in group(rows, keys).items():
        values = sorted(float(r[value]) for r in reps)
        if not all(v == v and abs(v) != float("inf") for v in values):
            raise ValueError(f"non-finite {value} in configuration {key}")
        out[key] = {"median": statistics.median(values), "low": values[0], "high": values[-1],
                    "n": len(values), "row": reps[0], "rows": reps}
    return out


def med(reps, column):
    return statistics.median(float(r[column]) for r in reps)


def passed(rows):
    """Rows whose validation did not fail (measurement runs without validation are kept).
    The audit allows failed repetitions only in the declared stress study, so in a
    performance file this filter removes nothing: no configuration is represented
    by a subset of its repetitions."""
    return [r for r in rows if r.get("validation") != "failed"]


def clock_table():
    """label -> clock and work per cycle of the control M0, from the perf records of suite clock.
    label is "packed-P<ranks>" or "spread-P<ranks>". Every rank ran under `perf stat`:
      ghz              cycles / CPU time, summed over the ranks (the clock they ran at, averaged over the run)
      cycles_per_pair  cycles / query-key pairs of all repetitions, warm-up included
      ipc              instructions / cycles
      time             median time of a measured repetition (the same runs)."""
    import re
    run = latest("e8-clock", "clock.csv")
    if not run:
        return {}
    by = summarise(read(run / "clock.csv"), ["purpose"])
    table = OrderedDict()
    for (purpose,), v in by.items():
        label = purpose.replace("final-clock-", "")
        row = v["row"]
        warm = int(re.search(r"--warmup (\d+)", row["command"]).group(1))
        reps = int(re.search(r"--reps (\d+)", row["command"]).group(1))
        if v["n"] != reps:
            raise ValueError(f"{label}: {v['n']} measured repetitions recorded, {reps} requested")
        ranks, n = int(row["ranks"]), int(row["N"])
        cycles = instructions = seconds = 0.0
        files = sorted((run / "clock").glob(f"perf-{label}.rank*.txt"))
        if len(files) != ranks:
            raise ValueError(f"{label}: {len(files)} perf records for {ranks} ranks")
        for path in files:
            for line in read_text(path).splitlines():
                parts = line.split(",")
                if len(parts) < 3 or line.startswith("#"):
                    continue
                if parts[2].startswith("cycles"):
                    cycles += float(parts[0])
                elif parts[2].startswith("instructions"):
                    instructions += float(parts[0])
                elif parts[2].startswith("task-clock"):
                    seconds += float(parts[0]) / 1e3
        pairs = float(warm + reps) * n * n
        table[label] = {"ranks": ranks, "N": n, "warmups": warm, "repetitions": reps, "ghz": cycles / seconds / 1e9,
                        "cycles_per_pair": cycles / pairs, "ipc": instructions / cycles, "time": v["median"],
                        "cpu_seconds": seconds, "node": row.get("cluster", "")}
    return table


def ring_exchange():
    """(ranks, rows per rank) -> seconds for one ring step without computation (maximum over ranks)."""
    rows = load("e8-clock", "ring-exchange.csv")
    return OrderedDict(((int(r["ranks"]), int(r["rows"])), float(r["seconds_per_step_max"])) for r in rows)
