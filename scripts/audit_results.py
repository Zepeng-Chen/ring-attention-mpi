#!/usr/bin/env python3
"""Audit the result records behind the report.

The figures, the tables and every number in the text are computed from the run
directories named in results/selected_runs.json. This script checks those
records and writes what it found:

  results/audit.json                       machine-readable result of every check
  report/generated/table_provenance.tex    experiment -> job -> files -> outputs

It exits non-zero if a hard rule is broken, so that a report cannot be built
from inconsistent data (scripts/build_report.sh runs it first).

Hard rules
  1. every run directory is either selected or excluded with a reason;
  2. a selected run ended with status 0, is not superseded, and ran on Vanda;
  3. a validated repetition may fail only in a file listed in DECLARED_FAILURES;
  4. every timed configuration has exactly three repetitions, numbered 0, 1, 2, with
     finite positive times, and CPU runs warm up for at least 5 s, unless the file
     is listed in PROTOCOL_EXCEPTIONS or REPEATED_ROUNDS with the reason;
  5. all CPU measurements were built from identical compute and driver sources,
     all GPU measurements from identical kernel sources, and the tree being
     submitted contains exactly those sources;
  6. the main CPU build used 512-bit vectors and 4096-byte alignment; tile sizes
     are the stated ones outside the declared sweeps;
  7. a source that was changed after its measurement is listed in
     POST_MEASUREMENT_CHANGES, and the condition under which the stored records
     are unaffected is checked on those records;
  8. the unit tests ran on the cluster on exactly the staged files of the submitted
     tree, and every one passed;
  9. the audit is never run on a preview of unfinished jobs (ATTN_PREVIEW);
 10. the failure log of a selected run lists only declared failures, and every result
     file still holds the number of configurations pinned when the run was selected
     (audit_results.py --pin).
Soft findings (recorded, not fatal): configurations whose three repetitions
differ by more than 10 %.
"""
import hashlib
import json
import re
import statistics
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results_io as io

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "report" / "generated"
HASHES = ROOT / "results" / "source_hashes"

# Files in which validated repetitions are allowed to exceed the tolerance, and why.
DECLARED_FAILURES = {
    "serial-error.csv": "stress study: queries multiplied by 50 and every row compared with fp64; "
                        "the single-precision scores exceed the fixed 1e-5 tolerance for N >= 4096 (reported as failures)",
}
# A job records every invocation that returned an error in failures.txt and carries on; its
# exit status does not show them. A selected run may list only these invocations there.
DECLARED_FAILURE_LINES = (r"^FAILED: serial-error .* --alpha 50$",)
# Files whose rows deliberately do not follow the timing protocol, and why.
PROTOCOL_EXCEPTIONS = {
    "memory.csv": "peak-memory runs: one repetition, no warm-up, no validation copy",
    "ramp.csv": "400 consecutive repetitions without warm-up: the object of the measurement is the warm-up itself",
    "align-perf.csv": "fixed amount of work under perf stat: one warm-up and three repetitions, no time-based warm-up",
    "netbench.csv": "microbenchmark with its own repetition scheme", "netbench-node.csv": "microbenchmark with its own repetition scheme",
    "gpu-validation.csv": "GPU unit tests: every query row compared with fp64, two repetitions without warm-up; correctness only",
    "clock.csv": "fixed work under perf stat: fixed numbers of warm-up and measured repetitions per rank count, no time-based warm-up",
}
# Files that hold several rounds of the same configurations on purpose.
REPEATED_ROUNDS = {
    "baseline.csv": (3, "three alternating rounds of the serial driver, one rank of M0 and one rank of M2"),
}
# Sources that changed after the records computed with them were taken. The stored
# records stay valid only under the stated condition, which rule 7 checks.
POST_MEASUREMENT_CHANGES = {
    "apps/attn_analyze.cpp": {
        "change": "two edge cases handled (a window whose weights all underflow relative to the row maximum; an exact output "
                  "equal to zero), absolute errors added as columns, MPI_THREAD_FUNNELED requested",
        "condition": "every field finite, every output norm positive and every window probability at least 1e-280: "
                     "the changed branches are then not reached and the arithmetic of the other branch is unchanged",
        "check": "analysis",
    },
    "CMakeLists.txt": {
        "change": "a target for the ring-exchange microbenchmark (apps/attn_ringbench.cpp) was added",
        "condition": "removing the three added lines restores the measured file byte for byte; no existing target is touched",
        "check": None,
        "revert": [
            ("    add_executable(attn_ringbench apps/attn_ringbench.cpp)\n"
             "    target_link_libraries(attn_ringbench PRIVATE MPI::MPI_CXX)\n"
             "    target_compile_options(attn_ringbench PRIVATE -Wall -Wextra -Wpedantic)\n", ""),
        ],
    },
    "apps/attn_bench.cpp": {
        "change": "the options --warmup-seconds and --tolerance-scale reject infinite values",
        "condition": "undoing the two changed conditions restores the measured file byte for byte, and every recorded command "
                     "gives both options a finite value, for which the old and the new condition are the same",
        "check": "commands",
        # (text in the submitted file, text in the measured file)
        "revert": [
            ('!std::isfinite(o.warmup_seconds) || o.warmup_seconds < 0) throw std::invalid_argument("invalid warm-up time");',
             '!(o.warmup_seconds >= 0)) throw std::invalid_argument("invalid warm-up time");'),
            ('!std::isfinite(o.tolerance_scale) || o.tolerance_scale < 0) throw std::invalid_argument("invalid tolerance scale");',
             '!(o.tolerance_scale >= 0)) throw std::invalid_argument("invalid tolerance scale");'),
        ],
    },
}
# Sweeps in which the tile sizes are the variable.
TILE_SWEEPS = ("serial-tiles.csv", "serial-w256.csv", "serial-error.csv")
TILES = ("128", "96")
CPU_SOURCES = ("include/attn/core.hpp", "include/attn/distributed.hpp", "src/core.cpp", "src/distributed.cpp",
               "apps/attn_bench.cpp", "apps/attn_netbench.cpp", "CMakeLists.txt")
GPU_SOURCES = ("include/attn/gpu.hpp", "src/gpu_kernels.cu")
ANALYSIS_SOURCES = ("apps/attn_analyze.cpp",)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(tag):
    path = HASHES / f"{tag}.sha256"
    if not path.exists():
        return None
    return {line.split(None, 1)[1].strip(): line.split()[0] for line in path.read_text().splitlines() if line.strip()}


def kind_of(run):
    if run.startswith("gpu-"):
        return "gpu"
    if run.startswith("real-e11"):
        return "analysis"
    if run.startswith("unit-tests"):
        return "tests"
    return "cpu"


def recorded_commands(path):
    """Checks the condition of POST_MEASUREMENT_CHANGES["apps/attn_bench.cpp"]: the values that the recorded
    command lines give to the two options whose checks were tightened."""
    commands, bad = set(), []
    for csv_path in sorted(path.glob("*.csv")):
        for row in io.read(csv_path):
            if "command" in row and "attn_bench" in row["command"]:
                commands.add(row["command"])
    for command in commands:
        tokens = command.split()
        for option in ("--warmup-seconds", "--tolerance-scale"):
            for i, token in enumerate(tokens[:-1]):
                if token == option:
                    try:
                        value = float(tokens[i + 1])
                    except ValueError:
                        value = float("nan")
                    if not (value == value and abs(value) != float("inf")):
                        bad.append(f"{option} {tokens[i + 1]}")
    return bool(commands) and not bad, {"commands": len(commands), "not_finite": bad[:5]}


def analysis_records(path):
    """Checks the condition of POST_MEASUREMENT_CHANGES["apps/attn_analyze.cpp"] on the stored window statistics."""
    records = fields = 0
    smallest_mass, smallest_norm, bad = float("inf"), float("inf"), []
    for csv_path in sorted(path.glob("*.csv")):
        for row in io.read(csv_path):
            if "output_norm" not in row:
                break
            records += 1
            for name, text in row.items():
                if name == "label":
                    continue
                value = float(text)
                fields += 1
                if value != value or abs(value) == float("inf"):
                    bad.append(f"{csv_path.name}: non-finite {name}")
                if name.startswith("mass_w"):
                    smallest_mass = min(smallest_mass, value)
            smallest_norm = min(smallest_norm, float(row["output_norm"]))
    ok = records > 0 and not bad and smallest_norm > 0 and smallest_mass >= 1e-280
    return ok, {"records": records, "fields": fields, "smallest_window_probability": smallest_mass,
                "smallest_output_norm": smallest_norm, "non_finite": bad[:5]}


def audit():
    problems, notes = [], []
    if io.preview():
        problems.append("ATTN_PREVIEW is set: the audit only accepts the explicit selection of finished runs")
    selection = json.loads(io.SELECTION.read_text())
    chosen, excluded = selection["runs"], selection["excluded_runs"]
    selected_dirs = sorted(set(chosen.values()))

    # 1. completeness of the selection
    for run in sorted(p.name for p in io.RUNS.iterdir() if p.is_dir()):
        if run not in selected_dirs and run not in excluded:
            problems.append(f"{run}: neither selected nor excluded")
        if run in selected_dirs and run in excluded:
            problems.append(f"{run}: both selected and excluded")
    for run, reason in excluded.items():
        if not reason or reason.startswith("UNCLASSIFIED"):
            problems.append(f"{run}: excluded without a reason")

    report = OrderedDict()
    spread, every_spread = [], []
    for run in selected_dirs:
        path = io.RUNS / run
        entry = OrderedDict(experiments=sorted(k for k, v in chosen.items() if v == run), kind=kind_of(run))
        status = (path / "exit-status.txt").read_text().strip() if (path / "exit-status.txt").exists() else None
        entry["exit_status"] = status
        if status != "0":
            problems.append(f"{run}: exit status {status}")
        if (path / "SUPERSEDED").exists():
            problems.append(f"{run}: selected although superseded")
        logged = [l for l in (path / "failures.txt").read_text().splitlines() if l.strip()] if (path / "failures.txt").exists() else []
        undeclared = [l for l in logged if not any(re.search(pattern, l) for pattern in DECLARED_FAILURE_LINES)]
        entry["logged_failures"] = len(logged)
        if undeclared:
            problems.append(f"{run}: failures.txt lists {len(undeclared)} invocation(s) that failed and are not declared, "
                            f"first: {undeclared[0][:120]}")
        env = (path / "environment.txt").read_text(errors="replace") if (path / "environment.txt").exists() else ""
        entry["job"] = (re.search(r"job=(\d+)", env) or [None, None])[1]
        entry["source_tag"] = (re.search(r"tag=(\w+)", env) or [None, None])[1]
        entry["nodes"] = sorted(set(re.findall(r"^\s+\d+ (\S+)", env.split("--- modules")[0], re.M)))
        entry["cpu_model"] = (re.search(r"Model name:\s*(.+)", env) or [None, None])[1]
        if not entry["job"]:
            problems.append(f"{run}: no job identifier recorded")
        field = lambda name: (re.search(rf"Resource_List\.{name} = (\S+)", env) or [None, None])[1]
        entry["allocated"] = {"nodes": field("nodect"), "cores": field("ncpus"), "gpus": field("ngpus")}
        cache = path / "build-gcc-native-w512.cache.txt"
        if entry["kind"] == "cpu" and cache.exists():
            text = cache.read_text()
            entry["build"] = {"alignment": (re.search(r"ATTN_ALIGNMENT:\w+=(\d+)", text) or [None, None])[1],
                              "extra_flags": (re.search(r"ATTN_EXTRA_FLAGS:\w+=(.*)", text) or [None, None])[1]}
            if entry["build"]["alignment"] != "4096" or "prefer-vector-width=512" not in (entry["build"]["extra_flags"] or ""):
                problems.append(f"{run}: main build is not the 512-bit, page-aligned one: {entry['build']}")
        files = OrderedDict()
        for csv_path in sorted(path.glob("*.csv")):
            rows = io.read(csv_path)
            info = OrderedDict(rows=len(rows), sha256=sha256(csv_path))
            name = csv_path.name
            if rows and "cluster" in rows[0] and {r["cluster"] for r in rows} != {"vanda"}:
                problems.append(f"{run}/{name}: rows not measured on Vanda")
            if rows and "validation" in rows[0]:
                info["validated"] = sum(r["validation"] != "not_requested" for r in rows)
                info["failed"] = sum(r["validation"] == "failed" for r in rows)
                if info["failed"] and name not in DECLARED_FAILURES:
                    problems.append(f"{run}/{name}: {info['failed']} repetitions failed validation")
                if info["failed"]:
                    info["declared_failure"] = DECLARED_FAILURES.get(name)
            timed = rows and "t_total_max" in rows[0]
            if timed:
                keys = [k for k in ("purpose", "impl", "kernel", "N", "ranks", "threads", "mask", "partition", "block_q", "block_k", "d", "H",
                                    "Hkv", "mha_control", "alpha", "poll_tiles", "fetch_order", "data", "comm", "schedule") if k in rows[0]]
                groups = io.group(rows, keys)
                reps = sorted({len(v) for v in groups.values()})
                info["configurations"], info["repetitions"] = len(groups), reps
                warm = sorted({float(r["warmup_seconds"]) for r in rows}) if "warmup_seconds" in rows[0] else None
                info["warmup_seconds"] = warm
                for r in rows:
                    for column in ("t_total_max", "t_total_mean"):
                        value = float(r[column])
                        if not (value == value and 0 < value < float("inf")):
                            problems.append(f"{run}/{name}: {column} = {r[column]}")
                if name in PROTOCOL_EXCEPTIONS:
                    info["protocol_exception"] = PROTOCOL_EXCEPTIONS[name]
                else:
                    rounds, reason = REPEATED_ROUNDS.get(name, (1, None))
                    if reason:
                        info["repeated_rounds"] = reason
                    for key, group in groups.items():
                        numbering = sorted(int(r["rep"]) for r in group)
                        if numbering != sorted([0, 1, 2] * rounds):
                            problems.append(f"{run}/{name}: configuration {dict(zip(keys, key))} has repetitions {numbering}, "
                                            f"expected {rounds} x (0, 1, 2)")
                    if entry["kind"] == "cpu" and (warm is None or min(warm) < 5):
                        problems.append(f"{run}/{name}: warm-up below 5 s ({warm})")
                if entry["kind"] == "cpu" and name not in TILE_SWEEPS and "block_q" in rows[0]:
                    tuned = [r for r in rows if r.get("kernel") == "tuned" and r["impl"] != "S0" and r["impl"] != "S0-fp32"]
                    tiles = sorted({(r["block_q"], r["block_k"]) for r in tuned})
                    # the adaptive query tile of threaded runs may be smaller than the requested one, never larger
                    bad = [t for t in tiles if t[1] != TILES[1] or int(t[0]) > int(TILES[0])]
                    if bad:
                        problems.append(f"{run}/{name}: tile sizes {bad}, expected {TILES}")
                for key, group in groups.items():
                    values = [float(r["t_total_max"]) for r in group]
                    if len(values) >= 3 and statistics.median(values) > 0:
                        rel = (max(values) - min(values)) / statistics.median(values)
                        if name not in PROTOCOL_EXCEPTIONS:
                            every_spread.append((rel, statistics.median(values), entry["kind"]))
                        if rel > 0.10:
                            spread.append({"run": run, "file": name, "configuration": dict(zip(keys, key)), "relative_spread": round(rel, 3),
                                           "median_seconds": statistics.median(values)})
            files[name] = info
        # The configurations of every file were counted when the run was selected (--pin); a file or a
        # configuration that disappears afterwards is noticed here.
        pinned = selection.get("configurations", {}).get(run)
        counted = {name: info.get("configurations", info["rows"]) for name, info in files.items()}
        entry["configurations"] = counted
        if "--pin" not in sys.argv:
            if pinned is None:
                problems.append(f"{run}: no pinned list of result files and configurations (run audit_results.py --pin)")
            elif pinned != counted:
                changed = sorted(n for n in set(pinned) | set(counted) if pinned.get(n) != counted.get(n))
                problems.append(f"{run}: result files or configurations differ from the pinned list: "
                                + ", ".join(f"{n} {pinned.get(n)} -> {counted.get(n)}" for n in changed[:6]))
        if entry["kind"] == "tests":
            outcome = {r["test"]: r["result"] for r in io.read(path / "unit-tests.csv")}
            entry["tests"] = outcome
            if not outcome or set(outcome.values()) != {"passed"}:
                problems.append(f"{run}: unit tests did not all pass: {outcome}")
            shot = snapshot(entry["source_tag"]) or {}
            differing = sorted(name for name, digest in shot.items()
                               if not (ROOT / name).exists() or sha256(ROOT / name) != digest)
            # The tag is a hash of every staged file, so equal tags mean the same files, none added or removed.
            import subprocess
            local_tag = subprocess.run(["bash", str(ROOT / "scripts" / "hpc" / "cluster.sh"), "tag", "-"],
                                       capture_output=True, text=True).stdout.strip()
            entry["submitted_tree_tag"] = local_tag
            entry["tested_source_equals_submitted_tree"] = bool(shot) and not differing and local_tag == entry["source_tag"]
            if not shot:
                problems.append(f"{run}: no file hashes recorded for the tested source snapshot {entry['source_tag']}")
            if differing or local_tag != entry["source_tag"]:
                problems.append(f"{run}: the unit tests ran on source {entry['source_tag']}, the submitted tree is {local_tag}"
                                + (f" (files that differ: {differing})" if differing else " (files were added or removed)"))
        entry["files"] = files
        entry["validated"] = sum(f.get("validated", 0) for f in files.values())
        entry["failed"] = sum(f.get("failed", 0) for f in files.values())
        report[run] = entry

    # 5. one source for all measurements of a kind, and the same source in the submitted tree
    sources = OrderedDict()
    for kind, required in (("cpu", CPU_SOURCES), ("gpu", GPU_SOURCES), ("analysis", ANALYSIS_SOURCES)):
        tags = sorted({e["source_tag"] for e in report.values() if e["kind"] == kind})
        shots = {t: snapshot(t) for t in tags}
        missing = [t for t, s in shots.items() if s is None]
        if missing:
            problems.append(f"{kind}: no file hashes recorded for source snapshot(s) {missing}")
            continue
        entry = OrderedDict(snapshots=tags, files=OrderedDict())
        for name in required:
            hashes = {t: shots[t].get(name) for t in tags}
            distinct = set(hashes.values())
            local = sha256(ROOT / name) if (ROOT / name).exists() else None
            declared = POST_MEASUREMENT_CHANGES.get(name)
            entry["files"][name] = {"sha256": sorted(distinct), "identical_across_snapshots": len(distinct) == 1,
                                    "submitted_tree_identical": distinct == {local}}
            if distinct == {local}:
                continue
            if not declared:
                if len(distinct) != 1:
                    problems.append(f"{kind}: {name} differs between source snapshots {tags}")
                problems.append(f"{kind}: {name} in the submitted tree differs from the measured source")
                continue
            # A declared change: snapshots may hold the file as it was measured or as it is submitted.
            record = OrderedDict(change=declared["change"], condition=declared["condition"])
            allowed = {local}
            if "revert" in declared:
                text = (ROOT / name).read_text()
                for submitted, measured in declared["revert"]:
                    if text.count(submitted) != 1:
                        problems.append(f"{kind}: {name}: a declared change is not in the submitted file exactly once")
                    text = text.replace(submitted, measured)
                restored = hashlib.sha256(text.encode()).hexdigest()
                allowed.add(restored)
                record["reverting_restores_measured_file"] = bool(distinct - {local}) and distinct - {local} == {restored}
                if not distinct <= allowed:
                    problems.append(f"{kind}: {name} differs from the measured source by more than the declared changes")
            elif len(distinct) != 1:
                problems.append(f"{kind}: {name} differs between source snapshots {tags}")
            record["snapshots_with_the_measured_version"] = sorted(t for t, h in hashes.items() if h != local)
            if declared["check"]:
                checker = analysis_records if declared["check"] == "analysis" else recorded_commands
                checks = OrderedDict()
                for run, e in report.items():
                    if e["kind"] == kind and hashes.get(e["source_tag"]) != local:
                        ok, stats = checker(io.RUNS / run)
                        checks[run] = dict(stats, condition_holds=ok)
                        if not ok:
                            problems.append(f"{kind}: {name} was changed after {run} was computed and the stored records "
                                            f"do not meet the condition under which they are unaffected: {stats}")
                record["records_checked"] = checks
            entry["files"][name]["changed_after_measurement"] = record
        sources[kind] = entry
    # the GPU driver differs between snapshots in logging and option checks only; record which version each job used
    sources.setdefault("gpu", OrderedDict())["driver_by_snapshot"] = {
        t: (snapshot(t) or {}).get("apps/attn_gpu_bench.cpp") for t in sorted({e["source_tag"] for e in report.values() if e["kind"] == "gpu"})}
    cpu_models = sorted({e["cpu_model"] for e in report.values() if e["kind"] == "cpu" and e["cpu_model"]})
    if len(cpu_models) > 1:
        problems.append(f"CPU measurements on different processor models: {cpu_models}")
    if spread:
        notes.append(f"{len(spread)} configurations have a repetition spread above 10% (listed under repetition_spread)")

    def summary(values):
        values = sorted(values)
        if not values:
            return None
        return {"configurations": len(values), "median": statistics.median(values), "p90": values[int(0.9 * (len(values) - 1))],
                "max": values[-1], "share_above_10_percent": sum(v > 0.10 for v in values) / len(values)}
    repeatability = {"definition": "(largest - smallest) / median of the repetitions of one configuration; threshold 0.05 s",
                     "at_least_50ms": summary([s for s, t, k in every_spread if t >= 0.05 and k == "cpu"]),
                     "below_50ms": summary([s for s, t, k in every_spread if t < 0.05 and k == "cpu"]),
                     "gpu_at_least_50ms": summary([s for s, t, k in every_spread if t >= 0.05 and k == "gpu"]),
                     "gpu_below_50ms": summary([s for s, t, k in every_spread if t < 0.05 and k == "gpu"])}

    result = OrderedDict(passed=not problems, problems=problems, notes=notes,
                         selected_run_directories=len(selected_dirs), excluded_run_directories=len(excluded),
                         repetitions_validated=sum(e["validated"] for e in report.values()),
                         repetitions_failed=sum(e["failed"] for e in report.values()),
                         cpu_model=cpu_models, sources=sources, repeatability=repeatability,
                         repetition_spread=sorted(spread, key=lambda s: -s["relative_spread"]), runs=report)
    (ROOT / "results" / "audit.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def used_by():
    """run directory -> the figures, tables and number groups computed from it."""
    store = GEN / "provenance.json"
    data = json.loads(store.read_text()) if store.exists() else {}
    out = {}
    for kind, items in data.items():
        for name, files in items.items():
            for run in {f.split("/")[0] for f in files}:
                out.setdefault(run, {}).setdefault(kind, set()).add(name)
    return out


FIGURE_LABEL = {"serial": "fig:serial", "tiles": "fig:tiles", "strong": "fig:strong", "collapse": "fig:collapse", "hardware": "fig:hardware",
                "small_c": "fig:smallc", "netbench": "fig:netbench", "traces": "fig:traces", "weak": "fig:weak", "causal": "fig:causal",
                "memory": "fig:memory", "hybrid": "fig:hybrid", "gpu_ladder": "fig:gpuladder", "gpu_ring": "fig:gpuring", "real_stats": "fig:realwindow",
                "real_heads": "fig:realheads"}
TABLE_LABEL = {"table_strong": "tab:strong", "table_align": "tab:align", "table_v7": "tab:v7"}


def table(result):
    uses = used_by()
    measures = json.loads(io.SELECTION.read_text()).get("measures", {})
    lines = [r"\begin{table}[!ht]\centering\footnotesize\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabularx}{\textwidth}{@{}l l l r >{\hsize=1.5\hsize}Y >{\hsize=0.5\hsize}Y@{}}", r"\toprule",
             r"PBS job & allocated & source & validated & measures & used by\\", r"\midrule"]
    for run, e in result["runs"].items():
        u = uses.get(run, {})
        labels = sorted(FIGURE_LABEL[f] for f in u.get("figures", ()) if f in FIGURE_LABEL) + \
                 sorted(TABLE_LABEL[x] for x in u.get("tables", ()) if x in TABLE_LABEL)
        refs = r"\cref{" + ",".join(labels) + "}" if labels else "text"
        what = "; ".join(measures.get(k, k) for k in e["experiments"]).replace(" x ", r" $\times$ ").replace("<=", r"$\le$").replace(">=", r"$\ge$")
        al = e["allocated"]
        allocated = f"{al['nodes']} $\\times$ {int(al['cores']) // int(al['nodes'])} cores" if al["cores"] and al["nodes"] else "?"
        if al["gpus"] not in (None, "0"):
            allocated += f", {al['gpus']} GPU"
        validated = f"{e['validated']:,}".replace(",", "{,}") + (f" ({e['failed']} failed)" if e["failed"] else "")
        if e["kind"] == "tests":
            validated = "all passed"
        lines.append(f"{e['job']} & {allocated} & \\code{{{e['source_tag']}}} & {validated} & {what} & {refs}\\\\")
    lines += [r"\bottomrule", r"\end{tabularx}",
              r"\caption{Provenance of the results: one line per Vanda job. ``Allocated'': nodes $\times$ cores per node given to "
              r"the job (a GPU job receives 36 cores per GPU); a job also measures configurations that use fewer cores than "
              r"it was given. ``Source'': identifier of the source snapshot the job was built from; the compute and benchmark-driver "
              r"sources of all CPU jobs are byte-identical, as are the kernel sources of all GPU jobs. The submitted compute and "
              r"kernel sources are these files; the submitted driver differs from the measured one in two option checks "
              r"(\cref{app:environment}). ``Validated'': repetitions compared with the double-precision reference on sampled rows; the failures are "
              r"those of the stress study of \cref{sec:validation}. ``Used by'': figures and tables computed from the job's files "
              r"(``text'': quoted numbers only).}", r"\label{tab:provenance}", r"\end{table}"]
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "table_provenance.tex").write_text("\n".join(lines) + "\n")


def pin(result):
    """Store the number of configurations of every result file of the selected runs in the selection."""
    selection = json.loads(io.SELECTION.read_text())
    selection["configurations"] = OrderedDict((run, e["configurations"]) for run, e in result["runs"].items())
    selection["configurations_note"] = ("Timed files: number of distinct configurations (each with its repetitions); other files: "
                                        "number of rows. Counted from the retrieved files when the runs were selected, after their "
                                        "failure logs had been checked; the audit compares every later state with these counts.")
    io.SELECTION.write_text(json.dumps(selection, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    outcome = audit()
    if "--pin" in sys.argv:
        pin(outcome)
        print("pinned the configurations of", len(outcome["runs"]), "runs in", io.SELECTION.name)
    table(outcome)
    print(f"audit: {'PASSED' if outcome['passed'] else 'FAILED'}; {outcome['selected_run_directories']} selected runs, "
          f"{outcome['excluded_run_directories']} excluded; {outcome['repetitions_validated']} repetitions validated, "
          f"{outcome['repetitions_failed']} failed (declared); {len(outcome['repetition_spread'])} configurations with spread > 10%")
    for problem in outcome["problems"]:
        print("  PROBLEM:", problem)
    for item in outcome["repetition_spread"][:8]:
        print("  spread %.0f%%: %s %s %s" % (100 * item["relative_spread"], item["run"].split("-2026")[0], item["file"],
                                           {k: v for k, v in item["configuration"].items() if k in ("impl", "N", "ranks", "threads", "poll_tiles")}))
    sys.exit(0 if outcome["passed"] else 1)
