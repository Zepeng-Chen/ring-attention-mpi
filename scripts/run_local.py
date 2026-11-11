#!/usr/bin/env python3
"""Run bounded local development checks and retain machine-readable evidence.

This runner never submits jobs or labels laptop timings as final cluster results.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def source_hashes():
    paths = [ROOT / "CMakeLists.txt"]
    for base in ["include", "src", "apps", "tests", "scripts"]:
        paths.extend(p for p in (ROOT / base).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts)
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def executable_hashes(build):
    names = ["ring_attention_mpi", "attn_core_tests", "attn_mpi_tests", "attn_bench"]
    return {name: hashlib.sha256((build / name).read_bytes()).hexdigest() for name in names}


def run(command, directory, name, env, timeout=180, expected=0):
    start = time.monotonic()
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=timeout, check=False)
        output, returncode = result.stdout, result.returncode
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        output += f"\nTimed out after {timeout} seconds\n"
        returncode = 124
    except OSError as error:
        output, returncode = str(error) + "\n", 127
    (directory / f"{name}.log").write_text(output)
    record = {"name": name, "command": command, "returncode": returncode, "expected_returncode": expected,
              "elapsed_seconds": time.monotonic() - start}
    print(f"{name}: {'PASS' if returncode == expected else 'FAIL'} ({record['elapsed_seconds']:.2f}s)", flush=True)
    if returncode != expected:
        print(output[-6000:], file=sys.stderr)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build-local")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 3, 4, 5, 7, 8])
    parser.add_argument("--mpiexec", default="mpirun")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.rounds < 1 or args.timeout < 1 or any(p < 1 for p in args.ranks) or len(set(args.ranks)) != len(args.ranks):
        parser.error("rounds, ranks and timeout must be positive; ranks must be unique")
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = (args.output or ROOT / "results" / f"local-{timestamp}").resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OMP_DYNAMIC="FALSE", OMP_PROC_BIND="false")
    # Laptop correctness tests deliberately allow oversubscription. Cluster
    # benchmark jobs must instead use explicitly allocated, bound resources.
    env.update(OMPI_MCA_rmaps_base_oversubscribe="1", OMPI_MCA_hwloc_base_binding_policy="none")
    build = args.build.resolve()
    bench = build / "attn_bench"
    mpi_tests = build / "attn_mpi_tests"
    commands = []
    records = []
    status = "failed"
    metadata = {"schema_version": 2, "started": timestamp, "purpose": "local development and correctness only",
                "platform": platform.platform(), "python": sys.version,
                "rounds": args.rounds, "ranks": args.ranks,
                "binding": "none; oversubscription allowed for correctness", "build_directory": str(build)}
    try:
        metadata["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        metadata["git_status"] = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
        metadata["source_sha256"] = source_hashes()
        cache = (build / "CMakeCache.txt").read_text()
        home_line = next((line.split("=", 1)[1] for line in cache.splitlines()
                          if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL=")), "")
        if not home_line or Path(home_line).resolve() != ROOT:
            raise RuntimeError("Build directory is not configured for this source tree")
        build_command = ["cmake", "--build", str(build), "--parallel", "4"]
        record = run(build_command, out, "build", env, args.timeout)
        records.append(record)
        (out / "runs.json").write_text(json.dumps(records, indent=2) + "\n")
        if record["returncode"]:
            return 1
        if source_hashes() != metadata["source_sha256"]:
            raise RuntimeError("Source changed while building; repeat with a stable source tree")
        cache_bytes = (build / "CMakeCache.txt").read_bytes()
        metadata["build_cache_sha256"] = hashlib.sha256(cache_bytes).hexdigest()
        metadata["build_type"] = next((line.split("=", 1)[1] for line in cache_bytes.decode().splitlines()
                                       if line.startswith("CMAKE_BUILD_TYPE:STRING=")), "unspecified") or "unspecified"
        metadata["executable_sha256"] = executable_hashes(build)
        commands.append(("ctest", ["ctest", "--test-dir", str(build), "--output-on-failure"]))
        for round_index in range(args.rounds):
            for ranks in args.ranks:
                commands.append((f"round{round_index + 1}-P{ranks}",
                                 [args.mpiexec, "-n", str(ranks), str(mpi_tests)]))
        # Check driver CSVs, nonblocking polling, trace output and serial baseline.
        for impl in ["S0", "S0-fp32", "S1", "overlap", "rma_get"]:
            command = [args.mpiexec, "-n", "1" if impl.startswith("S") else "3", str(bench),
                       "--impl", impl, "--N", "73", "--d", "17", "--dv", "9", "--H", "6", "--Hkv", "2",
                       "--mask", "sink-window", "--window", "11", "--sink", "4", "--partition", "zigzag",
                       "--block-q", "5", "--block-k", "13", "--alpha", "50", "--validate", "73",
                       "--warmup", "1", "--reps", "3", "--csv", str(out / "driver.csv")]
            if not impl.startswith("S"):
                command += ["--threads", "2", "--poll-tiles", "2", "--trace-csv", str(out / f"trace-{impl}"),
                            "--rank-csv", str(out / f"ranks-{impl}")]
            commands.append((f"driver-{impl}", command))
        # A small instrumented example tests a report hypothesis using work and
        # byte counts, without interpreting laptop runtimes as cluster evidence.
        for impl in ["overlap", "rma_get"]:
            for partition in ["contiguous", "zigzag"]:
                name = f"counts-{impl}-{partition}"
                commands.append((name, [args.mpiexec, "-n", "4", str(bench), "--impl", impl,
                                        "--N", "128", "--d", "16", "--dv", "16",
                                        "--mask", "causal", "--partition", partition,
                                        "--block-q", "8", "--block-k", "16", "--validate", "128",
                                        "--warmup", "0", "--reps", "1", "--csv", str(out / "counts.csv"),
                                        "--rank-csv", str(out / name)]))
        for name, command in commands:
            record = run(command, out, name, env, args.timeout)
            records.append(record)
            (out / "runs.json").write_text(json.dumps(records, indent=2) + "\n")
            if record["returncode"]:
                return 1
        with (out / "driver.csv").open(newline="") as source:
            rows = list(csv.DictReader(source))
        if len(rows) != 15 or any(r["validation"] != "passed" or int(r["nan_inf"]) for r in rows):
            raise RuntimeError("driver CSV validation failed")
        metadata["max_driver_abs_error"] = max(float(r["max_abs_err"]) for r in rows)
        with (out / "counts.csv").open(newline="") as source:
            counts = list(csv.DictReader(source))
        if len(counts) != 4 or any(r["validation"] != "passed" for r in counts):
            raise RuntimeError("work-count example failed validation")
        for impl in ["overlap", "rma_get"]:
            for partition in ["contiguous", "zigzag"]:
                measured = []
                for rank in range(4):
                    with (out / f"counts-{impl}-{partition}.rank{rank}.csv").open(newline="") as source:
                        measured.append(int(next(csv.DictReader(source))["useful_pairs"]))
                expected = [528, 1552, 2576, 3600] if partition == "contiguous" else [2064] * 4
                if measured != expected:
                    raise RuntimeError(f"causal work counts differ: {impl}, {partition}, {measured}")
        low_limit = dict(env, OMP_THREAD_LIMIT="1")
        record = run([args.mpiexec, "-n", "1", str(bench), "--impl", "ring", "--N", "7", "--threads", "2"],
                     out, "thread-limit-rejected", low_limit, args.timeout, expected=1)
        records.append(record)
        (out / "runs.json").write_text(json.dumps(records, indent=2) + "\n")
        if record["returncode"] != 1 or "OpenMP team size differs" not in (out / "thread-limit-rejected.log").read_text():
            raise RuntimeError("an unexpected OpenMP team size was not correctly rejected")
        if source_hashes() != metadata["source_sha256"] or executable_hashes(build) != metadata["executable_sha256"]:
            raise RuntimeError("Source or executables changed during validation; evidence cannot identify one build")
        status = "passed"
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        metadata["error"] = str(error)
        print(str(error), file=sys.stderr)
        return 1
    finally:
        metadata["status"] = status
        metadata["runs_completed"] = len(records)
        metadata["runs_planned"] = args.rounds * len(args.ranks) + 12
        (out / "runs.json").write_text(json.dumps(records, indent=2) + "\n")
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Evidence: {out}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
