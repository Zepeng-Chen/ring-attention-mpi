#!/usr/bin/env python3
"""List the result files that the report is computed from.

results/ holds every run retrieved from the cluster and is not under version
control. The files named by this script are the exception: the selected runs'
tables, their job records and the per-rank files that a figure, a table or a
number reads (report/generated/provenance.json), together with the selection,
the audit result and the source and data digests. With them the report can be
rebuilt from a clone of the repository. In these published copies the name of the
cluster account has been replaced by USER wherever a path or a command line
recorded it; nothing else differs from the files retrieved from the cluster.

    python3 scripts/published_results.py            # writes results/published_files.txt
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "hpc" / "vanda"
RECORDS = ("environment.txt", "gpu-environment.txt", "exit-status.txt", "finished-utc.txt", "nodes.txt", "submit.txt", "job.id",
           "failures.txt")       # the failure log: the audit checks that it lists only declared failures


def published():
    selected = sorted(set(json.loads((ROOT / "results" / "selected_runs.json").read_text())["runs"].values()))
    keep = {"results/selected_runs.json", "results/audit.json", "results/published_files.txt",
            "results/real_data/manifest.sha256", "results/real_data/meta.json"}
    keep |= {str(p.relative_to(ROOT)) for p in (ROOT / "results" / "source_hashes").glob("*.sha256")}
    for run in selected:
        for path in (RUNS / run).iterdir():
            if not path.is_file():
                continue
            name = path.name
            if (path.suffix == ".csv" or name in RECORDS or name.endswith(".cache.txt") or name.startswith("gpu-rank.")
                    or (run.startswith("unit-tests") and path.suffix == ".log" and not name.startswith("build-"))):
                keep.add(str(path.relative_to(ROOT)))
    provenance = json.loads((ROOT / "report" / "generated" / "provenance.json").read_text())
    for items in provenance.values():
        for files in items.values():
            for name in files:
                if (RUNS / name).is_file():
                    keep.add(str((RUNS / name).relative_to(ROOT)))
    return sorted(keep)


if __name__ == "__main__":
    files = published()
    (ROOT / "results" / "published_files.txt").write_text("\n".join(files) + "\n")
    size = sum((ROOT / f).stat().st_size for f in files if (ROOT / f).exists())
    print(f"{len(files)} files, {size / 1e6:.1f} MB -> results/published_files.txt")
