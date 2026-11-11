#!/usr/bin/env python3
"""Package a frozen source tree and manage a task-specific PBS run over SSH.

No credentials are stored here. All remote commands can be inspected with
--dry-run. Resource requests and module selection are explicit inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shlex
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS = ("include", "src", "apps", "tests", "scripts", "docs", "report", "skills")
ROOT_FILES = ("CMakeLists.txt", "README", "README.md", "AGENTS.md", "project_instructions.md", "spec.md", "REVIEW.md", "PLAN.md", ".gitignore")
EXCLUDED_SUFFIXES = {".pyc", ".o", ".a", ".so", ".dylib", ".pdf", ".png", ".jpg", ".log", ".csv", ".aux", ".out", ".toc", ".fls", ".fdb_latexmk"}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def capture(argv: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(argv, cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()


def run(argv: list[str], dry_run: bool = False) -> None:
    print(shlex.join(argv), flush=True)
    if not dry_run:
        subprocess.run(argv, check=True)


def endpoint(text: str) -> str:
    # SSH aliases, hostnames and user@host are accepted; option injection is not.
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", text):
        raise argparse.ArgumentTypeError("use an SSH alias or user@hostname")
    return text


def remote_path(text: str) -> str:
    # Restrict transfers to unambiguous POSIX paths; no ~ or shell expansion.
    p = PurePosixPath(text)
    if not text or text.startswith("-") or not re.fullmatch(r"[A-Za-z0-9_./-]+", text) or ".." in p.parts or str(p) in {".", "/"}:
        raise argparse.ArgumentTypeError("use a task-specific POSIX path without spaces, ~, or ..")
    return str(p)


def ssh(host: str, script: str) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host,
            "bash -l -c " + shlex.quote(script)]


def source_files(root: Path) -> dict[str, bytes]:
    paths = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for name in SOURCE_DIRS:
        directory = root / name
        if directory.is_dir():
            paths.extend(p for p in directory.rglob("*") if p.is_file())
    result = {}
    for path in sorted(set(paths)):
        relative = path.relative_to(root)
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts) and relative.as_posix() != ".gitignore":
            continue
        if path.suffix in EXCLUDED_SUFFIXES or any(part.startswith("build") for part in relative.parts):
            continue
        if path.is_symlink() or root not in path.resolve().parents:
            raise ValueError(f"refusing source symlink outside snapshot contract: {relative}")
        result[relative.as_posix()] = path.read_bytes()
    if "CMakeLists.txt" not in result:
        raise ValueError("snapshot source must be the project root")
    return result


def validate_archive(path: Path) -> None:
    """Only extract a snapshot created under this source manifest contract."""
    with tarfile.open(path, "r:gz") as tar:
        members = tar.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or "SNAPSHOT.json" not in names:
            raise ValueError("archive lacks a unique snapshot manifest")
        for member in members:
            name = PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or ".." in name.parts or str(name) != member.name:
                raise ValueError("archive contains an unsafe entry")
        stream = tar.extractfile("SNAPSHOT.json")
        assert stream is not None
        manifest = json.load(stream)
        hashes = manifest["files_sha256"]
        if set(hashes) != set(names) - {"SNAPSHOT.json"}:
            raise ValueError("archive entries differ from the manifest")
        if hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest() != manifest["source_sha256"]:
            raise ValueError("invalid source tree checksum")
        for name, digest in hashes.items():
            stream = tar.extractfile(name)
            assert stream is not None
            if hashlib.sha256(stream.read()).hexdigest() != digest:
                raise ValueError("archive file checksum mismatch: " + name)


def snapshot(args: argparse.Namespace) -> None:
    root = args.source.resolve()
    files = source_files(root)
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    tree_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    try:
        commit = capture(["git", "rev-parse", "HEAD"], root)
        dirty = bool(capture(["git", "status", "--porcelain", "--untracked-files=normal"], root))
    except subprocess.CalledProcessError:
        commit, dirty = "unavailable", True
    manifest = {
        "created_utc": utc_now(), "source_sha256": tree_hash,
        "git_commit": commit, "working_tree_dirty": dirty,
        "purpose": args.purpose, "timing_protocol": "one untimed warmup, three measured repetitions by default",
        "files_sha256": hashes,
    }
    files["SNAPSHOT.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / ("ring-attention-" + tree_hash[:16] + ".tar.gz")
    # Exclusive creation preserves earlier snapshots, even when the name matches.
    with archive.open("xb") as stream:
        with tarfile.open(fileobj=stream, mode="w:gz") as tar:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size, info.mtime, info.mode = len(data), 0, 0o644
                tar.addfile(info, io.BytesIO(data))
    manifest["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    sidecar = archive.with_suffix(".json")
    sidecar.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"archive": str(archive), "manifest": str(sidecar), **{k: manifest[k] for k in ("source_sha256", "archive_sha256", "git_commit", "working_tree_dirty")}}, indent=2))


def probe(args: argparse.Namespace) -> None:
    script = """set -eu
date -u '+%Y-%m-%dT%H:%M:%SZ'
hostname
id -un
pwd
uname -a
for tool in qsub qstat gstat cmake mpicxx mpirun nvidia-smi ncu nsys; do
    command -v "$tool" || true
done
if type module >/dev/null 2>&1; then module list 2>&1; fi
if command -v qstat >/dev/null 2>&1; then qstat -Q; fi
"""
    run(ssh(args.host, script), args.dry_run)


def push(args: argparse.Namespace) -> None:
    archive = args.archive.resolve()
    if not archive.is_file():
        raise ValueError("archive does not exist")
    validate_archive(archive)
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    run_id = "run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + checksum[:12]
    destination = args.remote_base + "/" + run_id
    q = shlex.quote
    prepare = f"set -eu\nmkdir -p {q(args.remote_base)}\nmkdir {q(destination)}\nmkdir {q(destination + '/source')} {q(destination + '/evidence')}\n"
    run(ssh(args.host, prepare), args.dry_run)
    run(["scp", "-o", "BatchMode=yes", str(archive), args.host + ":" + destination + "/source.tar.gz"], args.dry_run)
    extract = f"""set -eu
cd {q(destination)}
printf '%s  %s\\n' {q(checksum)} source.tar.gz | sha256sum -c -
tar -xzf source.tar.gz -C source
printf '%s\\n' {q(checksum)} > evidence/archive.sha256
cp source/SNAPSHOT.json evidence/SNAPSHOT.json
"""
    run(ssh(args.host, extract), args.dry_run)
    print("REMOTE_RUN=" + destination)


def submit(args: argparse.Namespace) -> None:
    # qsub runs from the frozen source directory, so PBS_O_WORKDIR points there.
    if PurePosixPath(args.script).is_absolute():
        raise ValueError("job script must be relative to the frozen source directory")
    command = ["qsub", "-N", args.name]
    for resource in args.resource:
        command += ["-l", resource]
    if args.queue:
        command += ["-q", args.queue]
    command += ["-o", "../evidence/scheduler.out", "-e", "../evidence/scheduler.err", args.script]
    script = f"""set -eu
cd {shlex.quote(args.run_dir)}
test ! -e evidence/job.id
cd source
test -f {shlex.quote(args.script)}
job_id=$({shlex.join(command)})
printf '%s\\n' "$job_id" > ../evidence/job.id
printf '%s\\n' {shlex.quote(shlex.join(command))} > ../evidence/submission-command.txt
date -u '+%Y-%m-%dT%H:%M:%SZ' > ../evidence/submitted-utc.txt
printf '%s\\n' "$job_id"
"""
    run(ssh(args.host, script), args.dry_run)


def status(args: argparse.Namespace) -> None:
    script = f"""set -eu
cd {shlex.quote(args.run_dir)}
job_id=$(cat evidence/job.id)
case "$job_id" in -*|*[!A-Za-z0-9_.\\[\\]-]*|'') printf '%s\\n' 'Invalid saved job id' >&2; exit 2;; esac
date -u '+%Y-%m-%dT%H:%M:%SZ'
record_found=0
if qstat -f "$job_id"; then
    record_found=1
elif qstat -xf "$job_id"; then
    record_found=1
else
    printf '%s\\n' 'Scheduler record unavailable; inspect retrieved exit status and outputs.'
fi
if test -f evidence/exit-status.txt; then
    payload_status=$(cat evidence/exit-status.txt)
    if [[ ! "$payload_status" =~ ^([0-9]|[1-9][0-9]{{1,2}})$ ]] || (( payload_status > 255 )); then
        printf '%s\\n' 'Invalid payload exit status' >&2
        exit 2
    fi
    printf 'payload_exit_status=%s\\n' "$payload_status"
    exit "$payload_status"
fi
if test "$record_found" -eq 0; then exit 3; fi
"""
    run(ssh(args.host, script), args.dry_run)


def fetch(args: argparse.Namespace) -> None:
    destination = args.output.resolve()
    if destination.exists():
        raise ValueError("fetch destination already exists; choose a new evidence directory")
    if not args.dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
    run(["scp", "-r", "-o", "BatchMode=yes", args.host + ":" + args.run_dir + "/evidence", str(destination)], args.dry_run)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    s = sub.add_parser("snapshot", help="freeze source bytes, including uncommitted changes")
    s.add_argument("--source", type=Path, default=ROOT)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--purpose", choices=("pilot", "validation", "final", "profile"), default="pilot")
    s.set_defaults(function=snapshot)
    for name, function in (("probe", probe), ("push", push), ("submit", submit), ("status", status), ("fetch", fetch)):
        s = sub.add_parser(name)
        s.add_argument("--host", type=endpoint, required=True)
        s.add_argument("--dry-run", action="store_true")
        s.set_defaults(function=function)
        if name == "push":
            s.add_argument("--archive", type=Path, required=True)
            s.add_argument("--remote-base", type=remote_path, default="ring-attention-runs")
        if name in {"submit", "status", "fetch"}:
            s.add_argument("--run-dir", type=remote_path, required=True)
        if name == "submit":
            s.add_argument("--script", type=remote_path, required=True)
            s.add_argument("--resource", action="append", required=True,
                           help="PBS resource request; repeat for select, walltime, etc.")
            s.add_argument("--queue")
            s.add_argument("--name", default="attn-probe")
        if name == "fetch":
            s.add_argument("--output", type=Path, required=True)
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    try:
        arguments.function(arguments)
    except (OSError, ValueError, KeyError, tarfile.TarError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
