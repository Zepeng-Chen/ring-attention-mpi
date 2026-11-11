# Shared preamble for every benchmark job. Sourced, not executed.
# Inputs (from qsub -v): ATTN_SRC, ATTN_RUN, ATTN_TAG, ATTN_BUILD_ROOT.
set -euo pipefail
: "${PBS_JOBID:?run through PBS}" "${ATTN_SRC:?}" "${ATTN_RUN:?}" "${ATTN_TAG:?}" "${ATTN_BUILD_ROOT:?}"
mkdir -p "$ATTN_RUN"
cd "$ATTN_RUN"
trap 'status=$?; printf "%s\n" "$status" > "$ATTN_RUN/exit-status.txt"; date -u "+%Y-%m-%dT%H:%M:%SZ" > "$ATTN_RUN/finished-utc.txt"' EXIT
# A job that is cancelled or reaches its walltime receives SIGTERM; without this
# the EXIT trap above would record the status of the last finished command (0).
trap 'exit 143' TERM INT
source "$ATTN_SRC/scripts/hpc/${ATTN_ENV:-vanda_env.sh}"
export OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=1
{
    date -u '+%Y-%m-%dT%H:%M:%SZ'; echo "job=$PBS_JOBID tag=$ATTN_TAG host=$(hostname)"
    cat "$ATTN_SRC/SOURCE_TAG"
    echo "--- nodes"; sort "$PBS_NODEFILE" | uniq -c
    echo "--- modules"; module list 2>&1
    echo "--- compilers"; g++ --version | head -1; mpirun --version | head -1; cmake --version | head -1
    echo "--- cpu"; lscpu | grep -E "Model name|Socket|Core|Thread|NUMA|L1d|L2|L3|MHz"
    echo "--- numa"; numactl -H 2>/dev/null | head -8 || true
    echo "--- limits"; qstat -f "$PBS_JOBID" | grep -E "Resource_List|exec_host|queue" || true
} > environment.txt 2>&1 || true   # a missing diagnostic tool must not stop the job
cp "$PBS_NODEFILE" nodes.txt
# The queue adds the memory of each chunk when a job is submitted. A select statement
# that is replaced later with qalter loses it: the vnodes then carry no memory, the
# control group allows about 1 GB per node and larger ranks are killed with signal 9.
# Such a job stops here instead of producing half a result.
vnodes_have_memory() {
    local vnodes; vnodes="$(qstat -f "$PBS_JOBID" 2>/dev/null | tr -d '\n\t' | sed -n 's/.*exec_vnode = \([^ ]*\).*/\1/p')"
    [[ -z "$vnodes" || "$vnodes" == *mem=* ]]
}
if ! vnodes_have_memory; then
    echo "no memory is assigned to the vnodes of this job (exec_vnode without mem=): submit the select statement with qsub, do not change it with qalter" >&2
    exit 5
fi
NODES=$(sort -u "$PBS_NODEFILE" | wc -l)
SLOTS=$(wc -l < "$PBS_NODEFILE")
NODES=$((NODES + 0)); SLOTS=$((SLOTS + 0))   # plain integers (some wc pad with spaces)

# Build directories are shared by the jobs of one source tag. A lock directory lets
# one job build at a time. A job that cannot get the lock stops instead of using a
# directory that another job may still be writing, and only the holder removes it.
# The holder file names the job; a waiting job takes over only a lock whose holder
# has finished (as reported by the scheduler), never one it merely waited long for.
lock_holder_finished() {
    local state
    state="$(qstat -x -f "${1%% *}" 2>/dev/null | awk -F' = ' '/job_state/ {print $2; exit}')" || true
    [[ "$state" == F || "$state" == X ]]
}
acquire_build_lock() {
    local lock="$1" limit="${2:-900}" waited=0 holder grave
    until mkdir "$lock" 2>/dev/null; do
        holder="$(cat "$lock/holder" 2>/dev/null || true)"
        if [[ -n "$holder" ]] && lock_holder_finished "$holder"; then
            # Move the lock away in one atomic step, then check that it is the stale one.
            grave="$lock.stale.$$"
            if mv "$lock" "$grave" 2>/dev/null; then
                if [[ "$(cat "$grave/holder" 2>/dev/null || true)" == "$holder" ]]; then
                    rm -rf "$grave"; echo "removed the build lock left by finished job ${holder%% *}" >&2; continue
                fi
                if [[ ! -e "$lock" ]]; then mv "$grave" "$lock"; else rm -rf "$grave"; fi
            fi
        fi
        sleep 3; waited=$((waited + 3))
        if (( waited > limit )); then
            echo "build lock $lock still held after ${limit}s (holder: ${holder:-unknown}); stopping" >&2
            return 1
        fi
    done
    echo "$PBS_JOBID on $(hostname)" > "$lock/holder"
}
release_build_lock() { rm -f "$1/holder"; rmdir "$1" 2>/dev/null || true; }

# build NAME [cmake args...] -> sets BUILD to an (immutable per tag) build directory
build() {
    local name="$1"; shift
    BUILD="$ATTN_BUILD_ROOT/$ATTN_TAG-$name"
    local lock="$BUILD.lock"
    acquire_build_lock "$lock" || return 1
    # Only a build that ran to the end (all targets) carries the marker; a directory
    # left by an interrupted build is built again.
    if [[ ! -f "$BUILD/.complete" ]]; then
        cmake -S "$ATTN_SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DATTN_NATIVE=ON \
            "-DMPIEXEC_PREFLAGS=-x;LD_LIBRARY_PATH;--bind-to;none" "$@" > "build-$name.log" 2>&1 \
            && cmake --build "$BUILD" --parallel 8 >> "build-$name.log" 2>&1 \
            && touch "$BUILD/.complete" || { release_build_lock "$lock"; echo "build failed: $name" >&2; tail -20 "build-$name.log" >&2; return 1; }
    fi
    release_build_lock "$lock"
    grep -E "CMAKE_CXX_COMPILER:|CMAKE_CXX_FLAGS_RELEASE:|ATTN_" "$BUILD/CMakeCache.txt" > "build-$name.cache.txt"
}
# Open MPI launcher with explicit binding: run RANKS THREADS -- program args
# The launched processes do not inherit the module environment by themselves
# (they would load the system libstdc++), so it is exported explicitly.
MPI_ENV=(-x LD_LIBRARY_PATH -x PATH -x OMP_NUM_THREADS -x OMP_DYNAMIC -x OMP_PROC_BIND -x OMP_PLACES)
# This Open MPI starts its per-node daemons over ssh, where no module is loaded;
# --prefix tells the remote shell where the daemon lives. The choice is tested
# once per job on every allocated node before any measurement.
MPI_LAUNCH=()
detect_launcher() {
    # The daemon (prted) may live in a separate PRRTE installation.
    local daemon; daemon="$(command -v prted || command -v orted || command -v mpirun)"
    local prefix; prefix="$(dirname "$(dirname "$daemon")")"
    local candidate
    : > launch-test.log
    # The first two candidates are the ones every CPU run used. The Open MPI of the
    # GPU toolchain (HPC-X) is built without PBS support: it does not see the
    # allocation, so a job on more than one node must also pass the node file.
    for candidate in "--prefix $prefix" "" "--prefix $prefix --hostfile $PBS_NODEFILE" "--hostfile $PBS_NODEFILE"; do
        echo "--- mpirun ${candidate:-(default)}" >> launch-test.log
        # --bind-to none: on shared GPU nodes the job owns only part of the cores.
        if timeout 60 mpirun $candidate -np "$NODES" --map-by ppr:1:node --bind-to none "${MPI_ENV[@]}" hostname > launch-attempt.log 2>&1 < /dev/null; then
            cat launch-attempt.log >> launch-test.log
            MPI_LAUNCH=($candidate)
            echo "launcher: mpirun ${candidate:-(default)} started on $(sort -u launch-attempt.log | grep -icE "^(cn|gn|in)-") of $NODES nodes" | tee -a environment.txt
            rm -f launch-attempt.log
            return 0
        fi
        cat launch-attempt.log >> launch-test.log
    done
    rm -f launch-attempt.log
    echo "no working mpirun launch configuration" >&2; cat launch-test.log >&2; return 1
}
PACKAGE_CORES=$(lscpu | awk -F: '/Core\(s\) per socket/ {gsub(/ /, "", $2); print $2}')
run() {
    local ranks="$1" threads="$2"; shift 3
    export OMP_NUM_THREADS="$threads"
    if (( threads <= PACKAGE_CORES )); then
        mpirun "${MPI_LAUNCH[@]}" -np "$ranks" --map-by "slot:PE=$threads" --bind-to core "${MPI_ENV[@]}" "$@" < /dev/null
    else
        # A rank wider than one socket cannot be bound by the slot mapper; place the
        # ranks per node and let the OpenMP runtime pin its threads to cores.
        local per_node=$(( SLOTS / NODES / threads ))
        mpirun "${MPI_LAUNCH[@]}" -np "$ranks" --map-by "ppr:$per_node:node" --bind-to none "${MPI_ENV[@]}" "$@" < /dev/null
    fi
}
