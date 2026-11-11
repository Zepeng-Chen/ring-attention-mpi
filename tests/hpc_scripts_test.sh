#!/bin/bash
# Local test of the job-script logic that cannot be tried on the cluster without
# paying for a job: the build lock, the launcher detection and the GPU selection
# of multi-node GPU jobs. Scheduler and system commands are replaced by mocks.
#   bash tests/hpc_scripts_test.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
unset ATTN_RUN ATTN_GPU_VNODES CUDA_VISIBLE_DEVICES   # nothing of a surrounding job leaks into the tests
trap 'rm -rf "$WORK"' EXIT
failures=0
check() {   # check NAME EXPECTED ACTUAL
    if [[ "$2" == "$3" ]]; then echo "ok    $1"; else echo "FAIL  $1: expected [$2], got [$3]"; failures=$((failures + 1)); fi
}

# ---------------------------------------------------------------- mocks
BIN="$WORK/bin"; mkdir -p "$BIN"
mock() { printf '#!/bin/bash\n%s\n' "$2" > "$BIN/$1"; chmod +x "$BIN/$1"; }
mock hostname 'echo "${MOCK_HOST:-GN-A40-035}"'
mock nvidia-smi 'case "$*" in
  *--query-gpu=index,uuid,pci.bus_id*) printf "%s" "$MOCK_GPUS";;
  *) :;;
esac'
mock taskset 'shift 2; echo "taskset:$MOCK_TASKSET_SEEN" > /dev/null; exec "$@"'
mock sleep ':'
mock timeout 'shift; exec "$@"'
mock module ':'
mock lscpu 'echo "Core(s) per socket:  36"'
mock numactl ':'
mock qstat 'case "$*" in
  *"-x -f"*) echo "    job_state = ${MOCK_HOLDER_STATE:-R}";;
  *) printf "    exec_host = x\n    exec_vnode = %s\n    Hold_Types = n\n" "${MOCK_EXEC_VNODE:-(cn-001[0]:mem=256465mb:ncpus=36+cn-001[1]:mem=2555}"
     [[ -z "${MOCK_EXEC_VNODE:-}" ]] && printf "\t35mb:ncpus=36)\n";;
esac'
# mpirun: prints one host name per node; refuses unless --hostfile is given when MOCK_MPI_NEEDS_HOSTFILE=1.
mock mpirun 'if [[ "${MOCK_MPI_NEEDS_HOSTFILE:-0}" == 1 && "$*" != *--hostfile* ]]; then echo "more processes than the ppr" >&2; exit 1; fi
if [[ "$1" == "--version" ]]; then echo "mpirun (Open MPI) mock"; exit 0; fi
echo "GN-A40-030"; echo "GN-A40-035"'
export PATH="$BIN:$PATH"

SYS="$WORK/sys"
mkdir -p "$SYS/bus/pci/devices/0000:17:00.0" "$SYS/bus/pci/devices/0000:ca:00.0" "$SYS/devices/system/node/node0" "$SYS/devices/system/node/node1"
echo 0 > "$SYS/bus/pci/devices/0000:17:00.0/numa_node"
echo 1 > "$SYS/bus/pci/devices/0000:ca:00.0/numa_node"
echo "0,2,4" > "$SYS/devices/system/node/node0/cpulist"
echo "1,3,5" > "$SYS/devices/system/node/node1/cpulist"
export ATTN_SYSFS="$SYS"
GPU0="0, GPU-aaaa, 00000000:17:00.0"$'\n'
GPU1="1, GPU-bbbb, 00000000:CA:00.0"$'\n'
RANK="$ROOT/scripts/hpc/jobs/gpu_rank.sh"
visible() { bash "$RANK" bash -c 'echo "${CUDA_VISIBLE_DEVICES:-unset}"' 2>/dev/null; echo "exit=$?"; }

# ---------------------------------------------------------------- exec_vnode parser
record='    exec_host = GN-A40-030/1*36+GN-A40-035/0*36
    exec_vnode = (GN-A40-030[1]:ncpus=36:ngpus=1:mem=240gb)+(GN-A40-035[0]:ncpu
	s=36:ngpus=1:mem=240gb)
    Hold_Types = n'
check "vnodes of a two-node job (wrapped line)" "gn-a40-030:1,gn-a40-035:0" "$(printf '%s\n' "$record" | bash "$RANK" --vnodes)"
record='    exec_vnode = (GN-A40-035[0]:mem=256465mb:ngpus=1:ncpus=36+GN-A40-035[1]:mem
	=235055mb:ngpus=1:ncpus=36)'
check "vnodes of a one-node two-GPU job" "gn-a40-035:0,gn-a40-035:1" "$(printf '%s\n' "$record" | bash "$RANK" --vnodes)"

# ---------------------------------------------------------------- GPU selection
export ATTN_GPU_VNODES="gn-a40-030:1,gn-a40-035:0" MOCK_HOST=GN-A40-035
MOCK_GPUS="$GPU0$GPU1" check "two GPUs visible, vnode 0 allocated -> GPU on NUMA 0" "GPU-aaaa exit=0" "$(MOCK_GPUS="$GPU0$GPU1" visible | tr '\n' ' ' | sed 's/ $//')"
export MOCK_HOST=GN-A40-030
check "two GPUs visible, vnode 1 allocated -> GPU on NUMA 1" "GPU-bbbb exit=0" "$(MOCK_GPUS="$GPU0$GPU1" visible | tr '\n' ' ' | sed 's/ $//')"
check "control group shows the allocated GPU -> unchanged" "unset exit=0" "$(MOCK_GPUS="$GPU1" visible | tr '\n' ' ' | sed 's/ $//')"
check "control group shows a GPU on the other NUMA node -> stop" "exit=1" "$(MOCK_GPUS="$GPU0" visible | tr '\n' ' ' | sed 's/ $//')"
check "no GPU visible -> stop" "exit=1" "$(MOCK_GPUS="" visible | tr '\n' ' ' | sed 's/ $//')"
export MOCK_HOST=GN-A40-099
check "host without a recorded vnode -> unchanged" "unset exit=0" "$(MOCK_GPUS="$GPU0$GPU1" visible | tr '\n' ' ' | sed 's/ $//')"
export ATTN_GPU_VNODES="gn-a40-035:0,gn-a40-035:1" MOCK_HOST=GN-A40-035
check "both vnodes of the node allocated -> unchanged" "unset exit=0" "$(MOCK_GPUS="$GPU0$GPU1" visible | tr '\n' ' ' | sed 's/ $//')"
export ATTN_GPU_VNODES="gn-a40-035:0"
echo -1 > "$SYS/bus/pci/devices/0000:17:00.0/numa_node"
check "NUMA node of the devices unknown -> stop" "exit=1" "$(MOCK_GPUS="$GPU0$GPU1" visible | tr '\n' ' ' | sed 's/ $//')"
check "NUMA node unknown but the control group already restricts -> unchanged" "unset exit=0" "$(MOCK_GPUS="$GPU0" visible | tr '\n' ' ' | sed 's/ $//')"
echo 0 > "$SYS/bus/pci/devices/0000:17:00.0/numa_node"
export ATTN_RUN="$WORK/run"; mkdir -p "$ATTN_RUN"
MOCK_GPUS="$GPU0$GPU1" bash "$RANK" --census > "$WORK/census.txt" 2>&1
check "census names the selection" "1" "$(grep -c 'restricted to vnode(s) \[0\] = GPU-aaaa' "$WORK/census.txt")"
check "the decision is recorded in the run directory" "1" "$(grep -c 'GPU-aaaa bus=00000000:17:00.0 numa=0' "$ATTN_RUN/gpu-rank.gn-a40-035.txt")"

# ---------------------------------------------------------------- common.sh: launcher and lock
SRC="$WORK/src"; mkdir -p "$SRC/scripts/hpc/jobs" "$WORK/build"
cp "$ROOT/scripts/hpc/jobs/common.sh" "$SRC/scripts/hpc/jobs/"
echo "export ATTN_CLUSTER=test" > "$SRC/scripts/hpc/test_env.sh"
echo "testtag" > "$SRC/SOURCE_TAG"
printf 'gn-a40-030.stdct.loc\ngn-a40-035.stdct.loc\n' > "$WORK/nodefile"
prologue='export PBS_JOBID=100.test ATTN_SRC="'$SRC'" ATTN_RUN="'$WORK'/job" ATTN_TAG=testtag ATTN_BUILD_ROOT="'$WORK'/build" ATTN_ENV=test_env.sh PBS_NODEFILE="'$WORK'/nodefile"; source "$ATTN_SRC/scripts/hpc/jobs/common.sh"'
check "launcher: first candidate when mpirun sees the allocation" "--prefix" \
    "$(bash -c "$prologue"'; detect_launcher > /dev/null; echo "${MPI_LAUNCH[0]}"' 2>/dev/null)"
check "launcher: node file when mpirun does not see the allocation" "--hostfile $WORK/nodefile" \
    "$(MOCK_MPI_NEEDS_HOSTFILE=1 bash -c "$prologue"'; detect_launcher > /dev/null; n=${#MPI_LAUNCH[@]}; echo "${MPI_LAUNCH[$((n - 2))]} ${MPI_LAUNCH[$((n - 1))]}"' 2>/dev/null)"
check "launcher: both nodes counted" "1" \
    "$(MOCK_MPI_NEEDS_HOSTFILE=1 bash -c "$prologue"'; detect_launcher' 2>/dev/null | grep -c 'started on 2 of 2 nodes')"

check "prologue: a job whose vnodes carry memory runs" "started" \
    "$(bash -c "$prologue"'; echo started' 2>/dev/null)"
check "prologue: a job whose vnodes lost their memory stops" "exit=5" \
    "$(MOCK_EXEC_VNODE="(cn-037[0]:ncpus=36+cn-037[1]:ncpus=36)" bash -c "$prologue"'; echo started' 2>/dev/null; echo "exit=$?")"

LOCK="$WORK/build/x.lock"
check "lock: free lock is taken and names its holder" "100.test" \
    "$(bash -c "$prologue"'; acquire_build_lock "'$LOCK'" 6 && cut -d" " -f1 "'$LOCK'/holder"' 2>/dev/null)"
check "lock: held by a running job -> give up, lock untouched" "failed 100.test" \
    "$(MOCK_HOLDER_STATE=R bash -c "$prologue"'; PBS_JOBID=200.test; acquire_build_lock "'$LOCK'" 6 || echo -n "failed "; cut -d" " -f1 "'$LOCK'/holder"' 2>/dev/null)"
check "lock: holder finished -> taken over" "200.test" \
    "$(MOCK_HOLDER_STATE=F bash -c "$prologue"'; PBS_JOBID=200.test; acquire_build_lock "'$LOCK'" 6 && cut -d" " -f1 "'$LOCK'/holder"' 2>/dev/null)"
check "lock: released by its holder" "gone" \
    "$(bash -c "$prologue"'; release_build_lock "'$LOCK'"; [[ -e "'$LOCK'" ]] && echo present || echo gone' 2>/dev/null)"

echo
if (( failures )); then echo "$failures test(s) failed"; exit 1; fi
echo "all job-script tests passed"
