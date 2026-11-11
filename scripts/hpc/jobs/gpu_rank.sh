#!/bin/bash
# Started by mpirun once per rank of a GPU job on more than one node:
#     mpirun ... -x ATTN_GPU_VNODES -x ATTN_RUN bash gpu_rank.sh PROGRAM ARGS...
# It restricts the rank to the GPU(s) that PBS allocated on its node and then runs
# the program.
#
# Why this is needed. The Open MPI of the GPU toolchain has no PBS support and
# starts the ranks of the other nodes over ssh. Such a rank is outside the job's
# control group: it sees every GPU of its node, including one that may belong to
# another user's job, and it is not confined to the job's cores. Vanda allocates a
# GPU node per NUMA domain (vnode host[k] = NUMA node k with the GPU attached to
# it), so the allocated GPU is the one whose PCI device sits on NUMA node k.
#
#   --vnodes   read `qstat -f JOB` on stdin, print "host:k,host:k,..." (exec_vnode)
#   --census   print what this node shows (devices, their NUMA nodes, processes)
#
# The rule is checked where it can be: on a node where the control group already
# hides the other GPU, the visible GPU must sit on the allocated NUMA node. If it
# does not, or if the allocated GPU cannot be identified, the rank stops instead
# of computing on a device that may not be ours.
set -uo pipefail
SYSFS="${ATTN_SYSFS:-/sys}"

if [[ "${1:-}" == "--vnodes" ]]; then
    # qstat wraps long values with "newline + tab"; join them, then list NAME[k].
    awk '{ sub(/\r$/, ""); if (substr($0, 1, 1) == "\t") line = line substr($0, 2); else { if (line != "") print line; line = $0 } }
         END { if (line != "") print line }' \
        | sed -n 's/^ *exec_vnode = //p' | tr '+' '\n' | tr -d '()' \
        | sed -n 's/^\([^:[]*\)\[\([0-9][0-9]*\)\].*$/\1:\2/p' | tr 'A-Z' 'a-z' | tr '\n' ',' | sed 's/,$//'
    echo
    exit 0
fi

host="$(hostname -s | tr 'A-Z' 'a-z')"
vnodes=""
for entry in $(printf '%s' "${ATTN_GPU_VNODES:-}" | tr ',' ' '); do
    [[ "${entry%%:*}" == "$host" ]] && vnodes="$vnodes ${entry##*:}"
done
vnodes="${vnodes# }"

# One line per visible GPU: "index uuid bus numa".
devices=""
count=0
while IFS=, read -r index uuid bus; do
    index="${index// /}"; uuid="${uuid// /}"; bus="${bus// /}"
    [[ -z "$uuid" ]] && continue
    # nvidia-smi prints an 8-digit PCI domain, sysfs uses 4 digits and lower case.
    path="$SYSFS/bus/pci/devices/$(printf '%s' "${bus#0000}" | tr 'A-Z' 'a-z')/numa_node"
    numa="?"; [[ -r "$path" ]] && numa="$(cat "$path")"
    devices="$devices$index $uuid $bus $numa"$'\n'
    count=$((count + 1))
done < <(nvidia-smi --query-gpu=index,uuid,pci.bus_id --format=csv,noheader 2>/dev/null)

wanted=0; for k in $vnodes; do wanted=$((wanted + 1)); done
selected=""; note=""; status=0
if (( count == 0 )); then
    note="no GPU visible"; status=1
elif (( wanted == 0 )); then
    note="no vnode recorded for this host; environment unchanged"
elif (( count == wanted )); then
    # The visible set is the allocation (control group, or the whole node is ours).
    # Where the NUMA node of the devices is known, check the rule this script relies on.
    if printf '%s' "$devices" | awk '$4 !~ /^[0-9]+$/ { unknown = 1 } END { exit !unknown }'; then
        note="visible GPUs = allocation; their NUMA node is not reported, rule not checked; environment unchanged"
    else
        for k in $vnodes; do
            if ! printf '%s' "$devices" | awk -v k="$k" '$4 == k { found = 1 } END { exit !found }'; then
                note="visible GPUs are not on the allocated NUMA node(s) [$vnodes]: the vnode rule does not hold"; status=1
            fi
        done
        [[ -z "$note" ]] && note="visible GPUs = allocated vnode(s) [$vnodes], rule confirmed; environment unchanged"
    fi
else
    # More GPUs visible than allocated: keep those on the allocated NUMA nodes.
    for k in $vnodes; do
        uuid="$(printf '%s' "$devices" | awk -v k="$k" '$4 == k { print $2 }')"
        if [[ -z "$uuid" || "$uuid" == *$'\n'* ]]; then
            note="cannot identify the GPU of vnode [$k] among $count visible devices"; status=1; break
        fi
        selected="${selected:+$selected,}$uuid"
    done
    if (( status == 0 )); then
        export CUDA_VISIBLE_DEVICES="$selected"
        note="$count GPUs visible, $wanted allocated: restricted to vnode(s) [$vnodes] = $selected"
    fi
fi

report() {
    echo "host=$host vnodes=[$vnodes] $note"
    printf '%s' "$devices" | while read -r index uuid bus numa; do echo "  gpu $index $uuid bus=$bus numa=$numa"; done
}
if [[ -n "${ATTN_RUN:-}" && -d "${ATTN_RUN:-}" ]]; then
    report > "$ATTN_RUN/gpu-rank.$host.$$.tmp" 2>/dev/null && mv "$ATTN_RUN/gpu-rank.$host.$$.tmp" "$ATTN_RUN/gpu-rank.$host.txt"
fi
if (( status != 0 )); then echo "gpu_rank.sh: $host: $note" >&2; exit 1; fi

if [[ "${1:-}" == "--census" ]]; then
    report
    nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed "s/^/  $host state: /"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null | sed "s/^/  $host process: /"
    exit 0
fi

# A rank outside the control group is also kept on the cores of its vnode(s).
if [[ -n "$selected" ]] && command -v taskset > /dev/null 2>&1; then
    cpus=""
    for k in $vnodes; do
        list="$(cat "$SYSFS/devices/system/node/node$k/cpulist" 2>/dev/null)" || list=""
        [[ -n "$list" ]] && cpus="${cpus:+$cpus,}$list"
    done
    [[ -n "$cpus" ]] && exec taskset -c "$cpus" "$@"
fi
exec "$@"
