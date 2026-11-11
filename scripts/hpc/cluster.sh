#!/bin/bash
# Minimal helper for the cluster runs. Usage (from the repository root):
#   scripts/hpc/cluster.sh push  HOST                 -> copies an immutable source tree, prints its tag
#   scripts/hpc/cluster.sh tag   -                    -> prints the tag of the local tree (nothing is copied)
#   scripts/hpc/cluster.sh submit HOST TAG JOB NAME [qsub args...]
#                                                      -> creates runs/NAME-<time>, submits scripts/hpc/jobs/JOB
#       Extra job variables: ATTN_JOB_ENV="A=1,B=x:y" (use ':' inside values, never ',').
#   scripts/hpc/cluster.sh fetch HOST                  -> copies all run directories back to results/hpc/HOST/
#   scripts/hpc/cluster.sh status HOST                 -> queue state of this user's jobs
# HOST is an SSH destination such as user@vanda.nus.edu.sg. No credentials are stored here.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BASE='ring-attention'
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20)
action="${1:?action}"; host="${2:?host}"
short="${host#*@}"; short="${short%%.*}"

# Everything a job can read is staged, and exactly what is staged is hashed.
STAGED=(include src apps tests CMakeLists.txt scripts/hpc scripts/extract_qkv.py)
FILES="-type f -not -name *.pyc -not -path */__pycache__/*"
tree_tag() {
    (cd "$ROOT" && set -f && LC_ALL=C find "${STAGED[@]}" $FILES -print0 | LC_ALL=C sort -z | xargs -0 shasum -a 256 | shasum -a 256 | cut -c1-10)
}

case "$action" in
tag)
    tree_tag
    ;;
push)
    tag="$(tree_tag)"
    if "${SSH[@]}" "$host" "test -d $BASE/src/$tag" 2>/dev/null; then
        echo "source $tag already staged on $short" >&2
    else
        "${SSH[@]}" "$host" "mkdir -p $BASE/src $BASE/build $BASE/runs && rm -rf $BASE/src/$tag.partial" 2>/dev/null
        (cd "$ROOT" && rsync -aR -e "${SSH[*]}" --exclude '__pycache__' --exclude '*.pyc' "${STAGED[@]}" "$host:$BASE/src/$tag.partial/" 2>/dev/null)
        # The staged copy must hash to the tag: the tree may have changed between hashing and copying.
        staged="$("${SSH[@]}" "$host" "cd $BASE/src/$tag.partial && set -f && LC_ALL=C find ${STAGED[*]} $FILES -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -c1-10" 2>/dev/null)"
        if [[ "$staged" != "$tag" ]]; then
            echo "staged copy hashes to '$staged', expected $tag; nothing staged" >&2
            "${SSH[@]}" "$host" "rm -rf $BASE/src/$tag.partial" 2>/dev/null
            exit 1
        fi
        git_state="$(cd "$ROOT" && git describe --always --dirty)"
        "${SSH[@]}" "$host" "cd $BASE/src && printf '%s\n' '$tag' '$git_state' > $tag.partial/SOURCE_TAG && mv $tag.partial $tag && chmod -R a-w $tag" 2>/dev/null
    fi
    echo "$tag"
    ;;
submit)
    tag="${3:?tag}"; job="${4:?job script}"; name="${5:?run name}"; shift 5
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    run="$BASE/runs/$name-$stamp"
    "${SSH[@]}" "$host" "bash -l -s" 2>/dev/null <<REMOTE
set -euo pipefail
mkdir -p "\$HOME/$run"
test -f "\$HOME/$BASE/src/$tag/scripts/hpc/jobs/$job"
# Submit from the (writable) run directory and let PBS place <name>.o<id> there.
# An explicit -o path made every start attempt fail on Vanda (direct-write output).
cd "\$HOME/$run"
job_id=\$(qsub -N "$name" -j oe \
    -v "ATTN_SRC=\$HOME/$BASE/src/$tag,ATTN_RUN=\$HOME/$run,ATTN_TAG=$tag,ATTN_BUILD_ROOT=\$HOME/$BASE/build${ATTN_JOB_ENV:+,$ATTN_JOB_ENV}" \
    $(printf '%q ' "$@") "\$HOME/$BASE/src/$tag/scripts/hpc/jobs/$job")
printf '%s\n' "\$job_id" > "\$HOME/$run/job.id"
printf '%s\n' "qsub $(printf '%q ' "$@") scripts/hpc/jobs/$job" > "\$HOME/$run/submit.txt"
printf '%s %s\n' "\$job_id" "$run"
REMOTE
    ;;
status)
    "${SSH[@]}" "$host" "qstat -u \$USER 2>/dev/null | tail -n +3; for f in $BASE/runs/*/job.id; do d=\$(dirname \$f); s=running/queued; test -f \$d/exit-status.txt && s=\"exit \$(cat \$d/exit-status.txt)\"; echo \"\$(cat \$f) \$(basename \$d) \$s\"; done" 2>/dev/null
    ;;
fetch)
    mkdir -p "$ROOT/results/hpc/$short"
    rsync -a -e "${SSH[*]}" "$host:$BASE/runs/" "$ROOT/results/hpc/$short/" 2>/dev/null
    ls "$ROOT/results/hpc/$short"
    ;;
*) echo "unknown action $action" >&2; exit 2;;
esac
