#!/bin/bash
# Option parsing of the CPU benchmark driver: a value that is not a plain finite
# number must be rejected before anything runs (an infinite warm-up time would
# never end, an infinite tolerance would accept every result).
#   bash tests/cli_test.sh build/attn_bench
#   bash tests/cli_test.sh mpirun -np 1 build/attn_bench
set -uo pipefail
(( $# >= 1 )) || { echo "usage: $0 [launcher ...] path/to/attn_bench"; exit 2; }
failures=0
# run SECONDS command... : exit status of the command, 124 if it had to be stopped
run() {
    local limit="$1"; shift
    "$@" > /dev/null 2>&1 < /dev/null &
    local pid=$! waited=0
    while kill -0 "$pid" 2>/dev/null; do
        if (( waited >= limit * 10 )); then kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; return 124; fi
        sleep 0.1; waited=$((waited + 1))
    done
    wait "$pid"
}
base=(--impl S1 --N 64 --validate 4 --warmup 1 --reps 1)
expect() {   # expect accepted|rejected NAME extra options...
    local want="$1" name="$2"; shift 2
    run 60 "${COMMAND[@]}" "${base[@]}" "$@"; local status=$?
    local got=rejected; (( status == 0 )) && got=accepted; (( status == 124 )) && got="did not stop"
    if [[ "$got" == "$want" ]]; then echo "ok    $name"; else echo "FAIL  $name: expected $want, got $got"; failures=$((failures + 1)); fi
}
COMMAND=("$@")
expect accepted "plain values"                    --warmup-seconds 0 --tolerance-scale 0
expect accepted "exponent notation"               --warmup-seconds 1e-3 --tolerance-scale 1e-3
expect rejected "infinite warm-up time"           --warmup-seconds inf
expect rejected "NaN warm-up time"                --warmup-seconds nan
expect rejected "negative warm-up time"           --warmup-seconds -1
expect rejected "warm-up time with trailing text" --warmup-seconds 1s
expect rejected "infinite tolerance scale"        --tolerance-scale inf
expect rejected "NaN tolerance scale"             --tolerance-scale nan
expect rejected "infinite alpha"                  --alpha inf
expect rejected "integer with trailing text"      --block-k 96x
expect rejected "fractional repetition count"     --threads 1.5
expect rejected "unknown option"                  --no-such-option 1
echo
if (( failures )); then echo "$failures test(s) failed"; exit 1; fi
echo "all option-parsing tests passed"
