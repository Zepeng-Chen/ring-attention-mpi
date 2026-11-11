#!/usr/bin/env python3
"""Tests of attn_analyze (the window statistics of the real-data section).

    python3 tests/analyze_test.py build/attn_analyze
    python3 tests/analyze_test.py mpirun -np 2 build/attn_analyze

1. An independent oracle: every column is recomputed here in plain Python from
   the definitions (log-sum-exp per row, no shared code) and compared.
2. A window whose weights all underflow relative to the row maximum.
3. An exact output that is the zero vector.
Only the standard library is used.
"""
import array
import csv
import math
import os
import random
import subprocess
import sys
import tempfile


def write(directory, name, values):
    with open(os.path.join(directory, name), "wb") as handle:
        array.array("f", values).tofile(handle)


def run(command, directory, n, heads, kv_heads, d, dv, windows, sink):
    out = os.path.join(directory, "out.csv")
    args = ["--data", directory, "--N", n, "--H", heads, "--Hkv", kv_heads, "--d", d, "--dv", dv,
            "--rows", 0, "--windows", ",".join(map(str, windows)), "--sink", sink, "--csv", out]
    env = dict(os.environ, OMP_NUM_THREADS="2")
    done = subprocess.run(command + [str(a) for a in args], env=env, capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit("analyzer failed:\n" + done.stdout + done.stderr)
    with open(out) as handle:
        return {(int(r["row"]), int(r["head"])): r for r in csv.DictReader(handle)}


def softmax_output(scores, values, keys, dv):
    """(normalised output, log of the sum of exp(scores)) over the listed keys."""
    top = max(scores[j] for j in keys)
    weights = [math.exp(scores[j] - top) for j in keys]
    total = math.fsum(weights)
    out = [math.fsum(w * values[j][a] for w, j in zip(weights, keys)) / total for a in range(dv)]
    return out, top + math.log(total)


def distance(a, b):
    return math.sqrt(math.fsum((x - y) ** 2 for x, y in zip(a, b)))


def close(got, want, tolerance=1e-8):
    got = float(got)
    if math.isinf(want) or math.isinf(got):
        return got == want
    return abs(got - want) <= tolerance * max(1.0, abs(want))


def oracle_case(command, failures):
    rng = random.Random(7)
    n, heads, kv_heads, d, dv, windows, sink = 48, 4, 2, 4, 3, [4, 16], 2
    f32 = lambda x: array.array("f", [x])[0]            # the files hold single precision
    q = [f32(rng.uniform(-3, 3)) for _ in range(n * heads * d)]
    k = [f32(rng.uniform(-3, 3)) for _ in range(n * kv_heads * d)]
    v = [f32(rng.uniform(-1, 1)) for _ in range(n * kv_heads * dv)]
    with tempfile.TemporaryDirectory() as directory:
        write(directory, "q.bin", q); write(directory, "k.bin", k); write(directory, "v.bin", v)
        rows = run(command, directory, n, heads, kv_heads, d, dv, windows, sink)
    group = heads // kv_heads
    checked = 0
    for i in range(n):
        for h in range(heads):
            kh = h // group
            scores = [math.fsum(q[(i * heads + h) * d + a] * k[(j * kv_heads + kh) * d + a] for a in range(d)) / math.sqrt(d)
                      for j in range(i + 1)]
            values = [[v[(j * kv_heads + kh) * dv + a] for a in range(dv)] for j in range(i + 1)]
            every = list(range(i + 1))
            exact, log_z = softmax_output(scores, values, every, dv)
            probability = [math.exp(s - log_z) for s in scores]
            norm = math.sqrt(math.fsum(x * x for x in exact))
            want = {
                "entropy": -math.fsum(p * math.log(p) for p in probability if p > 0),
                "max_prob": max(probability),
                "argmax_distance": i - max(every, key=lambda j: scores[j]),
                "mass_first": probability[0],
                "mass_sink": math.fsum(probability[:sink]),
                "output_norm": norm,
            }
            for w in windows:
                window = list(range(max(0, i + 1 - w), i + 1))
                union = sorted(set(window) | set(range(min(sink, i + 1))))
                out_w, _ = softmax_output(scores, values, window, dv)
                out_sw, _ = softmax_output(scores, values, union, dv)
                want["mass_w%d" % w] = math.fsum(probability[j] for j in window)
                want["abs_w%d" % w] = distance(out_w, exact)
                want["abs_sw%d" % w] = distance(out_sw, exact)
                want["err_w%d" % w] = distance(out_w, exact) / norm
                want["err_sw%d" % w] = distance(out_sw, exact) / norm
            for name, value in want.items():
                checked += 1
                if not close(rows[(i, h)][name], value):
                    failures.append("oracle row %d head %d %s: got %s, expected %.12g" % (i, h, name, rows[(i, h)][name], value))
    print("oracle: %d values compared" % checked)


def edge_case(command, failures, name, q, k, v, expect):
    with tempfile.TemporaryDirectory() as directory:
        write(directory, "q.bin", q); write(directory, "k.bin", k); write(directory, "v.bin", v)
        rows = run(command, directory, 2, 1, 1, 1, 1, [1], 0)
    for (row, column), value in expect.items():
        got = rows[(row, 0)][column]
        if not close(got, value):
            failures.append("%s row %d %s: got %s, expected %s" % (name, row, column, got, value))
    print("%s: %d values compared" % (name, len(expect)))


def main():
    command = sys.argv[1:]
    if not command:
        raise SystemExit(__doc__)
    failures = []
    oracle_case(command, failures)
    # Row 1 sees keys 0 and 1 with scores 1000 and 0: the exact output is 1 (the value of key 0);
    # the one-key window holds only key 1, whose weight exp(-1000) underflows, and outputs 2.
    edge_case(command, failures, "underflowing window", [1, 1], [1000, 0], [1, 2],
              {(1, "err_w1"): 1.0, (1, "abs_w1"): 1.0, (1, "err_sw1"): 1.0, (1, "mass_w1"): 0.0,
               (0, "err_w1"): 0.0, (0, "mass_w1"): 1.0})
    # All scores are zero and the values are +1 and -1: the exact output of row 1 is 0, the window gives -1.
    edge_case(command, failures, "zero exact output", [0, 0], [0, 0], [1, -1],
              {(1, "output_norm"): 0.0, (1, "abs_w1"): 1.0, (1, "err_w1"): math.inf, (1, "mass_w1"): 0.5,
               (0, "err_w1"): 0.0, (0, "abs_w1"): 0.0})
    for failure in failures:
        print("FAIL", failure)
    if failures:
        raise SystemExit("%d failure(s)" % len(failures))
    print("all analyzer tests passed")


if __name__ == "__main__":
    main()
