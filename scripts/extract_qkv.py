#!/usr/bin/env python3
"""Dump real attention inputs and the reference output of selected layers of a
causal language model (default: Qwen2.5-0.5B) on a long text.

For each requested layer and sequence length N the script writes, as raw
little-endian float32 token-major arrays:

    q.bin      [N][H][d]      queries  after the rotary position embedding (RoPE)
    k.bin      [N][Hkv][d]    keys     after RoPE
    v.bin      [N][Hkv][d]    values
    o_ref.bin  [N][H][d]      PyTorch's attention output, i.e. the input of o_proj
    meta.json  shapes, provenance and a self-check of the extraction

Python is used only for this data preparation. The attention itself is
recomputed by the C++ code; `o_ref.bin` is the independent reference.

The hooks are placed on q_proj/k_proj/v_proj (outputs) and o_proj (input),
which exist in every transformers version; RoPE is then applied here exactly
as the model does (cos/sin captured from the model when it exposes them).
"""
import argparse
import json
import math
import os
import re
import sys
import time

import torch


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope_tables(positions, dim, theta, device):
    """cos/sin tables as built by the Hugging Face rotary embedding (float32)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim))
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def clean_gutenberg(text):
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", text)
    return text[start.end() if start else 0:end.start() if end else len(text)].strip()


einsum_gap = [0.0]      # diagnostic: largest relative gap between einsum and explicit sums


@torch.no_grad()
def self_check(q, k, v, o_ref, heads, kv_heads, rows, check_einsum=False):
    """Recompute causal attention in float64 for a few rows from the dumped
    tensors and compare with PyTorch's output. Returns (max_abs, max_rel_to_scale)."""
    n, d = q.shape[0], q.shape[-1]
    group = heads // kv_heads
    worst_abs, worst_rel = 0.0, 0.0
    scale = 1.0 / math.sqrt(d)
    for i in rows:
        qi = q[i].double()                                   # [H, d]
        ki = k[: i + 1].double()                             # [i+1, Hkv, d]
        vi = v[: i + 1].double()
        kh = torch.arange(heads, device=q.device) // group    # KV head of each query head
        keys, values = ki[:, kh, :], vi[:, kh, :]            # [i+1, H, d]
        scores = (keys * qi[None, :, :]).sum(-1) * scale     # [i+1, H]
        probs = torch.softmax(scores, dim=0)                 # over keys
        out = (probs[:, :, None] * values).sum(0)            # [H, d]
        if check_einsum:
            alt = torch.einsum("hj,jhd->hd", torch.softmax(torch.einsum("hd,jhd->hj", qi, keys) * scale, dim=-1), values)
            einsum_gap[0] = max(einsum_gap[0], float((alt - out).abs().max() / out.abs().max().clamp_min(1e-30)))
        diff = (out - o_ref[i].double()).abs().max().item()
        ref_scale = o_ref[i].double().abs().max().item()
        worst_abs = max(worst_abs, diff)
        worst_rel = max(worst_rel, diff / max(ref_scale, 1e-30))
    return worst_abs, worst_rel


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="local directory of the model")
    parser.add_argument("--text", required=True, help="plain-text file (Project Gutenberg book)")
    parser.add_argument("--tokens", default="8192", help="comma-separated sequence lengths")
    parser.add_argument("--layers", default="0,11,23", help="comma-separated layer indices")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--diagnose", action="store_true", help="print per-row checks and continue on failure")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers

    lengths = [int(x) for x in args.tokens.split(",")]
    layers = [int(x) for x in args.layers.split(",")]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
                                                 attn_implementation="sdpa").to(args.device).eval()
    config = model.config
    heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
    d = config.hidden_size // heads
    theta = float(getattr(config, "rope_theta", 10000.0))
    print(f"model layers={config.num_hidden_layers} H={heads} Hkv={kv_heads} d={d} theta={theta}", flush=True)

    with open(args.text, encoding="utf-8") as source:
        text = clean_gutenberg(source.read())
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    print(f"text characters={len(text)} tokens={len(ids)}", flush=True)

    captured = {}

    def keep(name):
        def hook(_module, _inputs, output):
            captured[name] = output[0].detach()          # drop the batch dimension
        return hook

    def keep_input(name):
        def hook(_module, inputs):
            captured[name] = inputs[0][0].detach()
        return hook

    def keep_rope(_module, _inputs, output):
        captured["cos"], captured["sin"] = output[0][0].detach(), output[1][0].detach()

    # Ground truth of what the model computed: record the arguments and the
    # result of every scaled_dot_product_attention call (one per layer).
    import torch.nn.functional as F
    original_sdpa = F.scaled_dot_product_attention
    sdpa_calls = []

    def recording_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        result = original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)
        layer = len(sdpa_calls)
        record = {"is_causal": bool(is_causal), "mask": None if attn_mask is None else tuple(attn_mask.shape),
                  "scale": kwargs.get("scale"), "shape": tuple(query.shape)}
        if layer in layers:
            record.update(q=query[0].detach(), k=key[0].detach(), v=value[0].detach(), o=result[0].detach())
        sdpa_calls.append(record)
        return result

    F.scaled_dot_product_attention = recording_sdpa

    handles = []
    blocks = model.model.layers
    for layer in layers:
        attn = blocks[layer].self_attn
        handles += [attn.q_proj.register_forward_hook(keep(f"q{layer}")),
                    attn.k_proj.register_forward_hook(keep(f"k{layer}")),
                    attn.v_proj.register_forward_hook(keep(f"v{layer}")),
                    attn.o_proj.register_forward_pre_hook(keep_input(f"o{layer}"))]
    if hasattr(model.model, "rotary_emb"):
        handles.append(model.model.rotary_emb.register_forward_hook(keep_rope))

    for n in lengths:
        if n > len(ids):
            print(f"skip N={n}: the text has only {len(ids)} tokens", flush=True)
            continue
        captured.clear()
        sdpa_calls.clear()
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        begin = time.time()
        try:
            with torch.no_grad():
                model(input_ids=ids[:n][None].to(args.device), use_cache=False)
        except torch.OutOfMemoryError as error:
            print(f"N={n}: out of memory in the model forward pass ({str(error)[:120]})", flush=True)
            torch.cuda.empty_cache()
            continue
        forward_seconds = time.time() - begin
        peak_gb = torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else float("nan")
        positions = torch.arange(n, device=args.device)
        if "cos" in captured:
            cos, sin, rope_source = captured["cos"].float(), captured["sin"].float(), "captured from the model"
        else:
            cos, sin = rope_tables(positions, d, theta, args.device)
            rope_source = "recomputed (Hugging Face formula)"
        for layer in layers:
            q = captured[f"q{layer}"].float().view(n, heads, d)
            k = captured[f"k{layer}"].float().view(n, kv_heads, d)
            v = captured[f"v{layer}"].float().view(n, kv_heads, d)
            o_ref = captured[f"o{layer}"].float().view(n, heads, d)
            q = q * cos[:, None, :] + rotate_half(q) * sin[:, None, :]
            k = k * cos[:, None, :] + rotate_half(k) * sin[:, None, :]
            rows = sorted(set([0, 1, n // 3, n // 2, n - 2, n - 1] + list(range(0, n, max(1, n // 24)))))
            if len(sdpa_calls) == len(blocks):
                call = sdpa_calls[layer]                        # tensors are [H, N, d]
                tq, tk, tv, to = (call[x].float().transpose(0, 1) for x in ("q", "k", "v", "o"))
                group = heads // kv_heads
                rel = lambda a, b: float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))
                report = {
                    "sdpa_is_causal": call["is_causal"], "sdpa_mask": call["mask"], "sdpa_scale": call["scale"],
                    "q_hook_vs_sdpa": rel(q, tq), "k_hook_vs_sdpa": rel(k, tk[:, ::group, :]),
                    "v_hook_vs_sdpa": rel(v, tv[:, ::group, :]), "o_hook_vs_sdpa": rel(o_ref, to),
                    "kv_replication_exact": bool((tk.reshape(n, kv_heads, group, d) == tk[:, ::group, None, :]).all()),
                }
                print(f"  N={n} layer={layer} capture check: {report}", flush=True)
                # The recorded tensors are authoritative; keep one K/V head per group.
                q, k, v, o_ref = tq, tk[:, ::group, :], tv[:, ::group, :], to
                rope_source = "post-RoPE tensors recorded at the attention call"
                if not call["is_causal"] and call["mask"] is None and n > 1:
                    print("  WARNING: the attention call was neither causal nor masked", flush=True)
            max_abs, max_rel = self_check(q, k, v, o_ref, heads, kv_heads, rows, check_einsum=args.diagnose)
            if args.diagnose:
                print(f"    einsum vs explicit sums on the checked rows: largest relative gap {einsum_gap[0]:.2e}", flush=True)
                # Is the disagreement floating-point conditioning? Compare PyTorch's
                # float32 result with (a) PyTorch's own attention in float64 on the same
                # tensors and (b) the size of the terms that cancel inside a score.
                group = heads // kv_heads
                kn = k.double().norm(dim=-1)                       # [N, Hkv]
                qn = q.double().norm(dim=-1)
                print(f"    |k| per key: median={float(kn.median()):.3e} max={float(kn.max()):.3e} at position "
                      f"{int(kn.max(dim=1).values.argmax())}; |k_0|={[round(float(x), 2) for x in kn[0]]}; "
                      f"|q| median={float(qn.median()):.3e} max={float(qn.max()):.3e}", flush=True)
                if n <= 4096:
                    kx = k.double().repeat_interleave(group, dim=1)   # [N, H, d]
                    vx = v.double().repeat_interleave(group, dim=1)
                    o64 = original_sdpa(q.double().transpose(0, 1)[None], kx.transpose(0, 1)[None],
                                        vx.transpose(0, 1)[None], is_causal=True)[0].transpose(0, 1)
                    o32 = original_sdpa(q.transpose(0, 1)[None].contiguous(), kx.float().transpose(0, 1)[None].contiguous(),
                                        vx.float().transpose(0, 1)[None].contiguous(), is_causal=True)[0].transpose(0, 1)
                    row_scale = o64.abs().amax(dim=(1, 2)).clamp_min(1e-30)
                    err_model = (o_ref.double() - o64).abs().amax(dim=(1, 2)) / row_scale
                    err_again = (o32.double() - o64).abs().amax(dim=(1, 2)) / row_scale
                    bad = (err_model > 1e-3).nonzero().flatten()
                    print(f"    float64 PyTorch attention vs model float32 output: rows with rel error > 1e-3: {len(bad)} of {n}"
                          f" (first {bad[:6].tolist()}, last {bad[-6:].tolist()}); median {float(err_model.median()):.2e}, "
                          f"max {float(err_model.max()):.2e}; a second float32 call differs from float64 by max {float(err_again.max()):.2e}", flush=True)
                    a64, r64 = self_check(q, k, v, o64.float(), heads, kv_heads, rows)
                    print(f"    manual float64 rows vs PyTorch float64 attention: max rel {r64:.2e}", flush=True)
                    # Cancellation inside the scores of the worst row: sum |q_a k_a| vs |sum q_a k_a|.
                    worst = int(err_model.argmax())
                    terms = (q[worst].double()[:, None, :] * kx[: worst + 1].transpose(0, 1)).abs().sum(-1) / math.sqrt(d)
                    score = (q[worst].double()[:, None, :] * kx[: worst + 1].transpose(0, 1)).sum(-1) / math.sqrt(d)
                    top = score.max(dim=1).values
                    print(f"    worst row {worst}: largest sum|q_a k_a|/sqrt(d) = {float(terms.max()):.3e} (key {int(terms.max(dim=0).values.argmax())}), "
                          f"score range [{float(score.min()):.2f}, {float(score.max()):.2f}], top scores per head {[round(float(x), 2) for x in top[:4]]}", flush=True)
                for i in rows[:: max(1, len(rows) // 12)]:
                    a, r = self_check(q, k, v, o_ref, heads, kv_heads, [i])
                    print(f"    row {i}: abs={a:.3e} rel={r:.3e} |o_ref|max={float(o_ref[i].abs().max()):.3e} "
                          f"|q|rms={float(q[i].pow(2).mean().sqrt()):.3e}", flush=True)
            target = os.path.join(args.out, f"N{n}", f"layer{layer:02d}")
            os.makedirs(target, exist_ok=True)
            for name, tensor in (("q", q), ("k", k), ("v", v), ("o_ref", o_ref)):
                array = tensor.contiguous().cpu().numpy().astype("<f4")
                array.tofile(os.path.join(target, f"{name}.bin"))
            meta = {
                "model": os.path.basename(os.path.normpath(args.model)), "layer": layer,
                "N": n, "H": heads, "Hkv": kv_heads, "d": d, "dv": d,
                "layout": "token-major float32 little-endian: q[N][H][d], k[N][Hkv][d], v[N][Hkv][d], o_ref[N][H][d]",
                "rope_applied": True, "rope_theta": theta, "rope_source": rope_source,
                "kv_head_of_query_head": "h // (H // Hkv)", "score_scale": "1/sqrt(d)", "mask": "causal",
                "text": os.path.basename(args.text), "text_tokens_total": int(len(ids)),
                "tokenizer_special_tokens": False,
                "torch": torch.__version__, "transformers": transformers.__version__,
                "device": args.device, "model_dtype": "float32", "attn_implementation": "sdpa",
                "forward_seconds": forward_seconds, "forward_peak_gpu_gb": peak_gb,
                "self_check_rows": len(rows), "self_check_max_abs": max_abs,
                "self_check_max_rel_to_row_scale": max_rel,
                "o_ref_abs_max": float(o_ref.abs().max()), "o_ref_rms": float(o_ref.pow(2).mean().sqrt()),
                "q_rms": float(q.pow(2).mean().sqrt()), "k_rms": float(k.pow(2).mean().sqrt()),
            }
            with open(os.path.join(target, "meta.json"), "w") as sink:
                json.dump(meta, sink, indent=2)
            print(f"N={n} layer={layer}: forward {forward_seconds:.1f}s peak {peak_gb:.1f} GiB, "
                  f"self-check max_abs={max_abs:.3e} rel={max_rel:.3e} ({rope_source})", flush=True)
            if not max_rel < 1e-3:
                print("SELF-CHECK FAILED: dumped tensors do not reproduce the model's attention output", flush=True)
                if not args.diagnose:
                    sys.exit(3)
    for handle in handles:
        handle.remove()
    F.scaled_dot_product_attention = original_sdpa


if __name__ == "__main__":
    main()
