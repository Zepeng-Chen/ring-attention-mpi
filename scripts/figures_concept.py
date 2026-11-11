#!/usr/bin/env python3
"""Concept diagrams for the report (no measurements): the ring schedule and
the effect of a causal mask on per-rank work for two partitions."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plotstyle as ps
from plotstyle import plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"


def ring_schedule(ranks=4):
    """Rows: ranks. Columns: ring steps. Cell: which K/V shard the rank holds."""
    ps.apply()
    owner_colour = ps.ramp(ranks)                      # shard index is ordinal along the sequence
    fig, (left, right) = plt.subplots(1, 2, figsize=(ps.FULL, 2.7), gridspec_kw={"width_ratios": [1.0, 1.55], "wspace": 0.16})

    # (a) data ownership
    left.set_title("(a) Sequence-sharded ownership")
    left.set_xlim(0, ranks); left.set_ylim(-0.35, 3.1); left.axis("off")
    for r in range(ranks):
        for row, (name, colour) in enumerate((("Q", "#f0efec"), ("K", owner_colour[r]), ("V", owner_colour[r]))):
            y = 2.2 - row * 0.72
            left.add_patch(plt.Rectangle((r + 0.04, y), 0.92, 0.58, facecolor=colour, edgecolor="none"))
            dark = name != "Q" and r >= ranks // 2
            left.text(r + 0.5, y + 0.29, f"${name}_{r}$", ha="center", va="center", fontsize=ps.BASE,
                      color="#ffffff" if dark else ps.INK)
        left.text(r + 0.5, 2.92, f"rank {r}", ha="center", va="bottom", fontsize=ps.BASE, color=ps.INK2)
    left.annotate("", xy=(ranks, 0.42), xytext=(0, 0.42), arrowprops=dict(arrowstyle="->", color=ps.MUTED, lw=0.8))
    left.text(ranks / 2, 0.30, "token position $0 \\ldots N-1$,  $c=N/P$ rows per rank", ha="center", va="top",
              fontsize=ps.BASE, color=ps.INK2)
    left.text(ranks / 2, -0.12, "every query row needs the keys and values of every rank",
              ha="center", va="top", fontsize=ps.BASE, color=ps.INK2)

    # (b) schedule
    right.set_title("(b) Ring schedule: K/V shard held by each rank")
    right.set_xlim(-0.75, ranks + 0.05); right.set_ylim(-0.95, ranks + 0.45); right.axis("off")
    for step in range(ranks):
        right.text(step + 0.5, ranks + 0.12, f"step {step}", ha="center", va="bottom", fontsize=ps.BASE, color=ps.INK2)
        for r in range(ranks):
            owner = (r - step) % ranks
            y = ranks - 1 - r
            right.add_patch(plt.Rectangle((step + 0.05, y + 0.08), 0.74, 0.76, facecolor=owner_colour[owner], edgecolor="none"))
            right.text(step + 0.42, y + 0.46, f"$KV_{owner}$", ha="center", va="center", fontsize=ps.BASE,
                       color="#ffffff" if owner >= ranks // 2 else ps.INK)
            if step < ranks - 1:      # the shard moves to the right neighbour (rank r+1) for the next step
                target = ranks - 1 - ((r + 1) % ranks)
                right.annotate("", xy=(step + 1.05, target + 0.46), xytext=(step + 0.79, y + 0.46),
                               arrowprops=dict(arrowstyle="->", color=ps.MUTED, lw=0.7, shrinkA=0, shrinkB=1,
                                               connectionstyle="arc3,rad=0"))
    for r in range(ranks):
        right.text(-0.08, ranks - 1 - r + 0.46, f"rank {r}", ha="right", va="center", fontsize=ps.BASE, color=ps.INK2)
    right.text(ranks / 2 - 0.1, -0.2, "each step: fold the held shard into $(m,\\ell,a)$ of the local queries,\n"
               "then pass it on; after $P$ steps every rank has seen all $N$ keys",
               ha="center", va="top", fontsize=ps.BASE, color=ps.INK2, linespacing=1.4)
    ps.save(fig, OUT / "concept_ring")


def causal_partitions(n=32, ranks=4):
    """Allowed (query, key) pairs under a causal mask, coloured by the rank
    that owns the query row, for contiguous and zigzag partitions."""
    ps.apply()
    colours = ps.ramp(ranks)
    chunk = n // (2 * ranks)

    def owner_contiguous(i):
        return i // (n // ranks)

    def owner_zigzag(i):
        c = i // chunk
        return c if c < ranks else 2 * ranks - 1 - c

    fig, axes = plt.subplots(1, 3, figsize=(ps.FULL, 2.75), gridspec_kw={"width_ratios": [1, 1, 1.05], "wspace": 0.28})
    work = {}
    for ax, (name, owner) in zip(axes[:2], (("contiguous", owner_contiguous), ("zigzag", owner_zigzag))):
        image = np.ones((n, n, 3))
        counts = np.zeros(ranks)
        for i in range(n):
            rgb = np.array(matplotlib_rgb(colours[owner(i)]))
            for j in range(n):
                if j <= i:
                    image[i, j] = rgb
                    counts[owner(i)] += 1
                else:
                    image[i, j] = matplotlib_rgb("#f0efec")
        work[name] = counts
        ax.imshow(image, interpolation="nearest", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        for side in ("left", "bottom", "top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(False)
        ax.set_title(f"({'ab'[name == 'zigzag']}) {name.capitalize()} partition")
        ax.set_xlabel("key position $j$", labelpad=3)
        ax.set_ylabel("query position $i$", labelpad=3)
    ax = axes[2]
    width = 0.19
    for r in range(ranks):
        xs = np.array([0, 1]) + (r - (ranks - 1) / 2) * width
        heights = [work["contiguous"][r] / work["contiguous"].sum(), work["zigzag"][r] / work["zigzag"].sum()]
        ax.bar(xs, heights, width * 0.86, color=colours[r], label=f"rank {r}", linewidth=0)
    ax.axhline(1 / ranks, color=ps.MUTED, lw=0.8)
    ax.text(1.5, 1 / ranks + 0.008, "equal share", ha="right", va="bottom", fontsize=ps.SMALL, color=ps.INK2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["contiguous", "zigzag"])
    ax.set_ylabel("share of allowed pairs")
    ax.set_title("(c) Work per rank")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(0.0, 1.0), columnspacing=1.0, handlelength=1.0)
    ax.set_ylim(0, 0.56)
    ps.save(fig, OUT / "concept_causal")


def matplotlib_rgb(colour):
    from matplotlib.colors import to_rgb
    return to_rgb(colour)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ring_schedule()
    causal_partitions()
    print("wrote", OUT / "concept_ring.pdf", OUT / "concept_causal.pdf")
