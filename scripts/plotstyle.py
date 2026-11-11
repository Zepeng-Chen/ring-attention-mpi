"""Shared figure style for the report (static PDF figures made with matplotlib).

Colours follow one rule: a colour belongs to an entity, never to its rank in a
plot. The categorical hues are taken in a fixed order from a palette that was
checked for colour-vision-deficiency separation; ordered quantities (problem
size, rank count) use a single-hue light-to-dark ramp instead.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

# --- ink and chrome -------------------------------------------------------
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"
GRAY = "#a5a39c"          # de-emphasised context series

# --- categorical slots (fixed order) --------------------------------------
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")

# Communication variants keep one colour everywhere in the report.
VARIANT = {
    "ring": BLUE, "overlap": ORANGE, "rma_get": AQUA, "allgather": YELLOW,
    "rma_put": MAGENTA, "iallgather": GREEN, "local": GRAY,
}
VARIANT_LABEL = {
    "ring": "M2 ring, blocking", "overlap": "M3 ring, overlapped", "rma_get": "M7 one-sided Get",
    "allgather": "M1 all-gather", "rma_put": "M8 one-sided Put+fence", "iallgather": "M1b nonblocking all-gather",
    "local": "M0 no-communication control",
}
# Short legend labels (the two rings must stay distinguishable).
VARIANT_SHORT = {
    "ring": "M2 ring", "overlap": "M3 overlapped ring", "rma_get": "M7 one-sided Get", "allgather": "M1 all-gather",
    "rma_put": "M8 one-sided Put", "iallgather": "M1b nonblocking all-gather", "local": "M0 control (no communication)",
}
VARIANT_MARKER = {"ring": "o", "overlap": "s", "rma_get": "^", "allgather": "D", "rma_put": "v",
                  "iallgather": "P", "local": "X"}
KERNEL = {"reference": GRAY, "basic": BLUE, "tuned": ORANGE}
KERNEL_LABEL = {"reference": "S0-fp32 naive (two passes)", "basic": "S1 blocked, scalar", "tuned": "S1 blocked, tuned"}
GPU_KERNEL = {"k0": GRAY, "k1": BLUE, "k2": ORANGE}
GPU_LABEL = {"k0": "K0 naive", "k1": "K1 tiled, scores materialised", "k2": "K2 fused"}
PARTITION = {"contiguous": BLUE, "zigzag": ORANGE}

# Single-hue ramp for ordered series (light -> dark).
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]


def ramp(n):
    """n ordered steps of the blue ramp, light to dark."""
    if n <= 1:
        return [RAMP[3]]
    return [RAMP[round(i * (len(RAMP) - 1) / (n - 1))] for i in range(n)]


# --- typography and size --------------------------------------------------
# The report is set in Times at 10 pt on A4 with 22 mm margins (text width 166 mm).
# Figures use the same typeface and are drawn at their final size, so that they
# are included without scaling and their text has one size throughout the report:
# 8 pt for labels and titles, 7.5 pt for ticks and legends, never below 7 pt.
FULL = 6.5      # inches: a figure spanning the text width
HALF = 3.25     # inches: one panel of a two-panel figure
BASE, SMALL, TINY = 8.0, 7.5, 7.0
_FONTS_READY = False


def _register_text_font():
    """Use the report's typeface (TeX Gyre Termes, the base of newtx) if a TeX
    installation provides it; otherwise matplotlib falls back to Times."""
    global _FONTS_READY
    if _FONTS_READY:
        return
    _FONTS_READY = True
    import shutil
    import subprocess
    from matplotlib import font_manager
    if not shutil.which("kpsewhich"):
        return
    for style in ("regular", "bold", "italic", "bolditalic"):
        try:
            path = subprocess.run(["kpsewhich", f"texgyretermes-{style}.otf"], capture_output=True, text=True, timeout=20).stdout.strip()
            if path:
                font_manager.fontManager.addfont(path)
        except (OSError, subprocess.SubprocessError):
            return


def apply():
    _register_text_font()
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "serif", "font.serif": ["TeX Gyre Termes", "Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": BASE, "axes.titlesize": BASE, "axes.titleweight": "bold", "axes.labelsize": BASE,
        "axes.titlelocation": "left", "axes.titlepad": 6,
        "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": SMALL, "ytick.labelsize": SMALL,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 0, "ytick.minor.size": 0,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "lines.linewidth": 1.5, "lines.markersize": 5, "lines.markeredgecolor": SURFACE,
        "lines.markeredgewidth": 1.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": SMALL, "legend.handlelength": 1.8,
        "legend.borderaxespad": 0.2, "legend.labelcolor": INK2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
        "figure.dpi": 150, "savefig.dpi": 300,
    })


def log2_axis(ax, which="x", values=None):
    """Log axis with plain power-of-two tick labels (problem size, core count)."""
    setter = ax.set_xscale if which == "x" else ax.set_yscale
    setter("log", base=2)
    axis = ax.xaxis if which == "x" else ax.yaxis
    if values is not None:
        (ax.set_xticks if which == "x" else ax.set_yticks)(list(values))
    axis.set_major_formatter(FuncFormatter(lambda v, _: human(v)))
    axis.set_minor_formatter(NullFormatter())


def log10_axis(ax, which="y"):
    (ax.set_xscale if which == "x" else ax.set_yscale)("log")
    axis = ax.xaxis if which == "x" else ax.yaxis
    axis.set_minor_formatter(NullFormatter())


def log_ticks(ax, which, values, fmt="{:g}"):
    """Log axis whose ticks are the given values, written as plain numbers. Used for
    time axes that span less than a decade, where decade ticks would show nothing."""
    from matplotlib.ticker import NullLocator
    (ax.set_xscale if which == "x" else ax.set_yscale)("log")
    axis = ax.xaxis if which == "x" else ax.yaxis
    (ax.set_xticks if which == "x" else ax.set_yticks)(list(values))
    axis.set_major_formatter(FuncFormatter(lambda v, _: fmt.format(v)))
    axis.set_minor_locator(NullLocator())


def nice_ticks(low, high, target=4):
    """Round tick values covering [low, high] with about `target` ticks (for log_ticks)."""
    import math
    best = None
    for exponent in range(-4, 5):
        for mantissa in (1, 2, 5):
            step = mantissa * 10.0 ** exponent
            first = math.ceil(low / step - 1e-9)
            last = math.floor(high / step + 1e-9)
            count = last - first + 1
            if count >= 3 and (best is None or abs(count - target) < abs(best[0] - target)):
                best = (count, [round((first + i) * step, 10) for i in range(count)])
    return best[1] if best else [low, high]


def human(value):
    """1024 -> 1,024; 65536 -> 64K; 1048576 -> 1M (tick labels)."""
    value = float(value)
    if value >= 2**20 and value % 2**20 == 0:
        return f"{int(value / 2**20)}M"
    if value >= 16384 and value % 1024 == 0:
        return f"{int(value / 1024)}K"
    if value == int(value):
        return f"{int(value):,}"
    return f"{value:g}"


def end_label(ax, x, y, text, dx=4, dy=0, **kwargs):
    """Direct label next to the last point of a series, in ink (never in the series colour)."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", va="center", ha="left",
                fontsize=SMALL, color=INK2, **kwargs)


def fit_width(fig, target):
    """Resize the canvas until the saved (tightly cropped) figure is `target` inches
    wide. Margins hold text of fixed size, so the difference is added to the axes."""
    pad = plt.rcParams["savefig.pad_inches"]
    for _ in range(5):
        fig.canvas.draw()
        box = fig.get_tightbbox(fig.canvas.get_renderer())
        width = box.width + 2 * pad
        if abs(width - target) < 0.01:
            break
        current, height = fig.get_size_inches()
        fig.set_size_inches(max(current + (target - width), 1.0), height)


def save(fig, path_stem):
    """Save at the width given as figsize: that is the width on the page."""
    fit_width(fig, fig.get_size_inches()[0])
    fig.savefig(f"{path_stem}.pdf")
    fig.savefig(f"{path_stem}.png", dpi=200)
    plt.close(fig)


def check_widths(directory, limit=6.54):
    """Figures are included unscaled: none may be wider than the text (inches)."""
    try:
        import pypdf
    except ImportError:
        return []
    from pathlib import Path
    too_wide = []
    for path in sorted(Path(directory).glob("*.pdf")):
        width = float(pypdf.PdfReader(str(path)).pages[0].mediabox.width) / 72
        if width > limit:
            too_wide.append((path.name, round(width, 2)))
    return too_wide
