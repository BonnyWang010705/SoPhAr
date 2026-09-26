import matplotlib.pyplot as plt

MM = 1.0 / 25.4
MANUSCRIPT_WIDTH = 130.7 * MM   # \textwidth of sn-jnl, single column, 31 pc
SINGLE_COLUMN = 88 * MM
DOUBLE_COLUMN = 180 * MM
DPI = 330                        # matches the PNGs already in the manuscript

# Reference palette, light surface.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#c9c8c2"
NEUTRAL_LIGHT = "#e4e3df"
DAY = "#eb6834"      # categorical slot 2
NIGHT = "#2a78d6"    # categorical slot 1
LOW = "#d03b3b"
HIGH = "#0ca30c"

BASE_PT = 7
SMALL_PT = 5


def apply():
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": BASE_PT,
        "axes.titlesize": BASE_PT,
        "axes.labelsize": BASE_PT,
        "xtick.labelsize": SMALL_PT + 1,
        "ytick.labelsize": SMALL_PT + 1,
        "legend.fontsize": SMALL_PT + 1,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.0,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42,      # embed as TrueType, keeps text selectable
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
        "mathtext.default": "regular",
    })


def strip(ax, keep_left=False):
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    for side in ["top", "right"] + ([] if keep_left else ["left"]):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    if keep_left:
        ax.spines["left"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)


def panel_letter(ax, letter, dx=-0.055, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=BASE_PT + 1,
            fontweight="bold", color=INK, va="top", ha="left")


def save(fig, out_dir, stem):
    import os
    paths = []
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path)
        paths.append(path)
    return paths
