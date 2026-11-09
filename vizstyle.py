"""Shared print style for all figures (validated categorical slots 1-3 + neutral gray)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
GRAY, LIGHTGRAY, INK, INK2 = "#8a8985", "#d9d8d4", "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": "#ecebe8", "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 2,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})


def panel_label(ax, s):
    ax.text(-0.12, 1.04, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", color=INK)
