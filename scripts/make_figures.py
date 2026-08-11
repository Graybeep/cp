"""Paper figures 1 and 2.

Fig 1 (Detect):  per-seed paired endpoints, early vs delayed, one panel per size.
Fig 2 (Reopen):  absolute endpoints beside their nulls, with the pre-registered
                 same-endpoint band drawn.

Numbers are transcribed from their §6 owners (backbone_stats.md, gate_result.json,
week4_close/results.md); see NUMBERS below. `scripts/verify_figure_numbers.py`
re-reads all 27 of them from the runs' metrics.csv and fails loudly on any
disagreement -- run it after editing NUMBERS. It passed exactly (max delta 0.0000)
on 2026-08-07.

Note the one trap it encodes: the 23M d50 seed-1 null is 0.524 from
`..._seed1_recheck`, NOT 0.506 from `..._seed1`. Those are two different trained
models, and the pattern-generated run_id gives the wrong one.

Palette: dataviz categorical slots 1-3 (blue/orange/aqua), validated all-pairs in
light mode (worst CVD dE 9.2, normal-vision 24.0). Aqua sits below 3:1 contrast on
the surface, so the relief rule applies -> every series carries a visible direct
label, and marker shape is redundant with hue for grayscale print.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "logs/figures"
# Designed canvas size in inches, (width, height). At module level because the float's
# page cost follows from its ASPECT RATIO once it is scaled to the column or text width,
# and `scripts/plan_compile.py` prices it from these numbers. Restating them there would
# let the price and the figure drift apart.
FIG1_SIZE = (7.1, 2.45)
FIG2_SIZE = (7.1, 2.70)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
GRID = "#e6e5e1"

# ---- NUMBERS ---------------------------------------------------------------
# per-seed arithmetic exact match at step 5000, seeds 1/2/3
EARLY = {"11M": [0.556, 0.538, 0.520], "23M": [0.798, 0.792, 0.784],
         "60M": [0.798, 0.840, 0.848]}
# 23M seed 1 = 0.524, the canonical `..._seed1_recheck` model (see docstring)
DELAY = {"11M": [0.304, 0.332, 0.436], "23M": [0.524, 0.626, 0.582],
         "60M": [0.574, 0.646, 0.580]}
TREAT = {"23M": [0.752, 0.778, 0.772], "60M": [0.796, 0.818, 0.774]}
# n=1 dose points, 23M seed 1 only. d75 does NOT exist for arithmetic.
DOSE = {25: 0.724, 50: 0.524, 55: 0.504}
BAND = (0.722, 0.812)          # pre-registered same-endpoint band
FLOOR = 0.60                   # pre-registered anchor floor (excludes 11M)


def _style(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig1(path):
    fig, axes = plt.subplots(1, 4, figsize=FIG1_SIZE,
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.85]})
    fig.patch.set_facecolor("#fcfcfb")
    for ax, size in zip(axes[:3], ["11M", "23M", "60M"]):
        _style(ax)
        excluded = size == "11M"
        for i, (e, d) in enumerate(zip(EARLY[size], DELAY[size])):
            c = MUTED if excluded else INK2
            ax.plot([0, 1], [e, d], "-", color=c, lw=0.9, alpha=0.55, zorder=1)
            ax.plot(0, e, "o", ms=6, color=MUTED if excluded else BLUE,
                    mec="#fcfcfb", mew=1.2, zorder=3)
            ax.plot(1, d, "s", ms=5.5, color=MUTED if excluded else ORANGE,
                    mec="#fcfcfb", mew=1.2, zorder=3)
        ax.axhline(FLOOR, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=0)
        ax.set_xlim(-0.42, 1.42)
        ax.set_ylim(0.25, 0.92)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["early", "delayed"], fontsize=8.5)
        d_mean = np.mean(EARLY[size]) - np.mean(DELAY[size])
        title = f"{size}   $\\Delta$ {d_mean:.3f}"
        ax.set_title(title, fontsize=9, color=INK, pad=7,
                     fontweight="normal" if excluded else "bold")
        if excluded:
            # above the data, not through it: the 11M points run 0.30-0.56
            ax.text(0.5, 0.845, "excluded by\npre-registered floor", ha="center",
                    va="bottom", fontsize=7.2, color=MUTED, linespacing=1.25)
        if size == "11M":
            ax.set_ylabel("arithmetic exact match", fontsize=8.5, color=INK2)
            ax.text(1.40, FLOOR + 0.012, "floor 0.60", ha="right", va="bottom",
                    fontsize=7, color=MUTED)
        else:
            ax.set_yticklabels([])
    # direct labels on the first non-excluded panel
    axes[1].text(-0.30, EARLY["23M"][0] + 0.035, "early", fontsize=8,
                 color=BLUE, fontweight="bold")
    axes[1].text(1.32, DELAY["23M"][1] + 0.030, "delayed", fontsize=8,
                 color=ORANGE, fontweight="bold", ha="right")

    ax = axes[3]
    _style(ax)
    xs = sorted(DOSE)
    ax.plot(xs, [DOSE[x] for x in xs], "D", ms=5, color=MUTED,
            mec="#fcfcfb", mew=1.2, ls="none")
    # d50 and d55 are 5 points apart on a 50-point axis, so numeric ticks and a
    # shared label offset both collide. Direct labels, staggered: the label *is*
    # the x value, and dropping the tick axis also discourages reading a curve
    # through three n = 1 points.
    stagger = {25: (0, 8, "bottom"), 50: (0, 8, "bottom"), 55: (0, -9, "top")}
    for x in xs:
        dx, dy, va = stagger[x]
        ax.annotate(f"d{x}", (x, DOSE[x]), textcoords="offset points",
                    xytext=(dx, dy), ha="center", va=va, fontsize=7, color=MUTED)
    ax.set_xlim(15, 65)
    ax.set_ylim(0.25, 0.92)
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_xlabel("onset (% of training)", fontsize=8, color=INK2)
    ax.set_title("dose points, $n$ = 1", fontsize=9, color=INK, pad=7)
    ax.text(40, 0.285, "23M seed 1 only;\nno trend fitted", ha="center",
            va="bottom", fontsize=7.2, color=MUTED, linespacing=1.25)

    fig.tight_layout(w_pad=0.9)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=220, facecolor="#fcfcfb",
                    bbox_inches="tight")
    plt.close(fig)


def fig2(path):
    fig, ax = plt.subplots(figsize=FIG2_SIZE)
    fig.patch.set_facecolor("#fcfcfb")
    _style(ax)
    ax.axhspan(*BAND, color=AQUA, alpha=0.12, zorder=0)
    ax.axhline(BAND[0], color=AQUA, lw=0.8, alpha=0.55, zorder=0)
    ax.axhline(BAND[1], color=AQUA, lw=0.8, alpha=0.55, zorder=0)

    arms = [("delayed null", DELAY, ORANGE, "s"),
            ("+ LR shift", TREAT, BLUE, "o"),
            ("early anchor", EARLY, INK2, "^")]
    centers = {"23M": 0.0, "60M": 1.25}
    offs = {"delayed null": -0.30, "+ LR shift": 0.0, "early anchor": 0.30}
    for size, c0 in centers.items():
        for name, data, col, mk in arms:
            x = c0 + offs[name]
            vals = np.array(data[size])
            m, sd = vals.mean(), vals.std(ddof=1)
            ax.errorbar(x, m, yerr=sd, fmt="none", ecolor=col, elinewidth=1.6,
                        capsize=4, capthick=1.4, alpha=0.95, zorder=2)
            ax.plot([x] * len(vals), vals, mk, ms=4.5, color=col, alpha=0.30,
                    mec="none", zorder=2)
            ax.plot(x, m, mk, ms=9, color=col, mec="#fcfcfb", mew=1.4, zorder=4)
            ax.annotate(f"{m:.3f}", (x, m), textcoords="offset points",
                        xytext=(11, -3), fontsize=7.8, color=INK)
        ax.text(c0, 0.895, size, ha="center", fontsize=9.5, color=INK,
                fontweight="bold")
    # legend markers must be the series' own markers -- shape carries the series
    # identity in grayscale print, so a legend of uniform dots would contradict
    # the redundancy the palette choice depends on
    for name, _, col, mk in arms:
        ax.plot([], [], mk, color=col, label=name)
    ax.legend(loc="lower right", frameon=False, fontsize=8, ncol=3,
              labelcolor=INK2, handletextpad=0.4, columnspacing=1.4,
              bbox_to_anchor=(1.0, -0.02))
    # band label on the empty left flank, clear of the 60M early anchor
    ax.text(-0.56, np.mean(BAND), "pre-registered\nsame-endpoint band",
            ha="left", va="center", fontsize=7.4, color="#0f6b4c",
            linespacing=1.3)
    ax.set_xlim(-0.62, 1.98)
    ax.set_ylim(0.45, 0.93)
    ax.set_xticks([])
    ax.set_ylabel("arithmetic exact match", fontsize=8.5, color=INK2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=220, facecolor="#fcfcfb",
                    bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig1(f"{OUT}/fig1_detect")
    fig2(f"{OUT}/fig2_reopen")
    print(f"wrote {OUT}/fig1_detect.[pdf|png] and {OUT}/fig2_reopen.[pdf|png]")
