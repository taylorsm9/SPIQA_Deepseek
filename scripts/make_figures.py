"""README figures, drawn from runs/analysis.json and the timings below. Writes figures/*.png.

  envs/plots/bin/python scripts/make_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
R = json.load(open(ROOT / "runs/analysis.json"))

NAMES = {"A": "MinerU → V4 Flash", "C": "MinerU + figure images → V4 Flash Vision",
         "E": "MinerU + figure descriptions → V4 Flash", "G": "DeepSeek-OCR-2 → V4 Flash",
         "O": "Source text → V4 Flash"}
BASELINE = "Score difference from MinerU → V4 Flash"
COLORS = {"A": "#555555", "C": "#d1495b", "E": "#2e86ab", "G": "#edae49", "O": "#8a8a8a"}

# Conversion minutes for 300 papers on the RTX 4090, read from the MinerU and OCR-2 run logs. Answering hours are wall-clock from each run's generations.jsonl (A and O ran interleaved
# in one 8.8 h run, so A gets half).
CONVERT_MIN = {"A": 29, "C": 29, "E": 29 + 47, "G": 45}
ANSWER_H = {"A": 8.8 / 2, "C": 6.4, "E": 4.9, "G": 6.6}


def diffs(arm, split="overall"):
    """(difference vs A, lo, hi) from the analysis file."""
    if arm == "C":
        d = R["1_main_C_vs_A"][split]
    elif arm == "E":
        d = R["7_figure_descriptions_E"]["E_vs_A"][split]
    elif arm == "G":
        d = R["8_ocr2_pipeline_G"]["G_vs_A"][split]
    else:  # O: the file stores A - O, flip it to O - A
        d = R["3_text_loss_A_vs_O"][split]
        return -d["A-O"], -d["hi"], -d["lo"]
    return d[f"{arm}-A"], d["lo"], d["hi"]


def style(ax):
    ax.axvline(0, color="#999", lw=1, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#eee", zorder=-1)


def fig_main():
    arms = ["C", "E", "G", "O"]
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    for i, a in enumerate(reversed(arms)):
        d, lo, hi = diffs(a)
        ax.plot([lo, hi], [i, i], color=COLORS[a], lw=3, solid_capstyle="round")
        ax.plot(d, i, "o", color=COLORS[a], ms=9)
        ax.text(hi + 0.15, i, f"{d:+.1f}  ({lo:+.1f} to {hi:+.1f})", va="center", fontsize=9)
    ax.set_yticks(range(len(arms)), [NAMES[a] for a in reversed(arms)])
    ax.set_xlim(-3, 6.5)
    ax.set_xlabel(f"{BASELINE} (63.7), points, with 95% range")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "difference_from_text_only.png", dpi=160)


def fig_splits():
    arms = ["C", "E", "G"]
    splits = ["test-A", "test-B", "test-C"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.6), sharey=True)
    for ax, sp in zip(axes, splits):
        for i, a in enumerate(reversed(arms)):
            d, lo, hi = diffs(a, sp)
            ax.plot([lo, hi], [i, i], color=COLORS[a], lw=3, solid_capstyle="round")
            ax.plot(d, i, "o", color=COLORS[a], ms=8)
        ax.set_title(sp, fontsize=10)
        ax.set_xlim(-5, 7)
        style(ax)
    axes[0].set_yticks(range(len(arms)), [NAMES[a] for a in reversed(arms)])
    axes[1].set_xlabel(f"{BASELINE}, points, with 95% range")
    fig.tight_layout()
    fig.savefig(OUT / "difference_by_part.png", dpi=160)


def fig_reftype():
    groups = [("table", "Table"), ("plot", "Plot or chart"), ("schematic", "Diagram"),
              ("visualization", "Photo or visualization"), ("other figure", "Other figure"),
              ("multiple types", "Several of the above")]
    src = {"C": R["2_by_reference_type"], "E": R["7_figure_descriptions_E"]["E_vs_A_by_reference_type"],
           "G": R["8_ocr2_pipeline_G"]["G_vs_A_by_reference_type"]}
    arms = ["C", "E", "G"]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ticks, labels = [], []
    for gi, (key, label) in enumerate(reversed(groups)):
        base = gi * 4
        for j, a in enumerate(arms):
            d = src[a][key]
            y = base + (len(arms) - 1 - j)
            ax.plot([d["lo"], d["hi"]], [y, y], color=COLORS[a], lw=2.5, solid_capstyle="round")
            ax.plot(d[f"{a}-A"], y, "o", color=COLORS[a], ms=6, label=NAMES[a] if gi == 0 else None)
        ticks.append(base + 1)
        labels.append(f"{label}\n({src['C'][key]['n']} questions)")
    ax.set_yticks(ticks, labels, fontsize=9)
    ax.set_xlabel(f"{BASELINE}, points, with 95% range")
    style(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=1, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "difference_by_question_type.png", dpi=160)


def fig_time():
    arms = ["A", "C", "E", "G"]
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    for i, a in enumerate(reversed(arms)):
        conv, ans = CONVERT_MIN[a] / 60, ANSWER_H[a]
        ax.barh(i, conv, height=0.6, color=COLORS[a], alpha=0.4)
        ax.barh(i, ans, left=conv, height=0.6, color=COLORS[a])
        ax.text(conv + ans + 0.1, i, f"{conv + ans:.1f} h", va="center", fontsize=9)
    ax.set_yticks(range(len(arms)), [NAMES[a] for a in reversed(arms)])
    ax.set_xlim(0, 8.5)
    ax.set_xlabel("Hours for 300 papers and 967 questions")
    ax.set_title("Light: converting the PDFs.  Dark: answering the questions.", fontsize=9, loc="left")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#eee", zorder=-1)
    fig.tight_layout()
    fig.savefig(OUT / "pipeline_time.png", dpi=160)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    fig_main()
    fig_splits()
    fig_reftype()
    fig_time()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
