import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

R = Path("benchmark/results")
F = Path("benchmark/figures")
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
    }
)
GOOD, BAD, INK = "#5b8f6e", "#c0715f", "#1f3a5f"


def _roc(path, title, out):
    z = np.load(path)
    y = z["y"]
    fig, ax = plt.subplots(figsize=(5, 4.2))
    res = json.loads((R / "summary.json").read_text())
    task = "qa" if "qa" in path.name else "summarization"
    names = sorted([k for k in z.files if k not in ("y", "item")], key=lambda k: -res[task]["methods"][k]["auroc"])
    cmap = plt.get_cmap("viridis")
    for i, k in enumerate(names):
        fpr, tpr, _ = roc_curve(y, z[k])
        ax.plot(
            fpr,
            tpr,
            lw=2 if i == 0 else 1.1,
            color=cmap(i / max(1, len(names) - 1) * 0.85),
            label=f"{k} ({res[task]['methods'][k]['auroc']:.2f})",
        )
    ax.plot([0, 1], [0, 1], color="#bbb", lw=0.8, ls="--")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(title)
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(F / out)
    plt.close(fig)


def make_all():
    F.mkdir(parents=True, exist_ok=True)
    _roc(R / "qa_scores.npz", "halueval qa: detecting hallucinated answers", "roc_qa.png")
    _roc(R / "sum_scores.npz", "halueval summarization", "roc_summarization.png")

    # the original method's score distribution against its fixed thresholds
    z = np.load(R / "qa_scores.npz")
    y, c = z["y"], -z["original: cosine of nli logits"]
    fig, ax = plt.subplots(figsize=(5.5, 3))
    bins = np.linspace(min(c.min(), -1), 1, 60)
    ax.hist(c[y == 0], bins, color=GOOD, alpha=0.6, label="faithful answers")
    ax.hist(c[y == 1], bins, color=BAD, alpha=0.6, label="hallucinated answers")
    for t in (0.4, 0.8):
        ax.axvline(t, color="k", lw=0.8, ls="--")
    ax.set_xlabel("cosine similarity of nli logit vectors (original method)")
    ax.set_title("the original score barely separates the classes")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(F / "original_score_distribution.png")
    plt.close(fig)

    k = "deberta-v3 nli, 1 - p(entail)"
    s = z[k]
    fig, ax = plt.subplots(figsize=(5.5, 3))
    bins = np.linspace(0, 1, 41)
    ax.hist(s[y == 0], bins, color=GOOD, alpha=0.6, label="faithful answers")
    ax.hist(s[y == 1], bins, color=BAD, alpha=0.6, label="hallucinated answers")
    ax.set_yscale("log")
    ax.set_xlabel("1 - p(entailment | knowledge, answer), deberta-v3")
    ax.set_title("a cross-encoder pushes the classes to opposite ends")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(F / "cross_encoder_distribution.png")
    plt.close(fig)

    res = json.loads((R / "summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, task in zip(axes, ("qa", "summarization")):
        m = res[task]["methods"]
        names = sorted(m, key=lambda k: m[k]["auroc"])
        ax.barh(
            names,
            [m[n]["auroc"] for n in names],
            color=[BAD if n.startswith("original") else INK if "deberta" in n else "#bbb" for n in names],
        )
        ax.axvline(0.5, color="k", lw=0.6)
        ax.set_xlim(0.3, 1)
        ax.set_title(task)
        ax.set_xlabel("auroc")
        ax.tick_params(axis="y", labelsize=6.5)
    fig.tight_layout()
    fig.savefig(F / "auroc.png")
    plt.close(fig)
