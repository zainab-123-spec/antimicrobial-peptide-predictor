"""Figures for the README (reads results/ written by ampred.benchmark)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

SHOW = [
    "logreg[length]",
    "nearest_neighbour[3-mer]",
    "logreg[aac]",
    "logreg[aac+physchem]",
    "random_forest[aac+dpc+physchem]",
    "logreg[esm2_8M]",
    "ensemble[esm2+rf]",
]


def make_figures(results_dir: str = "results", fig_dir: str = "figures") -> list[Path]:
    res = json.loads((Path(results_dir) / "metrics.json").read_text())
    pred = np.load(Path(results_dir) / "test_predictions.npz")
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    y = pred["y"]
    paths = []

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name in SHOW:
        if name in pred.files:
            fpr, tpr, _ = roc_curve(y, pred[name])
            ax.plot(fpr, tpr, label=f"{name} (AUC {res['models'][name]['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="ROC on the official AMP Scanner v2 test set (n=1,424)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    paths.append(fig_dir / "roc_curves.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    names = [n for n in SHOW if n in res["models"]]
    mcc = [res["models"][n]["mcc"] for n in names]
    err = np.array([[m - res["models"][n]["mcc_95ci"][0], res["models"][n]["mcc_95ci"][1] - m]
                    for n, m in zip(names, mcc)]).T
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(names, mcc, xerr=err, color="#4C72B0", capsize=3)
    ax.axvline(0.8204, color="red", ls="--", lw=1, label="AMP Scanner v2 paper (0.820)")
    ax.set(xlabel="Matthews correlation (test set, 95% bootstrap CI)", xlim=(-0.1, 1))
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    paths.append(fig_dir / "mcc_comparison.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bins = np.linspace(0, 1, 26)
    ax.hist(pred["sim"][y == 1], bins=bins, alpha=0.6, label="AMPs")
    ax.hist(pred["sim"][y == 0], bins=bins, alpha=0.6, label="non-AMP decoys")
    ax.set(xlabel="Max 3-mer Jaccard similarity to any training peptide",
           ylabel="Test peptides", title="How 'new' is each test peptide?")
    ax.legend()
    fig.tight_layout()
    paths.append(fig_dir / "test_similarity.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    for p in make_figures():
        print(p)
