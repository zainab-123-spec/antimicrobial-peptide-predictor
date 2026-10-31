"""End-to-end benchmark: train on the official train split, pick the
regularisation strength on the official eval split, report on the untouched
official test split (exactly how AMP Scanner v2 was evaluated)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_split
from .features import featurize
from .metrics import binary_metrics, bootstrap_ci
from .similarity import max_similarity_to_reference

C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def logreg(c: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))


def tune_logreg(x_tr, y_tr, x_ev, y_ev) -> float:
    """Pick C with the best eval-split ROC-AUC."""
    best_c, best_auc = C_GRID[0], -1.0
    for c in C_GRID:
        prob = logreg(c).fit(x_tr, y_tr).predict_proba(x_ev)[:, 1]
        auc = binary_metrics(y_ev, prob)["auc"]
        if auc > best_auc:
            best_c, best_auc = c, auc
    return best_c


def run(
    data_dir: str = "data/raw",
    out_dir: str = "results",
    use_esm: bool = True,
    seed: int = 0,
) -> dict:
    tr, ev, te = (load_split(s, data_dir) for s in ("train", "eval", "test"))
    y_tr, y_ev, y_te = (np.array(s.labels) for s in (tr, ev, te))
    feats: dict[str, tuple] = {}
    for kind in ("length", "aac", "aac+physchem", "aac+dpc+physchem"):
        feats[kind] = tuple(featurize(s.seqs, kind) for s in (tr, ev, te))
    if use_esm:
        from .embed import esm_embed

        feats["esm2_8M"] = tuple(
            esm_embed(s.seqs, cache=f"cache/esm_{name}.npy")
            for s, name in ((tr, "train"), (ev, "eval"), (te, "test"))
        )

    results: dict = {"n": {"train": len(tr), "eval": len(ev), "test": len(te)}, "models": {}}
    probs: dict[str, np.ndarray] = {}

    dummy = DummyClassifier(strategy="prior").fit(feats["length"][0], y_tr)
    probs["majority_class"] = dummy.predict_proba(feats["length"][2])[:, 1]

    for kind in feats:
        x_tr, x_ev, x_te = feats[kind]
        c = tune_logreg(x_tr, y_tr, x_ev, y_ev)
        model = logreg(c).fit(np.vstack([x_tr, x_ev]), np.concatenate([y_tr, y_ev]))
        probs[f"logreg[{kind}]"] = model.predict_proba(x_te)[:, 1]
        results["models"][f"logreg[{kind}]"] = {"C": c}

    x_tr, x_ev, x_te = feats["aac+dpc+physchem"]
    rf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
    rf.fit(np.vstack([x_tr, x_ev]), np.concatenate([y_tr, y_ev]))
    probs["random_forest[aac+dpc+physchem]"] = rf.predict_proba(x_te)[:, 1]

    # "Memorisation" baseline: is a test peptide closer to a known AMP or a known decoy?
    tr_all = tr.seqs + ev.seqs
    y_all = np.concatenate([y_tr, y_ev])
    amp_ref = [s for s, y in zip(tr_all, y_all) if y == 1]
    dec_ref = [s for s, y in zip(tr_all, y_all) if y == 0]
    nn_score = max_similarity_to_reference(te.seqs, amp_ref) - max_similarity_to_reference(
        te.seqs, dec_ref
    )
    probs["nearest_neighbour[3-mer]"] = (nn_score + 1) / 2

    if use_esm:
        p = (probs["logreg[esm2_8M]"] + probs["random_forest[aac+dpc+physchem]"]) / 2
        probs["ensemble[esm2+rf]"] = p

    for name, p in probs.items():
        m = binary_metrics(y_te, p)
        if name != "majority_class":
            m["auc_95ci"] = bootstrap_ci(y_te, p, "auc", seed=seed)
            m["mcc_95ci"] = bootstrap_ci(y_te, p, "mcc", seed=seed)
        results["models"].setdefault(name, {}).update(m)

    # Homology check: does performance hold on test peptides unlike training ones?
    sim = max_similarity_to_reference(te.seqs, tr.seqs + ev.seqs, k=3)
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 1.01)]
    homology = {}
    for lo, hi in bins:
        mask = (sim >= lo) & (sim < hi)
        entry = {"n": int(mask.sum()), "frac_amp": float(y_te[mask].mean()) if mask.any() else None}
        if mask.sum() > 20 and len(np.unique(y_te[mask])) == 2:
            for name in probs:
                if name == "majority_class":
                    continue
                entry[name] = binary_metrics(y_te[mask], probs[name][mask])["auc"]
        homology[f"{lo:.1f}-{min(hi, 1.0):.1f}"] = entry
    results["homology_bins_auc"] = homology
    results["test_similarity_median"] = float(np.median(sim))
    results["test_similarity_median_by_class"] = {
        "amp": float(np.median(sim[y_te == 1])),
        "non_amp": float(np.median(sim[y_te == 0])),
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    np.savez(out / "test_predictions.npz", y=y_te, sim=sim, **{k: v for k, v in probs.items()})
    return results


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Run the AMP benchmark")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--no-esm", action="store_true", help="skip ESM-2 (no torch needed)")
    args = ap.parse_args()
    res = run(args.data_dir, args.out_dir, use_esm=not args.no_esm)
    for name, m in res["models"].items():
        if "auc" in m:
            print(f"{name:38s} AUC={m['auc']:.3f} MCC={m['mcc']:.3f} ACC={m['accuracy']:.3f}")


if __name__ == "__main__":
    main()
