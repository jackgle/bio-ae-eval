#!/usr/bin/env python3
"""
quick evaluator for precomputed embeddings arranged like:

embeddings_root/
  method_1/
    tp/
      embeddings.joblib
      labels.json
    fp/
      embeddings.joblib
      labels.json
  method_2/
    ...

for each method, evaluates how well embeddings for each class cluster together
against negatives defined as: fp of the same class + tp of all other classes.

outputs per-class and aggregated AUROC (and optional AP) using a simple centroid
scorer. distances can be cosine or euclidean.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import normalize as sk_normalize


# ---------------------------- io helpers ----------------------------

def _ensure_1d(vec: np.ndarray) -> np.ndarray:
    """
    make sure a single sample embedding is shape (d,).
    if vec has shape (1, d) -> squeeze to (d,)
    if vec has shape (k, d) with k>1 -> average over axis 0
    if vec has shape (d,) -> return as is
    """
    arr = np.asarray(vec)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if arr.shape[0] == 1:
            return arr[0]
        # multiple per-sample vectors: pool by mean
        return arr.mean(axis=0)
    # unexpected shape: flatten last dimension smartly
    return arr.reshape(-1, arr.shape[-1]).mean(axis=0)


def _load_split(split_dir: Path) -> Tuple[List[np.ndarray], List[str]]:
    """
    load embeddings and labels from a tp/ or fp/ directory.
    embeddings.joblib is expected to be a list-like of arrays per sample.
    labels.json is expected to be a list of class labels per sample.
    """
    emb_path = split_dir / "embeddings.joblib"
    lab_path = split_dir / "labels.json"
    if not emb_path.exists():
        raise FileNotFoundError(f"missing {emb_path}")
    if not lab_path.exists():
        raise FileNotFoundError(f"missing {lab_path}")

    embeddings_obj = joblib.load(emb_path)
    # handle cases where a single big array was saved instead of list
    if isinstance(embeddings_obj, np.ndarray):
        # if shape is (n, d) or (n, k, d)
        if embeddings_obj.ndim == 2:
            emb_list = [embeddings_obj[i] for i in range(embeddings_obj.shape[0])]
        else:
            emb_list = [embeddings_obj[i] for i in range(embeddings_obj.shape[0])]
    else:
        emb_list = list(embeddings_obj)

    with open(lab_path, "r") as f:
        labels_list = json.load(f)

    if len(emb_list) != len(labels_list):
        raise ValueError(
            f"mismatch in {split_dir}: {len(emb_list)} embeddings vs {len(labels_list)} labels"
        )

    # normalize shapes
    emb_list = [_ensure_1d(e) for e in emb_list]

    return emb_list, labels_list


def load_method(method_dir: Path) -> List[Dict]:
    """
    return a list of dicts with keys: 'embedding', 'class_label', 'split' ('tp'|'fp')
    """
    records: List[Dict] = []
    for split in ("tp", "fp"):
        split_path = method_dir / split
        if not split_path.exists():
            continue
        emb_list, labels_list = _load_split(split_path)
        for emb, lab in zip(emb_list, labels_list):
            records.append({
                "embedding": np.asarray(emb, dtype=float),
                "class_label": str(lab),
                "split": split,
            })
    if not records:
        raise RuntimeError(f"no data found under {method_dir}")
    return records


# ---------------------------- scoring ----------------------------

def l2_normalize(X: np.ndarray) -> np.ndarray:
    """row-wise l2 normalization"""
    return sk_normalize(X, norm="l2")


def compute_centroid(vectors: np.ndarray) -> np.ndarray:
    """simple arithmetic mean"""
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2D (n, d)")
    return vectors.mean(axis=0)


def score_similarity(X: np.ndarray, center: np.ndarray, metric: str) -> np.ndarray:
    """
    produce a score where higher is more likely to be the target class.
    for cosine: score is cosine similarity
    for euclidean: score is negative euclidean distance (so larger is better)
    """
    if metric == "cosine":
        # cosine similarity = (x·c)/(||x||*||c||)
        c = center / (np.linalg.norm(center) + 1e-12)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        return (Xn @ c)
    elif metric == "euclidean":
        diffs = X - center[None, :]
        dists = np.linalg.norm(diffs, axis=1)
        return -dists
    else:
        raise ValueError(f"unknown metric: {metric}")


# ---------------------------- evaluation ----------------------------

def evaluate_records(records: List[Dict], metric: str = "cosine", l2norm: bool = True,
                     compute_ap: bool = False) -> Dict:
    """
    perform per-class auroc where positives are tp of class c, negatives are fp of class c + tp of all other classes.
    returns per-class metrics and macro/micro aggregates.
    """
    # build arrays
    X = np.stack([r["embedding"] for r in records])
    y_class = np.array([r["class_label"] for r in records])
    y_split = np.array([r["split"] for r in records])

    if l2norm:
        X = l2_normalize(X)

    classes = sorted(set(y_class[y_split == "tp"]))
    results = {"per_class": {}, "macro": {}, "micro": {}}

    # micro: concatenate all per-class decisions by reusing per-class centroids and labels
    micro_scores: List[float] = []
    micro_truth: List[int] = []

    for c in classes:
        pos_idx = np.where((y_class == c) & (y_split == "tp"))[0]
        neg_idx_same_class_fp = np.where((y_class == c) & (y_split == "fp"))[0]
        neg_idx_other_classes_tp = np.where((y_class != c) & (y_split == "tp"))[0]
        neg_idx = np.concatenate([neg_idx_same_class_fp, neg_idx_other_classes_tp])

        if pos_idx.size < 2:
            # need at least 2 to form a stable centroid and compute auc
            results["per_class"][c] = {"n_pos": int(pos_idx.size), "n_neg": int(neg_idx.size), "auroc": np.nan}
            continue

        # centroid from positives only
        center = compute_centroid(X[pos_idx])

        eval_idx = np.concatenate([pos_idx, neg_idx])
        y_true = np.concatenate([np.ones(pos_idx.size, dtype=int), np.zeros(neg_idx.size, dtype=int)])
        scores = score_similarity(X[eval_idx], center, metric)

        try:
            auroc = roc_auc_score(y_true, scores)
            ap = average_precision_score(y_true, scores) if compute_ap else None
        except ValueError:
            auroc = np.nan
            ap = np.nan if compute_ap else None

        results["per_class"][c] = {
            "n_pos": int(pos_idx.size),
            "n_neg": int(neg_idx.size),
            "auroc": float(auroc),
        }
        if compute_ap:
            results["per_class"][c]["ap"] = float(ap) if ap is not None else None

        micro_scores.extend(scores.tolist())
        micro_truth.extend(y_true.tolist())

    # macro average
    valid_aurocs = [v["auroc"] for v in results["per_class"].values() if not np.isnan(v["auroc"])]
    results["macro"]["auroc"] = float(np.mean(valid_aurocs)) if valid_aurocs else np.nan
    if compute_ap:
        valid_aps = [v["ap"] for v in results["per_class"].values() if ("ap" in v and not np.isnan(v["ap"]))]
        results["macro"]["ap"] = float(np.mean(valid_aps)) if valid_aps else np.nan

    # micro average
    if micro_scores:
        results["micro"]["auroc"] = float(roc_auc_score(np.array(micro_truth), np.array(micro_scores)))
        if compute_ap:
            results["micro"]["ap"] = float(average_precision_score(np.array(micro_truth), np.array(micro_scores)))
    else:
        results["micro"]["auroc"] = np.nan
        if compute_ap:
            results["micro"]["ap"] = np.nan

    return results


# ---------------------------- runner ----------------------------

def run(root: Path, metric: str, l2norm: bool, compute_ap: bool, save_csv: Path | None) -> None:
    methods = sorted([p for p in root.iterdir() if p.is_dir()])
    if not methods:
        raise RuntimeError(f"no method_* directories found under {root}")

    # collect a friendly table
    rows = []

    for method_dir in methods:
        try:
            records = load_method(method_dir)
        except Exception as e:
            print(f"warning: skipping {method_dir.name}: {e}")
            continue
        res = evaluate_records(records, metric=metric, l2norm=l2norm, compute_ap=compute_ap)

        # print per-class summary
        print(f"\n=== {method_dir.name} ===")
        print(f"micro AUROC: {res['micro'].get('auroc', np.nan):.4f}")
        if compute_ap:
            print(f"micro AP:    {res['micro'].get('ap', np.nan):.4f}")
        print(f"macro AUROC: {res['macro'].get('auroc', np.nan):.4f}")
        if compute_ap:
            print(f"macro AP:    {res['macro'].get('ap', np.nan):.4f}")
        print("per-class:")
        for c, v in sorted(res["per_class"].items()):
            if np.isnan(v["auroc"]):
                print(f"  {c:20s} n_pos={v['n_pos']:4d} n_neg={v['n_neg']:5d} AUROC=nan")
            else:
                ap_str = f" AP={v['ap']:.4f}" if compute_ap and 'ap' in v else ""
                print(f"  {c:20s} n_pos={v['n_pos']:4d} n_neg={v['n_neg']:5d} AUROC={v['auroc']:.4f}{ap_str}")

        # add to rows for csv
        rows.append({
            "method": method_dir.name,
            "micro_auroc": res["micro"].get("auroc", np.nan),
            "macro_auroc": res["macro"].get("auroc", np.nan),
            **{f"class_{c}_auroc": v["auroc"] for c, v in res["per_class"].items()},
        })
        if compute_ap:
            rows[-1].update({
                "micro_ap": res["micro"].get("ap", np.nan),
                "macro_ap": res["macro"].get("ap", np.nan),
                **{f"class_{c}_ap": v.get("ap", np.nan) for c, v in res["per_class"].items()},
            })

    # optional csv
    if save_csv is not None and rows:
        import csv
        # collect all keys
        keys = sorted({k for row in rows for k in row.keys()})
        with open(save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"\nsaved summary: {save_csv}")


# ---------------------------- cli ----------------------------

def main():
    parser = argparse.ArgumentParser(description="evaluate class separation from embeddings")
    parser.add_argument("root", type=Path, help="path to embeddings_root")
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine",
                        help="similarity metric for scoring")
    parser.add_argument("--no-l2", dest="l2norm", action="store_false", help="disable l2 normalization")
    parser.add_argument("--ap", dest="compute_ap", action="store_true", help="also compute average precision")
    parser.add_argument("--csv", type=Path, default=None, help="optional path to write a summary csv")
    args = parser.parse_args()

    run(args.root, metric=args.metric, l2norm=args.l2norm, compute_ap=args.compute_ap, save_csv=args.csv)


if __name__ == "__main__":
    main()
