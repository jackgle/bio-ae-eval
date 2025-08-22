#!/usr/bin/env python3
"""
evaluate class separation using class prototypes built from a *cropped* set
and scores computed on a separate *full* set.

directory layout:

embeddings_cropped/
  tp/
    embeddings.joblib
    labels.json
    filenames.json
  fp/
    embeddings.joblib
    labels.json
    filenames.json

embeddings_full/
  tp/
    embeddings.joblib
    labels.json
    filenames.json
  fp/
    embeddings.joblib
    labels.json
    filenames.json

for each class c:
  - build a centroid prototype from N randomly sampled TP embeddings of class c
    drawn from embeddings_cropped/tp
  - score all evaluation samples from embeddings_full using the centroid for c
  - positives: full/tp of class c
  - negatives: full/fp of class c + full/tp of all other classes

outputs per-class and aggregated AUROC (and optional AP). supports cosine or
euclidean distance. optionally writes a csv summary.

note: if a class has fewer than N tp samples in cropped, use all of them.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import normalize as sk_normalize


# ---------------------------- io helpers ----------------------------

def _ensure_1d(vec: np.ndarray) -> np.ndarray:
    """
    make sure a single sample embedding is shape (d,)
    - if vec has shape (1, d) -> squeeze to (d,)
    - if vec has shape (k, d) with k>1 -> average over axis 0
    - if vec has shape (d,) -> return as is
    """
    arr = np.asarray(vec)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if arr.shape[0] == 1:
            return arr[0]
        return arr.mean(axis=0)
    return arr.reshape(-1, arr.shape[-1]).mean(axis=0)


def _load_split(split_dir: Path) -> Tuple[List[np.ndarray], List[str], Optional[List[str]]]:
    """
    load embeddings, labels, filenames from a tp/ or fp/ directory
    - embeddings.joblib: list-like of arrays per sample, or ndarray (n, d) / (n, k, d)
    - labels.json: list of class labels per sample
    - filenames.json: optional list of filenames per sample (ignored if missing)
    """
    emb_path = split_dir / "embeddings.joblib"
    lab_path = split_dir / "labels.json"
    fn_path = split_dir / "filenames.json"
    if not emb_path.exists():
        raise FileNotFoundError(f"missing {emb_path}")
    if not lab_path.exists():
        raise FileNotFoundError(f"missing {lab_path}")

    embeddings_obj = joblib.load(emb_path)
    if isinstance(embeddings_obj, np.ndarray):
        emb_list = [embeddings_obj[i] for i in range(embeddings_obj.shape[0])]
    else:
        emb_list = list(embeddings_obj)

    with open(lab_path, "r") as f:
        labels_list = json.load(f)

    filenames_list: Optional[List[str]] = None
    if fn_path.exists():
        with open(fn_path, "r") as f:
            filenames_list = json.load(f)

    if len(emb_list) != len(labels_list):
        raise ValueError(
            f"mismatch in {split_dir}: {len(emb_list)} embeddings vs {len(labels_list)} labels"
        )
    if filenames_list is not None and len(filenames_list) != len(labels_list):
        raise ValueError(
            f"mismatch in {split_dir}: {len(filenames_list)} filenames vs {len(labels_list)} labels"
        )

    emb_list = [_ensure_1d(e) for e in emb_list]
    return emb_list, labels_list, filenames_list


def load_dir(root_dir: Path) -> List[Dict]:
    """
    return list of dicts with keys:
      'embedding' : np.ndarray
      'class_label' : str
      'split' : 'tp' | 'fp'
      'filename' : Optional[str]
    """
    records: List[Dict] = []
    for split in ("tp", "fp"):
        split_path = root_dir / split
        if not split_path.exists():
            continue
        emb_list, labels_list, filenames_list = _load_split(split_path)
        for i, (emb, lab) in enumerate(zip(emb_list, labels_list)):
            rec = {
                "embedding": np.asarray(emb, dtype=float),
                "class_label": str(lab),
                "split": split,
            }
            if filenames_list is not None:
                rec["filename"] = filenames_list[i]
            records.append(rec)
    if not records:
        raise RuntimeError(f"no data found under {root_dir}")
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
    produce a score where higher is more likely to be the target class
    - cosine: cosine similarity
    - euclidean: negative euclidean distance
    """
    if metric == "cosine":
        c = center / (np.linalg.norm(center) + 1e-12)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        return Xn @ c
    elif metric == "euclidean":
        diffs = X - center[None, :]
        dists = np.linalg.norm(diffs, axis=1)
        return -dists
    else:
        raise ValueError(f"unknown metric: {metric}")


# ---------------------------- prototypes ----------------------------

def build_prototypes_from_cropped(
    cropped_records: List[Dict],
    n_proto: int,
    l2norm: bool,
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    build per-class centroid prototypes from cropped tp samples
    - sample up to n_proto tp embeddings per class without replacement
    - apply l2 normalization before centroid if l2norm is True (keeps train/test consistent)
    """
    rng = np.random.default_rng(seed)
    # collect tp embeddings per class
    class_to_embs: Dict[str, List[np.ndarray]] = {}
    for r in cropped_records:
        if r["split"] != "tp":
            continue
        c = r["class_label"]
        class_to_embs.setdefault(c, []).append(r["embedding"])

    # stack, normalize (optional), sample, then centroid
    prototypes: Dict[str, np.ndarray] = {}
    for c, embs in class_to_embs.items():
        X = np.stack(embs) if len(embs) else None
        if X is None or X.shape[0] == 0:
            continue
        if l2norm:
            X = l2_normalize(X)
        k = min(n_proto, X.shape[0])
        idx = rng.choice(X.shape[0], size=k, replace=False)
        center = compute_centroid(X[idx])
        prototypes[c] = center
    return prototypes


# ---------------------------- evaluation ----------------------------

def evaluate_with_prototypes(
    prototypes: Dict[str, np.ndarray],
    full_records: List[Dict],
    metric: str = "cosine",
    l2norm: bool = True,
    compute_ap: bool = False,
) -> Dict:
    """
    evaluate per-class auroc where positives are full tp of class c,
    negatives are full fp of class c + full tp of all other classes.
    prototypes are fixed centroids computed from cropped tp samples.
    """
    # build arrays from full set
    X = np.stack([r["embedding"] for r in full_records])
    y_class = np.array([r["class_label"] for r in full_records])
    y_split = np.array([r["split"] for r in full_records])

    if l2norm:
        X = l2_normalize(X)

    classes = sorted(list(prototypes.keys()))
    results = {
        "per_class": {},
        "macro": {},
        "micro": {},
        "meta": {
            "n_eval": int(X.shape[0]),
            "n_classes": len(classes),
        },
    }

    micro_scores: List[float] = []
    micro_truth: List[int] = []

    for c in classes:
        if c not in prototypes:
            continue

        pos_idx = np.where((y_class == c) & (y_split == "tp"))[0]
        neg_idx_same_class_fp = np.where((y_class == c) & (y_split == "fp"))[0]
        neg_idx_other_classes_tp = np.where((y_class != c) & (y_split == "tp"))[0]
        # neg_idx = np.concatenate([neg_idx_same_class_fp, neg_idx_other_classes_tp])
        neg_idx = neg_idx_same_class_fp
        
        # need at least 1 pos and 1 neg to compute auc
        if pos_idx.size == 0 or neg_idx.size == 0:
            results["per_class"][c] = {
                "n_pos": int(pos_idx.size),
                "n_neg": int(neg_idx.size),
                "auroc": float("nan"),
            }
            if compute_ap:
                results["per_class"][c]["ap"] = float("nan")
            continue

        center = prototypes[c]
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
    results["macro"]["auroc"] = float(np.mean(valid_aurocs)) if valid_aurocs else float("nan")
    if compute_ap:
        valid_aps = [v["ap"] for v in results["per_class"].values() if ("ap" in v and not np.isnan(v["ap"]))]
        results["macro"]["ap"] = float(np.mean(valid_aps)) if valid_aps else float("nan")

    # micro average
    if micro_scores:
        results["micro"]["auroc"] = float(roc_auc_score(np.array(micro_truth), np.array(micro_scores)))
        if compute_ap:
            results["micro"]["ap"] = float(average_precision_score(np.array(micro_truth), np.array(micro_scores)))
    else:
        results["micro"]["auroc"] = float("nan")
        if compute_ap:
            results["micro"]["ap"] = float("nan")

    return results


# ---------------------------- runner ----------------------------

def run(
    cropped_root: Path,
    full_root: Path,
    metric: str,
    l2norm: bool,
    compute_ap: bool,
    n_proto: int,
    seed: int,
    save_csv: Optional[Path],
) -> None:
    # load datasets
    cropped_records = load_dir(cropped_root)
    full_records = load_dir(full_root)

    # build prototypes from cropped tp samples
    prototypes = build_prototypes_from_cropped(
        cropped_records=cropped_records,
        n_proto=n_proto,
        l2norm=l2norm,
        seed=seed,
    )

    if not prototypes:
        raise RuntimeError("no prototypes could be built; check cropped tp data")

    # evaluate on full
    res = evaluate_with_prototypes(
        prototypes=prototypes,
        full_records=full_records,
        metric=metric,
        l2norm=l2norm,
        compute_ap=compute_ap,
    )

    # print summary
    print("\n=== prototype eval (cropped -> full) ===")
    print(f"cropped: {cropped_root}")
    print(f"full:    {full_root}")
    print(f"classes with prototypes: {len(prototypes)}")
    print(f"metric: {metric} | l2norm: {l2norm} | n_proto: {n_proto} | seed: {seed}")
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

    # optional csv
    if save_csv is not None:
        # build a single-row csv with class metrics expanded
        row = {
            "cropped_dir": str(cropped_root),
            "full_dir": str(full_root),
            "metric": metric,
            "l2norm": l2norm,
            "n_proto": n_proto,
            "seed": seed,
            "micro_auroc": res["micro"].get("auroc", np.nan),
            "macro_auroc": res["macro"].get("auroc", np.nan),
        }
        if compute_ap:
            row.update({
                "micro_ap": res["micro"].get("ap", np.nan),
                "macro_ap": res["macro"].get("ap", np.nan),
            })
        for c, v in res["per_class"].items():
            row[f"class_{c}_auroc"] = v["auroc"]
            if compute_ap:
                row[f"class_{c}_ap"] = v.get("ap", np.nan)

        keys = sorted(row.keys())
        with open(save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerow(row)
        print(f"\nsaved summary: {save_csv}")


# ---------------------------- cli ----------------------------

def main():
    parser = argparse.ArgumentParser(description="prototype-based eval: cropped -> full")
    parser.add_argument("cropped_root", type=Path, help="path to embeddings_cropped (with tp/fp)")
    parser.add_argument("full_root", type=Path, help="path to embeddings_full (with tp/fp)")
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine",
                        help="similarity metric for scoring")
    parser.add_argument("--no-l2", dest="l2norm", action="store_false", help="disable l2 normalization")
    parser.add_argument("--ap", dest="compute_ap", action="store_true", help="also compute average precision")
    parser.add_argument("--n-proto", type=int, default=5, help="number of cropped tp samples per class for prototype")
    parser.add_argument("--seed", type=int, default=123, help="rng seed for prototype sampling")
    parser.add_argument("--csv", type=Path, default=None, help="optional path to write a summary csv")
    args = parser.parse_args()

    run(
        cropped_root=args.cropped_root,
        full_root=args.full_root,
        metric=args.metric,
        l2norm=args.l2norm,
        compute_ap=args.compute_ap,
        n_proto=args.n_proto,
        seed=args.seed,
        save_csv=args.csv,
    )


if __name__ == "__main__":
    main()
