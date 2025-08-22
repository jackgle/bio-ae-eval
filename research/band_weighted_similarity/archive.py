"""
band-selective vector search POC

- learns a linear spectral probe mapping embeddings -> band-masked spectral profiles
- derives class-specific band weights from predicted spectra
- compares baseline cosine on embeddings vs. band-selective cosine on predicted spectra

note: single-line comments keep lowercase per user preference.
"""

from __future__ import annotations

# --- imports -----------------------------------------------------------------
import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
from tqdm import tqdm
from scipy.io import wavfile
import scipy.signal as sg
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# --- metrics -----------------------------------------------------------------

def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """cosine similarity rows(a) vs rows(b). returns (a_rows, b_rows)."""
    # ensure 2d
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    na = np.linalg.norm(a, axis=1) + eps
    nb = np.linalg.norm(b, axis=1) + eps
    return (a @ b.T) / (na[:, None] * nb[None, :] + eps)


def precision_at_k(rel_idx_set: Set[int], ranking: np.ndarray, k: int = 10) -> float:
    # compute precision@k for a single query
    topk = ranking[:k]
    return float(sum(int(i in rel_idx_set) for i in topk) / max(1, k))


def average_precision(rel_idx_set: Set[int], ranking: np.ndarray) -> float:
    # compute average precision for a single query
    num_rel = 0
    s = 0.0
    for r, i in enumerate(ranking, 1):
        if i in rel_idx_set:
            num_rel += 1
            s += num_rel / r
    return float(s / max(1, len(rel_idx_set)))


def evaluate_method(sim_rows: np.ndarray, relevant_sets: List[Set[int]], ks: Tuple[int, ...] = (10,)) -> Dict[str, float]:
    # evaluate a retrieval similarity matrix against relevance sets
    out = {f"P@{k}": [] for k in ks}
    mAP = []
    for q, sims in enumerate(sim_rows):
        ranking = np.argsort(-sims)
        rel = relevant_sets[q]
        for k in ks:
            out[f"P@{k}"].append(precision_at_k(rel, ranking, k))
        mAP.append(average_precision(rel, ranking))
    return {k: float(np.mean(v)) for k, v in out.items()} | {"mAP": float(np.mean(mAP))}


# --- spectral probe -----------------------------------------------------------

class SpectralProbe:
    """linear probe: embeddings (n,d) -> band-masked spectral profiles (n,K)."""

    def __init__(self, alpha: float = 1e-2):
        self.alpha = alpha
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.reg: Ridge | None = None
        self.W: np.ndarray | None = None  # (d, K)
        self.intercept_: np.ndarray | None = None  # (K,)

    def fit(self, E: np.ndarray, Y: np.ndarray) -> "SpectralProbe":
        # standardize embeddings for stable coefficients
        X = self.scaler.fit_transform(E)
        self.reg = Ridge(alpha=self.alpha, fit_intercept=True)
        self.reg.fit(X, Y)  # multi-output
        W_std = self.reg.coef_.T  # (d_std, K)
        self.W = W_std / (self.scaler.scale_[:, None] + 1e-12)  # map back to original space
        self.intercept_ = self.reg.intercept_.copy()
        return self

    def predict(self, E: np.ndarray) -> np.ndarray:
        # predict nonnegative spectral profiles from embeddings
        assert self.W is not None and self.intercept_ is not None, "probe not fitted"
        Y_hat = E @ self.W + self.intercept_[None, :]
        return np.maximum(Y_hat, 0.0)


# --- band weighting -----------------------------------------------------------

def learn_band_weights(
    Y_hat_train: np.ndarray,
    Y_multilabel_train: np.ndarray,
    smooth_sigma_bins: int | float = 2,
) -> np.ndarray:
    """
    learn class-specific band weights from predicted spectra.

    Y_hat_train: (n_train, K) predicted spectra from probe
    Y_multilabel_train: (n_train, C) binary matrix (1 if class present)

    returns: (C, K) nonnegative weights per class, each normalized to [0, 1]
    """
    C = Y_multilabel_train.shape[1]
    K = Y_hat_train.shape[1]
    Wc = np.zeros((C, K), dtype=float)

    for c in range(C):
        pos = Y_multilabel_train[:, c] == 1
        neg = ~pos
        if pos.sum() == 0 or neg.sum() == 0:
            continue
        mu_pos = Y_hat_train[pos].mean(axis=0)
        mu_neg = Y_hat_train[neg].mean(axis=0)
        w = np.clip(mu_pos - mu_neg, 0.0, None)  # emphasize class-specific excess energy

        # optional gaussian smoothing (in frequency-bin domain)
        if smooth_sigma_bins and smooth_sigma_bins > 0:
            r = int(4 * smooth_sigma_bins + 1)
            x = np.arange(-r, r + 1)
            g = np.exp(-0.5 * (x / (smooth_sigma_bins + 1e-12)) ** 2)
            g /= g.sum()
            w = np.convolve(w, g, mode="same")

        if w.max() > 0:
            w = w / w.max()
        Wc[c] = w

    return Wc


def band_weighted_similarity(Yq: np.ndarray, Y_db: np.ndarray, w: np.ndarray) -> np.ndarray:
    # cosine similarity on spectra, weighted by per-class band mask w (length K)
    Yw = Y_db * w[None, :]
    q = Yq * w
    denom = (np.linalg.norm(Yw, axis=1) * (np.linalg.norm(q) + 1e-12) + 1e-12)
    return (Yw @ q) / denom


# --- targets (band-masked spectral profiles) ---------------------------------

def band_masked_profile(
    x: np.ndarray,
    sr: int,
    f_min: float,
    f_max: float,
    n_fft: int = 1024,
    hop: int = 256,
    n_bins: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    compute a band-masked frequency profile for a target event.
    - masks spectrogram to [f_min, f_max] across all times (use time-cropped clips upstream)
    - averages over time and bins into n_bins across [fmin, fmax]

    returns: (profile[K], f_centers[K])
    """
    if fmax is None:
        fmax = sr / 2

    # compute spectrogram (magnitude, power-scaled)
    f, t, S = sg.spectrogram(
        x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, scaling="spectrum", mode="magnitude"
    )

    # frequency mask
    f_mask = (f >= f_min) & (f <= f_max)
    S_masked = np.zeros_like(S)
    S_masked[f_mask] = S[f_mask]

    # average over time
    S_band = S_masked.mean(axis=1)  # (n_freq,)

    # bin to fixed grid
    edges = np.linspace(fmin, fmax, n_bins + 1)
    y = np.zeros(n_bins, dtype=float)
    for i in range(n_bins):
        in_bin = (f >= edges[i]) & (f < edges[i + 1])
        if np.any(in_bin):
            y[i] = S_band[in_bin].mean()
    y = np.log1p(y)  # mild compression

    f_centers = 0.5 * (edges[:-1] + edges[1:])
    return y, f_centers


# --- helpers for dataset-specific bits --------------------------------------

def get_freq_range(filename: str, annotations_df: pd.DataFrame) -> Tuple[float, float] | None:
    # fetch (f_min, f_max) from a dataframe using recording_id parsed from filename
    base = os.path.basename(filename)
    rec_id = base.split("_")[0]
    row = annotations_df.loc[annotations_df["recording_id"] == rec_id]
    if not row.empty:
        return float(row.iloc[0]["f_min"]), float(row.iloc[0]["f_max"])
    return None


# --- end-to-end POC (class-wise evaluation) ---------------------------------

# def poc_band_selective_search(
#     E: np.ndarray,                 # (n, d) embeddings for all clips (raw, time-cropped)
#     Y_multilabel: np.ndarray,      # (n, C) binary presence matrix (one class per clip ok)
#     band_masked_spectral_profiles: np.ndarray,  # (n, K) targets built from raw clips + f-bands
#     train_idx: np.ndarray,
#     val_idx: np.ndarray,
#     db_idx: np.ndarray,
#     ks: Tuple[int, ...] = (1, 5, 10),
# ) -> Tuple[Dict[int, Dict[str, Dict[str, float]]], np.ndarray, SpectralProbe]:
#     # fit spectral probe on train
#     probe = SpectralProbe(alpha=1e-2).fit(E[train_idx], band_masked_spectral_profiles[train_idx])

#     # predict spectra for splits
#     Yhat_train = probe.predict(E[train_idx])
#     Yhat_val = probe.predict(E[val_idx])
#     Yhat_db = probe.predict(E[db_idx])

#     # learn class-specific band weights on train
#     Wc = learn_band_weights(Yhat_train, Y_multilabel[train_idx], smooth_sigma_bins=2)

#     # baseline cosine on embeddings
#     E_val = E[val_idx]
#     E_db = E[db_idx]
#     sims_base_all = cosine(E_val, E_db)  # (n_val, n_db)

#     # class-wise evaluation
#     results: Dict[int, Dict[str, Dict[str, float]]] = {}
#     class_ids = np.arange(Y_multilabel.shape[1])

#     # map absolute class ids if Y_multilabel columns map differently than labels
#     for c in class_ids:
#         q_mask = Y_multilabel[val_idx, c] == 1
#         if not np.any(q_mask):
#             continue
#         q_rows = np.where(q_mask)[0]

#         # relevance: db items with same class
#         db_mask = Y_multilabel[db_idx, c] == 1
#         rel_idx = set(np.where(db_mask)[0])
#         rel_sets = [rel_idx for _ in q_rows]

#         # skip if weight is empty
#         w_c = Wc[c]
#         if w_c.max() == 0:
#             continue

#         # band-weighted similarity in spectral space
#         sims_band = np.vstack([
#             band_weighted_similarity(Yhat_val[q], Yhat_db, w_c) for q in q_rows
#         ])

#         sims_base = sims_base_all[q_rows]

#         res_base = evaluate_method(sims_base, rel_sets, ks=ks)
#         res_band = evaluate_method(sims_band, rel_sets, ks=ks)
#         results[int(c)] = {"baseline": res_base, "band_selective": res_band}

#     return results, Wc, probe, Yhat_val

def normalize_profiles(Y):
    Y = np.array(Y, dtype=np.float32)
    sums = Y.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return Y / sums

def poc_band_selective_search(
    E: np.ndarray,
    Y_multilabel: np.ndarray,
    band_masked_spectral_profiles: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    db_idx: np.ndarray,
    ks: Tuple[int, ...] = (1, 5, 10),
):
    # normalize targets before fitting
    band_masked_spectral_profiles = normalize_profiles(band_masked_spectral_profiles)

    # fit spectral probe on train
    probe = SpectralProbe(alpha=1e-2).fit(
        E[train_idx], band_masked_spectral_profiles[train_idx]
    )

    # predict spectra for splits
    Yhat_train = normalize_profiles(probe.predict(E[train_idx]))
    Yhat_val   = normalize_profiles(probe.predict(E[val_idx]))
    Yhat_db    = normalize_profiles(probe.predict(E[db_idx]))

    # learn class-specific band weights on train
    Wc = learn_band_weights(Yhat_train, Y_multilabel[train_idx], smooth_sigma_bins=2)

    # baseline cosine on embeddings
    E_val = E[val_idx]
    E_db = E[db_idx]
    sims_base_all = cosine(E_val, E_db)

    # class-wise evaluation loop (unchanged)
    results: Dict[int, Dict[str, Dict[str, float]]] = {}
    class_ids = np.arange(Y_multilabel.shape[1])

    for c in class_ids:
        q_mask = Y_multilabel[val_idx, c] == 1
        if not np.any(q_mask):
            continue
        q_rows = np.where(q_mask)[0]
        db_mask = Y_multilabel[db_idx, c] == 1
        rel_idx = set(np.where(db_mask)[0])
        rel_sets = [rel_idx for _ in q_rows]

        w_c = Wc[c]
        if w_c.max() == 0:
            continue

        sims_band = np.vstack([
            band_weighted_similarity(Yhat_val[q], Yhat_db, w_c) for q in q_rows
        ])
        sims_base = sims_base_all[q_rows]

        res_base = evaluate_method(sims_base, rel_sets, ks=ks)
        res_band = evaluate_method(sims_band, rel_sets, ks=ks)
        results[int(c)] = {"baseline": res_base, "band_selective": res_band}

    return results, Wc, probe, Yhat_val



# --- dataset loading sketch (customize paths) --------------------------------
if __name__ == "__main__":
    # config
    n_bins = 128
    fmin, fmax = 50.0, 16000.0

    # user paths (adjust)
    embeddings_dir = "../../artifacts/embeddings/20250811-075752/perch_bird/tp/"
    clips_dir = "../../data/audio/clips/tp/"
    annotations_csv = "../../data/annotations/train_tp.csv"

    # load metadata
    print("loading metadata …")
    anno = pd.read_csv(annotations_csv)
    embeddings_list = joblib.load(os.path.join(embeddings_dir, "embeddings.joblib"))
    labels_raw = json.load(open(os.path.join(embeddings_dir, "labels.json"), "r"))
    filenames = json.load(open(os.path.join(embeddings_dir, "filenames.json"), "r"))

    # label mapping (single-class per clip)
    class_to_int = {name: idx for idx, name in enumerate(sorted(set(labels_raw)))}
    classes = np.array([class_to_int[name] for name in labels_raw], dtype=int)

    # stack embeddings
    E = np.vstack(embeddings_list).astype(np.float32)

    # build band-masked spectral profiles from raw, time-cropped clips
    print("building band-masked spectral profiles …")
    profiles = []
    for file in tqdm(filenames, desc="profiles"):
        sr, audio = wavfile.read(os.path.join(clips_dir, file))
        # convert to float [-1,1] if integer PCM
        if np.issubdtype(audio.dtype, np.integer):
            maxv = np.iinfo(audio.dtype).max
            audio = audio.astype(np.float32) / max(1.0, maxv)
        fr = get_freq_range(file, anno)
        if fr is None:
            # if missing annotation, fall back to zeros
            y = np.zeros(n_bins, dtype=float)
        else:
            f_min, f_max = fr
            y, _ = band_masked_profile(
                audio, sr, f_min, f_max, n_fft=1024, hop=256, n_bins=n_bins, fmin=fmin, fmax=fmax
            )
        profiles.append(y)

    band_masked_spectral_profiles = np.vstack(profiles).astype(np.float32)

    # one-hot for classes
    class_ids = np.unique(classes)
    cid2idx = {cid: i for i, cid in enumerate(class_ids)}
    C = len(class_ids)
    Y_multilabel = np.zeros((len(E), C), dtype=int)
    for i, cid in enumerate(classes):
        Y_multilabel[i, cid2idx[cid]] = 1

    # splits
    rng = np.random.default_rng(0)
    idx = np.arange(len(E))
    rng.shuffle(idx)
    n_train = int(0.6 * len(idx))
    n_val = int(0.2 * len(idx))
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    db_idx = idx[n_train + n_val :]

    # run POC
    print("running POC …")
    results, class_specific_band_weights, probe, Yhat_val = poc_band_selective_search(
        E, Y_multilabel, band_masked_spectral_profiles, train_idx, val_idx, db_idx, ks=(1, 5, 10)
    )

#     # pretty print results
#     import pprint as _pp

#     _pp.pprint(results)
