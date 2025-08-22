#!/usr/bin/env python3
"""
clip_extractor.py — robust batch snippet extractor for long .flac recordings

updates in this version
- clips are centered on the annotation and padded with zeros on both sides to meet the target length (no surrounding context audio)
- if the annotated duration exceeds the target, the clip is a centered crop of the annotation (no padding)
- optional frequency-band filtering per-annotation when f_min/f_max columns are present in the csv
- preserves prior features (resampling, rich metadata, dry-run, progress)

notes
- if scipy is installed, uses butterworth band-pass with filtfilt; otherwise falls back to an fft mask
- the frequency filter is applied after constructing the clip (padding remains zeros)

examples
  # process with zero-padding around detections and band filtering when columns present
  python clip_extractor.py \
    --input-root ../data/audio/source \
    --csv ../data/train_tp.csv:tp --csv ../data/train_fp.csv:fp \
    --out-root ../data/audio/clips \
    --seconds 3.0 --sr 32000 --metadata out_meta.csv

  # disable frequency filtering even if f_min/f_max exist
  python clip_extractor.py --no-freq-filter \
    --csv ../data/train_tp.csv:tp --input-root ../data/audio/source --out-root x --dry-run
"""
from __future__ import annotations

import argparse
import csv as _csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd
import soundfile as sf

try:
    import tensorflow_io as tfio  # type: ignore
    _HAS_TFIO = True
except Exception:
    _HAS_TFIO = False

# optional resampling helper (no hard dependency)
try:
    from scipy.signal import resample_poly, butter, filtfilt  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# ----------------------------- audio ops -----------------------------

def extract_centered_zero_padded(
    y: np.ndarray,
    sr: int,
    t_min: float,
    t_max: float,
    target_seconds: float,
) -> tuple[np.ndarray, dict]:
    """build a clip centered on the annotation, padding with zeros on both sides.

    behavior
    - if annotated segment is shorter than target, return zeros + segment + zeros, centered
    - if annotated segment is longer than target, return a centered crop of the segment
    - never includes context audio outside [t_min, t_max]
    """
    # compute targets in samples
    samples_target = int(round(target_seconds * sr))

    # parse segment endpoints in samples (clamp to recording)
    seg_start = max(0, int(round(t_min * sr)))
    seg_end = min(len(y), int(round(t_max * sr)))
    seg = y[seg_start:seg_end]
    seg_len = len(seg)

    meta = {
        "samples_target": samples_target,
        "seg_len": seg_len,
        "method": "",
        "pad_left": 0,
        "pad_right": 0,
    }

    if seg_len >= samples_target:
        # crop centered within the segment
        center = seg_len // 2
        half = samples_target // 2
        start = max(0, center - half)
        end = start + samples_target
        if end > seg_len:
            end = seg_len
            start = end - samples_target
        clip = seg[start:end]
        meta["method"] = "crop"
    else:
        # pad with zeros to center the segment within the target window
        rem = samples_target - seg_len
        pad_left = rem // 2
        pad_right = rem - pad_left
        clip = np.pad(seg, (pad_left, pad_right), mode="constant")
        meta.update({
            "method": "zero_pad",
            "pad_left": int(pad_left),
            "pad_right": int(pad_right),
        })

    # guard for off-by-one due to rounding
    if len(clip) != samples_target:
        if len(clip) > samples_target:
            clip = clip[:samples_target]
        else:
            clip = np.pad(clip, (0, samples_target - len(clip)), mode="constant")

    return clip.astype(np.float32, copy=False), meta


def _read_soundfile(path: Path) -> Tuple[np.ndarray, int]:
    """try reading with soundfile."""
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        # mixdown: simple first-channel pick to keep maximal speed; change if you prefer mean
        audio = audio[:, 0]
    return audio.astype(np.float32, copy=False), int(sr)


def _read_tfio(path: Path) -> Tuple[np.ndarray, int]:
    """fallback reader using tensorflow-io (slower but robust for some flac variants)."""
    if not _HAS_TFIO:
        raise RuntimeError("tensorflow-io not available")
    audio_tensor = tfio.audio.AudioIOTensor(str(path))
    audio = audio_tensor.to_tensor().numpy()
    if audio.ndim == 1:
        mono = audio
    else:
            # take first channel
            mono = audio[:, 0]
    return mono.astype(np.float32, copy=False), int(audio_tensor.rate.numpy())


def load_audio(path: Path) -> Tuple[np.ndarray, int]:
    """load mono audio (float32) and sample rate, trying sf then tfio."""
    try:
        return _read_soundfile(path)
    except Exception as e_sf:
        try:
            return _read_tfio(path)
        except Exception as e_t:
            raise RuntimeError(f"failed to read {path.name} via soundfile ({e_sf}) and tfio ({e_t})")


def maybe_resample(y: np.ndarray, sr: int, target_sr: Optional[int]) -> Tuple[np.ndarray, int]:
    """resample if target_sr is set and differs; uses resample_poly if scipy is installed."""
    if target_sr is None or target_sr == sr:
        return y, sr
    if not _HAS_SCIPY:
        raise RuntimeError("--sr requested but scipy is not installed; install scipy or omit --sr")
    # rational factor resampling with reasonable default gcd handling
    from math import gcd
    g = gcd(sr, target_sr)
    up = target_sr // g
    down = sr // g
    y_rs = resample_poly(y, up, down)
    return y_rs.astype(np.float32, copy=False), target_sr

# ----------------------------- frequency filtering -----------------------------

def _butter_bandpass(sig: np.ndarray, sr: int, fmin: float, fmax: float) -> np.ndarray:
    """apply a zero-phase butterworth band-pass; clamps edges safely."""
    nyq = sr * 0.5
    lo = max(1.0, float(fmin)) / nyq
    hi = min(float(fmax), nyq - 1.0) / nyq
    if not (0 < lo < hi < 1):
        # fall back to passthrough if invalid
        return sig
    b, a = butter(N=4, Wn=[lo, hi], btype="band")
    return filtfilt(b, a, sig).astype(np.float32, copy=False)


def _fft_bandpass(sig: np.ndarray, sr: int, fmin: float, fmax: float) -> np.ndarray:
    """simple real-fft mask band-pass, keeps bins in [fmin, fmax]."""
    n = len(sig)
    spec = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mask = (freqs >= float(fmin)) & (freqs <= float(fmax))
    spec[~mask] = 0
    out = np.fft.irfft(spec, n=n)
    return out.astype(np.float32, copy=False)


def maybe_band_filter(
    clip: np.ndarray,
    sr: int,
    fmin: Optional[float],
    fmax: Optional[float],
    enable: bool,
) -> tuple[np.ndarray, str]:
    """apply band-pass when fmin/fmax provided and enabled; returns (clip, method)."""
    if not enable or fmin is None or fmax is None or not np.isfinite([fmin, fmax]).all():
        return clip, "none"
    if fmax <= 0 or fmin >= fmax:
        return clip, "none"
    method = "scipy_butter" if _HAS_SCIPY else "fft_mask"
    if _HAS_SCIPY:
        return _butter_bandpass(clip, sr, float(fmin), float(fmax)), method
    else:
        return _fft_bandpass(clip, sr, float(fmin), float(fmax)), method

# ----------------------------- io + metadata -----------------------------

@dataclass
class Row:
    recording_id: str
    species_id: int
    songtype_id: int
    t_min: float
    t_max: float
    f_min: Optional[float] = None
    f_max: Optional[float] = None


def read_annotations(csv_path: Path) -> pd.DataFrame:
    # be forgiving with whitespace
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    required = {"recording_id", "species_id", "songtype_id", "t_min", "t_max"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")
    # make optional frequency columns standardized if present
    if "f_min" in df.columns and "f_max" in df.columns:
        df["f_min"] = pd.to_numeric(df["f_min"], errors="coerce")
        df["f_max"] = pd.to_numeric(df["f_max"], errors="coerce")
    else:
        # add as NaN columns for uniform downstream handling
        df["f_min"] = np.nan
        df["f_max"] = np.nan
    return df


# ----------------------------- core routine -----------------------------

def process_csv(
    csv_path: Path,
    split_name: str,
    input_root: Path,
    out_root: Path,
    seconds: float,
    target_sr: Optional[int],
    overwrite: bool,
    dry_run: bool,
    apply_freq_filter: bool,
    metadata_rows: list,
) -> Tuple[int, int]:
    # read annotations
    df = read_annotations(csv_path)
    grouped = df.groupby("recording_id")

    processed_count = 0
    error_count = 0

    total_groups = len(grouped)
    print(f"\nprocessing {csv_path.name} -> split '{split_name}' ({total_groups} files)")

    for gi, (rec_id, group) in enumerate(tqdm(grouped, total=total_groups), start=1):
        input_file = input_root / f"{rec_id}.flac"
        if not input_file.exists():
            print(f"  warning: file not found - {input_file}")
            error_count += len(group)
            continue
        try:
            audio, sr = load_audio(input_file)
            audio, sr = maybe_resample(audio, sr, target_sr)
        except Exception as e:
            print(f"  error: could not load {input_file.name} - {e}")
            error_count += len(group)
            continue

        # per-clip processing
        for idx, row in group.iterrows():
            try:
                clip, clip_meta = extract_centered_zero_padded(
                    audio,
                    sr,
                    float(row["t_min"]),
                    float(row["t_max"]),
                    seconds,
                )

                # optional frequency filter using annotation band
                fmin = row.get("f_min", np.nan)
                fmax = row.get("f_max", np.nan)
                fmin_val = float(fmin) if pd.notna(fmin) else None
                fmax_val = float(fmax) if pd.notna(fmax) else None
                clip, filt_method = maybe_band_filter(clip, sr, fmin_val, fmax_val, apply_freq_filter)

                # destination: out_root/split/speciesX_songtypeY/
                class_folder = f"species{int(row['species_id'])}_songtype{int(row['songtype_id'])}"
                class_path = out_root / split_name / class_folder
                out_name = f"{rec_id}_{float(row['t_min']):.2f}_{float(row['t_max']):.2f}.wav"
                out_path = class_path / out_name

                if dry_run:
                    # skip writing; still record metadata for inspection
                    processed_count += 1
                else:
                    class_path.mkdir(parents=True, exist_ok=True)
                    if out_path.exists() and not overwrite:
                        # skip existing unless overwrite
                        pass
                    else:
                        sf.write(out_path, clip, sr, format="WAV")
                        processed_count += 1

                # collect metadata
                metadata_rows.append({
                    "split": split_name,
                    "recording_id": rec_id,
                    "species_id": int(row["species_id"]),
                    "songtype_id": int(row["songtype_id"]),
                    "t_min": float(row["t_min"]),
                    "t_max": float(row["t_max"]),
                    "f_min": fmin_val,
                    "f_max": fmax_val,
                    "sr": int(sr),
                    "seconds": float(seconds),
                    "seg_seconds": float(max(0.0, float(row["t_max"]) - float(row["t_min"]))),
                    "method": clip_meta["method"],
                    "pad_left": int(clip_meta["pad_left"]),
                    "pad_right": int(clip_meta["pad_right"]),
                    "freq_filter": filt_method,
                    "outfile": str(out_path if not dry_run else class_path / out_name),
                })
            except Exception as e:
                print(f"  error processing clip row {idx} ({rec_id}): {e}")
                error_count += 1

    return processed_count, error_count


# ----------------------------- cli -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="extract fixed-length wav clips centered on annotations with zero padding and optional band-pass")
    p.add_argument("--input-root", type=Path, required=True, help="folder containing .flac files")
    p.add_argument("--out-root", type=Path, required=True, help="output root folder for clips")
    p.add_argument("--csv", action="append", required=True,
                   help="annotation csv with optional split name suffix ':/name' (e.g., path/to.csv:tp). repeatable.")
    p.add_argument("--seconds", type=float, default=3.0, help="target clip length in seconds (default: 3.0)")
    p.add_argument("--sr", type=int, default=None, help="optional resample rate (requires scipy)")
    p.add_argument("--overwrite", action="store_true", help="overwrite existing wav files")
    p.add_argument("--metadata", type=Path, default=None, help="optional path to write per-clip metadata csv")
    p.add_argument("--dry-run", action="store_true", help="don't write audio, just simulate and build metadata")
    p.add_argument("--no-freq-filter", action="store_true", help="disable band-pass even if f_min/f_max columns exist")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # parse csv specs into (path, split) pairs
    csv_specs: list[Tuple[Path, str]] = []
    for spec in args.csv:
        if ":" in spec:
            path_str, split = spec.rsplit(":", 1)
            path = Path(path_str)
        else:
            path = Path(spec)
            split = path.stem  # fallback: use filename stem
        csv_specs.append((path, split))

    # summary trackers
    grand_ok = 0
    grand_err = 0
    metadata_rows: list[dict] = []

    for csv_path, split_name in csv_specs:
        ok, err = process_csv(
            csv_path=csv_path,
            split_name=split_name,
            input_root=args.input_root,
            out_root=args.out_root,
            seconds=args.seconds,
            target_sr=args.sr,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            apply_freq_filter=not args.no_freq_filter,
            metadata_rows=metadata_rows,
        )
        grand_ok += ok
        grand_err += err

    # write metadata if requested
    if args.metadata is not None:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        # write with deterministic column order
        cols = [
            "split", "recording_id", "species_id", "songtype_id",
            "t_min", "t_max", "f_min", "f_max", "sr", "seconds", "seg_seconds",
            "method", "pad_left", "pad_right", "freq_filter", "outfile",
        ]
        # include any extra columns if present
        extra = [c for c in metadata_rows[0].keys() if c not in cols] if metadata_rows else []
        with open(args.metadata, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=cols + extra)
            writer.writeheader()
            for r in metadata_rows:
                writer.writerow(r)
        print(f"\nsaved metadata: {args.metadata}")

    # final summary
    print("\n=== batch complete ===")
    print(f"created clips: {grand_ok}")
    print(f"errors:        {grand_err}")
    if args.dry_run:
        print("note: dry run (no audio written)")
    if args.sr is not None and not _HAS_SCIPY:
        print("note: scipy not installed; resampling used fft (if available) or was skipped")


if __name__ == "__main__":
    main()
