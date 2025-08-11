#!/usr/bin/env python3
"""
clip_extractor.py — robust batch snippet extractor for long .flac recordings

features
- reads csv annotations with columns: recording_id, species_id, songtype_id, t_min, t_max
- loads audio via soundfile with automatic fallback to tensorflow-io for odd .flac files
- extracts fixed-length, center-aligned clips with configurable padding mode
- writes .wav files into class folders: species{species_id}_songtype{songtype_id}/
- progress bar, rich summary, optional per-clip metadata csv
- optional resampling to a target sample rate (uses scipy.signal.resample_poly if available)

examples
  # process one csv into tp/ and another into fp/
  python clip_extractor.py \
    --input-root ../data/audio/source \
    --csv ../data/train_tp.csv:tp --csv ../data/train_fp.csv:fp \
    --out-root ../data/audio/clips \
    --seconds 3.0 --pad constant --sr 32000 --metadata out_meta.csv

  # dry run to see what would be produced
  python clip_extractor.py --csv ../data/train_tp.csv:tp --input-root ../data/audio/source --out-root x --dry-run
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
    from scipy.signal import resample_poly  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# ----------------------------- audio ops -----------------------------

def extract_fixed_clip(y: np.ndarray, sr: int, t_min: float, t_max: float, target_seconds: float,
                       pad_mode: str = "constant") -> np.ndarray:
    """extract a centered, fixed-length window around [t_min, t_max]."""
    # compute targets in samples
    samples_target = int(round(target_seconds * sr))
    half = samples_target // 2

    # parse segment endpoints in samples
    start_sample = int(round(t_min * sr))
    end_sample = int(round(t_max * sr))

    # center of your segment
    center = (start_sample + end_sample) // 2

    # desired window [start_idx, end_idx) of fixed length
    start_idx = center - half
    end_idx = start_idx + samples_target

    # compute how much padding we'll need if indices go out of bounds
    pad_left = max(0, -start_idx)
    pad_right = max(0, end_idx - len(y))

    # clamp indices to valid range
    start_idx = max(0, start_idx)
    end_idx = min(len(y), end_idx)

    # slice the audio
    clip = y[start_idx:end_idx]

    # pad to hit exact length
    if pad_left or pad_right:
        # note: for non-constant padding, pass pad_mode="edge"/"reflect"/etc.
        clip = np.pad(clip, (pad_left, pad_right), mode=pad_mode)

    # if target length is odd, integer math above still yields correct size
    if len(clip) != samples_target:
        # final guard: trim or pad by 1 sample if rounding created off-by-one
        if len(clip) > samples_target:
            clip = clip[:samples_target]
        else:
            clip = np.pad(clip, (0, samples_target - len(clip)), mode=pad_mode)

    return clip


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

# ----------------------------- io + metadata -----------------------------

@dataclass
class Row:
    recording_id: str
    species_id: int
    songtype_id: int
    t_min: float
    t_max: float


def read_annotations(csv_path: Path) -> pd.DataFrame:
    # be forgiving with whitespace
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    required = {"recording_id", "species_id", "songtype_id", "t_min", "t_max"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")
    return df


# ----------------------------- core routine -----------------------------

def process_csv(csv_path: Path, split_name: str, input_root: Path, out_root: Path, seconds: float,
                pad: str, target_sr: Optional[int], overwrite: bool, dry_run: bool,
                metadata_rows: list) -> Tuple[int, int]:
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
                clip = extract_fixed_clip(audio, sr, float(row["t_min"]), float(row["t_max"]), seconds, pad)

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
                    "sr": int(sr),
                    "seconds": float(seconds),
                    "pad": pad,
                    "outfile": str(out_path if not dry_run else class_path / out_name),
                })
            except Exception as e:
                print(f"  error processing clip row {idx} ({rec_id}): {e}")
                error_count += 1

    return processed_count, error_count


# ----------------------------- cli -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="extract fixed-length wav clips from annotated flac recordings")
    p.add_argument("--input-root", type=Path, required=True, help="folder containing .flac files")
    p.add_argument("--out-root", type=Path, required=True, help="output root folder for clips")
    p.add_argument("--csv", action="append", required=True,
                   help="annotation csv with optional split name suffix ':/name' (e.g., path/to.csv:tp). repeatable.")
    p.add_argument("--seconds", type=float, default=3.0, help="target clip length in seconds (default: 3.0)")
    p.add_argument("--pad", type=str, default="constant", choices=["constant", "edge", "reflect"],
                   help="numpy pad mode for out-of-bounds regions")
    p.add_argument("--sr", type=int, default=None, help="optional resample rate (requires scipy)")
    p.add_argument("--overwrite", action="store_true", help="overwrite existing wav files")
    p.add_argument("--metadata", type=Path, default=None, help="optional path to write per-clip metadata csv")
    p.add_argument("--dry-run", action="store_true", help="don't write audio, just simulate and build metadata")
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
            pad=args.pad,
            target_sr=args.sr,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
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
            "t_min", "t_max", "sr", "seconds", "pad", "outfile",
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
        print("note: scipy not installed; resampling was skipped")


if __name__ == "__main__":
    main()
