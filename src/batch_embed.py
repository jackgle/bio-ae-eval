#!/usr/bin/env python3
"""
batch_embed.py — simple batch embedding generator using bacpipe

requirements
- bacpipe installed/importable with `Loader` and `Embedder` at bacpipe.generate_embeddings
- input dirs should contain class subfolders with wav/flac files (e.g., speciesX_songtypeY/...)

examples
  # process two splits (tp, fp) with a subset of models
  python batch_embed.py \
    --audio-dir /path/to/clips/tp \
    --audio-dir /path/to/clips/fp \
    --model birdnet --model audiomae --out-root embeddings_tp_fp

  # scan all immediate subfolders of a root dir
  python batch_embed.py --audio-root /path/to/clips --model biolingual
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

# quiet tensorflow/absl logging early
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # keep errors only
import tensorflow as tf
from absl import logging as absl_logging

tf.get_logger().setLevel("ERROR")
absl_logging.set_verbosity(absl_logging.ERROR)

from bacpipe.generate_embeddings import Loader, Embedder
import joblib


# ----------------------------- helpers -----------------------------

def discover_dirs(audio_root: Path) -> List[Path]:
    """return immediate child dirs of audio_root (e.g., tp/, fp/)."""
    return [p for p in audio_root.iterdir() if p.is_dir()]


def infer_split_name(audio_dir: Path) -> str:
    """use the directory name as split (e.g., 'tp' or 'fp')."""
    return audio_dir.name


def rel_from(file_path: Path, start_dir: Path) -> str:
    """relative path string from start_dir to file_path."""
    try:
        return str(file_path.relative_to(start_dir))
    except Exception:
        return file_path.name


# ----------------------------- core -----------------------------

def run(audio_dirs: List[Path], models: List[str], out_root: Path, recursive: bool) -> None:
    out_root.mkdir(parents=True, exist_ok=True)

    for model in models:
        print(f"\nmodel: {model}")
        for audio_dir in audio_dirs:
            split = infer_split_name(audio_dir)
            print(f"  scanning: {audio_dir} -> split '{split}'")

            loader = Loader(
                audio_dir=str(audio_dir),
                check_if_combination_exists=False,
                model_name=model,
                testing=False,
                recursively=recursive,
            )

            files = list(loader.files)
            if not files:
                print(f"  warning: no files found in {audio_dir}")
                continue

            embedder = Embedder(model, testing=False)

            embeddings = []
            filenames = []
            labels = []

            for i, file_path in enumerate(files, start=1):
                try:
                    emb = embedder.get_embeddings_from_model(file_path)
                    embeddings.append(emb)
                    filenames.append(rel_from(Path(file_path), audio_dir))
                    labels.append(Path(file_path).parent.name)
                except Exception as e:
                    print(f"  warning: failed on {Path(file_path).name}: {e}")
                if i % 100 == 0:
                    print(f"    processed {i}/{len(files)}")

            # save outputs
            out_dir = out_root / model / split
            out_dir.mkdir(parents=True, exist_ok=True)

            joblib.dump(embeddings, out_dir / "embeddings.joblib")
            with open(out_dir / "labels.json", "w") as f:
                json.dump(labels, f)
            with open(out_dir / "filenames.json", "w") as f:
                json.dump(filenames, f)

            print(f"  saved {len(filenames)} items -> {out_dir}")


# ----------------------------- cli -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="generate embeddings from audio folders using bacpipe models")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--audio-root", type=Path, help="root dir; process each immediate subdir as a split")
    g.add_argument("--audio-dir", type=Path, action="append", help="specific dir(s) to process; repeatable")

    p.add_argument("--model", action="append", required=True, help="model name; repeat for multiple")
    p.add_argument("--out-root", type=Path, required=True, help="output root for embeddings")
    p.add_argument("--recursive", action="store_true", help="recurse within each audio dir")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.audio_root is not None:
        audio_dirs = discover_dirs(args.audio_root)
        if not audio_dirs:
            raise SystemExit(f"no subdirectories found under {args.audio_root}")
    else:
        audio_dirs = args.audio_dir

    run(audio_dirs=audio_dirs, models=args.model, out_root=args.out_root, recursive=args.recursive)


if __name__ == "__main__":
    main()
