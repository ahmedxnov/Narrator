#!/usr/bin/env python
"""Create train/validation split from cleaned dataset.

Usage:
    python create_train_val_split.py --input-dir cleaned_dataset --output-dir data --train-ratio 0.9
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def create_split(
    input_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.9,
    seed: int = 42
) -> None:
    """Split cleaned texts into train and validation sets.
    
    Args:
        input_dir: Directory containing cleaned text files
        output_dir: Directory to create train/ and val/ subdirectories
        train_ratio: Proportion of data for training (default: 0.9)
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Get all text files
    txt_files = list(input_dir.glob("*.txt"))
    
    if not txt_files:
        print(f"Error: No text files found in {input_dir}")
        return
    
    print(f"Found {len(txt_files)} text files")
    
    # Shuffle files
    random.shuffle(txt_files)
    
    # Calculate split point
    n_train = int(len(txt_files) * train_ratio)
    n_val = len(txt_files) - n_train
    
    train_files = txt_files[:n_train]
    val_files = txt_files[n_train:]
    
    print(f"\nSplit: {n_train} train, {n_val} validation ({train_ratio:.0%}/{1-train_ratio:.0%})")
    
    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files to train directory
    print(f"\nCopying {n_train} files to {train_dir}...")
    for i, file in enumerate(train_files, 1):
        shutil.copy2(file, train_dir / file.name)
        if i % 500 == 0:
            print(f"  Copied {i}/{n_train} train files...")
    
    # Copy files to val directory
    print(f"\nCopying {n_val} files to {val_dir}...")
    for i, file in enumerate(val_files, 1):
        shutil.copy2(file, val_dir / file.name)
        if i % 100 == 0:
            print(f"  Copied {i}/{n_val} val files...")
    
    print(f"\n{'='*60}")
    print(f"Split complete!")
    print(f"  Train: {n_train} files in {train_dir}")
    print(f"  Val:   {n_val} files in {val_dir}")
    print(f"  Random seed: {seed} (use same seed for reproducibility)")
    print(f"{'='*60}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create train/val split from cleaned dataset")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="cleaned_dataset",
        help="Directory containing cleaned text files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="split_data",
        help="Directory to create train/val subdirectories"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Proportion of data for training (default: 0.9 for 90/10 split)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return 1
    
    if not 0 < args.train_ratio < 1:
        print(f"Error: train_ratio must be between 0 and 1, got {args.train_ratio}")
        return 1
    
    create_split(input_dir, output_dir, args.train_ratio, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
