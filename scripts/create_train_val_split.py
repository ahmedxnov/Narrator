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
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def copy_file(args):
    """Copy a single file (for parallel processing).
    
    Args:
        args: Tuple of (source_file, dest_dir)
        
    Returns:
        filename
    """
    source_file, dest_dir = args
    shutil.copy2(source_file, dest_dir / source_file.name)
    return source_file.name


def create_split(
    input_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.9,
    seed: int = 42,
    num_workers: int = None
) -> None:
    """Split cleaned texts into train and validation sets with parallel file copying.
    
    Args:
        input_dir: Directory containing cleaned text files
        output_dir: Directory to create train/ and val/ subdirectories
        train_ratio: Proportion of data for training (default: 0.9)
        seed: Random seed for reproducibility
        num_workers: Number of parallel workers (None = auto)
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
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    print(f"\nCopying files with {num_workers} workers...")
    
    # Copy files to train directory in parallel
    print(f"\nCopying {n_train} files to {train_dir}...")
    train_args = [(file, train_dir) for file in train_files]
    
    with Pool(num_workers) as pool:
        list(tqdm(
            pool.imap(copy_file, train_args),
            total=len(train_files),
            desc="Train files"
        ))
    
    # Copy files to val directory in parallel
    print(f"\nCopying {n_val} files to {val_dir}...")
    val_args = [(file, val_dir) for file in val_files]
    
    with Pool(num_workers) as pool:
        list(tqdm(
            pool.imap(copy_file, val_args),
            total=len(val_files),
            desc="Val files"
        ))
    
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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 1)"
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
    
    create_split(input_dir, output_dir, args.train_ratio, args.seed, args.num_workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
