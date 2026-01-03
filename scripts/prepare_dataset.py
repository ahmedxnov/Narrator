#!/usr/bin/env python
"""Prepare dataset for fine-tuning - from text splits to tokenized Arrow format.

Combines chunking and tokenization into one optimized pipeline with parallel processing.

Usage:
    python scripts/prepare_dataset.py --max-length 1024 --output-dir tokenized_data_1024
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator
from multiprocessing import Pool, cpu_count

from transformers import AutoTokenizer
from datasets import Dataset
from tqdm import tqdm
import os

# Suppress tokenizer warnings globally
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_stories(data_dir: Path) -> list[str]:
    """Load all stories from a directory.
    
    Args:
        data_dir: Directory containing .txt story files
        
    Returns:
        List of story texts
    """
    stories = []
    txt_files = sorted(data_dir.glob("*.txt"))
    
    for txt_file in txt_files:
        try:
            text = txt_file.read_text(encoding='utf-8', errors='ignore')
            if text.strip():
                stories.append(text.strip())
        except Exception as e:
            print(f"Warning: Failed to read {txt_file.name}: {e}")
            continue
    
    return stories


def tokenize_story(args):
    """Tokenize a single story (for parallel processing).
    
    Args:
        args: Tuple of (story_text, tokenizer_path, eos_token)
        
    Returns:
        List of token IDs
    """
    story, tokenizer_path, eos_token = args
    # Suppress warnings in worker process
    import warnings
    import logging
    warnings.filterwarnings('ignore')
    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    # Load tokenizer from local path (already cached)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True)
    
    story_with_sep = f"{story}\n{eos_token}\n"
    tokens = tokenizer.encode(story_with_sep, add_special_tokens=False, truncation=False)
    return tokens


def chunk_tokenized_stories(
    all_tokens: list[list[int]],
    max_length: int,
    overlap: int = 128
) -> list[list[int]]:
    """Chunk pre-tokenized stories into fixed-length sequences.
    
    Args:
        all_tokens: List of tokenized stories
        max_length: Maximum sequence length in tokens
        overlap: Number of overlapping tokens between chunks
        
    Returns:
        List of token ID lists (chunked)
    """
    chunks = []
    buffer = []
    
    for tokens in tqdm(all_tokens, desc="Chunking tokenized stories"):
        buffer.extend(tokens)
        
        # Yield chunks when buffer is full
        while len(buffer) >= max_length:
            chunks.append(buffer[:max_length])
            buffer = buffer[max_length - overlap:]
    
    # Yield remaining buffer if it has meaningful content
    if len(buffer) > max_length // 2:
        chunks.append(buffer)
    
    return chunks


def prepare_and_tokenize_split(
    data_dir: Path,
    tokenizer: AutoTokenizer,
    max_length: int,
    overlap: int,
    split_name: str,
    num_workers: int = None
) -> Dataset:
    """Prepare and tokenize a data split with parallel processing.
    
    Args:
        data_dir: Directory containing story files
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
        overlap: Overlap between chunks
        split_name: Name of the split (for logging)
        num_workers: Number of parallel workers (None = auto)
        
    Returns:
        Tokenized Dataset in Arrow format
    """
    print(f"\nProcessing {split_name} set from {data_dir}...")
    
    # Load all stories
    stories = load_stories(data_dir)
    print(f"  Loaded {len(stories):,} stories")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    eos_token = tokenizer.eos_token
    
    # Use sequential processing for 1 worker (avoids multiprocessing overhead)
    if num_workers == 1:
        print(f"  Tokenizing sequentially (1 worker)...")
        # Suppress warnings for long sequences
        import warnings
        import logging
        warnings.filterwarnings('ignore')
        logging.getLogger("transformers").setLevel(logging.ERROR)
        
        all_tokens = []
        for story in tqdm(stories, total=len(stories), desc="Tokenizing stories"):
            story_with_sep = f"{story}\n{eos_token}\n"
            tokens = tokenizer.encode(story_with_sep, add_special_tokens=False, truncation=False)
            all_tokens.append(tokens)
    else:
        # Save tokenizer locally for workers to use (avoids HF rate limits)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer_path = Path(tmpdir) / "tokenizer"
            tokenizer.save_pretrained(tokenizer_path)
            
            # Parallel tokenization using local tokenizer
            print(f"  Tokenizing with {num_workers} workers...")
            
            args_list = [(story, str(tokenizer_path), eos_token) for story in stories]
            
            with Pool(num_workers) as pool:
                all_tokens = list(tqdm(
                    pool.imap(tokenize_story, args_list),
                    total=len(stories),
                    desc="Tokenizing stories"
                ))
    
    # Sequential chunking (maintains context flow across stories)
    token_chunks = chunk_tokenized_stories(all_tokens, max_length, overlap)
    
    print(f"  Created {len(token_chunks):,} chunks")
    
    # Free memory from all_tokens before creating dataset
    del all_tokens
    
    # Create dataset using generator to reduce peak memory usage
    def chunk_generator():
        for chunk in token_chunks:
            yield {"input_ids": chunk, "labels": chunk}
    
    print(f"  Converting to Arrow format (memory-efficient)...")
    dataset = Dataset.from_generator(chunk_generator, num_proc=1)
    
    # Free the token_chunks list
    del token_chunks
    
    return dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare dataset from text splits to tokenized Arrow format"
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="split_data/train",
        help="Directory containing training stories"
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default="split_data/val",
        help="Directory containing validation stories"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tokenized_data",
        help="Directory to save tokenized Arrow datasets"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="Hugging Face model name for tokenizer"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length in tokens"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=128,
        help="Number of overlapping tokens between chunks"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 1)"
    )
    
    args = parser.parse_args()
    
    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    output_dir = Path(args.output_dir)
    
    # Validate input directories
    if not train_dir.exists():
        print(f"Error: Training directory {train_dir} does not exist")
        return 1
    
    if not val_dir.exists():
        print(f"Error: Validation directory {val_dir} does not exist")
        return 1
    
    print("="*60)
    print("Preparing Dataset: Text Splits → Tokenized Arrow Format")
    print("="*60)
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    
    # Process train split
    train_dataset = prepare_and_tokenize_split(
        train_dir,
        tokenizer,
        args.max_length,
        args.overlap,
        "train",
        args.num_workers
    )
    
    # Process validation split
    val_dataset = prepare_and_tokenize_split(
        val_dir,
        tokenizer,
        args.max_length,
        args.overlap,
        "validation",
        args.num_workers
    )
    
    # Save to disk in Arrow format
    print(f"\nSaving tokenized datasets to {output_dir}/...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from datasets import DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
    })
    
    dataset_dict.save_to_disk(str(output_dir))
    
    print("\n" + "="*60)
    print("  Dataset preparation complete!")
    print(f"   Output: {output_dir}/")
    print(f"   Train: {len(train_dataset):,} chunks")
    print(f"   Validation: {len(val_dataset):,} chunks")
    print(f"   Max length: {args.max_length} tokens")
    print(f"   Overlap: {args.overlap} tokens")
    print("\nUpdate training_config.yaml:")
    print(f'  data:\n    tokenized_dir: "{output_dir}"')
    print("="*60)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
