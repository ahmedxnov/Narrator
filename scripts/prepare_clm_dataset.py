#!/usr/bin/env python
"""Prepare dataset for CLM fine-tuning of Llama 3.2 3B.

Creates tokenized chunks suitable for training on RTX 5070 12GB with QLoRA.

Usage:
    python prepare_clm_dataset.py --model-name meta-llama/Llama-3.2-3B --max-length 2048
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

from transformers import AutoTokenizer
from tqdm import tqdm


def load_stories(data_dir: Path) -> Iterator[str]:
    """Load all stories from a directory.
    
    Args:
        data_dir: Directory containing .txt story files
        
    Yields:
        Story text content
    """
    txt_files = sorted(data_dir.glob("*.txt"))
    for txt_file in txt_files:
        try:
            text = txt_file.read_text(encoding='utf-8', errors='ignore')
            if text.strip():
                yield text.strip()
        except Exception as e:
            print(f"Warning: Failed to read {txt_file.name}: {e}")
            continue


def chunk_stories(
    stories: Iterator[str],
    tokenizer: AutoTokenizer,
    max_length: int,
    overlap: int = 128
) -> Iterator[dict]:
    """Chunk stories into fixed-length sequences for CLM.
    
    Args:
        stories: Iterator of story texts
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length in tokens
        overlap: Number of overlapping tokens between chunks
        
    Yields:
        Dictionary with 'text' field containing chunked story
    """
    buffer = []
    buffer_length = 0
    
    for story in stories:
        # Add story separator token
        story_with_sep = f"{story}\n{tokenizer.eos_token}\n"
        
        # Tokenize the story
        tokens = tokenizer.encode(story_with_sep, add_special_tokens=False)
        
        # Add to buffer
        buffer.extend(tokens)
        buffer_length = len(buffer)
        
        # Yield chunks when buffer is full
        while buffer_length >= max_length:
            chunk_tokens = buffer[:max_length]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=False)
            
            yield {"text": chunk_text}
            
            # Remove processed tokens, keep overlap
            buffer = buffer[max_length - overlap:]
            buffer_length = len(buffer)
    
    # Yield remaining buffer if it has meaningful content
    if buffer_length > max_length // 2:  # At least 50% of max_length
        chunk_text = tokenizer.decode(buffer, skip_special_tokens=False)
        yield {"text": chunk_text}


def prepare_dataset(
    train_dir: Path,
    val_dir: Path,
    output_dir: Path,
    model_name: str,
    max_length: int,
    overlap: int
) -> None:
    """Prepare train and validation datasets for CLM fine-tuning.
    
    Args:
        train_dir: Directory with training stories
        val_dir: Directory with validation stories
        output_dir: Directory to save processed datasets
        model_name: Hugging Face model name for tokenizer
        max_length: Maximum sequence length
        overlap: Overlap between chunks
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"\nTokenizer info:")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    
    # Process training set
    print(f"\nProcessing training set from {train_dir}...")
    train_stories = load_stories(train_dir)
    train_chunks = chunk_stories(train_stories, tokenizer, max_length, overlap)
    
    train_output = output_dir / "train.jsonl"
    train_count = 0
    
    with train_output.open('w', encoding='utf-8') as f:
        for chunk in tqdm(train_chunks, desc="Creating train chunks"):
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            train_count += 1
    
    print(f"  Saved {train_count} training chunks to {train_output}")
    
    # Process validation set
    print(f"\nProcessing validation set from {val_dir}...")
    val_stories = load_stories(val_dir)
    val_chunks = chunk_stories(val_stories, tokenizer, max_length, overlap)
    
    val_output = output_dir / "val.jsonl"
    val_count = 0
    
    with val_output.open('w', encoding='utf-8') as f:
        for chunk in tqdm(val_chunks, desc="Creating val chunks"):
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            val_count += 1
    
    print(f"  Saved {val_count} validation chunks to {val_output}")
    
    # Save dataset info
    info = {
        "model_name": model_name,
        "max_length": max_length,
        "overlap": overlap,
        "train_chunks": train_count,
        "val_chunks": val_count,
        "train_file": str(train_output),
        "val_file": str(val_output),
    }
    
    info_file = output_dir / "dataset_info.json"
    with info_file.open('w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Dataset preparation complete!")
    print(f"  Train: {train_count} chunks")
    print(f"  Val:   {val_count} chunks")
    print(f"  Max length: {max_length} tokens")
    print(f"  Overlap: {overlap} tokens")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare dataset for CLM fine-tuning of Llama 3.2 3B"
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
        default="processed_data",
        help="Directory to save processed datasets"
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
        default=2048,
        help="Maximum sequence length in tokens (default: 2048 for 12GB VRAM)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=128,
        help="Number of overlapping tokens between chunks (default: 128)"
    )
    
    args = parser.parse_args()
    
    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    output_dir = Path(args.output_dir)
    
    if not train_dir.exists():
        print(f"Error: Training directory {train_dir} does not exist")
        return 1
    
    if not val_dir.exists():
        print(f"Error: Validation directory {val_dir} does not exist")
        return 1
    
    prepare_dataset(
        train_dir,
        val_dir,
        output_dir,
        args.model_name,
        args.max_length,
        args.overlap
    )
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
