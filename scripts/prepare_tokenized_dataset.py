#!/usr/bin/env python
"""Pre-tokenize and save dataset in Arrow format for fast training.

This eliminates the tokenization bottleneck during training.

Usage:
    python scripts/prepare_tokenized_dataset.py --max_length 1024 --output_dir ~/tokenized_data_1024
"""
from __future__ import annotations

import argparse
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_length', type=int, default=2048, help='Max sequence length')
    parser.add_argument('--output_dir', type=str, default='tokenized_data', help='Output directory')
    parser.add_argument('--model_path', type=str, default='models/llama-3.2-3b-hf', help='Model path')
    parser.add_argument('--train_file', type=str, default='processed_data/train.jsonl', help='Train file')
    parser.add_argument('--val_file', type=str, default='processed_data/val.jsonl', help='Val file')
    args = parser.parse_args()
    
    print("="*60)
    print("Pre-tokenizing Dataset for Fast Training")
    print("="*60)
    
    # Use command-line arguments
    train_file = args.train_file
    val_file = args.val_file
    output_dir = args.output_dir
    model_path = args.model_path
    max_length = args.max_length
    
    print(f"\nLoading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
    
    print(f"\nLoading datasets...")
    print(f"  Train: {train_file}")
    print(f"  Val: {val_file}")
    
    dataset = load_dataset(
        'json',
        data_files={
            'train': train_file,
            'validation': val_file,
        }
    )
    
    print(f"  Train samples: {len(dataset['train']):,}")
    print(f"  Val samples: {len(dataset['validation']):,}")
    
    def tokenize_function(examples):
        """Tokenize text and prepare for CLM."""
        outputs = tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        # For CLM, labels are the same as input_ids
        outputs['labels'] = outputs['input_ids'].copy()
        return outputs
    
    print(f"\nTokenizing datasets (max_length={max_length})...")
    print("This will take a few minutes but only needs to be done once...")
    
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing",
        num_proc=1,  # Single process for Windows compatibility
    )
    
    print(f"\nSaving tokenized datasets to {output_dir}/...")
    Path(output_dir).mkdir(exist_ok=True)
    
    tokenized_datasets.save_to_disk(output_dir)
    
    print("\n" + "="*60)
    print("✅ Success! Tokenized datasets saved.")
    print(f"   Location: {output_dir}/")
    print(f"   Train: {len(tokenized_datasets['train']):,} samples")
    print(f"   Val: {len(tokenized_datasets['validation']):,} samples")
    print("\nNow update training_config.yaml to use tokenized_data/")
    print("="*60)


if __name__ == "__main__":
    main()
