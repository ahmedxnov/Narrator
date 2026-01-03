#!/usr/bin/env python
"""Convert Meta Llama checkpoint to Hugging Face format.

Usage:
    python convert_meta_to_hf.py --input-dir "C:/Users/TotsPC/.llama/checkpoints/Llama3.2-3B" --output-dir "models/llama-3.2-3b-hf"
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer


def convert_meta_checkpoint(
    input_dir: Path,
    output_dir: Path,
    model_size: str = "3B"
) -> None:
    """Convert Meta Llama checkpoint to Hugging Face format.
    
    Args:
        input_dir: Directory containing Meta checkpoint files
        output_dir: Directory to save HF format
        model_size: Model size (3B, 1B, etc.)
    """
    print(f"Converting Meta Llama {model_size} checkpoint to Hugging Face format")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    
    # Check required files exist
    required_files = [
        "consolidated.00.pth",
        "params.json",
        "tokenizer.model"
    ]
    
    for fname in required_files:
        fpath = input_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Required file not found: {fpath}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load params
    print("\nLoading Meta checkpoint configuration...")
    with (input_dir / "params.json").open("r") as f:
        params = json.load(f)
    
    print(f"Model parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Create HF config from Meta params
    print("\nCreating Hugging Face configuration...")
    from transformers import LlamaConfig
    
    # Calculate intermediate size: ffn_dim = int(2 * hidden_dim / 3)
    # Then round up to multiple of `multiple_of`
    hidden_dim = params["dim"]
    multiple_of = params.get("multiple_of", 256)
    ffn_dim_multiplier = params.get("ffn_dim_multiplier", 1.0)
    
    # Llama's SwiGLU: intermediate_size = int(2/3 * 4 * hidden_size) rounded to multiple_of
    intermediate_size = int(2 * hidden_dim * 4 / 3)
    intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
    intermediate_size = int(ffn_dim_multiplier * intermediate_size)
    
    config = LlamaConfig(
        vocab_size=params.get("vocab_size", 128256),
        hidden_size=hidden_dim,
        intermediate_size=intermediate_size,
        num_hidden_layers=params["n_layers"],
        num_attention_heads=params["n_heads"],
        num_key_value_heads=params.get("n_kv_heads", params["n_heads"]),
        max_position_embeddings=params.get("max_seq_len", 8192),
        rms_norm_eps=params.get("norm_eps", 1e-5),
        rope_theta=params.get("rope_theta", 500000.0),
        attention_bias=False,
        tie_word_embeddings=False,
    )
    
    # Save config
    config.save_pretrained(output_dir)
    print(f"  Saved config to {output_dir / 'config.json'}")
    
    # Load and convert weights
    print("\nLoading Meta checkpoint weights...")
    checkpoint_path = input_dir / "consolidated.00.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    print(f"  Loaded {len(checkpoint)} weight tensors")
    
    # Initialize HF model
    print("\nInitializing Hugging Face model...")
    model = LlamaForCausalLM(config)
    
    # Convert weight names from Meta to HF format
    print("Converting weight format...")
    hf_state_dict = {}
    
    # Meta -> HF mapping
    for key, value in checkpoint.items():
        # Remove 'tok_embeddings' prefix and map to 'model.embed_tokens'
        if key == "tok_embeddings.weight":
            hf_state_dict["model.embed_tokens.weight"] = value
        
        # Map output layer
        elif key == "output.weight":
            hf_state_dict["lm_head.weight"] = value
        
        # Map norm
        elif key == "norm.weight":
            hf_state_dict["model.norm.weight"] = value
        
        # Map layer weights
        elif key.startswith("layers."):
            # layers.0.attention.wq.weight -> model.layers.0.self_attn.q_proj.weight
            new_key = key.replace("layers.", "model.layers.")
            new_key = new_key.replace("attention.wq.", "self_attn.q_proj.")
            new_key = new_key.replace("attention.wk.", "self_attn.k_proj.")
            new_key = new_key.replace("attention.wv.", "self_attn.v_proj.")
            new_key = new_key.replace("attention.wo.", "self_attn.o_proj.")
            new_key = new_key.replace("feed_forward.w1.", "mlp.gate_proj.")
            new_key = new_key.replace("feed_forward.w2.", "mlp.down_proj.")
            new_key = new_key.replace("feed_forward.w3.", "mlp.up_proj.")
            new_key = new_key.replace("attention_norm.", "input_layernorm.")
            new_key = new_key.replace("ffn_norm.", "post_attention_layernorm.")
            hf_state_dict[new_key] = value
        else:
            print(f"  Warning: Unknown key {key}, skipping")
    
    # Load converted weights into model
    print("\nLoading converted weights into model...")
    missing_keys, unexpected_keys = model.load_state_dict(hf_state_dict, strict=False)
    
    if missing_keys:
        print(f"  Warning: Missing keys: {missing_keys[:5]}..." if len(missing_keys) > 5 else missing_keys)
    if unexpected_keys:
        print(f"  Warning: Unexpected keys: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else unexpected_keys)
    
    # Save model
    print("\nSaving Hugging Face model...")
    model.save_pretrained(output_dir, safe_serialization=True)
    print(f"  Saved model weights to {output_dir}")
    
    # Convert and save tokenizer
    print("\nConverting tokenizer...")
    from transformers import LlamaTokenizerFast
    
    # Copy tokenizer.model
    tokenizer_src = input_dir / "tokenizer.model"
    tokenizer_dst = output_dir / "tokenizer.model"
    shutil.copy2(tokenizer_src, tokenizer_dst)
    
    # Create HF tokenizer
    try:
        tokenizer = LlamaTokenizerFast(
            vocab_file=str(tokenizer_dst),
            legacy=False,
            add_bos_token=True,
            add_eos_token=False,
        )
        tokenizer.save_pretrained(output_dir)
        print(f"  Saved tokenizer to {output_dir}")
    except Exception as e:
        print(f"  Warning: Could not create LlamaTokenizerFast: {e}")
        print(f"  Copied tokenizer.model to {output_dir}")
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  HF model saved to: {output_dir}")
    print(f"  You can now load it with:")
    print(f"    from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"    model = AutoModelForCausalLM.from_pretrained('{output_dir}')")
    print(f"    tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")
    print(f"{'='*60}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Meta Llama checkpoint to Hugging Face format"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing Meta checkpoint (consolidated.00.pth, params.json, tokenizer.model)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/llama-3.2-3b-hf",
        help="Directory to save converted HF model"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="3B",
        help="Model size (default: 3B)"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return 1
    
    try:
        convert_meta_checkpoint(input_dir, output_dir, args.model_size)
    except Exception as e:
        print(f"\nError during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
