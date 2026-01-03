#!/usr/bin/env python
"""Fine-tune Llama 3.2 3B on children's stories using LoRA.

Optimized for RTX 5070 12GB VRAM.

Usage:
    python scripts/train.py --config configs/training_config.yaml
"""
from __future__ import annotations

import argparse
from datetime import datetime

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    BitsAndBytesConfig,
)


class TimeDisplayCallback(TrainerCallback):
    """Callback to display current time and elapsed time during training."""
    
    def __init__(self):
        self.start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.now()
        print(f"\n⏰ Training started at: {self.start_time.strftime('%I:%M %p')}")
        print("="*60)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.start_time and state.global_step % 100 == 0:
            elapsed = datetime.now() - self.start_time
            hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            current_time = datetime.now().strftime('%I:%M %p')
            print(f"⏰ Current: {current_time} | Elapsed: {hours:02d}h {minutes:02d}m {seconds:02d}s")


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model_and_tokenizer(config: dict):
    """Load model with optional 4-bit quantization (QLoRA) and tokenizer."""
    
    print(f"Loading model: {config['model']['name']}")
    
    # Check if using 4-bit quantization (QLoRA)
    if config['quantization'].get('load_in_4bit', False):
        print("  Using 4-bit QLoRA quantization")
        
        # Configure 4-bit quantization
        compute_dtype = torch.bfloat16 if config['quantization']['bnb_4bit_compute_dtype'] == 'bfloat16' else torch.float16
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=config['quantization']['bnb_4bit_quant_type'],
            bnb_4bit_use_double_quant=config['quantization']['bnb_4bit_use_double_quant'],
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            config['model']['name'],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # EXPLICITLY disable gradient checkpointing if config says false
        if not config['training'].get('gradient_checkpointing', False):
            if hasattr(model, 'gradient_checkpointing_disable'):
                model.gradient_checkpointing_disable()
            model.config.use_cache = True  # Re-enable cache for speed
    else:
        # Load in BF16 without quantization
        print("  Using BF16 precision")
        model = AutoModelForCausalLM.from_pretrained(
            config['model']['name'],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
    
    # Gradient checkpointing (if enabled)
    if config['training'].get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        print("  Using gradient checkpointing")
    else:
        print("  Gradient checkpointing disabled")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'],
        trust_remote_code=True,
    )
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    print(f"  Model loaded with {model.num_parameters():,} parameters")
    print(f"  Tokenizer vocab size: {len(tokenizer)}")
    
    return model, tokenizer


def setup_lora(model, config: dict):
    """Configure and apply LoRA to the model."""
    
    print("Configuring LoRA...")
    
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type=config['lora']['task_type'],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Compile model for faster training (PyTorch 2.x, RTX 50-series benefits greatly)
    if hasattr(torch, 'compile') and config['training'].get('torch_compile', False):
        print("Compiling model with torch.compile (this may take a minute)...")
        model = torch.compile(model, mode="reduce-overhead")
        print("  Model compiled!")
    
    return model


def load_datasets(config: dict):
    """Load pre-tokenized datasets from disk."""
    
    print(f"Loading pre-tokenized datasets...")
    print(f"  From: {config['data']['tokenized_dir']}")
    
    # Load from pre-tokenized Arrow format
    from datasets import load_from_disk
    dataset = load_from_disk(config['data']['tokenized_dir'])
    
    print(f"  Train samples: {len(dataset['train']):,}")
    print(f"  Val samples:   {len(dataset['validation']):,}")
    
    return dataset





def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.2 3B with QLoRA")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training_config.yaml',
        help='Path to training configuration file'
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    torch.manual_seed(config['training']['seed'])
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Apply LoRA
    model = setup_lora(model, config)
    
    # Load pre-tokenized datasets
    tokenized_datasets = load_datasets(config)
    
    # Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_ratio=config['training']['warmup_ratio'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        optim=config['training']['optim'],
        max_grad_norm=config['training']['max_grad_norm'],
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        fp16=config['training']['fp16'],
        bf16=config['training']['bf16'],
        logging_steps=config['training']['logging_steps'],
        eval_strategy=config['training']['eval_strategy'],
        eval_steps=config['training']['eval_steps'],
        save_strategy=config['training']['save_strategy'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        dataloader_num_workers=config['training']['dataloader_num_workers'],
        dataloader_pin_memory=config['training']['dataloader_pin_memory'],
        seed=config['training']['seed'],
        report_to="wandb" if config['wandb']['enabled'] else "none",
        run_name=config['wandb'].get('run_name', None) if config['wandb']['enabled'] else None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Initialize Trainer with custom callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=data_collator,
        callbacks=[TimeDisplayCallback()],
    )
    
    # Print training info
    print("\n" + "="*60)
    print("Training Configuration:")
    print(f"  Model: {config['model']['name']}")
    print(f"  Total epochs: {config['training']['num_train_epochs']}")
    print(f"  Batch size per device: {config['training']['per_device_train_batch_size']}")
    print(f"  Gradient accumulation: {config['training']['gradient_accumulation_steps']}")
    print(f"  Effective batch size: {config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Max sequence length: {config['model']['max_seq_length']}")
    print(f"  LoRA rank: {config['lora']['r']}")
    print(f"  Output dir: {config['training']['output_dir']}")
    print("="*60 + "\n")
    
    # Train
    print("Starting training...")
    train_result = trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model()
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Final evaluation
    print("\nRunning final evaluation...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"  Final train loss: {metrics['train_loss']:.4f}")
    print(f"  Final eval loss:  {eval_metrics['eval_loss']:.4f}")
    print(f"  Model saved to:   {config['training']['output_dir']}")
    print("="*60 + "\n")
    
    # Save tokenizer
    tokenizer.save_pretrained(config['training']['output_dir'])
    print(f"Tokenizer saved to {config['training']['output_dir']}")


if __name__ == "__main__":
    main()