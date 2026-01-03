# Narrator - Fine-tuning Llama 3.2 3B on Children's Stories

A complete pipeline for fine-tuning Meta's Llama 3.2 3B model on high-quality children's stories from Project Gutenberg using QLoRA (Quantized Low-Rank Adaptation) on consumer GPUs.

## 🎯 Project Overview

This project demonstrates how to:
- Filter and curate 4,000+ child-friendly stories from Project Gutenberg's 71,000+ book collection
- Clean and prepare text data for language model training
- Convert Meta Llama checkpoints to Hugging Face format
- Fine-tune Llama 3.2 3B using LoRA on a 12GB GPU (RTX 5070)
- Generate child-friendly narratives using the trained model

## 📊 Dataset Statistics

- **Source**: Project Gutenberg (71,000+ books)
- **Filtering Criteria**:
  - Language: English only
  - Categories: 45 curated child-friendly shelves (animals, fairy tales, children's fiction, mythology, etc.)
  - Flesch Reading Ease Score: 81-100 (very easy to read)
- **Final Dataset**: 4,036 stories
- **Train/Val Split**: 3,631 train / 405 validation (90/10)
- **Training Chunks**: 106,598 chunks (1,024 tokens each for speed/memory)

## 🗂️ Project Structure

```
Narrator/
├── scripts/                          # Python scripts
│   ├── download_gutenberg.py        # Download books from Project Gutenberg
│   ├── clean_dataset.py             # Remove boilerplate, clean text
│   ├── create_train_val_split.py    # Split into train/validation
│   ├── prepare_clm_dataset.py       # Tokenize and chunk for CLM
│   └── convert_meta_to_hf.py        # Convert Meta checkpoint to HF format
│
├── notebooks/                        # Jupyter notebooks
│   └── EDA_books.ipynb              # Exploratory data analysis & filtering
│
├── configs/                          # Configuration files
│   └── training_config.yaml         # Training hyperparameters
│
├── split_data/                      # Train/val split (gitignored)
│   ├── train/                       # Training stories (3,631 files)
│   └── val/                         # Validation stories (404 files)
│
├── processed_data/                  # Tokenized chunks (gitignored)
│   ├── train.jsonl                  # 106,598 training chunks
│   ├── val.jsonl                    # 11,449 validation chunks
│   └── dataset_info.json            # Dataset metadata
│
├── models/                          # Model checkpoints (gitignored)
│   └── llama-3.2-3b-hf/            # Converted Hugging Face model
│
├── cleaned_dataset/                 # Cleaned stories (gitignored)
├── curated_ids.txt                  # 4,036 filtered book IDs
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/narrator.git
cd narrator

# Create a conda environment
conda create -n narrator python=3.10
conda activate narrator

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

#### Option A: Use Pre-filtered Book IDs (Recommended)

```bash
# Download the curated stories (4,035 books)
python scripts/download_gutenberg.py --ids-file curated_ids.txt --output-dir dataset

# Clean the downloaded texts
python scripts/clean_dataset.py --input-dir dataset --output-dir cleaned_dataset

# Create train/val split
python scripts/create_train_val_split.py --input-dir cleaned_dataset --output-dir split_data --train-ratio 0.9

# Prepare for CLM training
python scripts/prepare_clm_dataset.py --model-name meta-llama/Llama-3.2-3B --max-length 2048
```

#### Option B: Filter Books Yourself

See `notebooks/EDA_books.ipynb` for the complete filtering process.

### 3. Model Preparation

#### Option A: Convert Meta Checkpoint (Current Setup)

1. Download Llama 3.2 3B from Meta AI
2. Convert to HuggingFace format:

```bash
python scripts/convert_checkpoint.py \
    --input-dir "path/to/meta/checkpoint" \
    --output-dir "models/llama-3.2-3b-hf"
```

The model will be saved to `models/llama-3.2-3b-hf/` and training will use this local path.

#### Option B: Use HuggingFace Hub (Alternative)

Change `training_config.yaml`:
```yaml
model:
  name: "meta-llama/Llama-3.2-3B"  # Instead of models/llama-3.2-3b-hf
```
Requires HuggingFace access token. Model auto-downloads on first run.

### 4. Pre-tokenize Dataset (Critical for Speed)

```bash
# Pre-tokenize to Arrow format for fast data loading
python scripts/prepare_tokenized_dataset.py \
    --max_length 1024 \
    --output_dir ~/tokenized_data_1024

# Update configs/training_config.yaml to point to tokenized data
```

### 5. Fine-tuning

```bash
# WSL2/Ubuntu recommended (Windows has 50-100x slowdown with gradient checkpointing)
python -m scripts.train
```

## 💾 Hardware Requirements

- **GPU**: 12GB VRAM minimum (tested on RTX 5070 with QLoRA 4-bit)
- **RAM**: 32GB recommended
- **Storage**: ~50GB for dataset and model checkpoints
- **OS**: WSL2 Ubuntu 22.04 REQUIRED (Windows native has critical PyTorch bugs causing 50-100x slowdown)

## 📚 Dataset Curation Process

The dataset was carefully curated through multiple filtering steps:

1. **Language Filter**: English books only
2. **Category Filter**: 45 child-friendly categories including:
   - Animals (domestic, wild, birds, insects, etc.)
   - Children's fiction and literature
   - Fairy tales and mythology
   - Educational content for children
   - Children's magazines (Harper's Young People, St. Nicholas, etc.)
3. **Readability Filter**: Flesch Reading Ease score 81-100 (very easy)
4. **Quality Control**: Manual verification of categories

See `notebooks/EDA_books.ipynb` for detailed analysis.

## 🛠️ Key Features

- **LoRA Training**: Parameter-efficient fine-tuning in BF16 precision
- **Automated Pipeline**: From raw data to trained model
- **Clean Codebase**: Modular scripts for each step
- **Reproducible**: Fixed random seeds and documented process
- **GPU Optimized**: Designed for consumer GPUs (12GB VRAM)
- **Memory Efficient**: QLoRA reduces 30GB to 8GB requirement

## 📖 Scripts Documentation

### `download_gutenberg.py`
Downloads books from Project Gutenberg given a list of IDs.

```bash
python scripts/download_gutenberg.py \
    --ids-file curated_ids.txt \
    --output-dir dataset \
    --resume  # Skip already downloaded files
```

### `clean_dataset.py`
Removes Project Gutenberg boilerplate and cleans text.

```bash
python scripts/clean_dataset.py \
    --input-dir dataset \
    --output-dir cleaned_dataset \
    --min-length 1000  # Minimum characters after cleaning
```

### `create_train_val_split.py`
Creates train/validation split with reproducible shuffling.

```bash
python scripts/create_train_val_split.py \
    --input-dir cleaned_dataset \
    --output-dir split_data \
    --train-ratio 0.9 \
    --seed 42
```

### `prepare_clm_dataset.py`
Tokenizes and chunks stories for causal language modeling.

```bash
python scripts/prepare_clm_dataset.py \
    --train-dir data/train \
    --val-dir data/val \
    --output-dir processed_data \
    --model-name meta-llama/Llama-3.2-3B \
    --max-length 2048 \
    --overlap 128
```

## 🎓 Training Configuration

Training uses QLoRA (4-bit quantization + LoRA) with:
- **Base Model**: Llama 3.2 3B (4-bit NF4 quantization)
- **Precision**: BFloat16 compute, 4-bit weights
- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **Target Modules**: q_proj, k_proj, v_proj, o_proj
- **Sequence Length**: 1,024 tokens
- **Batch Size**: 1 per device
- **Learning Rate**: 2e-4 with cosine schedule
- **Optimizer**: paged_adamw_8bit
- **Gradient Checkpointing**: DISABLED (causes massive slowdown)
- **Speed**: ~5.35s/step on RTX 5070 12GB

## 📝 License

This project is for educational purposes. Individual components:
- **Code**: MIT License
- **Dataset**: Project Gutenberg texts are in the public domain
- **Model**: Llama 3.2 is subject to Meta's license agreement

## 🙏 Acknowledgments

- [Project Gutenberg](https://www.gutenberg.org/) for providing public domain books
- Meta AI for releasing Llama 3.2
- Hugging Face for the Transformers library
- The open-source community for QLoRA and PEFT

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.

---

**Note**: Training in progress. Model adapters will be released upon completion (~6-7 days on RTX 5070).
