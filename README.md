# Narrator - Fine-tuning Llama 3.2 3B on Children's Stories

A complete pipeline for fine-tuning Meta's Llama 3.2 3B model on high-quality children's stories from Project Gutenberg using QLoRA (Quantized Low-Rank Adaptation) on consumer GPUs.

## 🎯 Project Overview

This project provides an end-to-end pipeline for fine-tuning Llama 3.2 3B on children's stories:

- **4,036 High-Quality Stories**: Curated from Project Gutenberg with Flesch readability scores 81-100, filtered across 45 child-friendly categories
- **Parallel Processing Throughout**: Multi-core optimization reduces data preparation from hours to minutes (5-10x speedup on cleaning, splitting, and tokenization)
- **Pre-tokenized Arrow Datasets**: 228,425 training chunks ready for instant loading—no tokenization bottleneck during training
- **Fits on Free Hardware**: QLoRA 4-bit quantization + optimized hyperparameters complete training in 9-10 hours on Google Colab T4 (12GB VRAM)
- **Production-Grade Pipeline**: Modular scripts with error handling, reproducible configurations, comprehensive documentation

## 📊 Dataset Statistics

- **Source**: Project Gutenberg (71,000+ books)
- **Filtering Criteria**:
  - Language: English only
  - Categories: 45 curated child-friendly shelves (animals, fairy tales, children's fiction, mythology, etc.)
  - Flesch Reading Ease Score: 81-100 (very easy to read)
- **Final Dataset**: 4,036 stories
- **Train/Val Split**: 3,631 train / 405 validation (90/10)
- **Training Chunks**: 228,425 train / 24,533 validation (1,024 tokens each)
- **Optimized Training**: ~9-10 hours on Google Colab T4 (50% of data)

## 🗂️ Project Structure

```
Narrator/
├── scripts/                          # Python scripts
│   ├── download_gutenberg.py        # Download books from Project Gutenberg
│   ├── clean_dataset.py             # Remove boilerplate, clean text
│   ├── create_train_val_split.py    # Split into train/validation
│   ├── prepare_dataset.py           # Chunk stories and tokenize to Arrow format
│   └── train.py                     # Fine-tune model with QLoRA
│
├── notebooks/                        # Jupyter notebooks
│   └── EDA_books.ipynb              # Exploratory data analysis & filtering
│
├── configs/                          # Configuration files
│   └── training_config.yaml         # Training hyperparameters
│
├── split_data/                      # Train/val split (gitignored)
│   ├── train/                       # Training stories (3,631 files)
│   └── val/                         # Validation stories (405 files)
│
├── tokenized_data_1024/             # Pre-tokenized datasets (gitignored)
│   ├── train/                       # Tokenized training chunks (Arrow format)
│   └── validation/                  # Tokenized validation chunks (Arrow format)
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

### 2. HuggingFace Setup (Required)

Before downloading or tokenizing data, you need access to Llama 3.2 3B:

**Step 1: Create HuggingFace Access Token**
1. Go to https://huggingface.co/settings/tokens
2. Click your **Profile** → **Settings** → **Access Tokens**
3. Click **"Create new token"**
4. Choose **"Read"** type (recommended for model access)
5. Copy the generated token

**Step 2: Login via Terminal**
```bash
# Open terminal and run
hf auth login

# Paste your access token when prompted
# Token: [paste your token here]
```

**Step 3: Request Model Access**
1. Visit https://huggingface.co/meta-llama/Llama-3.2-3B
2. Click **"Request Access"** button
3. Accept Meta's license agreement
4. Wait for approval (usually takes a few minutes to hours)
5. Once approved, you can use the model and tokenizer

### 3. Data Preparation

#### Option A: Use Pre-filtered Book IDs (Recommended)

```bash
# Download the curated stories (4,036 books)
python scripts/download_gutenberg.py --ids-file curated_ids.txt --output-dir dataset

# Clean the downloaded texts (parallel processing auto-enabled)
python scripts/clean_dataset.py --input-dir dataset --output-dir cleaned_dataset

# Create train/val split (parallel file copying auto-enabled)
python scripts/create_train_val_split.py --input-dir cleaned_dataset --output-dir split_data --train-ratio 0.9

# Prepare tokenized dataset (requires HuggingFace access - see step 2)
python scripts/prepare_dataset.py --max-length 1024 --output-dir tokenized_data_1024
```

#### Option B: Filter Books Yourself

See `notebooks/EDA_books.ipynb` for the complete filtering process.

### 4. Fine-tuning

The model will automatically download from HuggingFace on first run (requires access from step 2).

```bash
# Google Colab or WSL2/Ubuntu recommended
python scripts/train.py
```

**Important Notes:**
- First run will download the 4-bit quantized model (~2-3GB)
- Training saves checkpoints to `./checkpoints/` every 3,000 steps
- Google Colab: Complete within 12-hour session limit (~9-10 hours for 50% of data)
- Local: No time limits, but use WSL2 on Windows (native Windows is 50-100x slower)

## 💾 Hardware Requirements

### Option 1: Google Colab (Recommended)
- **GPU**: T4 (16GB VRAM) - Free tier
- **Training Time**: ~9-10 hours for 50% of dataset
- **Storage**: Upload `tokenized_data_1024/` (~5GB compressed)
- **Pros**: No setup, more VRAM, free
- **Cons**: 12-hour session limit, internet required

### Option 2: Local GPU
- **GPU**: 12GB VRAM minimum (tested on RTX 5070 with QLoRA 4-bit)
- **RAM**: 32GB recommended
- **Storage**: ~50GB for dataset and model checkpoints
- **OS**: WSL2 Ubuntu 22.04 REQUIRED (Windows native has critical PyTorch bugs causing 50-100x slowdown)
- **Pros**: No time limits, faster for local data
- **Cons**: Setup complexity, electricity costs

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

- **Parallel Data Processing**: Multi-core parallelization across the entire pipeline
  - Text cleaning: ~5x faster with parallel processing
  - Train/val splitting: Parallel file copying
  - Tokenization: ~6x faster with 11 workers (5 min vs 30+ min)
- **QLoRA 4-bit Training**: Parameter-efficient fine-tuning reduces VRAM from 30GB to 8GB
- **Pre-tokenized Datasets**: Arrow format eliminates tokenization bottleneck during training
- **Optimized for Speed**: Completes in ~9-10 hours on Google Colab T4 (50% of dataset)
- **Clean & Modular**: Separate scripts for each pipeline stage with comprehensive error handling
- **Reproducible**: Fixed random seeds, documented hyperparameters, version-controlled configs

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
Removes Project Gutenberg boilerplate and cleans text with parallel processing.

```bash
python scripts/clean_dataset.py \
    --input-dir dataset \
    --output-dir cleaned_dataset \
    --min-length 1000 \
    --num-workers 8  # Parallel processing (optional)
```

### `create_train_val_split.py`
Creates train/validation split with reproducible shuffling and parallel file copying.

```bash
python scripts/create_train_val_split.py \
    --input-dir cleaned_dataset \
    --output-dir split_data \
    --train-ratio 0.9 \
    --seed 42 \
    --num-workers 8  # Parallel file copying (optional)
```

### `prepare_dataset.py`
Chunks stories and tokenizes them directly to Arrow format with parallel processing.

```bash
python scripts/prepare_dataset.py \
    --train-dir split_data/train \
    --val-dir split_data/val \
    --output-dir tokenized_data_1024 \
    --model-name meta-llama/Llama-3.2-3B \
    --max-length 1024 \
    --overlap 128 \
    --num-workers 11  # Parallel tokenization (auto-detects if omitted)
```

**Features:**
- Combines chunking and tokenization in one optimized step
- Parallel tokenization across CPU cores (~11 workers on modern CPUs)
- Saves directly to Arrow format for instant data loading
- Completes in ~5 minutes with parallelization (vs ~30+ minutes sequential)
- Output: 228,425 train chunks, 24,533 validation chunks

## 🎓 Training Configuration

Training uses QLoRA (4-bit quantization + LoRA) optimized for Google Colab T4:
- **Base Model**: Llama 3.2 3B (4-bit NF4 quantization)
- **Precision**: BFloat16 compute, 4-bit weights
- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **Target Modules**: q_proj, k_proj, v_proj, o_proj
- **Sequence Length**: 1,024 tokens
- **Batch Size**: 8 per device (with gradient accumulation = 2)
- **Effective Batch Size**: 16
- **Epochs**: 0.5 (50% of data, ~114K samples)
- **Learning Rate**: 2e-4 with cosine schedule
- **Optimizer**: paged_adamw_8bit
- **Gradient Checkpointing**: DISABLED (for speed)
- **Training Time**: ~9-10 hours on T4 16GB
- **VRAM Usage**: ~12-14GB during training

## 📝 License

This project is for educational purposes. Individual components:
- **Code**: MIT License
- **Dataset**: Project Gutenberg texts are in the public domain
- **Model**: Llama 3.2 is subject to Meta's license agreement

## 🙏 Acknowledgments

- [Project Gutenberg](https://www.gutenberg.org/) for providing public domain books
- [Meta AI](https://www.meta.ai/) for releasing Llama 3.2
- [Hugging Face](https://huggingface.co/) for the Transformers library
- The open-source community for QLoRA and PEFT

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.

---

**Note**: Optimized for fast iteration. Training completes in ~9-10 hours on Google Colab T4 with 50% of dataset. Increase `num_train_epochs` in `configs/training_config.yaml` for full training.
