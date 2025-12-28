# PLUM Implementation

This repository contains a PyTorch implementation of the [PLUM]([url](https://arxiv.org/abs/2510.07784)) framework (Pre-trained Language Models for Industrial-scale Generative Recommendations).

## Components

1.  **Semantic ID (SID) Model** (`plum/sid_model.py`):
    *   `MultiModalEncoder`: Fuses multiple content embeddings.
    *   `RQVAE`: Residual Quantized VAE with multi-resolution codebooks and progressive masking.
    *   `ContrastiveLoss`: Co-occurrence contrastive loss.

2.  **Generative Retrieval (LLM)** (`plum/llm_model.py`):
    *   `SIDTokenizer`: Handles text and Semantic ID tokens.
    *   `CausalTransformer`: Decoder-only Transformer for next-token prediction.

## Configuration

Configuration is managed via `plum/config.py`.
*   `PLUMConfig`: Main configuration class.
*   `DatasetConfig`: Dataset-specific configuration (paths, dimensions).

## Supported Datasets

*   `movielens-1m`: Default dataset (1 million ratings).
*   `movielens-10m`: Larger dataset (10 million ratings).

## Usage

### 1. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch numpy pandas transformers sentence-transformers tqdm scikit-learn matplotlib seaborn tensorboard
```

### 2. Prepare Data

**MovieLens 1M (Default):**
```bash
python -m plum.prepare_movielens
```

**MovieLens 10M:**
```bash
python -m plum.prepare_movielens_10m
```

### 3. Train Semantic ID Model

Train the SID model. You can specify the dataset and batch size.

```bash
# For MovieLens 1M
python -m plum.train_sid --dataset movielens-1m --epochs 20

# For MovieLens 10M (Recommended)
python -m plum.train_sid --dataset movielens-10m --epochs 10 --batch_size 1024
```

### 4. Generate SIDs

Generate Semantic IDs using the trained model.

```bash
python -m plum.generate_sids --dataset movielens-10m
```

### 5. Prepare LLM Data

Convert user sequences to SID tokens and create train/val splits.

```bash
python -m plum.prepare_llm_data --dataset movielens-10m
```

### 6. Validate Embeddings (Optional)

Check the quality and diversity of input embeddings.

```bash
python -m plum.validate_embeddings --dataset movielens-10m
```

### 7. Train LLM

Train the generative retrieval model.

```bash
# For MovieLens 10M (Adjust batch size for GPU memory)
python -m plum.train_llm --dataset movielens-10m --epochs 1 --batch_size 8
```

### 8. Run Inference

Generate recommendations for random user sequences.

```bash
python -m plum.inference --dataset movielens-10m --num_samples 5
```

## File Structure

*   `plum/`
    *   `config.py`: Configuration and dataset registry.
    *   `data.py`: Dataset classes.
    *   `sid_model.py`: SID model architecture.
    *   `llm_model.py`: LLM architecture.
    *   `prepare_movielens.py`: Data prep for ML-1M.
    *   `prepare_movielens_10m.py`: Data prep for ML-10M.
    *   `train_sid.py`: SID training script.
    *   `generate_sids.py`: SID generation script.
    *   `prepare_llm_data.py`: LLM data prep script.
    *   `validate_embeddings.py`: Embedding validation script.
    *   `train_llm.py`: LLM training script.
    *   `inference.py`: Inference demo.
    *   `utils.py`: Shared utilities.
    *   `data/`: Data directory.
