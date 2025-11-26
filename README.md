# PLUM Implementation

This repository contains a PyTorch implementation of the PLUM framework (Pre-trained Language Models for Industrial-scale Generative Recommendations).

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

Supported datasets are defined in the `datasets` registry in `PLUMConfig`. The default dataset is `movielens-1m`.

## Usage

### 1. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch numpy pandas transformers sentence-transformers tqdm
```

### 2. Prepare Data (MovieLens 1M)

Download and preprocess the MovieLens 1M dataset. This will:
*   Download `ml-1m.zip` to `plum/data/movielens-1m/raw/`.
*   Generate movie embeddings using TinyBERT.
*   Create user sequences.
*   Save processed data to `plum/data/movielens-1m/`.

```bash
python -m plum.prepare_movielens
```

### 3. Train Semantic ID Model

Train the SID model to generate discrete codes for items.

```bash
python -m plum.train_sid
```

This will save the model to `plum/data/movielens-1m/checkpoints/sid_model.pth`.

### 4. Generate SIDs

Generate Semantic IDs for all movies using the trained SID model.

```bash
python -m plum.generate_sids
```

This saves `movie_sids.json` and `movie_sids.pt`.

### 5. Prepare LLM Data

Convert user movie sequences into Semantic ID token sequences for LLM training.

```bash
python -m plum.prepare_llm_data
```

### 6. Train LLM

Train the LLM (distilgpt2) to predict the next Semantic ID.

```bash
python -m plum.train_llm
```

This saves checkpoints to `plum/data/movielens-1m/checkpoints/`.

### 7. Run Inference Demo

Simulate a user history, generate the next Semantic ID using the LLM, and decode it back to recommended movies.

```bash
python -m plum.inference
```

## File Structure

*   `plum/`
    *   `config.py`: Configuration and dataset registry.
    *   `data.py`: Dataset classes (`PLUMSIDDataset`, `MovieLensLLMDataset`).
    *   `sid_model.py`: SID model architecture.
    *   `llm_model.py`: LLM architecture.
    *   `prepare_movielens.py`: Data preparation script.
    *   `train_sid.py`: Training script for SID.
    *   `generate_sids.py`: Script to generate SIDs from trained model.
    *   `prepare_llm_data.py`: Script to prepare data for LLM.
    *   `train_llm.py`: Training script for LLM.
    *   `inference.py`: End-to-end inference demo.
    *   `data/`: Directory containing dataset-specific files.
        *   `movielens-1m/`:
            *   `raw/`: Raw downloaded data.
            *   `checkpoints/`: Model checkpoints.
            *   `*.pt`, `*.json`: Processed data files.
