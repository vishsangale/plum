# PLUM Implementation

This repository contains a PyTorch implementation of the PLUM framework (Pre-trained Language Models for Industrial-scale Generative Recommendations) as described in the paper.

## Components

1.  **Semantic ID (SID) Model** (`plum/sid_model.py`):
    *   `MultiModalEncoder`: Fuses multiple content embeddings.
    *   `RQVAE`: Residual Quantized VAE with multi-resolution codebooks and progressive masking.
    *   `ContrastiveLoss`: Co-occurrence contrastive loss.

2.  **Generative Retrieval (LLM)** (`plum/llm_model.py`):
    *   `SIDTokenizer`: Handles text and Semantic ID tokens.
    *   `CausalTransformer`: Decoder-only Transformer for next-token prediction.

## Usage

### 1. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch numpy
```

### 2. Train Semantic ID Model

Train the SID model on synthetic multi-modal data.

```bash
python -m plum.train_sid
```

This will save the model to `checkpoints/sid_model.pth`.

### 3. Train LLM (Continued Pre-training)

Train the LLM on synthetic user behavior sequences and metadata.

```bash
python -m plum.train_llm
```

This will save the model to `checkpoints/llm_model.pth`.

### 4. Run Inference Demo

Simulate a user history, generate the next Semantic ID using the LLM, and decode it back to an item embedding.

```bash
python -m plum.inference
```

## File Structure

*   `plum/`
    *   `sid_model.py`: SID model architecture.
    *   `llm_model.py`: LLM architecture (GPT-2) and tokenizer.
    *   `data.py`: Synthetic data generation for SID.
    *   `train_sid.py`: Training script for SID.
    *   `train_llm.py`: Training script for LLM.
    *   `inference.py`: End-to-end inference demo.
