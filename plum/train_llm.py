import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial
from torch.utils.data import DataLoader
from plum.llm_model import PLUM_LLM
from plum.config import PLUMConfig
from plum.data import MovieLensLLMDataset, collate_fn
from plum.utils import load_plum_models
import os
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

import argparse

def train_llm():
    parser = argparse.ArgumentParser(description="Train PLUM LLM")
    parser.add_argument("--dataset", type=str, default="movielens-1m", help="Dataset name (e.g., movielens-1m, movielens-10m)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of steps per epoch")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    args = parser.parse_args()
    
    config = PLUMConfig(dataset_name=args.dataset)
    if args.epochs:
        config.active_llm_config.epochs = args.epochs
    if args.batch_size:
        config.active_llm_config.batch_size = args.batch_size
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {config.dataset_name}")
    
    # Setup TensorBoard
    writer = SummaryWriter(log_dir=f"runs/{config.dataset_name}_llm")
    
    # Initialize Model
    # Using distilgpt2 for faster training
    # Note: load_plum_models is for inference usually (loading checkpoints), 
    # but we can use PLUM_LLM directly for training from scratch/pretrained base.
    # We need to know codebook sizes to initialize the tokenizer correctly
    plum_model = PLUM_LLM(
        model_name=config.active_llm_config.model_name,
        num_levels=config.active_model_config.num_levels,
        codebook_sizes=config.active_model_config.codebook_sizes
    )
    
    # Move to device
    plum_model.to(device)
    plum_model.train()
    
    # Load Datasets
    dataset_path = config.active_dataset.llm_dataset_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
    train_dataset = MovieLensLLMDataset(dataset_path, max_len=config.active_llm_config.max_seq_len, split='train')
    val_dataset = MovieLensLLMDataset(dataset_path, max_len=config.active_llm_config.max_seq_len, split='val')
    
    pad_token_id = plum_model.tokenizer.pad_token_id
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.active_llm_config.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_token_id=pad_token_id)
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.active_llm_config.batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, pad_token_id=pad_token_id)
    )
    
    optimizer = optim.AdamW(plum_model.parameters(), lr=config.active_llm_config.learning_rate)
    
    print(f"Starting LLM Training ({config.active_llm_config.model_name}) on {device}...")
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config.active_llm_config.epochs):
        # --- Training Loop ---
        plum_model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.active_llm_config.epochs} [Train]")
        
        for batch_idx, (input_ids, attention_mask) in enumerate(progress_bar):
            if args.max_steps and batch_idx >= args.max_steps:
                break
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # GPT-2 forward computes loss automatically if labels are provided
            outputs = plum_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log to TensorBoard
            writer.add_scalar('LLM/Train_Loss', loss.item(), global_step)
            
            # Save checkpoint every 100 steps
            if global_step > 0 and global_step % 100 == 0:
                step_path = os.path.join(config.active_dataset.checkpoint_dir, f"plum_llm_step_{global_step}")
                plum_model.save_pretrained(step_path)
                # Also update 'final' for immediate use
                final_path = os.path.join(config.active_dataset.checkpoint_dir, config.llm_checkpoint_dir)
                plum_model.save_pretrained(final_path)
                
            global_step += 1
            
        avg_train_loss = total_loss / len(train_dataloader)
        
        # --- Validation Loop ---
        plum_model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for input_ids, attention_mask in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{config.active_llm_config.epochs} [Val]"):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                
                outputs = plum_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                total_val_loss += outputs.loss.item()
                
        avg_val_loss = total_val_loss / len(val_dataloader)
        perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val PPL: {perplexity:.2f}")
        
        writer.add_scalar('LLM/Epoch_Train_Loss', avg_train_loss, epoch)
        writer.add_scalar('LLM/Epoch_Val_Loss', avg_val_loss, epoch)
        writer.add_scalar('LLM/Epoch_Val_Perplexity', perplexity, epoch)
        
        # Save checkpoint every epoch
        os.makedirs(config.active_dataset.checkpoint_dir, exist_ok=True)
        plum_model.save_pretrained(os.path.join(config.active_dataset.checkpoint_dir, f"plum_llm_epoch_{epoch+1}"))
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(config.active_dataset.checkpoint_dir, "plum_llm_best")
            plum_model.save_pretrained(best_path)
            print(f"New best model saved to {best_path}")
        
    print("Training complete!")
    final_path = os.path.join(config.active_dataset.checkpoint_dir, config.llm_checkpoint_dir)
    plum_model.save_pretrained(final_path)
    print(f"Final model saved to {final_path}")

if __name__ == "__main__":
    train_llm()
