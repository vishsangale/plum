import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial
from torch.utils.data import DataLoader, Dataset
from plum.llm_model import PLUM_LLM
from plum.config import PLUMConfig
import os
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

class MovieLensLLMDataset(Dataset):
    def __init__(self, data_path: str, max_len: int = 1024):
        self.data = torch.load(data_path)
        # Use subset for faster training on CPU
        self.data = self.data[:500] 
        self.max_len = max_len
        print(f"Loaded {len(self.data)} sequences for LLM training (Subset).")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        # Truncate if too long (keep recent history)
        if len(seq) > self.max_len:
            seq = seq[-self.max_len:]
        return torch.tensor(seq, dtype=torch.long)

def collate_fn(batch, pad_token_id: int):
    # Pad sequences to max length in batch using the tokenizer's pad id
    max_len = max(len(seq) for seq in batch)
    
    padded_batch = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_masks = torch.zeros(len(batch), max_len, dtype=torch.long)
    
    for i, seq in enumerate(batch):
        l = len(seq)
        padded_batch[i, :l] = seq
        attention_masks[i, :l] = 1 # 1 for valid tokens, 0 for padding
        
    return padded_batch, attention_masks

def train_llm():
    config = PLUMConfig()
    # Hyperparameters from config
    # batch_size, lr, epochs are now in config.active_model_config
    
    # Setup TensorBoard
    writer = SummaryWriter('runs/llm_training')
    
    # Initialize Model
    # Using distilgpt2 for faster training
    plum_model = PLUM_LLM(model_name='distilgpt2', num_levels=config.active_model_config.num_levels, base_codebook_size=config.active_model_config.base_codebook_size)
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plum_model.to(device)
    plum_model.train()
    
    # Load Dataset
    dataset_path = config.active_dataset.llm_dataset_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
    dataset = MovieLensLLMDataset(dataset_path)
    pad_token_id = plum_model.tokenizer.pad_token_id
    dataloader = DataLoader(
        dataset,
        batch_size=config.active_model_config.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_token_id=pad_token_id)
    )
    
    optimizer = optim.AdamW(plum_model.parameters(), lr=config.active_model_config.learning_rate)
    
    print(f"Starting LLM Training (distilgpt2) on {device}...")
    
    global_step = 0
    for epoch in range(config.active_model_config.epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.active_model_config.epochs}")
        
        for batch_idx, (input_ids, attention_mask) in enumerate(progress_bar):
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
            writer.add_scalar('LLM/Loss', loss.item(), global_step)
            global_step += 1
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{config.active_model_config.epochs}, Avg Loss: {avg_loss:.4f}")
        writer.add_scalar('LLM/Epoch_Loss', avg_loss, epoch)
        
        # Save checkpoint every epoch
        os.makedirs(config.active_dataset.checkpoint_dir, exist_ok=True)
        plum_model.save_pretrained(os.path.join(config.active_dataset.checkpoint_dir, f"plum_llm_epoch_{epoch+1}"))
        
    print("Training complete!")
    final_path = os.path.join(config.active_dataset.checkpoint_dir, config.llm_checkpoint_dir)
    plum_model.save_pretrained(final_path)
    print(f"Final model saved to {final_path}")

if __name__ == "__main__":
    train_llm()
