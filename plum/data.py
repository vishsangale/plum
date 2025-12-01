import torch
from torch.utils.data import Dataset
from plum.config import DatasetConfig

class SyntheticSIDDataset(Dataset):
    """
    Generates synthetic data for training the PLUM SID model.
    Each item has multiple modality embeddings.
    We also generate a 'positive' co-occurring item for contrastive learning.
    """
    def __init__(self, num_samples: int, input_dims: list[int]):
        self.num_samples = num_samples
        self.input_dims = input_dims
        
        # Generate random embeddings for all samples
        self.data = []
        for _ in range(num_samples):
            item_embeddings = [torch.randn(dim) for dim in input_dims]
            self.data.append(item_embeddings)
            
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Anchor item
        anchor_embeddings = self.data[idx]
        
        # Positive item (simulate co-occurrence)
        # For synthetic data, let's create a positive that is somewhat similar to the anchor
        # e.g., add some noise to the anchor embeddings
        positive_embeddings = [emb + 0.1 * torch.randn_like(emb) for emb in anchor_embeddings]
        
        return anchor_embeddings, positive_embeddings

class PLUMSIDDataset(Dataset):
    """
    Generic dataset for SID training.
    Uses pre-computed embeddings and sequences.
    """
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.embeddings = torch.load(config.embeddings_path, map_location='cpu')
        self.sequences = torch.load(config.user_sequences_path)
        
        # Pre-compute pairs for faster training
        self.pairs = []
        for seq in self.sequences:
            # seq is list of movie indices
            # Generate pairs (seq[i], seq[i+1])
            for i in range(len(seq) - 1):
                self.pairs.append((seq[i], seq[i+1]))
                
        print(f"Loaded {len(self.embeddings)} items and {len(self.pairs)} co-occurrence pairs.")
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        idx1, idx2 = self.pairs[idx]
        
        # Embeddings are single tensors (BERT output)
        # SID model expects list of tensors (multi-modal)
        # We wrap them in a list
        emb1 = [self.embeddings[idx1]]
        emb2 = [self.embeddings[idx2]]
        
        return emb1, emb2

class MovieLensLLMDataset(Dataset):
    def __init__(self, data_path: str, max_len: int = 256, split: str = 'train'):
        loaded_data = torch.load(data_path, weights_only=False)
        
        if isinstance(loaded_data, dict):
            if split not in loaded_data:
                raise ValueError(f"Split '{split}' not found in dataset. Available: {list(loaded_data.keys())}")
            self.data = loaded_data[split]
        else:
            print("Warning: Dataset is a list (legacy format), using all data.")
            self.data = loaded_data
            
        self.max_len = max_len
        print(f"Loaded {len(self.data)} sequences for LLM training (Split: {split}).")
        
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
