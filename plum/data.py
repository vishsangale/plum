import torch
from torch.utils.data import Dataset

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

class MovieLensSIDDataset(Dataset):
    """
    Real MovieLens data for SID training.
    Uses pre-computed BERT embeddings.
    Generates pairs (Movie A, Movie B) where B follows A in a user's history.
    """
    def __init__(self, embeddings_path: str, sequences_path: str):
        self.embeddings = torch.load(embeddings_path) # (Num_Movies, 384)
        self.sequences = torch.load(sequences_path)
        
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
