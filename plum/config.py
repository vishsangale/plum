from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class DatasetConfig:
    """
    Configuration for a specific dataset.
    """
    name: str
    input_dims: List[int]
    embeddings_path: str
    user_sequences_path: str
    movie_sids_json_path: str
    movie_sids_pt_path: str
    llm_dataset_path: str
    checkpoint_dir: str
    raw_data_dir: str

@dataclass
class PLUMConfig:
    """
    Configuration for PLUM SID model and training.
    """
    # Model Architecture
    # input_dims is now part of DatasetConfig
    latent_dim: int = 256
    output_dim: int = 256
    num_levels: int = 3
    base_codebook_size: int = 512
    
    # Training
    batch_size: int = 128
    learning_rate: float = 1e-3
    epochs: int = 5
    contrastive_temperature: float = 0.07
    
    # Datasets
    dataset_name: str = "movielens-1m"
    datasets: Dict[str, DatasetConfig] = field(default_factory=lambda: {
        "movielens-1m": DatasetConfig(
            name="movielens-1m",
            input_dims=[128],
            embeddings_path="plum/data/movielens-1m/movie_embeddings.pt",
            user_sequences_path="plum/data/movielens-1m/user_sequences.pt",
            movie_sids_json_path="plum/data/movielens-1m/movie_sids.json",
            movie_sids_pt_path="plum/data/movielens-1m/movie_sids.pt",
            llm_dataset_path="plum/data/movielens-1m/llm_dataset.pt",
            checkpoint_dir="plum/data/movielens-1m/checkpoints",
            raw_data_dir="plum/data/movielens-1m/raw"
        )
    })
    
    # Global Paths
    # checkpoint_dir removed, use active_dataset.checkpoint_dir
    sid_model_checkpoint: str = "sid_model.pth" # Relative to checkpoint_dir
    llm_checkpoint_dir: str = "plum_llm_final" # Relative to checkpoint_dir
    
    @property
    def active_dataset(self) -> DatasetConfig:
        if self.dataset_name not in self.datasets:
            raise ValueError(f"Dataset {self.dataset_name} not found in registry.")
        return self.datasets[self.dataset_name]
    
    def __post_init__(self):
        pass
