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
    model_config_name: str = "default"

@dataclass
class ModelConfig:
    """
    Configuration for the PLUM model architecture.
    """
    name: str
    latent_dim: int
    output_dim: int
    num_levels: int
    base_codebook_size: int
    
    # Training
    batch_size: int
    learning_rate: float
    epochs: int
    contrastive_temperature: float
    commitment_beta: float = 0.25
    recon_weight: float = 500.0
    contrastive_weight: float = 1.0
    level_dropout_prob: float = 0.0
    enable_dead_code_revival: bool = True  # Enable/disable dead code revival during training
    enable_progressive_masking: bool = True  # Enable/disable progressive masking (random depth r during training)



@dataclass
class PLUMConfig:
    """
    Configuration for PLUM SID model and training.
    """
    
    # Datasets
    dataset_name: str = "movielens-1m"
    datasets: Dict[str, DatasetConfig] = field(default_factory=lambda: {
        "movielens-1m": DatasetConfig(
            name="movielens-1m",
            input_dims=[384],
            embeddings_path="plum/data/movielens-1m/movie_embeddings.pt",
            user_sequences_path="plum/data/movielens-1m/user_sequences.pt",
            movie_sids_json_path="plum/data/movielens-1m/movie_sids.json",
            movie_sids_pt_path="plum/data/movielens-1m/movie_sids.pt",
            llm_dataset_path="plum/data/movielens-1m/llm_dataset.pt",
            checkpoint_dir="plum/data/movielens-1m/checkpoints",
            raw_data_dir="plum/data/movielens-1m/raw",
            model_config_name="movielens-1m"
        ),
        "movielens-10m": DatasetConfig(
            name="movielens-10m",
            input_dims=[384],
            embeddings_path="plum/data/movielens-10m/movie_embeddings.pt",
            user_sequences_path="plum/data/movielens-10m/user_sequences.pt",
            movie_sids_json_path="plum/data/movielens-10m/movie_sids.json",
            movie_sids_pt_path="plum/data/movielens-10m/movie_sids.pt",
            llm_dataset_path="plum/data/movielens-10m/llm_dataset.pt",
            checkpoint_dir="plum/data/movielens-10m/checkpoints",
            raw_data_dir="plum/data/movielens-10m/raw",
            model_config_name="movielens-10m"
        )
    })
    
    # Model Configs
    model_configs: Dict[str, ModelConfig] = field(default_factory=lambda: {
        "default": ModelConfig(
            name="default",
            latent_dim=256,
            output_dim=256,
            num_levels=3,
            base_codebook_size=512,
            batch_size=128,
            learning_rate=1e-3,
            epochs=3,
            contrastive_temperature=0.07,
            commitment_beta=0.25,
            recon_weight=500.0,
            contrastive_weight=1.0
        ),
        "movielens-1m": ModelConfig(
            name="movielens-1m",
            latent_dim=256,
            output_dim=256,
            num_levels=3,
            base_codebook_size=64, # Set to 64 (Levels: 64, 32, 16) -> Capacity ~32k
            batch_size=128,
            learning_rate=1e-3,
            epochs=3,
            contrastive_temperature=0.07,
            commitment_beta=0.5,
            recon_weight=500.0,
            contrastive_weight=1.0,
            level_dropout_prob=0.0,  # Disabled
        ),
        "movielens-10m": ModelConfig(
            name="movielens-10m",
            latent_dim=256,
            output_dim=256,
            num_levels=3,
            base_codebook_size=512,
            batch_size=256, # Larger batch size for larger dataset
            learning_rate=1e-3,
            epochs=10, # More epochs for larger dataset
            contrastive_temperature=0.07,
            commitment_beta=0.25,
            recon_weight=500.0,
            contrastive_weight=1.0
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
        
    @property
    def active_model_config(self) -> ModelConfig:
        model_name = self.active_dataset.model_config_name
        if model_name not in self.model_configs:
            raise ValueError(f"Model config {model_name} not found in registry.")
        return self.model_configs[model_name]
    
    def __post_init__(self):
        pass
