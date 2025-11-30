import torch
import os
import json
from plum.sid_model import PLUM_SID
from plum.llm_model import PLUM_LLM
from plum.config import PLUMConfig

def load_plum_models(config: PLUMConfig, device=None):
    """
    Load both SID and LLM models based on configuration.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load SID Model
    sid_model = PLUM_SID(
        config.active_dataset.input_dims, 
        config.active_model_config.latent_dim, 
        config.active_model_config.output_dim, 
        config.active_model_config.codebook_sizes, 
        kmeans_init=config.active_model_config.kmeans_init
    )
    sid_checkpoint_path = os.path.join(config.active_dataset.checkpoint_dir, config.sid_model_checkpoint)
    if os.path.exists(sid_checkpoint_path):
        sid_model.load_state_dict(torch.load(sid_checkpoint_path, map_location=device, weights_only=False))
        print(f"SID Model loaded from {sid_checkpoint_path}")
    else:
        print(f"Warning: SID checkpoint not found at {sid_checkpoint_path}. Using initialized model.")
        
    sid_model.to(device)
    sid_model.eval()
    
    # 2. Load LLM
    llm_path = os.path.join(config.active_dataset.checkpoint_dir, config.llm_checkpoint_dir)
    if not os.path.exists(llm_path):
        # Fallback to epoch 1 if final not ready
        llm_path = os.path.join(config.active_dataset.checkpoint_dir, "plum_llm_epoch_1")
        
    if not os.path.exists(llm_path):
        print("Warning: LLM Checkpoint not found, using base distilgpt2 (untrained)")
        llm_path = "distilgpt2"
        
    llm_model = PLUM_LLM(model_name=llm_path, num_levels=config.active_model_config.num_levels, codebook_sizes=config.active_model_config.codebook_sizes)
    llm_model.to(device)
    llm_model.eval()
    print(f"LLM loaded from {llm_path}.")
    
    return sid_model, llm_model, device

def load_mappings(config: PLUMConfig):
    """
    Load MovieID -> SID and SID -> MovieID mappings.
    """
    # Load SID -> MovieID mapping
    if not os.path.exists(config.active_dataset.movie_sids_json_path):
        raise FileNotFoundError(f"Movie SIDs file not found at {config.active_dataset.movie_sids_json_path}")
        
    with open(config.active_dataset.movie_sids_json_path, 'r') as f:
        movie_sids = json.load(f) # MovieID -> Dict or String
        
    # Create reverse mapping: SID String -> List of MovieIDs
    sid_to_movies = {}
    for mid, data in movie_sids.items():
        if isinstance(data, dict):
            sid = data.get('sid')
        else:
            sid = data
            
        if sid:
            if sid not in sid_to_movies:
                sid_to_movies[sid] = []
            sid_to_movies[sid].append(mid)
        
    return movie_sids, sid_to_movies
