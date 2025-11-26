import torch
from torch.utils.data import DataLoader
from plum.sid_model import PLUM_SID
import json
import os
from tqdm import tqdm
from plum.config import PLUMConfig

def generate_sids():
    # Configuration
    config = PLUMConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load embeddings
    print("Loading embeddings...")
    embeddings_path = config.active_dataset.embeddings_path
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found at {embeddings_path}")
        
    movie_embeddings = torch.load(embeddings_path) # (Num_Movies, 128)
    print(f"Loaded {len(movie_embeddings)} movie embeddings.")
    
    # Load Model
    print("Loading model...")
    model = PLUM_SID(config.active_dataset.input_dims, config.active_model_config.latent_dim, config.active_model_config.output_dim, config.active_model_config.num_levels, config.active_model_config.base_codebook_size)
    checkpoint_path = os.path.join(config.active_dataset.checkpoint_dir, config.sid_model_checkpoint)
    
    if not os.path.exists(checkpoint_path):
         raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
         
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Generate SIDs
    print("Generating SIDs...")
    batch_size = 256
    all_codes = []
    
    # Process in batches
    num_movies = len(movie_embeddings)
    with torch.no_grad():
        for i in tqdm(range(0, num_movies, batch_size)):
            batch = movie_embeddings[i:i+batch_size].to(device)
            # Normalize inputs as done in training
            batch = torch.nn.functional.normalize(batch, p=2, dim=-1)
            
            # Forward pass (just encoder + rqvae)
            # We need to wrap batch in a list because MultiModalEncoder expects a list
            # We need to wrap batch in a list because MultiModalEncoder expects a list
            z = model.encoder([batch])
            
            # Get codes (training=False for deterministic inference)
            _, codes, _ = model.rqvae(z, training=False)
            
            all_codes.append(codes.cpu())
            
    all_codes = torch.cat(all_codes, dim=0) # (Num_Movies, num_levels)
    
    # Convert to string format
    sid_map = {}
    unique_sids = set()
    for idx, codes in enumerate(all_codes):
        # Convert tensor codes to string "c1-c2-c3"
        sid_str = "-".join([str(c.item()) for c in codes])
        sid_map[idx] = sid_str
        unique_sids.add(sid_str)
        
    # Uniqueness Check
    num_unique = len(unique_sids)
    print(f"\nUniqueness Check:")
    print(f"Total Movies: {num_movies}")
    print(f"Unique SIDs: {num_unique}")
    print(f"Collisions: {num_movies - num_unique}")
    print(f"Uniqueness Rate: {num_unique / num_movies * 100:.2f}%")
        
    # Save results
    output_path_json = config.active_dataset.movie_sids_json_path
    output_path_pt = config.active_dataset.movie_sids_pt_path
    
    with open(output_path_json, 'w') as f:
        json.dump(sid_map, f, indent=2)
        
    torch.save(all_codes, output_path_pt)
    
    print(f"Saved SIDs to {output_path_json} and {output_path_pt}")
    
    # Print some examples
    print("\nExample SIDs:")
    for i in range(5):
        print(f"Movie {i}: {sid_map[i]}")

if __name__ == "__main__":
    generate_sids()
