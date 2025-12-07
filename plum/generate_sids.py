import torch
from torch.utils.data import DataLoader
from plum.sid_model import PLUM_SID
import json
import os
from tqdm import tqdm
from plum.config import PLUMConfig

import argparse

def generate_sids():
    parser = argparse.ArgumentParser(description="Generate SIDs for Movies")
    parser.add_argument("--dataset", type=str, default="movielens-1m", help="Dataset name (e.g., movielens-1m, movielens-10m)")
    args = parser.parse_args()
    
    # Configuration
    config = PLUMConfig(dataset_name=args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataset: {config.dataset_name}")
    
    # Load embeddings
    print("Loading embeddings...")
    embeddings_path = config.active_dataset.embeddings_path
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found at {embeddings_path}")
        
    movie_embeddings = torch.load(embeddings_path, weights_only=False) # (Num_Movies, 128)
    print(f"Loaded {len(movie_embeddings)} movie embeddings.")
    
    # Load Model
    print("Loading model...")
    model = PLUM_SID(
        input_dims=config.active_dataset.input_dims,
        latent_dim=config.active_model_config.latent_dim,
        output_dim=config.active_model_config.output_dim,
        codebook_sizes=config.active_model_config.codebook_sizes
    )
    checkpoint_path = os.path.join(config.active_dataset.checkpoint_dir, config.sid_model_checkpoint)
    
    if not os.path.exists(checkpoint_path):
         raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
         
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    
    # Generate SIDs
    print("Generating SIDs...")
    batch_size = 4096 # Increased for speed
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
            z = model.encoder([batch])
            
            # Get codes (training=False for deterministic inference)
            _, codes, _ = model.rqvae(z, training=False)
            
            all_codes.append(codes.cpu())
            
    all_codes = torch.cat(all_codes, dim=0) # (Num_Movies, num_levels)
    
    # Convert to string format and enrich with metadata
    sid_map = {}
    unique_sids = set()
    
    # Load Metadata if available
    import pandas as pd
    # We don't strictly need movie_id_map.pt if we assume indices match, but let's try to be safe.
    # Actually, prepare_movielens_10m.py doesn't save movie_id_map.pt by default (commented out).
    # But it saves embeddings in order of movies_df.
    # So idx corresponds to row in movies_df.
    
    raw_dir = config.active_dataset.raw_data_dir
    movies_df = None
    
    if config.dataset_name == "movielens-1m":
        movies_path = os.path.join(raw_dir, "ml-1m/movies.dat")
    elif config.dataset_name == "movielens-10m":
        movies_path = os.path.join(raw_dir, "ml-10M100K/movies.dat")
    elif config.dataset_name == "movielens-25m":
        movies_path = os.path.join(raw_dir, "ml-25m/movies.csv")
    else:
        movies_path = None
        
    if movies_path and os.path.exists(movies_path):
        print(f"Loading metadata from {movies_path}")
        if config.dataset_name == "movielens-25m":
             movies_df = pd.read_csv(movies_path)
             # Rename columns to match expected format if needed, or adjust access below
             # ML-25M: movieId, title, genres
             movies_df = movies_df.rename(columns={"movieId": "MovieID", "title": "Title", "genres": "Genres"})
        else:
            movies_df = pd.read_csv(
                movies_path, 
                sep="::", 
                engine="python", 
                names=["MovieID", "Title", "Genres"],
                encoding="latin-1"
            )
        # Assuming embeddings were generated from this DF in order
        # We can just use iloc
    else:
        print("Metadata file not found or dataset unknown. SIDs will lack titles.")

    for idx, codes in enumerate(all_codes):
        # Convert tensor codes to string "c1-c2-c3"
        sid_str = "-".join([str(c.item()) for c in codes])
        
        title = "Unknown"
        genres = "Unknown"
        
        if movies_df is not None and idx < len(movies_df):
            title = movies_df.iloc[idx]["Title"]
            genres = movies_df.iloc[idx]["Genres"]
        
        sid_map[idx] = {
            "sid": sid_str,
            "title": title,
            "genres": genres
        }
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
    for i in range(min(10, len(sid_map))): 
        entry = sid_map[i]
        print(f"Movie {i}: {entry['sid']} | {entry['title']} | {entry['genres']}")

if __name__ == "__main__":
    generate_sids()
