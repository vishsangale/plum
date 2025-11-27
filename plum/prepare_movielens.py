import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from urllib.request import urlretrieve
import zipfile
from plum.config import PLUMConfig

def download_movielens():
    config = PLUMConfig(dataset_name="movielens-1m")
    raw_dir = config.active_dataset.raw_data_dir
    os.makedirs(raw_dir, exist_ok=True)
    
    zip_path = os.path.join(raw_dir, "ml-1m.zip")
    extracted_path = os.path.join(raw_dir, "ml-1m")
    
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    if not os.path.exists(zip_path):
        print("Downloading MovieLens 1M...")
        urlretrieve(url, zip_path)
        
    if not os.path.exists(extracted_path):
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)

def prepare_data():
    download_movielens()
    
    # Load Movies
    # MovieID::Title::Genres
    print("Loading Movies...")
    config = PLUMConfig(dataset_name="movielens-1m")
    raw_dir = config.active_dataset.raw_data_dir
    movies_df = pd.read_csv(
        os.path.join(raw_dir, "ml-1m/movies.dat"), 
        sep="::", 
        engine="python", 
        names=["MovieID", "Title", "Genres"],
        encoding="latin-1"
    )
    
    # Create text for embedding: "Title. Genres"
    movies_df["Text"] = movies_df["Title"] + ". " + movies_df["Genres"].str.replace("|", ", ")
    
    # Generate Embeddings using all-MiniLM-L6-v2 (384 dim)
    print("Generating all-MiniLM-L6-v2 Embeddings (384 dim)...")
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    texts = movies_df["Text"].tolist()
    
    # Process in batches to avoid OOM
    batch_size = 64
    print(f"Processing {len(texts)} texts in batches of {batch_size}...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)
    
    # L2 normalize embeddings
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    print(f"Generated embeddings shape: {embeddings.shape}")
    print(f"Embeddings L2 normalized: norm = {torch.norm(embeddings[0]).item():.4f} (should be ~1.0)")
    
    # Map MovieID to Embedding
    # Note: MovieIDs are not contiguous (some missing). We'll use a dictionary or a large tensor if we re-map IDs.
    # For simplicity, let's re-map MovieIDs to contiguous indices 0..N-1
    movie_id_to_idx = {mid: i for i, mid in enumerate(movies_df["MovieID"].values)}
    idx_to_movie_id = {i: mid for mid, i in movie_id_to_idx.items()}
    
    # Save embeddings tensor
    os.makedirs(os.path.dirname(config.active_dataset.embeddings_path), exist_ok=True)
    torch.save(embeddings, config.active_dataset.embeddings_path)
    # Save ID map in the same directory
    id_map_path = os.path.join(os.path.dirname(config.active_dataset.embeddings_path), "movie_id_map.pt")
    torch.save(movie_id_to_idx, id_map_path)
    print(f"Saved embeddings for {len(embeddings)} movies.")
    
    # Load Ratings (User History)
    # UserID::MovieID::Rating::Timestamp
    print("Loading Ratings...")
    ratings_df = pd.read_csv(
        os.path.join(raw_dir, "ml-1m/ratings.dat"), 
        sep="::", 
        engine="python", 
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        encoding="latin-1"
    )
    
    # Sort by User and Time
    ratings_df = ratings_df.sort_values(by=["UserID", "Timestamp"])
    
    # Group by User to get sequences
    print("Processing User Sequences...")
    user_sequences = []
    for user_id, group in ratings_df.groupby("UserID"):
        # Get sequence of Movie indices
        movie_ids = group["MovieID"].values
        # Filter out movies not in our map (shouldn't happen with ML-1M but good practice)
        indices = [movie_id_to_idx[mid] for mid in movie_ids if mid in movie_id_to_idx]
        
        if len(indices) >= 5: # Only keep sequences with at least 5 items
            user_sequences.append(indices)
            
    torch.save(user_sequences, config.active_dataset.user_sequences_path)
    print(f"Saved {len(user_sequences)} user sequences.")

if __name__ == "__main__":
    prepare_data()
