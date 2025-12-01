import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from urllib.request import urlretrieve
import zipfile
from plum.config import PLUMConfig

def download_movielens_10m():
    config = PLUMConfig(dataset_name="movielens-10m")
    raw_dir = config.active_dataset.raw_data_dir
    os.makedirs(raw_dir, exist_ok=True)
    
    zip_path = os.path.join(raw_dir, "ml-10m.zip")
    extracted_path = os.path.join(raw_dir, "ml-10M100K")
    
    url = "https://files.grouplens.org/datasets/movielens/ml-10m.zip"
    if not os.path.exists(zip_path):
        print("Downloading MovieLens 10M...")
        urlretrieve(url, zip_path)
        
    if not os.path.exists(extracted_path):
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)

def prepare_data():
    download_movielens_10m()
    
    config = PLUMConfig(dataset_name="movielens-10m")
    
    # Load Movies
    # MovieID::Title::Genres
    print("Loading Movies...")
    raw_dir = config.active_dataset.raw_data_dir
    movies_path = os.path.join(raw_dir, "ml-10M100K/movies.dat")
    
    movies_df = pd.read_csv(
        movies_path, 
        sep="::", 
        engine="python", 
        names=["MovieID", "Title", "Genres"],
        encoding="latin-1" # Usually safe for MovieLens
    )
    
    # Create text for embedding: "Title. Genres"
    movies_df["Text"] = movies_df["Title"] + ". " + movies_df["Genres"].str.replace("|", ", ")
    
    # Generate Embeddings using all-MiniLM-L6-v2 (384 dim)
    print("Generating Embeddings (384 dim)...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    texts = movies_df["Text"].tolist()
    
    # SentenceTransformer handles batching internally, but we can be explicit if needed
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    
    # L2 normalize embeddings (SentenceTransformer usually does this for cosine similarity models, but let's ensure)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    
    # L2 normalize embeddings
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    print(f"Generated embeddings shape: {embeddings.shape}")
    print(f"Embeddings L2 normalized: norm = {torch.norm(embeddings[0]).item():.4f} (should be ~1.0)")
    
    # Map MovieID to Embedding
    movie_id_to_idx = {mid: i for i, mid in enumerate(movies_df["MovieID"].values)}
    
    # Save embeddings tensor
    os.makedirs(os.path.dirname(config.active_dataset.embeddings_path), exist_ok=True)
    torch.save(embeddings, config.active_dataset.embeddings_path)
    # We don't strictly need movie_id_map.pt if we use json for SIDs, but good to have
    # torch.save(movie_id_to_idx, "plum/data/movielens-10m/movie_id_map.pt") 
    print(f"Saved embeddings for {len(embeddings)} movies.")
    
    # Load Ratings (User History)
    # UserID::MovieID::Rating::Timestamp
    print("Loading Ratings...")
    ratings_path = os.path.join(raw_dir, "ml-10M100K/ratings.dat")
    ratings_df = pd.read_csv(
        ratings_path, 
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
        # Filter out movies not in our map
        indices = [movie_id_to_idx[mid] for mid in movie_ids if mid in movie_id_to_idx]
        
        if len(indices) >= 5: # Only keep sequences with at least 5 items
            # Store both indices and ratings
            ratings = group["Rating"].values
            # Filter ratings to match indices (since we filtered indices)
            # Note: indices construction above iterates movie_ids. We need to be careful to keep alignment.
            
            # Let's redo the filtering to be safe and keep them aligned
            seq_indices = []
            seq_ratings = []
            
            for mid, rating in zip(movie_ids, ratings):
                if mid in movie_id_to_idx:
                    seq_indices.append(movie_id_to_idx[mid])
                    seq_ratings.append(rating)
            
            if len(seq_indices) >= 5:
                user_sequences.append({
                    "items": seq_indices,
                    "ratings": seq_ratings
                })
            
    torch.save(user_sequences, config.active_dataset.user_sequences_path)
    print(f"Saved {len(user_sequences)} user sequences.")

if __name__ == "__main__":
    prepare_data()
