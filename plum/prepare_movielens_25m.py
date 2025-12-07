import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from urllib.request import urlretrieve
import zipfile
from plum.config import PLUMConfig

def download_movielens_25m():
    config = PLUMConfig(dataset_name="movielens-25m")
    raw_dir = config.active_dataset.raw_data_dir
    os.makedirs(raw_dir, exist_ok=True)
    
    zip_path = os.path.join(raw_dir, "ml-25m.zip")
    extracted_path = os.path.join(raw_dir, "ml-25m")
    
    url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    if not os.path.exists(zip_path):
        print("Downloading MovieLens 25M...")
        urlretrieve(url, zip_path)
        
    if not os.path.exists(extracted_path):
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)

def prepare_data():
    download_movielens_25m()
    
    config = PLUMConfig(dataset_name="movielens-25m")
    
    # Load Movies
    # movieId,title,genres
    print("Loading Movies...")
    raw_dir = config.active_dataset.raw_data_dir
    movies_path = os.path.join(raw_dir, "ml-25m/movies.csv")
    
    movies_df = pd.read_csv(movies_path)
    
    # Create text for embedding: "Title. Genres"
    movies_df["Text"] = movies_df["title"] + ". " + movies_df["genres"].str.replace("|", ", ")
    
    # Generate Embeddings using all-MiniLM-L6-v2 (384 dim)
    print("Generating Embeddings (384 dim)...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    texts = movies_df["Text"].tolist()
    
    # SentenceTransformer handles batching internally
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    
    # L2 normalize embeddings
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    print(f"Generated embeddings shape: {embeddings.shape}")
    print(f"Embeddings L2 normalized: norm = {torch.norm(embeddings[0]).item():.4f} (should be ~1.0)")
    
    # Map MovieID to Embedding
    movie_id_to_idx = {mid: i for i, mid in enumerate(movies_df["movieId"].values)}
    
    # Save embeddings tensor
    os.makedirs(os.path.dirname(config.active_dataset.embeddings_path), exist_ok=True)
    torch.save(embeddings, config.active_dataset.embeddings_path)
    print(f"Saved embeddings for {len(embeddings)} movies.")
    
    # Load Ratings (User History)
    # userId,movieId,rating,timestamp
    print("Loading Ratings...")
    ratings_path = os.path.join(raw_dir, "ml-25m/ratings.csv")
    ratings_df = pd.read_csv(ratings_path)
    
    # Sort by User and Time
    ratings_df = ratings_df.sort_values(by=["userId", "timestamp"])
    
    # Group by User to get sequences
    print("Processing User Sequences...")
    user_sequences = []
    
    # Optimization: Use pandas groupby aggregation for speed on large data
    # Filter movies first? No, we need to filter during sequence creation or before.
    # Let's filter ratings for movies that exist (though they should all exist in ML-25m)
    
    # Group and aggregate
    grouped = ratings_df.groupby("userId")
    
    for user_id, group in tqdm(grouped, desc="Grouping Users"):
        movie_ids = group["movieId"].values
        ratings = group["rating"].values
        
        # Filter and map to indices
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
    from tqdm import tqdm
    prepare_data()
