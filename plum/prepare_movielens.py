import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from urllib.request import urlretrieve
import zipfile
from plum.config import PLUMConfig

def download_movielens():
    config = PLUMConfig()
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
    config = PLUMConfig()
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
    
    # Generate Embeddings using TinyBERT (128 dim)
    print("Generating TinyBERT Embeddings (128 dim)...")
    from transformers import AutoTokenizer, AutoModel
    
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    
    texts = movies_df["Text"].tolist()
    embeddings_list = []
    
    # Process in batches to avoid OOM
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            batch_embeddings = sum_embeddings / sum_mask
            
            embeddings_list.append(batch_embeddings)
            
    embeddings = torch.cat(embeddings_list, dim=0)
    
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
    os.makedirs("plum/data/movielens-1m", exist_ok=True)
    torch.save(embeddings, "plum/data/movielens-1m/movie_embeddings.pt")
    torch.save(movie_id_to_idx, "plum/data/movielens-1m/movie_id_map.pt")
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
            
    torch.save(user_sequences, "plum/data/movielens-1m/user_sequences.pt")
    print(f"Saved {len(user_sequences)} user sequences.")

if __name__ == "__main__":
    prepare_data()
