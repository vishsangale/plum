import torch
import json
import os
import random
from tqdm import tqdm
from plum.llm_model import PLUM_LLM
from plum.config import PLUMConfig

import argparse

def prepare_llm_data():
    parser = argparse.ArgumentParser(description="Prepare LLM Training Data")
    parser.add_argument("--dataset", type=str, default="movielens-1m", help="Dataset name (e.g., movielens-1m, movielens-10m)")
    args = parser.parse_args()
    
    print("Initializing PLUM LLM to get tokenizer...")
    config = PLUMConfig(dataset_name=args.dataset)
    print(f"Dataset: {config.dataset_name}")
    
    plum_model = PLUM_LLM(num_levels=config.active_model_config.num_levels, codebook_sizes=config.active_model_config.codebook_sizes, load_model=False)
    tokenizer = plum_model.tokenizer
    
    print("Loading data...")
    sequences_path = config.active_dataset.user_sequences_path
    sids_path = config.active_dataset.movie_sids_json_path
    
    if not os.path.exists(sequences_path) or not os.path.exists(sids_path):
        raise FileNotFoundError("Data files not found. Run generate_sids.py first.")
        
    user_sequences = torch.load(sequences_path, weights_only=False) # List[List[int]]
    with open(sids_path, 'r') as f:
        movie_sids = json.load(f) # Dict[str, dict] "0": {"sid": "...", "title": "...", "genres": "..."}
        
    print(f"Loaded {len(user_sequences)} user sequences.")
    
    # --- Optimization: Pre-compute Movie Tokens ---
    print("Pre-computing movie tokens...")
    movie_token_cache = {}
    
    # Static tokens
    start_prompt_tokens = tokenizer.encode("User History: ")
    separator_tokens = tokenizer.encode(", ")
    
    for mid_str, data in tqdm(movie_sids.items(), desc="Caching Movies"):
        if not isinstance(data, dict): continue
        
        sid_str = data.get("sid")
        title = data.get("title", "Unknown")
        genres = data.get("genres", "Unknown")
        
        if not sid_str: continue
        
        # 1. Text Tokens (without rating)
        # "Movie: Title (Genres) "
        base_text = f"Movie: {title} ({genres}) "
        text_tokens = tokenizer.encode(base_text)
        
        # 2. SID Tokens
        sid_tokens = []
        codes = [int(c) for c in sid_str.split('-')]
        for level, code in enumerate(codes):
            tid = plum_model.get_sid_token_id(level, code)
            if tid is not None:
                sid_tokens.append(tid)
                
        movie_token_cache[mid_str] = {
            "text": text_tokens,
            "sid": sid_tokens
        }
        
    # Cache rating tokens: "Rating: X "
    # We'll bucket/int cast ratings: 0, 1, 2, 3, 4, 5
    rating_token_cache = {}
    for r in range(6):
        rating_token_cache[r] = tokenizer.encode(f"Rating: {r} ")
        # Also handle halves if we want, but let's stick to ints for speed/vocab as requested?
        # User asked: "bucket or int-cast ratings (e.g., R=5, R=Med/High/Low)"
        # Let's do int casting.
        
    llm_dataset = []
    skipped_movies = 0
    total_movies = 0
    
    # 1. Enriched User Sequences
    # Format: "Movie: <Title> (<Genres>) Rating: <Rating> <SID>"
    print("Generating Enriched User Sequences...")
    
    for seq_data in tqdm(user_sequences, desc="User Sequences"):
        token_seq = []
        
        # Handle both old (list) and new (dict) formats
        if isinstance(seq_data, dict):
            seq = seq_data["items"]
            ratings = seq_data["ratings"]
        else:
            seq = seq_data
            ratings = [None] * len(seq)
            
        # Truncate history up front!
        # Limit to last 20 items
        MAX_HISTORY = 20
        if len(seq) > MAX_HISTORY:
            seq = seq[-MAX_HISTORY:]
            ratings = ratings[-MAX_HISTORY:]
        
        token_seq.extend(start_prompt_tokens)
        
        for i, movie_idx in enumerate(seq):
            total_movies += 1
            mid_str = str(movie_idx)
            
            cached = movie_token_cache.get(mid_str)
            if not cached:
                skipped_movies += 1
                continue
            
            # Add Text
            token_seq.extend(cached["text"])
            
            # Add Rating
            rating = ratings[i]
            if rating is not None:
                # Int cast
                r_int = int(round(float(rating)))
                r_int = max(0, min(5, r_int)) # Clamp 0-5
                token_seq.extend(rating_token_cache[r_int])
            
            # Add SID
            token_seq.extend(cached["sid"])
            
            # Add Separator
            token_seq.extend(separator_tokens)
            
        if len(token_seq) > 0:
            llm_dataset.append(token_seq)
 
    # 2. SID Grounding Tasks
    # Format: "<SID> is movie <Title> (<Genres>)"
    # And: "Movie <Title> (<Genres>) has ID <SID>"
    grounding_dataset = []
    
    if config.ablation.enable_grounding_task:
        print("Generating SID Grounding Tasks...")
        
        for movie_idx, data in tqdm(movie_sids.items(), desc="Grounding Tasks"):
            if not isinstance(data, dict): continue
            
            sid_str = data.get("sid")
            title = data.get("title")
            genres = data.get("genres")
            
            if not sid_str or not title: continue
            
            codes = [int(c) for c in sid_str.split('-')]
            sid_tokens = []
            for level, code in enumerate(codes):
                tid = plum_model.get_sid_token_id(level, code)
                if tid is not None:
                    sid_tokens.append(tid)
            
            if not sid_tokens: continue
            
            # Task A: SID -> Text
            # "<SID> is movie Title (Genres)"
            task_a_tokens = []
            task_a_tokens.extend(sid_tokens)
            task_a_tokens.extend(tokenizer.encode(f" is movie {title} ({genres})"))
            grounding_dataset.append(task_a_tokens)
            
            # Task B: Text -> SID
            # "Movie Title (Genres) has ID <SID>"
            task_b_tokens = []
            task_b_tokens.extend(tokenizer.encode(f"Movie {title} ({genres}) has ID "))
            task_b_tokens.extend(sid_tokens)
            grounding_dataset.append(task_b_tokens)
            
        print(f"Generated {len(grounding_dataset)} grounding examples.")
    else:
        print("Skipping SID Grounding Tasks (Ablation).")
    
    # Combine datasets
    # We can upsample grounding tasks if needed, but for now just mix them
    full_dataset = llm_dataset + grounding_dataset
    random.seed(42) # Ensure deterministic split
    random.shuffle(full_dataset)
    
    # Split into Train/Val
    val_split = config.active_llm_config.validation_split
    split_idx = int(len(full_dataset) * (1 - val_split))
    
    train_data = full_dataset[:split_idx]
    val_data = full_dataset[split_idx:]
    
    print(f"Total sequences: {len(full_dataset)}")
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")
    print(f"Skipped {skipped_movies}/{total_movies} movies in sequences.")
    
    output_path = config.active_dataset.llm_dataset_path
    save_data = {
        'train': train_data,
        'val': val_data
    }
    torch.save(save_data, output_path)
    print(f"Saved enriched LLM dataset (Train/Val) to {output_path}")

if __name__ == "__main__":
    prepare_llm_data()
