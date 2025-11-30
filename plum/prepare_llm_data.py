import torch
import json
import os
import random
from tqdm import tqdm
from plum.llm_model import PLUM_LLM
from plum.config import PLUMConfig

def prepare_llm_data():
    print("Initializing PLUM LLM to get tokenizer...")
    config = PLUMConfig()
    plum_model = PLUM_LLM(num_levels=config.active_model_config.num_levels, codebook_sizes=config.active_model_config.codebook_sizes)
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
    
    llm_dataset = []
    skipped_movies = 0
    total_movies = 0
    
    # 1. Enriched User Sequences
    # Format: "Movie: <Title> (<Genres>) <SID>"
    print("Generating Enriched User Sequences...")
    for seq in tqdm(user_sequences, desc="User Sequences"):
        token_seq = []
        
        # Add a start prompt for the sequence
        # "User History: "
        token_seq.extend(tokenizer.encode("User History: "))
        
        for movie_idx in seq:
            total_movies += 1
            sid_data = movie_sids.get(str(movie_idx))
            
            if not sid_data or not isinstance(sid_data, dict):
                skipped_movies += 1
                continue
            
            sid_str = sid_data.get("sid")
            title = sid_data.get("title", "Unknown")
            genres = sid_data.get("genres", "Unknown")
            
            if not sid_str:
                skipped_movies += 1
                continue
                
            # Text Part: "Movie: Title (Genres) "
            text_prompt = f"Movie: {title} ({genres}) "
            token_seq.extend(tokenizer.encode(text_prompt))
            
            # SID Part: <SID_Tokens>
            codes = [int(c) for c in sid_str.split('-')]
            for level, code in enumerate(codes):
                tid = plum_model.get_sid_token_id(level, code)
                if tid is not None:
                    token_seq.append(tid)
            
            # Add a separator (comma or space)
            token_seq.extend(tokenizer.encode(", "))
            
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
    random.shuffle(full_dataset)
    
    print(f"Total processed sequences: {len(full_dataset)}")
    print(f"Skipped {skipped_movies}/{total_movies} movies in sequences.")
    
    output_path = config.active_dataset.llm_dataset_path
    torch.save(full_dataset, output_path)
    print(f"Saved enriched LLM dataset to {output_path}")

if __name__ == "__main__":
    prepare_llm_data()
