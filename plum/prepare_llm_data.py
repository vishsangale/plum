import torch
import json
import os
from tqdm import tqdm
from plum.llm_model import PLUM_LLM

def prepare_llm_data():
    print("Initializing PLUM LLM to get tokenizer...")
    # Initialize model to get the tokenizer with special tokens
    # Using defaults: num_levels=3, base_codebook_size=512
    plum_model = PLUM_LLM()
    
    print("Loading data...")
    sequences_path = "plum/user_sequences.pt"
    sids_path = "plum/movie_sids.json"
    
    if not os.path.exists(sequences_path) or not os.path.exists(sids_path):
        raise FileNotFoundError("Data files not found. Run generate_sids.py first.")
        
    user_sequences = torch.load(sequences_path) # List[List[int]]
    with open(sids_path, 'r') as f:
        movie_sids = json.load(f) # Dict[str, str] "0": "c1-c2-c3"
        
    print(f"Loaded {len(user_sequences)} user sequences.")
    
    llm_dataset = []
    skipped_movies = 0
    total_movies = 0
    
    print("Converting sequences to tokens...")
    for seq in tqdm(user_sequences):
        token_seq = []
        for movie_idx in seq:
            total_movies += 1
            sid_str = movie_sids.get(str(movie_idx))
            
            if not sid_str:
                skipped_movies += 1
                continue
                
            # Parse SID string "c1-c2-c3"
            codes = [int(c) for c in sid_str.split('-')]
            
            # Convert to tokens
            for level, code in enumerate(codes):
                tid = plum_model.get_sid_token_id(level, code)
                if tid is not None:
                    token_seq.append(tid)
                else:
                    print(f"Warning: Could not find token for Level {level} Code {code}")
                    
        if len(token_seq) > 0:
            llm_dataset.append(token_seq)
            
    print(f"Processed {len(llm_dataset)} sequences.")
    print(f"Skipped {skipped_movies}/{total_movies} movies (missing SIDs).")
    
    output_path = "plum/llm_dataset.pt"
    torch.save(llm_dataset, output_path)
    print(f"Saved LLM dataset to {output_path}")

if __name__ == "__main__":
    prepare_llm_data()
