import torch
from plum.llm_model import PLUM_LLM
import json
import os
import pandas as pd

def probe_llm():
    print("Loading metadata...")
    # Load Movies to get Titles
    movies_df = pd.read_csv(
        "ml-1m/movies.dat", 
        sep="::", 
        engine="python", 
        names=["MovieID", "Title", "Genres"],
        encoding="latin-1"
    )
    movie_id_to_title = dict(zip(movies_df["MovieID"], movies_df["Title"]))
    
    # Load ID Mapping (Original MovieID -> Internal Index)
    movie_id_to_idx = torch.load("plum/movie_id_map.pt")
    idx_to_movie_id = {v: k for k, v in movie_id_to_idx.items()}
    
    # Load SID Mapping (Internal Index -> SID)
    with open("plum/movie_sids.json", 'r') as f:
        idx_to_sid = json.load(f)
        
    # Reverse SID Mapping (SID -> List of Internal Indices)
    sid_to_indices = {}
    for idx, sid in idx_to_sid.items():
        if sid not in sid_to_indices:
            sid_to_indices[sid] = []
        sid_to_indices[sid].append(int(idx))
        
    def get_titles_for_sid(sid_str):
        indices = sid_to_indices.get(sid_str, [])
        titles = []
        for idx in indices:
            original_id = idx_to_movie_id.get(idx)
            if original_id:
                title = movie_id_to_title.get(original_id, "Unknown")
                titles.append(title)
        return titles

    print("Loading model from checkpoint...")
    checkpoint_path = "checkpoints/plum_llm_epoch_1"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found. Trying 'checkpoints/plum_llm_final'")
        checkpoint_path = "checkpoints/plum_llm_final"
        
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("No checkpoints found.")
        
    # Load PLUM model (wrapper)
    plum_model = PLUM_LLM(model_name=checkpoint_path, num_levels=3, base_codebook_size=512)
    
    # Get underlying HF model and tokenizer
    model = plum_model.model
    tokenizer = plum_model.tokenizer
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print("Model loaded. Running probes...")
    
    def generate_text(prompt, max_new_tokens=10):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True, # Add sampling for variety
                top_k=50,
                top_p=0.95
            )
            
        return tokenizer.decode(outputs[0], skip_special_tokens=False)

    def sid_to_tokens_str(sid_str):
        codes = [int(c) for c in sid_str.split('-')]
        tokens = []
        for l, code in enumerate(codes):
            tokens.append(f"<SID_L{l}_{code}>")
        return "".join(tokens)

    # --- Test 1: SID Sequence Completion ---
    print("\n--- Test 1: SID Sequence Completion ---")
    
    # Pick 3 random SIDs from the map
    sids = list(idx_to_sid.values())[:3]
    
    print("Prompt Items:")
    for s in sids:
        titles = get_titles_for_sid(s)
        print(f"  SID {s}: {titles[:3]}...") # Show first 3 titles if collision
        
    prompt_sids = [sid_to_tokens_str(s) for s in sids]
    prompt_text = "".join(prompt_sids)
    
    generated_text = generate_text(prompt_text, max_new_tokens=10)
    
    # Extract new tokens
    new_text = generated_text[len(prompt_text):]
    print(f"\nGenerated Raw: {new_text}")
    
    # Try to parse generated SID
    # This is tricky because it might generate partial tokens or multiple SIDs
    # Let's just look for <SID_L... tokens
    import re
    tokens = re.findall(r"<SID_L(\d+)_(\d+)>", new_text)
    
    if tokens:
        # Group by levels (0, 1, 2)
        # Assuming it generates valid sequence 0->1->2
        current_sid_codes = []
        for level, code in tokens:
            current_sid_codes.append(code)
            if len(current_sid_codes) == 3:
                sid_str = "-".join(current_sid_codes)
                print(f"Generated SID: {sid_str}")
                titles = get_titles_for_sid(sid_str)
                print(f"Mapped Titles: {titles}")
                current_sid_codes = []
    else:
        print("Could not parse valid SID from output.")
    
    # --- Test 3: Behavior Test ---
    print("\n--- Test 3: Behavior Test ---")
    prompt = f"User watched {prompt_sids[0]} and {prompt_sids[1]}. Next they watched"
    output = generate_text(prompt, max_new_tokens=10)
    print(f"Input: User watched {sids[0]} and {sids[1]}...")
    print(f"Output: {output}")

if __name__ == "__main__":
    probe_llm()
