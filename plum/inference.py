import torch
from plum.sid_model import PLUM_SID
from plum.llm_model import PLUM_LLM
from plum.config import PLUMConfig
import json
import os

def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load SID Model
    config = PLUMConfig()
    
    sid_model = PLUM_SID(config.active_dataset.input_dims, config.active_model_config.latent_dim, config.active_model_config.output_dim, config.active_model_config.num_levels, config.active_model_config.base_codebook_size)
    sid_checkpoint_path = os.path.join(config.active_dataset.checkpoint_dir, config.sid_model_checkpoint)
    sid_model.load_state_dict(torch.load(sid_checkpoint_path, map_location=device))
    sid_model.to(device)
    sid_model.eval()
    print("SID Model loaded.")
    
    # 2. Load LLM
    # Load from final checkpoint
    llm_path = os.path.join(config.active_dataset.checkpoint_dir, config.llm_checkpoint_dir)
    if not os.path.exists(llm_path):
        # Fallback to epoch 1 if final not ready
        llm_path = os.path.join(config.active_dataset.checkpoint_dir, "plum_llm_epoch_1")
        
    if not os.path.exists(llm_path):
        print("Warning: Checkpoint not found, using base distilgpt2 (untrained)")
        llm_path = "distilgpt2"
        
    llm_model = PLUM_LLM(model_name=llm_path, num_levels=config.active_model_config.num_levels, base_codebook_size=config.active_model_config.base_codebook_size)
    llm_model.to(device)
    llm_model.eval()
    print(f"LLM loaded from {llm_path}.")
    
    return sid_model, llm_model, device

def load_mappings(config):
    # Load SID -> MovieID mapping
    with open(config.active_dataset.movie_sids_json_path, 'r') as f:
        movie_sids = json.load(f) # MovieID -> SID String
        
    # Create reverse mapping: SID String -> List of MovieIDs
    sid_to_movies = {}
    for mid, sid in movie_sids.items():
        if sid not in sid_to_movies:
            sid_to_movies[sid] = []
        sid_to_movies[sid].append(mid)
        
    return movie_sids, sid_to_movies

def recommend_next_movie(history_movie_ids, sid_model, llm_model, movie_sids, sid_to_movies, device, config):
    # 1. Convert History to SID Tokens
    history_tokens = []
    
    for mid in history_movie_ids:
        sid_str = movie_sids.get(str(mid))
        if not sid_str:
            continue
            
        codes = [int(c) for c in sid_str.split('-')]
        for l, code in enumerate(codes):
            tid = llm_model.get_sid_token_id(l, code)
            history_tokens.append(tid)
            
    input_ids = torch.tensor([history_tokens]).to(device)
    
    # 2. Generate Next SID (3 tokens)
    num_levels = config.active_model_config.num_levels
    generated_ids = input_ids
    
    print(f"Generating next item (History length: {len(history_movie_ids)})...")
    
    with torch.no_grad():
        for _ in range(num_levels):
            outputs = llm_model(input_ids=generated_ids)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Mask EOS token
            next_token_logits[:, 50256] = -float('inf')
            
            # Greedy decode
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
    # Extract generated tokens
    new_tokens = generated_ids[0, -num_levels:].tolist()
    
    # 3. Decode Tokens to SID String
    generated_codes = []
    for tid in new_tokens:
        token_str = llm_model.tokenizer.convert_ids_to_tokens(tid)
        try:
            # <SID_L{l}_{c}>
            content = token_str.replace("<SID_L", "").replace(">", "")
            _, code_str = content.split("_")
            generated_codes.append(code_str)
        except:
            pass
            
    generated_sid_str = "-".join(generated_codes)
    print(f"Generated SID: {generated_sid_str}")
    
    # 4. Find Matching Movies
    recommended_movies = sid_to_movies.get(generated_sid_str, [])
    
    return recommended_movies, generated_sid_str

def run_inference():
    config = PLUMConfig()
    sid_model, llm_model, device = load_models()
    movie_sids, sid_to_movies = load_mappings(config)
    
    # Simulate User History (from real data)
    # Let's pick a sequence from the dataset
    user_sequences = torch.load(config.active_dataset.user_sequences_path)
    test_seq = user_sequences[0][:5] # First 5 movies of first user
    
    print(f"\nUser History (Movie IDs): {test_seq}")
    print("SIDs:")
    for mid in test_seq:
        print(f"  {mid}: {movie_sids.get(str(mid))}")
        
    recommendations, gen_sid = recommend_next_movie(test_seq, sid_model, llm_model, movie_sids, sid_to_movies, device, config)
    
    print(f"\nRecommended Movies: {recommendations}")
    if not recommendations:
        print("  (No exact match found for generated SID)")

if __name__ == "__main__":
    run_inference()
