import torch
import torch.nn.functional as F
from plum.sid_model import PLUM_SID
from plum.llm_model import PLUM_LLM
from plum.config import PLUMConfig
import json
import os

def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load SID Model
    config = PLUMConfig()
    
    sid_model = PLUM_SID(config.active_dataset.input_dims, config.active_model_config.latent_dim, config.active_model_config.output_dim, config.active_model_config.codebook_sizes, kmeans_init=config.active_model_config.kmeans_init)
    sid_checkpoint_path = os.path.join(config.active_dataset.checkpoint_dir, config.sid_model_checkpoint)
    sid_model.load_state_dict(torch.load(sid_checkpoint_path, map_location=device, weights_only=False))
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
        
    llm_model = PLUM_LLM(model_name=llm_path, num_levels=config.active_model_config.num_levels, codebook_sizes=config.active_model_config.codebook_sizes)
    llm_model.to(device)
    llm_model.eval()
    print(f"LLM loaded from {llm_path}.")
    
    return sid_model, llm_model, device

def load_mappings(config):
    # Load SID -> MovieID mapping
    with open(config.active_dataset.movie_sids_json_path, 'r') as f:
        movie_sids = json.load(f) # MovieID -> Dict or String
        
    # Create reverse mapping: SID String -> List of MovieIDs
    sid_to_movies = {}
    for mid, data in movie_sids.items():
        if isinstance(data, dict):
            sid = data.get('sid')
        else:
            sid = data
            
        if sid:
            if sid not in sid_to_movies:
                sid_to_movies[sid] = []
            sid_to_movies[sid].append(mid)
        
    return movie_sids, sid_to_movies

def beam_search_decode(model, input_ids, num_levels, beam_width=10, device='cuda'):
    """
    Perform beam search to generate the next SID (sequence of tokens).
    """
    # Each candidate is (sequence_tensor, score)
    candidates = [(input_ids, 0.0)]
    
    # We need to generate `num_levels` tokens
    for level in range(num_levels):
        next_candidates = []
        
        for seq, score in candidates:
            with torch.no_grad():
                outputs = model(input_ids=seq)
                next_token_logits = outputs.logits[:, -1, :] # (1, vocab_size)
                
                # Mask EOS/Pad tokens to force SID generation if possible
                # (Optional: depends on if we want to allow early stopping, but SIDs are fixed length)
                next_token_logits[:, 50256] = -float('inf') 
                
                # Get top-k probabilities
                probs = F.log_softmax(next_token_logits, dim=-1)
                topk_probs, topk_ids = torch.topk(probs, beam_width, dim=-1)
                
                for i in range(beam_width):
                    token_id = topk_ids[0, i].unsqueeze(0).unsqueeze(0) # (1, 1)
                    token_prob = topk_probs[0, i].item()
                    
                    new_seq = torch.cat([seq, token_id], dim=1)
                    new_score = score + token_prob
                    next_candidates.append((new_seq, new_score))
        
        # Select top-k globally for this level
        candidates = sorted(next_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        
    return candidates

def recommend_next_movie(history_movie_ids, sid_model, llm_model, movie_sids, sid_to_movies, device, config, beam_width=10):
    # 1. Convert History to Enriched Prompt
    # Format: "User History: Movie: <Title> (<Genres>) <SID>, ..."
    
    tokenizer = llm_model.tokenizer
    prompt_tokens = tokenizer.encode("User History: ")
    
    skipped_ids = []
    
    for mid in history_movie_ids:
        sid_data = movie_sids.get(str(mid))
        if not sid_data:
            skipped_ids.append(mid)
            continue
            
        if isinstance(sid_data, dict):
            sid_str = sid_data.get("sid")
            title = sid_data.get("title", "Unknown")
            genres = sid_data.get("genres", "Unknown")
        else:
            sid_str = sid_data
            title = "Unknown"
            genres = "Unknown"
            
        if not sid_str:
            skipped_ids.append(mid)
            continue
            
        # Text Part
        text_prompt = f"Movie: {title} ({genres}) "
        prompt_tokens.extend(tokenizer.encode(text_prompt))
        
        # SID Part
        codes = [int(c) for c in sid_str.split('-')]
        for l, code in enumerate(codes):
            tid = llm_model.get_sid_token_id(l, code)
            if tid is not None:
                prompt_tokens.append(tid)
        
        # Separator
        prompt_tokens.extend(tokenizer.encode(", "))

    if len(prompt_tokens) == 0:
        raise ValueError(f"Cannot run inference: no tokens built.")

    input_ids = torch.tensor([prompt_tokens]).to(device)
    
    # 2. Run Beam Search
    print(f"Generating next item using Beam Search (k={beam_width})...")
    num_levels = config.active_model_config.num_levels
    
    # We want to predict the NEXT movie, so we expect the model to continue with "Movie:" or just the SID?
    # Based on training, it predicts "Movie: Title..." then SID.
    # BUT, for Generative Retrieval, we usually want to skip the text generation and go straight to SID if possible,
    # OR we let it generate the text and then the SID.
    # Given we trained it to generate text first, we should probably let it generate text.
    # HOWEVER, that's slow.
    # Let's try to force it to generate the SID directly by appending the prompt "Movie: " or just letting it run.
    # Actually, the paper says "Given user history... generate the SID of the next item".
    # If we trained it with "Movie: Title <SID>", it will want to generate "Movie: Title" first.
    # Let's see what happens if we just let it generate.
    
    # For this demo, let's assume we want to generate the SID directly.
    # But since we changed the training data, the model expects text.
    # Let's try to generate the whole next sequence "Movie: Title (Genre) <SID>"
    # But that's hard to control with fixed beam search for just SID tokens.
    
    # Alternative: We can just prompt it with "Movie: " and let it generate the Title and then SID.
    # But for strict SID retrieval, we might want to just predict the SID.
    # Let's try to prompt it to generate the SID directly by appending "Movie: Unknown (Unknown) " ? No that's bad.
    
    # Let's stick to the plan: The model was trained to predict the next token.
    # If we want to retrieve items, we should let it generate the SID.
    # But we trained it to generate text first.
    # Let's try to generate the text AND the SID.
    # We will run beam search for a longer sequence length, stopping when we get a full SID.
    
    # Actually, to keep it simple and robust:
    # Let's just generate tokens until we find a full SID pattern or hit a limit.
    
    # REVISED PLAN for Beam Search with Text:
    # It's complex to do beam search over text + SID.
    # Let's try to just generate the next 20 tokens and see if we find an SID.
    # But the user asked for Beam Search "like the paper".
    # The paper likely uses SIDs only or uses the SID as the target.
    # Since we added text, we made it a "text generation" task.
    
    # Let's try a hybrid:
    # We will use `model.generate` from HuggingFace which supports beam search out of the box!
    # This is much better than writing our own loop for variable length text.
    
    outputs = llm_model.model.generate(
        input_ids, 
        max_new_tokens=100, 
        num_beams=beam_width, 
        num_return_sequences=beam_width,
        early_stopping=True,
        pad_token_id=llm_model.tokenizer.pad_token_id,
        repetition_penalty=1.2
    )
    
    recommendations = []
    
    for i, output_seq in enumerate(outputs):
        # Decode the generated part
        generated_part = output_seq[len(input_ids[0]):]
        
        # Extract SID tokens
        # We look for the pattern of SID tokens in the generated sequence
        sid_codes = []
        decoded_text = llm_model.tokenizer.decode(generated_part, skip_special_tokens=False)
        
        # Parse tokens manually to find SIDs
        # This is safer than regex on string because of special tokens
        current_sid = []
        for tid in generated_part:
            token_str = llm_model.tokenizer.convert_ids_to_tokens(tid.item())
            if token_str.startswith("<SID_L"):
                try:
                    content = token_str.replace("<SID_L", "").replace(">", "")
                    _, code_str = content.split("_")
                    current_sid.append(code_str)
                except:
                    pass
            elif len(current_sid) > 0:
                # End of an SID sequence
                if len(current_sid) == num_levels:
                    sid_codes = current_sid
                    break # Found one
                current_sid = []
                
        if len(current_sid) == num_levels:
            sid_codes = current_sid
            
        if sid_codes:
            sid_str = "-".join(sid_codes)
            movies = sid_to_movies.get(sid_str, [])
            recommendations.append({
                "rank": i+1,
                "sid": sid_str,
                "text": decoded_text,
                "movies": movies
            })
            
    return recommendations, outputs, input_ids

def run_inference():
    config = PLUMConfig()
    sid_model, llm_model, device = load_models()
    movie_sids, sid_to_movies = load_mappings(config)
    
    # Simulate User History (from real data)
    user_sequences = torch.load(config.active_dataset.user_sequences_path, weights_only=False)
    # Pick a sequence that has some length
    test_seq = []
    for seq in user_sequences:
        if len(seq) >= 5:
            test_seq = seq[:5]
            break
            
    print(f"\nUser History (Movie IDs): {test_seq}")
    print("History Details:")
    for mid in test_seq:
        data = movie_sids.get(str(mid))
        if isinstance(data, dict):
            print(f"  {mid}: {data['title']} ({data['genres']}) [{data['sid']}]")
        else:
            print(f"  {mid}: {data}")
        
    recommendations, outputs, input_ids = recommend_next_movie(test_seq, sid_model, llm_model, movie_sids, sid_to_movies, device, config, beam_width=5)
    
    print(f"\nTop Recommendations (Beam Search):")
    if not recommendations:
        print("No valid SIDs found in beam search outputs.")
        print("Raw outputs:")
        for i, output_seq in enumerate(outputs):
             generated_part = output_seq[len(input_ids[0]):]
             print(f"  Beam {i}: {llm_model.tokenizer.decode(generated_part)}")
             
    for rec in recommendations:
        print(f"\nRank {rec['rank']}:")
        print(f"  Generated Text: {rec['text']}")
        print(f"  Detected SID: {rec['sid']}")
        if rec['movies']:
            print(f"  Matched Movies ({len(rec['movies'])}):")
            for mid in rec['movies'][:3]: # Show top 3 matches
                data = movie_sids.get(str(mid))
                if isinstance(data, dict):
                     print(f"    - {data['title']} ({data['genres']})")
                else:
                     print(f"    - ID {mid}")
        else:
            print("  (No exact match found for SID)")

if __name__ == "__main__":
    run_inference()
