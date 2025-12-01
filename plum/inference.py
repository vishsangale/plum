import torch
import torch.nn.functional as F
from plum.config import PLUMConfig
from plum.utils import load_plum_models, load_mappings
import json
import os

def recommend_next_movie(history_movie_ids, sid_model, llm_model, movie_sids, sid_to_movies, device, config):
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
    
    # 2. Run Inference (Beam Search or Greedy)
    beam_width = config.ablation.beam_width if config.ablation.enable_beam_search else 1
    do_sample = False # Deterministic for now
    
    if config.ablation.enable_beam_search:
        print(f"Generating next item using Beam Search (k={beam_width})...")
    else:
        print(f"Generating next item using Greedy Decoding...")
        
    num_levels = config.active_model_config.num_levels
    
    outputs = llm_model.model.generate(
        input_ids, 
        max_new_tokens=100, 
        num_beams=beam_width, 
        num_return_sequences=beam_width if config.ablation.enable_beam_search else 1,
        early_stopping=True if config.ablation.enable_beam_search else False,
        pad_token_id=llm_model.tokenizer.pad_token_id,
        repetition_penalty=1.2
    )
    
    recommendations = []
    
    # Handle single output for greedy
    if not config.ablation.enable_beam_search:
        outputs = [outputs[0]]
    
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

import argparse

def run_inference():
    parser = argparse.ArgumentParser(description="Run PLUM Inference")
    parser.add_argument("--dataset", type=str, default="movielens-1m", help="Dataset name (e.g., movielens-1m, movielens-10m)")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of random user sequences to test")
    args = parser.parse_args()
    
    config = PLUMConfig(dataset_name=args.dataset)
    print(f"Dataset: {config.dataset_name}")
    
    sid_model, llm_model, device = load_plum_models(config)
    movie_sids, sid_to_movies = load_mappings(config)
    
    # Simulate User History (from real data)
    user_sequences = torch.load(config.active_dataset.user_sequences_path, weights_only=False)
    
    # Pick random sequences
    import random
    valid_sequences = [seq for seq in user_sequences if len(seq) >= 5]
    test_sequences = random.sample(valid_sequences, min(args.num_samples, len(valid_sequences)))
    
    for i, test_seq in enumerate(test_sequences):
        test_seq = test_seq[:5] # Use first 5 items as history
        
        print(f"\n{'='*40}")
        print(f"Sample {i+1}/{len(test_sequences)}")
        print(f"{'='*40}")
        
        print(f"User History (Movie IDs): {test_seq}")
        print("History Details:")
        for mid in test_seq:
            data = movie_sids.get(str(mid))
            if isinstance(data, dict):
                print(f"  {mid}: {data['title']} ({data['genres']}) [{data['sid']}]")
            else:
                print(f"  {mid}: {data}")
            
        recommendations, outputs, input_ids = recommend_next_movie(test_seq, sid_model, llm_model, movie_sids, sid_to_movies, device, config)
        
        print(f"\nTop Recommendations:")
        if not recommendations:
            print("No valid SIDs found in outputs.")
            print("Raw outputs:")
            for j, output_seq in enumerate(outputs):
                 generated_part = output_seq[len(input_ids[0]):]
                 print(f"  Rank {j+1}: {llm_model.tokenizer.decode(generated_part)}")
                 
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
