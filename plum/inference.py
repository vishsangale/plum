import torch
import torch.nn.functional as F
from plum.config import PLUMConfig
from plum.utils import load_plum_models, load_mappings
import json
import os

def recommend_next_movie(history_movie_ids, sid_model, llm_model, movie_sids, sid_to_movies, device, config, ratings=None, sampling_args=None):
    # 1. Convert History to Enriched Prompt
    # Format: "User History: Movie: <Title> (<Genres>) Rating: <Rating> <SID>, ..."
    
    tokenizer = llm_model.tokenizer
    prompt_tokens = tokenizer.encode("User History: ")
    
    skipped_ids = []
    
    if ratings is None:
        ratings = [None] * len(history_movie_ids)
    
    for i, mid in enumerate(history_movie_ids):
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
            
        rating = ratings[i]
            
        if not sid_str:
            skipped_ids.append(mid)
            continue
            
        # Text Part
        if rating is not None:
             text_prompt = f"Movie: {title} ({genres}) Rating: {rating} "
        else:
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
    attention_mask = torch.ones_like(input_ids).to(device)
    
    
    # 2. Run Inference
    if sampling_args is None:
        sampling_args = {"do_sample": True, "temperature": 0.8, "top_p": 0.9}
        
    do_sample = sampling_args["do_sample"]
    beam_width = config.ablation.beam_width if (config.ablation.enable_beam_search and not do_sample) else 1
    
    if do_sample:
        print(f"Generating next item using Sampling (temp={sampling_args['temperature']}, top_p={sampling_args['top_p']})...")
    elif config.ablation.enable_beam_search:
        print(f"Generating next item using Beam Search (k={beam_width})...")
    else:
        print(f"Generating next item using Greedy Decoding...")
        
    num_levels = config.active_model_config.num_levels
        
    outputs = llm_model.model.generate(
        input_ids, 
        attention_mask=attention_mask,
        max_new_tokens=100, 
        num_beams=beam_width, 
        num_return_sequences=beam_width if (config.ablation.enable_beam_search and not do_sample) else 1,
        early_stopping=True if (config.ablation.enable_beam_search and not do_sample) else False,
        pad_token_id=llm_model.tokenizer.pad_token_id,
        repetition_penalty=1.2,
        do_sample=do_sample,
        temperature=sampling_args["temperature"],
        top_p=sampling_args["top_p"]
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
    parser.add_argument("--do_sample", action="store_true", default=True, help="Enable sampling (default: True)")
    parser.add_argument("--no_sample", action="store_false", dest="do_sample", help="Disable sampling")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling probability")
    parser.add_argument("--split", type=str, default="random", choices=["random", "val"], help="Data split to use (random sample or validation set)")
    args = parser.parse_args()
    
    config = PLUMConfig(dataset_name=args.dataset)
    # Override config with args if needed, or just pass to function
    config.ablation.enable_beam_search = not args.do_sample # If sampling, disable beam search usually, or use beam sampling

    print(f"Dataset: {config.dataset_name}")
    
    sid_model, llm_model, device = load_plum_models(config)
    movie_sids, sid_to_movies = load_mappings(config)
    
    # Simulate User History (from real data)
    user_sequences = torch.load(config.active_dataset.user_sequences_path, weights_only=False)
    
    # Pick sequences based on split
    import random
    
    def get_seq_len(seq):
        if isinstance(seq, dict):
            return len(seq["items"])
        return len(seq)
        
    if args.split == "val":
        print("Selecting Validation Set...")
        # Replicate logic from prepare_llm_data.py
        # 1. Enriched User Sequences
        llm_dataset = []
        for seq in user_sequences:
            # We just need the objects to shuffle, content doesn't matter for index
            llm_dataset.append(seq)
            
        # 2. SID Grounding Tasks
        # We need to know how many grounding tasks were generated to replicate the shuffle
        # This is tricky because we don't want to regenerate them here.
        # BUT, wait. prepare_llm_data saves 'train' and 'val' lists in the output .pt file.
        # We should probably load THAT file if we want exact validation set.
        # However, that file contains TOKENS, not original movie IDs.
        # Inference needs original movie IDs to check ground truth.
        
        # Alternative: We can try to replicate the shuffle if we know the counts.
        # But grounding tasks depend on movie_sids count.
        
        # Let's count grounding tasks
        grounding_count = 0
        if config.ablation.enable_grounding_task:
             for movie_idx, data in movie_sids.items():
                if not isinstance(data, dict): continue
                sid_str = data.get("sid")
                title = data.get("title")
                if not sid_str or not title: continue
                
                codes = [int(c) for c in sid_str.split('-')]
                sid_tokens = []
                for level, code in enumerate(codes):
                    tid = llm_model.get_sid_token_id(level, code)
                    if tid is not None:
                        sid_tokens.append(tid)
                if not sid_tokens: continue
                # Task A and Task B
                grounding_count += 2
        
        print(f"Estimated grounding tasks: {grounding_count}")
        
        # Create a list of indices or placeholders
        # We have len(user_sequences) user tasks + grounding_count grounding tasks
        # We want to find which USER SEQUENCES ended up in val.
        
        # Let's assign IDs to user sequences: 0 to N-1
        # And IDs to grounding tasks: N to N+M-1
        
        all_indices = list(range(len(user_sequences) + grounding_count))
        random.seed(42) # Fixed seed
        random.shuffle(all_indices)
        
        val_split = config.active_llm_config.validation_split
        split_idx = int(len(all_indices) * (1 - val_split))
        
        val_indices = set(all_indices[split_idx:])
        
        # Filter user sequences that are in val_indices
        valid_sequences = []
        for i, seq in enumerate(user_sequences):
            if i in val_indices:
                if get_seq_len(seq) >= 5:
                    valid_sequences.append(seq)
                    
        print(f"Found {len(valid_sequences)} user sequences in Validation Set.")
        
        # Sample from validation set
        test_sequences = random.sample(valid_sequences, min(args.num_samples, len(valid_sequences)))
        
    else:
        # Random split (original behavior)
        valid_sequences = [seq for seq in user_sequences if get_seq_len(seq) >= 5]
        test_sequences = random.sample(valid_sequences, min(args.num_samples, len(valid_sequences)))
    
    total_recall = 0.0
    num_valid_samples_for_recall = 0
    
    for i, test_seq_data in enumerate(test_sequences):
        # Handle both old (list) and new (dict) formats
        if isinstance(test_seq_data, dict):
            full_seq = test_seq_data["items"]
            full_ratings = test_seq_data["ratings"]
        else:
            full_seq = test_seq_data
            full_ratings = [None] * len(full_seq)

        test_seq = full_seq[:5] # Use first 5 items as history
        test_ratings = full_ratings[:5]
        
        future_seq = full_seq[5:]
        future_ratings = full_ratings[5:]
        
        # Calculate Relevant Items (Ground Truth with Rating > 3)
        relevant_items = set()
        for idx, mid in enumerate(future_seq):
            r = future_ratings[idx]
            if r is not None and float(r) > 3.0:
                relevant_items.add(mid)
        
        print(f"\n{'='*40}")
        print(f"Sample {i+1}/{len(test_sequences)}")
        print(f"{'='*40}")
        
        print(f"User History (Movie IDs): {test_seq}")
        print(f"Future Sequence Length: {len(future_seq)}")
        print(f"Relevant Future Items (Rating > 3): {len(relevant_items)}")
        print("History Details:")
        for idx, mid in enumerate(test_seq):
            data = movie_sids.get(str(mid))
            rating_str = f"Rating: {test_ratings[idx]}" if test_ratings[idx] is not None else ""
            if isinstance(data, dict):
                print(f"  {mid}: {data['title']} ({data['genres']}) {rating_str} [{data['sid']}]")
            else:
                print(f"  {mid}: {data} {rating_str}")
                
        if len(future_seq) > 0:
            print("\nGround Truth Next Items (First 5):")
            for idx, mid in enumerate(future_seq[:5]):
                data = movie_sids.get(str(mid))
                rating_str = f"Rating: {future_ratings[idx]}" if future_ratings[idx] is not None else ""
                if isinstance(data, dict):
                    print(f"  {mid}: {data['title']} ({data['genres']}) {rating_str} [{data['sid']}]")
                else:
                    print(f"  {mid}: {data} {rating_str}")
            
        sampling_args = {
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p
        }
        recommendations, outputs, input_ids = recommend_next_movie(test_seq, sid_model, llm_model, movie_sids, sid_to_movies, device, config, ratings=test_ratings, sampling_args=sampling_args)
        
        print(f"\nTop Recommendations:")
        if not recommendations:
            print("No valid SIDs found in outputs.")
            print("Raw outputs:")
            for j, output_seq in enumerate(outputs):
                 generated_part = output_seq[len(input_ids[0]):]
                 print(f"  Rank {j+1}: {llm_model.tokenizer.decode(generated_part)}")
                 
        # Calculate Recall@5
        top_5_recs = recommendations[:5]
        hits = 0
        
        for rec in recommendations:
            print(f"\nRank {rec['rank']}:")
            print(f"  Generated Text: {rec['text']}")
            print(f"  Detected SID: {rec['sid']}")
            
            # Check against ground truth
            matched_in_future = False
            is_hit_for_recall = False
            
            if rec['movies']:
                print(f"  Matched Movies ({len(rec['movies'])}):")
                for mid in rec['movies']:
                    data = movie_sids.get(str(mid))
                    
                    # Check if this movie is in the future sequence
                    is_ground_truth = False
                    gt_rating = None
                    if mid in future_seq:
                        is_ground_truth = True
                        # Find index in future_seq
                        try:
                            future_idx = future_seq.index(mid)
                            gt_rating = future_ratings[future_idx]
                        except ValueError:
                            pass
                    
                    if mid in relevant_items:
                        is_hit_for_recall = True
                    
                    gt_str = f" [GROUND TRUTH! Rating: {gt_rating}]" if is_ground_truth else ""
                    if is_ground_truth:
                        matched_in_future = True
                    
                    if isinstance(data, dict):
                         print(f"    - {data['title']} ({data['genres']}){gt_str}")
                    else:
                         print(f"    - ID {mid}{gt_str}")
            else:
                print("  (No exact match found for SID)")
                
            if not matched_in_future:
                 # Maybe the user didn't watch it, or it wasn't in the immediate future list we have?
                 pass
                 
            # Only count hits for top 5
            if rec['rank'] <= 5 and is_hit_for_recall:
                hits += 1
                
        if len(relevant_items) > 0:
            recall = hits / len(relevant_items)
            total_recall += recall
            num_valid_samples_for_recall += 1
            print(f"\nRecall@5: {recall:.4f} (Hits: {hits}, Relevant: {len(relevant_items)})")
        else:
            print(f"\nRecall@5: N/A (No relevant items in future)")

    if num_valid_samples_for_recall > 0:
        avg_recall = total_recall / num_valid_samples_for_recall
        print(f"\n{'='*40}")
        print(f"Average Recall@5: {avg_recall:.4f} (over {num_valid_samples_for_recall} samples)")
        print(f"{'='*40}")
    else:
        print("\nAverage Recall@5: N/A")

if __name__ == "__main__":
    run_inference()
