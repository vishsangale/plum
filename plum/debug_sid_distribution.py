"""
Debug script to analyze SID generation patterns and understand low uniqueness
"""

import json
import torch
from collections import Counter
from plum.config import PLUMConfig

import argparse

def analyze_sid_distribution():
    parser = argparse.ArgumentParser(description="Analyze SID Distribution")
    parser.add_argument("--dataset", type=str, default="movielens-1m", help="Dataset name")
    args = parser.parse_args()

    config = PLUMConfig(dataset_name=args.dataset)
    
    # Load SIDs
    print(f"Analyzing dataset: {args.dataset}")
    print(f"Loading SIDs from {config.active_dataset.movie_sids_json_path}")
    with open(config.active_dataset.movie_sids_json_path, 'r') as f:
        sids = json.load(f)
    
    print("="*70)
    print("SID DISTRIBUTION ANALYSIS")
    print("="*70)
    
    # Get codebook sizes from config
    codebook_sizes = config.active_model_config.codebook_sizes
    num_levels = len(codebook_sizes)
    
    # Analyze SID structure
    level_codes = [[] for _ in range(num_levels)]
    full_sids = []
    
    for entry in sids.values():
        sid_str = entry['sid']
        parts = sid_str.split('-')
        for i in range(min(len(parts), num_levels)):
            level_codes[i].append(int(parts[i]))
        full_sids.append(sid_str)
    
    # Count unique values at each level
    print(f"\nLevel-wise unique code usage:")
    level_counts = []
    for i in range(num_levels):
        unique_used = len(set(level_codes[i]))
        total_available = codebook_sizes[i]
        percent = (unique_used / total_available) * 100 if total_available > 0 else 0
        level_counts.append(Counter(level_codes[i]))
        print(f"  Level {i} ({total_available} codes available): {unique_used} unique used ({percent:.1f}%)")
    
    # Most common SIDs
    sid_counts = Counter(full_sids)
    print(f"\nTop 10 most common SIDs:")
    for sid, count in sid_counts.most_common(10):
        print(f"  {sid}: {count} movies ({count/len(sids)*100:.1f}%)")
    
    # Distribution of collision sizes
    collision_sizes = [count for sid, count in sid_counts.items() if count > 1]
    print(f"\nCollision analysis:")
    print(f"  Unique SIDs: {len([c for c in sid_counts.values() if c == 1])}")
    print(f"  SIDs with collisions: {len(collision_sizes)}")
    print(f"  Total unique SIDs: {len(sid_counts)}")
    print(f"  Average collision size: {sum(collision_sizes)/len(collision_sizes) if collision_sizes else 0:.2f}")
    print(f"  Max collision: {max(sid_counts.values())} movies")
    
    # Level-wise distribution
    for i in range(num_levels):
        print(f"\nLevel {i} code distribution (top 10):")
        for code, count in level_counts[i].most_common(10):
            print(f"  Code {code}: {count} times ({count/len(level_codes[i])*100:.1f}%)")
    
    # Check for dominant patterns
    print("\n" + "="*70)
    print("POTENTIAL ISSUES")
    print("="*70)
    
    # Check if certain codes dominate
    for i in range(num_levels):
        if not level_counts[i]: continue
        top_code, top_count = level_counts[i].most_common(1)[0]
        if top_count > len(level_codes[i]) * 0.1:  # If one code is used > 10% of the time
            print(f"⚠️  Level {i}: Code {top_code} dominates with {top_count/len(level_codes[i])*100:.1f}% usage")
    
    # Expected vs actual uniqueness
    theoretical_max = 1
    for size in codebook_sizes:
        theoretical_max *= size
        
    actual_combinations = len(set(full_sids))
    print(f"\nTheoretical max SIDs: {theoretical_max:,}")
    print(f"Actual unique SIDs: {actual_combinations:,} ({actual_combinations/theoretical_max*100:.2f}% of capacity)")

if __name__ == "__main__":
    analyze_sid_distribution()
