"""
Debug script to analyze SID generation patterns and understand low uniqueness
"""

import json
import torch
from collections import Counter
from plum.config import PLUMConfig

def analyze_sid_distribution():
    config = PLUMConfig(dataset_name="movielens-1m")
    
    # Load SIDs
    with open(config.active_dataset.movie_sids_json_path, 'r') as f:
        sids = json.load(f)
    
    print("="*70)
    print("SID DISTRIBUTION ANALYSIS")
    print("="*70)
    
    # Analyze SID structure
    level_0_codes = []
    level_1_codes = []
    level_2_codes = []
    full_sids = []
    
    for sid_str in sids.values():
        parts = sid_str.split('-')
        level_0_codes.append(int(parts[0]))
        level_1_codes.append(int(parts[1]))
        level_2_codes.append(int(parts[2]))
        full_sids.append(sid_str)
    
    # Count unique values at each level
    print(f"\nLevel-wise unique code usage:")
    print(f"  Level 0 (64 codes available): {len(set(level_0_codes))} unique used ({len(set(level_0_codes))/64*100:.1f}%)")
    print(f"  Level 1 (32 codes available): {len(set(level_1_codes))} unique used ({len(set(level_1_codes))/32*100:.1f}%)")
    print(f"  Level 2 (16 codes available): {len(set(level_2_codes))} unique used ({len(set(level_2_codes))/16*100:.1f}%)")
    
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
    print(f"\nLevel 0 code distribution (top 10):")
    l0_counts = Counter(level_0_codes)
    for code, count in l0_counts.most_common(10):
        print(f"  Code {code}: {count} times ({count/len(level_0_codes)*100:.1f}%)")
    
    print(f"\nLevel 1 code distribution (top 10):")
    l1_counts = Counter(level_1_codes)
    for code, count in l1_counts.most_common(10):
        print(f"  Code {code}: {count} times ({count/len(level_1_codes)*100:.1f}%)")
    
    print(f"\nLevel 2 code distribution (top 10):")
    l2_counts = Counter(level_2_codes)
    for code, count in l2_counts.most_common(10):
        print(f"  Code {code}: {count} times ({count/len(level_2_codes)*100:.1f}%)")
    
    # Check for dominant patterns
    print("\n" + "="*70)
    print("POTENTIAL ISSUES")
    print("="*70)
    
    # Check if certain codes dominate
    for level, counts, total in [(0, l0_counts, 64), (1, l1_counts, 32), (2, l2_counts, 16)]:
        top_code, top_count = counts.most_common(1)[0]
        if top_count > len(level_0_codes) * 0.1:  # If one code is used > 10% of the time
            print(f"⚠️  Level {level}: Code {top_code} dominates with {top_count/len(level_0_codes)*100:.1f}% usage")
    
    # Expected vs actual uniqueness
    theoretical_max = 64 * 32 * 16
    actual_combinations = len(set(full_sids))
    print(f"\nTheoretical max SIDs: {theoretical_max:,}")
    print(f"Actual unique SIDs: {actual_combinations:,} ({actual_combinations/theoretical_max*100:.2f}% of capacity)")

if __name__ == "__main__":
    analyze_sid_distribution()
