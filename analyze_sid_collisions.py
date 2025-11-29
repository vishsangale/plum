import json
import os
from collections import defaultdict
from plum.config import PLUMConfig

def analyze_collisions():
    config = PLUMConfig()
    sids_path = config.active_dataset.movie_sids_json_path
    
    if not os.path.exists(sids_path):
        print(f"Error: {sids_path} not found.")
        return

    with open(sids_path, 'r') as f:
        sid_map = json.load(f)
        
    # Group by SID
    sid_to_movies = defaultdict(list)
    for idx, entry in sid_map.items():
        sid = entry['sid']
        sid_to_movies[sid].append(entry)
        
    # Analyze collisions
    collisions = {sid: movies for sid, movies in sid_to_movies.items() if len(movies) > 1}
    
    print(f"Total SIDs: {len(sid_to_movies)}")
    print(f"Colliding SIDs: {len(collisions)}")
    
    # Sort collisions by size
    sorted_collisions = sorted(collisions.items(), key=lambda x: len(x[1]), reverse=True)
    
    print("\nTop 10 Collisions:")
    for i, (sid, movies) in enumerate(sorted_collisions[:10]):
        print(f"\n{i+1}. SID: {sid} (Count: {len(movies)})")
        # Check genre consistency
        genres = [m['genres'] for m in movies]
        unique_genres = set(genres)
        print(f"   Genres: {unique_genres}")
        print("   Movies:")
        for m in movies[:5]: # Show first 5
            print(f"     - {m['title']} ({m['genres']})")
        if len(movies) > 5:
            print(f"     ... and {len(movies)-5} more")
            
    # Check if collisions share genres generally
    print("\nGenre Consistency Analysis:")
    same_genre_count = 0
    total_collisions = len(collisions)
    
    for sid, movies in collisions.items():
        first_genre = movies[0]['genres']
        if all(m['genres'] == first_genre for m in movies):
            same_genre_count += 1
            
    print(f"Collisions with exact same genre string: {same_genre_count}/{total_collisions} ({same_genre_count/total_collisions*100:.1f}%)")

    # Analyze unique codes per level
    print("\nUnique Codes per Level:")
    level_codes = defaultdict(set)
    for sid in sid_map.values():
        codes = sid['sid'].split('-')
        for i, code in enumerate(codes):
            level_codes[i].add(code)
            
    for level, codes in sorted(level_codes.items()):
        print(f"  Level {level}: {len(codes)} unique codes used")

if __name__ == "__main__":
    analyze_collisions()
