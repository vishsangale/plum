"""
Experiment: Enrich text input to improve embedding diversity

This script compares:
1. Original: "Title. Genres"
2. Enhanced: More descriptive text with genre descriptions

Run this to see if text enrichment helps with the current 128-dim model.
"""

import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from plum.config import PLUMConfig
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def create_enriched_text(title, genres):
    """Create more descriptive text from movie data"""
    # Extract year from title if present
    year = ""
    if "(" in title and ")" in title:
        year_part = title[title.rfind("("):title.rfind(")")+1]
        year = f"Released {year_part.strip('()')}"
    
    # Expand genre abbreviations and add descriptive context
    genre_list = genres.split("|")
    genre_descriptions = {
        "Action": "action-packed adventure",
        "Adventure": "adventurous journey",
        "Animation": "animated feature",
        "Children's": "family-friendly children's movie",
        "Comedy": "comedic entertainment",
        "Crime": "crime thriller",
        "Documentary": "documentary film",
        "Drama": "dramatic story",
        "Fantasy": "fantasy world",
        "Film-Noir": "film noir classic",
        "Horror": "horror suspense",
        "Musical": "musical performance",
        "Mystery": "mysterious plot",
        "Romance": "romantic story",
        "Sci-Fi": "science fiction",
        "Thriller": "thrilling suspense",
        "War": "war epic",
        "Western": "western tale"
    }
    
    # Build enriched description
    genre_desc = ", ".join([genre_descriptions.get(g, g.lower()) for g in genre_list])
    
    enriched = f"{title}. {year}. A {genre_desc} film"
    return enriched.strip()

def generate_embeddings(texts, model_name="prajjwal1/bert-tiny"):
    """Generate embeddings for given texts"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    embeddings_list = []
    batch_size = 64
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                          return_tensors="pt", max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            batch_embeddings = sum_embeddings / sum_mask
            embeddings_list.append(batch_embeddings)
    
    embeddings = torch.cat(embeddings_list, dim=0)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    return embeddings

def compare_text_strategies():
    """Compare original vs enriched text embeddings"""
    
    config = PLUMConfig(dataset_name="movielens-1m")
    raw_dir = config.active_dataset.raw_data_dir
    
    # Load movies
    print("Loading MovieLens data...")
    movies_df = pd.read_csv(
        os.path.join(raw_dir, "ml-1m/movies.dat"),
        sep="::", engine="python",
        names=["MovieID", "Title", "Genres"],
        encoding="latin-1"
    )
    
    # Strategy 1: Original (simple concatenation)
    print("\n" + "="*60)
    print("STRATEGY 1: Original Text")
    print("="*60)
    original_texts = (movies_df["Title"] + ". " + 
                     movies_df["Genres"].str.replace("|", ", ")).tolist()
    print(f"Example: {original_texts[0]}")
    
    print("\nGenerating embeddings...")
    original_embeddings = generate_embeddings(original_texts)
    
    # Compute stats
    sample_size = min(1000, len(original_embeddings))
    sample_indices = np.random.choice(len(original_embeddings), sample_size, replace=False)
    sample_emb = original_embeddings.numpy()[sample_indices]
    sim_matrix = cosine_similarity(sample_emb)
    upper_tri = np.triu_indices_from(sim_matrix, k=1)
    similarities = sim_matrix[upper_tri]
    
    print(f"Mean similarity: {similarities.mean():.4f}")
    print(f"Std similarity: {similarities.std():.4f}")
    print(f"Median similarity: {np.median(similarities):.4f}")
    
    # Strategy 2: Enriched text
    print("\n" + "="*60)
    print("STRATEGY 2: Enriched Text")
    print("="*60)
    enriched_texts = [create_enriched_text(row["Title"], row["Genres"]) 
                     for _, row in movies_df.iterrows()]
    print(f"Example: {enriched_texts[0]}")
    
    print("\nGenerating embeddings...")
    enriched_embeddings = generate_embeddings(enriched_texts)
    
    # Compute stats
    sample_emb_enriched = enriched_embeddings.numpy()[sample_indices]
    sim_matrix_enriched = cosine_similarity(sample_emb_enriched)
    similarities_enriched = sim_matrix_enriched[upper_tri]
    
    print(f"Mean similarity: {similarities_enriched.mean():.4f}")
    print(f"Std similarity: {similarities_enriched.std():.4f}")
    print(f"Median similarity: {np.median(similarities_enriched):.4f}")
    
    # Comparison
    print("\n" + "="*60)
    print("COMPARISON & RECOMMENDATION")
    print("="*60)
    
    improvement = similarities.mean() - similarities_enriched.mean()
    print(f"\nMean similarity change: {improvement:+.4f}")
    
    if abs(improvement) < 0.02:
        print("\n⚠️  MINIMAL IMPROVEMENT from text enrichment")
        print("   → Text enrichment alone won't solve the diversity issue")
        print("   → Consider using a larger model OR different embedding approach")
    elif improvement > 0:
        print(f"\n✓ IMPROVEMENT: {improvement:.4f} reduction in similarity")
        print("   → Text enrichment helps!")
        print("   → Consider using enriched text for SID training")
    else:
        print(f"\n⚠️  WORSE: {abs(improvement):.4f} increase in similarity")
        print("   → Text enrichment didn't help")
    
    # Model size analysis
    print("\n" + "="*60)
    print("MODEL SIZE ANALYSIS")
    print("="*60)
    
    print("\nCurrent setup (TinyBERT 128-dim):")
    print(f"  - Mean similarity: {similarities.mean():.4f}")
    print(f"  - Issue: Embeddings too similar → Poor SID diversity")
    
    print("\nYour concern about larger models:")
    print("  - Valid: More params, larger memory footprint")
    print("  - However: ALL dimensions get used, not 'wasted'")
    print("  - Trade-off: Better semantic capacity vs. efficiency")
    
    print("\nRecommended approach:")
    print("  1. Try 384-dim model (all-MiniLM-L6-v2) - good balance")
    print("  2. OR stick with 128-dim but use sentence-transformers")
    print("     which are fine-tuned for semantic similarity")
    print("  3. Compare SID codebook usage after training")
    
    return {
        'original_mean_sim': similarities.mean(),
        'enriched_mean_sim': similarities_enriched.mean(),
        'improvement': improvement
    }

if __name__ == "__main__":
    results = compare_text_strategies()
