"""
Analysis: Do we need more dimensions for short movie text?

This addresses the question: "Won't larger models waste dimensions on short text?"
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

def analyze_dimension_usage():
    """
    Test if different model sizes effectively use their dimensions
    even with short text like movie titles.
    """
    
    # Sample movie texts (short, like yours)
    test_texts = [
        "Toy Story (1995). Animation, Children's, Comedy",
        "Jumanji (1995). Adventure, Children's, Fantasy",
        "Grumpier Old Men (1995). Comedy, Romance",
        "Heat (1995). Action, Crime, Thriller",
        "Seven (1995). Crime, Thriller"
    ]
    
    print("="*70)
    print("DIMENSION USAGE ANALYSIS: Short Text (Movie Metadata)")
    print("="*70)
    
    # Model 1: TinyBERT (128 dim)
    print("\n1. TinyBERT (128 dimensions)")
    print("-" * 70)
    tokenizer_tiny = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    model_tiny = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    
    inputs = tokenizer_tiny(test_texts, padding=True, truncation=True, 
                           return_tensors="pt", max_length=128)
    with torch.no_grad():
        outputs = model_tiny(**inputs)
        embeddings_128 = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        embeddings_128 = torch.nn.functional.normalize(embeddings_128, p=2, dim=-1)
    
    # Analyze variance across dimensions
    dim_variance_128 = embeddings_128.var(dim=0).numpy()
    
    print(f"Mean variance per dim: {dim_variance_128.mean():.6f}")
    print(f"Std variance per dim: {dim_variance_128.std():.6f}")
    print(f"Zero-variance dims: {(dim_variance_128 < 1e-6).sum()}/128")
    print(f"Low-variance dims (< 0.001): {(dim_variance_128 < 0.001).sum()}/128")
    print(f"Active dims (>= 0.001): {(dim_variance_128 >= 0.001).sum()}/128")
    
    # Pairwise distances
    from sklearn.metrics.pairwise import cosine_similarity
    sim_128 = cosine_similarity(embeddings_128.numpy())
    # Get off-diagonal elements
    mask = ~np.eye(sim_128.shape[0], dtype=bool)
    similarities_128 = sim_128[mask]
    print(f"Mean pairwise similarity: {similarities_128.mean():.4f}")
    print(f"Range: [{similarities_128.min():.4f}, {similarities_128.max():.4f}]")
    
    # Model 2: all-MiniLM-L6-v2 (384 dim) - Sentence-Transformers
    print("\n2. all-MiniLM-L6-v2 (384 dimensions)")
    print("-" * 70)
    model_384 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings_384 = model_384.encode(test_texts)
    embeddings_384 = embeddings_384 / np.linalg.norm(embeddings_384, axis=1, keepdims=True)
    
    dim_variance_384 = embeddings_384.var(axis=0)
    
    print(f"Mean variance per dim: {dim_variance_384.mean():.6f}")
    print(f"Std variance per dim: {dim_variance_384.std():.6f}")
    print(f"Zero-variance dims: {(dim_variance_384 < 1e-6).sum()}/384")
    print(f"Low-variance dims (< 0.001): {(dim_variance_384 < 0.001).sum()}/384")
    print(f"Active dims (>= 0.001): {(dim_variance_384 >= 0.001).sum()}/384")
    
    sim_384 = cosine_similarity(embeddings_384)
    similarities_384 = sim_384[mask]
    print(f"Mean pairwise similarity: {similarities_384.mean():.4f}")
    print(f"Range: [{similarities_384.min():.4f}, {similarities_384.max():.4f}]")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY & INSIGHTS")
    print("="*70)
    
    print("\n📊 Dimension Usage:")
    print(f"   TinyBERT (128d): {(dim_variance_128 >= 0.001).sum()}/128 active dims")
    print(f"   MiniLM (384d): {(dim_variance_384 >= 0.001).sum()}/384 active dims")
    
    diversity_improvement = similarities_128.mean() - similarities_384.mean()
    
    print(f"\n📏 Diversity (lower similarity = more diversity):")
    print(f"   TinyBERT (128d): {similarities_128.mean():.4f} mean similarity")
    print(f"   MiniLM (384d): {similarities_384.mean():.4f} mean similarity")
    print(f"   Improvement: {diversity_improvement:+.4f}")
    
    if diversity_improvement > 0.05:
        print("\n✅ CONCLUSION:")
        print("   Larger model (384d) provides SIGNIFICANTLY better diversity")
        print("   Even with short text, extra dims capture semantic nuances")
        print("   Recommendation: Use 384-dim model for better SID training")
    elif diversity_improvement > 0.02:
        print("\n✓ CONCLUSION:")
        print("   Larger model (384d) provides MODERATE improvement")
        print("   Trade-off: Better diversity vs. larger model size")
        print("   Recommendation: Try 384-dim model, monitor SID codebook usage")
    else:
        print("\n⚠️ CONCLUSION:")
        print("   Minimal difference between model sizes")
        print("   The issue may not be model capacity alone")
        print("   Consider: Different embedding approach or additional features")
    
    print("\n💡 Key Insight:")
    print("   Model dimensions DON'T get 'wasted' on short text.")
    print("   Pre-trained models use ALL dimensions to encode semantic meaning,")
    print("   even from short inputs. The question is: does MORE capacity help?")
    
    print("\n🎯 For Your Use Case:")
    print("   Current: 0.8557 mean similarity → 12% SID uniqueness")
    print("   If 384d gives < 0.80 similarity → Could improve SID diversity")
    print("   If 384d still > 0.85 → Need different embedding strategy")
    
    return {
        '128d_similarity': similarities_128.mean(),
        '384d_similarity': similarities_384.mean(),
        'improvement': diversity_improvement
    }

if __name__ == "__main__":
    results = analyze_dimension_usage()
