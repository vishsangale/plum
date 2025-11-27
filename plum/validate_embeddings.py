import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from plum.config import PLUMConfig

def validate_embeddings():
    """Validate BERT embeddings generated for MovieLens dataset"""
    
    config = PLUMConfig(dataset_name="movielens-1m")
    
    # Load embeddings
    print("Loading embeddings...")
    embeddings = torch.load(config.active_dataset.embeddings_path)
    embeddings_np = embeddings.numpy()
    
    # Load movie metadata
    raw_dir = config.active_dataset.raw_data_dir
    movies_df = pd.read_csv(
        os.path.join(raw_dir, "ml-1m/movies.dat"), 
        sep="::", 
        engine="python", 
        names=["MovieID", "Title", "Genres"],
        encoding="latin-1"
    )
    
    print(f"\n{'='*60}")
    print("EMBEDDING VALIDATION REPORT")
    print(f"{'='*60}\n")
    
    # 1. Basic Statistics
    print("1. BASIC STATISTICS")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Dimension: {embeddings.shape[1]}")
    print(f"   Number of items: {embeddings.shape[0]}")
    print(f"   Mean norm: {torch.norm(embeddings, dim=1).mean():.4f}")
    print(f"   Std norm: {torch.norm(embeddings, dim=1).std():.4f}")
    print(f"   Min/Max values: [{embeddings_np.min():.4f}, {embeddings_np.max():.4f}]")
    print(f"   Mean value: {embeddings_np.mean():.4f}")
    print(f"   Std value: {embeddings_np.std():.4f}")
    
    # 2. Cosine Similarity Analysis
    print("\n2. COSINE SIMILARITY ANALYSIS")
    print("   Computing pairwise similarities (this may take a moment)...")
    
    # Sample if dataset is too large
    sample_size = min(1000, len(embeddings))
    sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
    sample_embeddings = embeddings_np[sample_indices]
    
    sim_matrix = cosine_similarity(sample_embeddings)
    # Get upper triangle (excluding diagonal) to avoid counting each pair twice
    upper_tri_idx = np.triu_indices_from(sim_matrix, k=1)
    similarities = sim_matrix[upper_tri_idx]
    
    print(f"   Mean similarity: {similarities.mean():.4f}")
    print(f"   Std similarity: {similarities.std():.4f}")
    print(f"   Min similarity: {similarities.min():.4f}")
    print(f"   Max similarity: {similarities.max():.4f}")
    print(f"   Median similarity: {np.median(similarities):.4f}")
    
    # Distribution of similarities
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("\n   Similarity Percentiles:")
    for p in percentiles:
        val = np.percentile(similarities, p)
        print(f"      {p}th: {val:.4f}")
    
    # 3. Check for Duplicates/Near-Duplicates
    print("\n3. DUPLICATE ANALYSIS")
    very_similar_threshold = 0.99
    high_sim_pairs = np.sum(similarities > very_similar_threshold)
    print(f"   Pairs with similarity > {very_similar_threshold}: {high_sim_pairs}")
    
    if high_sim_pairs > 0:
        print(f"   WARNING: Found {high_sim_pairs} very similar pairs!")
        # Find and display some examples
        high_sim_idx = np.where(sim_matrix > very_similar_threshold)
        for i in range(min(5, len(high_sim_idx[0]))):
            idx1, idx2 = high_sim_idx[0][i], high_sim_idx[1][i]
            if idx1 != idx2:  # Skip diagonal
                real_idx1 = sample_indices[idx1]
                real_idx2 = sample_indices[idx2]
                print(f"      Movies {real_idx1} and {real_idx2}: {sim_matrix[idx1, idx2]:.4f}")
                print(f"        - {movies_df.iloc[real_idx1]['Title']}")
                print(f"        - {movies_df.iloc[real_idx2]['Title']}")
    
    # 4. Dimension-wise Analysis
    print("\n4. DIMENSION-WISE ANALYSIS")
    dim_means = embeddings_np.mean(axis=0)
    dim_stds = embeddings_np.std(axis=0)
    
    print(f"   Mean activation per dim - Mean: {dim_means.mean():.4f}, Std: {dim_means.std():.4f}")
    print(f"   Std activation per dim - Mean: {dim_stds.mean():.4f}, Std: {dim_stds.std():.4f}")
    print(f"   Min std across dimensions: {dim_stds.min():.4f}")
    print(f"   Max std across dimensions: {dim_stds.max():.4f}")
    
    # Check for dead dimensions (low variance)
    dead_dim_threshold = 0.01
    dead_dims = np.sum(dim_stds < dead_dim_threshold)
    print(f"   Dead dimensions (std < {dead_dim_threshold}): {dead_dims}/{embeddings.shape[1]}")
    
    # 5. Genre-based Analysis
    print("\n5. GENRE-BASED ANALYSIS")
    # Extract primary genre
    movies_df['PrimaryGenre'] = movies_df['Genres'].str.split('|').str[0]
    
    # Get top genres
    top_genres = movies_df['PrimaryGenre'].value_counts().head(5)
    print(f"   Analyzing top {len(top_genres)} genres...")
    
    for genre in top_genres.index[:3]:  # Show top 3
        genre_mask = movies_df['PrimaryGenre'] == genre
        genre_indices = np.where(genre_mask)[0]
        
        if len(genre_indices) > 1:
            genre_embeddings = embeddings_np[genre_indices]
            genre_sim = cosine_similarity(genre_embeddings)
            
            # Get upper triangle
            upper_tri = np.triu_indices_from(genre_sim, k=1)
            intra_genre_sim = genre_sim[upper_tri]
            
            print(f"\n   {genre} ({len(genre_indices)} movies):")
            print(f"      Intra-genre mean similarity: {intra_genre_sim.mean():.4f}")
            print(f"      Intra-genre std similarity: {intra_genre_sim.std():.4f}")
    
    # 6. Visualization Suggestions
    print("\n6. GENERATING VISUALIZATIONS...")
    
    # Create output directory
    viz_dir = os.path.join(os.path.dirname(config.active_dataset.embeddings_path), "validation_plots")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Plot 1: Similarity Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Pairwise Cosine Similarities')
    plt.axvline(similarities.mean(), color='r', linestyle='--', label=f'Mean: {similarities.mean():.3f}')
    plt.legend()
    plt.tight_layout()
    sim_dist_path = os.path.join(viz_dir, "similarity_distribution.png")
    plt.savefig(sim_dist_path, dpi=150)
    print(f"   Saved: {sim_dist_path}")
    plt.close()
    
    # Plot 2: Dimension Statistics
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(dim_means)), dim_means)
    plt.xlabel('Dimension')
    plt.ylabel('Mean Activation')
    plt.title('Mean Activation per Dimension')
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(dim_stds)), dim_stds)
    plt.xlabel('Dimension')
    plt.ylabel('Std Activation')
    plt.title('Standard Deviation per Dimension')
    plt.axhline(dead_dim_threshold, color='r', linestyle='--', label=f'Dead threshold: {dead_dim_threshold}')
    plt.legend()
    
    plt.tight_layout()
    dim_stats_path = os.path.join(viz_dir, "dimension_statistics.png")
    plt.savefig(dim_stats_path, dpi=150)
    print(f"   Saved: {dim_stats_path}")
    plt.close()
    
    # Plot 3: Heatmap of Sample Similarities
    plt.figure(figsize=(10, 8))
    sample_size_heatmap = min(50, sample_size)
    sns.heatmap(
        sim_matrix[:sample_size_heatmap, :sample_size_heatmap],
        cmap='coolwarm',
        center=0.5,
        vmin=0,
        vmax=1,
        square=True
    )
    plt.title(f'Cosine Similarity Heatmap ({sample_size_heatmap} samples)')
    plt.tight_layout()
    heatmap_path = os.path.join(viz_dir, "similarity_heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    print(f"   Saved: {heatmap_path}")
    plt.close()
    
    # Plot 4: Embedding Value Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(embeddings_np.flatten(), bins=100, edgecolor='black', alpha=0.7)
    plt.xlabel('Embedding Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of All Embedding Values')
    plt.axvline(0, color='r', linestyle='--', label='Zero')
    plt.legend()
    plt.tight_layout()
    val_dist_path = os.path.join(viz_dir, "value_distribution.png")
    plt.savefig(val_dist_path, dpi=150)
    print(f"   Saved: {val_dist_path}")
    plt.close()
    
    print(f"\n{'='*60}")
    print("SUMMARY & RECOMMENDATIONS")
    print(f"{'='*60}\n")
    
    # Provide actionable insights
    if similarities.mean() > 0.7:
        print("⚠️  HIGH AVERAGE SIMILARITY detected!")
        print("   - Embeddings may lack diversity")
        print("   - Consider using a larger/different BERT model")
        print("   - Or add more distinguishing features to the text\n")
    elif similarities.mean() < 0.3:
        print("✓  GOOD DIVERSITY detected")
        print("   - Embeddings are sufficiently different from each other\n")
    
    if dead_dims > embeddings.shape[1] * 0.1:
        print(f"⚠️  MANY DEAD DIMENSIONS detected ({dead_dims}/{embeddings.shape[1]})")
        print("   - Some dimensions may not be contributing")
        print("   - This is okay for SID training but good to be aware\n")
    
    if high_sim_pairs > sample_size * 0.05:
        print("⚠️  MANY NEAR-DUPLICATE embeddings detected")
        print("   - Check if movies have very similar titles/genres")
        print("   - May affect SID uniqueness\n")
    
    print("📊 Validation complete! Check the plots in:")
    print(f"   {viz_dir}\n")
    
    return {
        'mean_similarity': similarities.mean(),
        'std_similarity': similarities.std(),
        'dead_dimensions': dead_dims,
        'near_duplicates': high_sim_pairs,
        'embedding_shape': embeddings.shape
    }

if __name__ == "__main__":
    results = validate_embeddings()
