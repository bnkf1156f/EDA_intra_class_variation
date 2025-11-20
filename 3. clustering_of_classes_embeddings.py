"""
Clustering using: Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
    - unique approach to identifying clusters based on the density of data points
    - MinPts (Minimum Points), is a parameter that specifies min number of points required to form a dense region, which is consider a cluster.
    - Epsilon - key parameter, is max distance between two points for them to be considered as part of the same neighborhood.
    - Unlike some clustering algorithms, DBSCAN does not predict the cluster membership of new, unseen data points. 
      Once the model is trained, it is applied to the existing dataset without the ability to generalize to new observations outside the training set.
"""

"""
Analyze intra-class variations using DBSCAN clustering on DINOv2 embeddings.

This script clusters embeddings WITHIN each class to discover sub-groups,
outliers, and understand class heterogeneity. Saves sample images from each cluster.

Usage:
    python '.\EDA_intra_class_variation_scripts\3. clustering_of_classes_embeddings.py' --auto_tune --cross_class --save_montage
"""

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import umap
from pathlib import Path
import numpy as np
import argparse
import seaborn as sns
from PIL import Image
import shutil
import pandas as pd

def auto_tune_eps(embeddings, min_samples, percentile=90):
    """
        Automatically find optimal eps using k-nearest neighbors.
        What this does?
        - For every point in your data, find its `min_samples` closest neighbors
        - Calculates the distance to each neighbor using cosine distance
        - Returns a matrix: `distances[i, j]` = distance from point `i` to its `j-th` nearest neighbor

        Example: If `min_samples=3` and you have 200 images:
        ```
        Point 0: [0.05, 0.12, 0.18]  ← distances to 1st, 2nd, 3rd nearest neighbors
        Point 1: [0.08, 0.15, 0.22]
        Point 2: [0.03, 0.10, 0.14]
        ...
        Point 199: [0.06, 0.13, 0.19]
    """

    # Converts each 768D embedding vector to unit length to make cosine distance calculations work properly
    embeddings_norm = normalize(embeddings, norm='l2')
    
    # Find k-nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=min_samples, metric='cosine')
    neighbors.fit(embeddings_norm)
    distances, _ = neighbors.kneighbors(embeddings_norm)
    
    # Use the distance to the k-th nearest neighbor --- distance to the **farthest** of the k-nearest neighbors
    k_distances = np.sort(distances[:, -1])
    
    # Choose eps at a percentile (avoids extreme outliers skewing the value) -- This means: "90% of points have their k-th neighbor within this distance" so as less as possible outliers
    optimal_eps = np.percentile(k_distances, percentile)
    
    return optimal_eps

def cluster_single_class(embeddings, class_name, eps, min_samples):
    """Apply DBSCAN to a single class's embeddings."""
    # DBSCAN works better on normalized embeddings in high-D space
    embeddings_norm = normalize(embeddings, norm='l2')
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    cluster_labels = dbscan.fit_predict(embeddings_norm)
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_outliers = np.sum(cluster_labels == -1)
    
    print(f"\n{class_name}:")
    print(f"  - Found {n_clusters} sub-clusters")
    print(f"  - Identified {n_outliers} outliers ({n_outliers/len(embeddings)*100:.1f}%)")
    
    return cluster_labels, n_clusters, n_outliers

def save_cluster_samples(image_files, cluster_labels, class_name, output_dir, max_samples=5):
    """Save sample images from each cluster and outliers."""
    cluster_dir = output_dir / f"{class_name}_samples"
    cluster_dir.mkdir(exist_ok=True)
    
    # Get unique clusters (excluding -1 which is outliers)
    unique_clusters = sorted(set(cluster_labels))
    
    print(f"  - Saving sample images to: {cluster_dir}")
    
    for cluster_id in unique_clusters:
        # Get indices for this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        # Determine folder name
        if cluster_id == -1:
            folder_name = "outliers"
        else:
            folder_name = f"cluster_{cluster_id}"
        
        cluster_folder = cluster_dir / folder_name
        cluster_folder.mkdir(exist_ok=True)
        
        # Sample random images from this cluster
        n_samples = min(max_samples, len(cluster_indices))
        sampled_indices = np.random.choice(cluster_indices, n_samples, replace=False)
        
        # Copy images
        for idx in sampled_indices:
            src_path = image_files[idx]
            dst_path = cluster_folder / src_path.name
            shutil.copy2(src_path, dst_path)
        
        print(f"    - {folder_name}: saved {n_samples}/{len(cluster_indices)} samples")

def create_cluster_montage(image_files, cluster_labels, class_name, output_path, max_per_cluster=5):
    """Create a montage showing sample images from each cluster."""
    unique_clusters = sorted(set(cluster_labels))
    n_clusters = len(unique_clusters)
    
    # Calculate grid size
    n_cols = max_per_cluster
    n_rows = n_clusters
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    
    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, cluster_id in enumerate(unique_clusters):
        # Get indices for this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        # Sample images
        n_samples = min(max_per_cluster, len(cluster_indices))
        sampled_indices = np.random.choice(cluster_indices, n_samples, replace=False)
        
        # Determine title
        if cluster_id == -1:
            row_title = f"Outliers (n={len(cluster_indices)})"
            color = 'red'
        else:
            row_title = f"Cluster {cluster_id} (n={len(cluster_indices)})"
            color = 'blue'
        
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            
            if col_idx < len(sampled_indices):
                # Load and display image
                img_path = image_files[sampled_indices[col_idx]]
                img = Image.open(img_path)
                ax.imshow(img)
                ax.set_title(img_path.name, fontsize=8)
            
            ax.axis('off')
        
        # Add row label
        axes[row_idx, 0].text(-0.1, 0.5, row_title, 
                             transform=axes[row_idx, 0].transAxes,
                             fontsize=12, fontweight='bold', color=color,
                             rotation=90, va='center', ha='right')
    
    plt.suptitle(f'Cluster Samples: {class_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  - Saved montage: {output_path}")
    plt.close()

def visualize_intra_class(X_2d, cluster_labels, class_name, output_path):
    """Visualize clusters within a single class."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot 1: Full scatter plot
    outlier_mask = cluster_labels == -1
    
    # Plot outliers
    if np.any(outlier_mask):
        ax1.scatter(X_2d[outlier_mask, 0], X_2d[outlier_mask, 1],
                   c='red', marker='x', s=100, alpha=0.6,
                   label='Outliers', linewidths=2)
    
    # Plot clusters
    unique_clusters = sorted(set(cluster_labels[~outlier_mask]))
    colors = sns.color_palette('husl', n_colors=len(unique_clusters))
    
    for i, cluster in enumerate(unique_clusters):
        mask = cluster_labels == cluster
        ax1.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=[colors[i]], s=100, alpha=0.7,
                   label=f'Sub-cluster {cluster} (n={np.sum(mask)})')
    
    ax1.set_title(f'Intra-Class Variation: {class_name}')
    ax1.set_xlabel('UMAP Dimension 1')
    ax1.set_ylabel('UMAP Dimension 2')
    ax1.legend()
    
    # Plot 2: Cluster size distribution
    cluster_counts = [np.sum(cluster_labels == c) for c in unique_clusters]
    outlier_count = np.sum(outlier_mask)
    
    labels = [f'C{c}' for c in unique_clusters] + ['Outliers']
    counts = cluster_counts + [outlier_count]
    colors_bar = list(colors) + ['red']
    
    ax2.bar(labels, counts, color=colors_bar, alpha=0.7)
    ax2.set_title(f'Cluster Size Distribution')
    ax2.set_xlabel('Cluster ID')
    ax2.set_ylabel('Number of Samples')
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (label, count) in enumerate(zip(labels, counts)):
        ax2.text(i, count + 2, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  - Saved visualization: {output_path}")
    plt.close()

def visualize_all_classes_overview(all_results, output_path):
    """Create overview visualization showing all classes together."""
    n_classes = len(all_results)
    n_cols = 3
    n_rows = (n_classes + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten()
    
    for idx, (class_name, X_2d, cluster_labels) in enumerate(all_results):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        outlier_mask = cluster_labels == -1
        
        # Plot outliers
        if np.any(outlier_mask):
            ax.scatter(X_2d[outlier_mask, 0], X_2d[outlier_mask, 1],
                      c='red', marker='x', s=50, alpha=0.6, linewidths=1.5)
        
        # Plot clusters
        unique_clusters = sorted(set(cluster_labels[~outlier_mask]))
        colors = sns.color_palette('husl', n_colors=len(unique_clusters))
        
        for i, cluster in enumerate(unique_clusters):
            mask = cluster_labels == cluster
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                      c=[colors[i]], s=50, alpha=0.7)
        
        ax.set_title(f'{class_name}\n({len(unique_clusters)} clusters, {np.sum(outlier_mask)} outliers)')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
    
    # Hide unused subplots
    for idx in range(len(all_results), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Intra-Class Variation Overview (All Classes)', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved overview: {output_path}")
    plt.close()

def analyze_cross_class_separability(X, y, class_names, eps, min_samples, output_path):
    """Optional: Check if DBSCAN discovers class boundaries naturally."""
    print("\n" + "="*60)
    print("CROSS-CLASS ANALYSIS (Optional)")
    print("="*60)
    
    # Reduce dimensions
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_2d = reducer.fit_transform(X)
    
    # Apply DBSCAN to all data
    embeddings_norm = normalize(X, norm='l2')
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    cluster_labels = dbscan.fit_predict(embeddings_norm)
    
    print(f"Discovered {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)} clusters across all classes")
    
    # Visualize
    plt.figure(figsize=(15, 10))
    colors = sns.color_palette('husl', n_colors=len(class_names))
    
    for i, class_name in enumerate(class_names):
        mask = y == i
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=[colors[i]], s=50, alpha=0.6, label=class_name)
    
    plt.title('Cross-Class Embedding Space\n(Colored by ground-truth class)')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved cross-class visualization: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="cropped_images")
    parser.add_argument("--eps", type=float, default=0.15,
                       help="DBSCAN eps for cosine distance (try 0.1-0.3)")
    parser.add_argument("--min_samples", type=int, default=3,
                       help="Min samples to form cluster")
    parser.add_argument("--auto_tune", action="store_true",
                       help="Automatically find optimal eps for each class")
    parser.add_argument("--output_dir", type=str, default="clustering_results")
    parser.add_argument("--max_samples", type=int, default=5,
                       help="Max sample images per cluster to save")
    parser.add_argument("--cross_class", action="store_true",
                       help="Also perform cross-class analysis")
    parser.add_argument("--save_montage", action="store_true",
                       help="Create image montage for each class")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load all embeddings
    root = Path(args.root)
    all_embeddings = []
    all_labels = []
    class_names = []
    all_results = []
    stats_data = []
    
    print("="*60)
    print("INTRA-CLASS CLUSTERING ANALYSIS")
    print("="*60)
    
    # Process each class separately
    for cls_folder in sorted(root.iterdir()):
        if not cls_folder.is_dir():
            continue
            
        emb_path = cls_folder / "embeddings_dinov2.npy"
        if not emb_path.exists():
            print(f"\n⚠️  Skipping {cls_folder.name} - no embeddings found.")
            continue
        
        # Load embeddings
        embeddings = np.load(str(emb_path))
        class_name = cls_folder.name
        
        # Load image mapping file to ensure correct alignment
        mapping_path = cls_folder / "embeddings_dinov2_image_list.txt"
        if mapping_path.exists():
            # Use the saved mapping (ensures correct alignment even if some images failed)
            with open(mapping_path, 'r') as f:
                image_names = [line.strip() for line in f.readlines()]
            image_files = [cls_folder / name for name in image_names]
            print(f"\n✓ Using saved image mapping from: {mapping_path.name}")
        else:
            # Fallback to sorted list (legacy behavior, may be misaligned if images failed)
            image_files = sorted([f for f in cls_folder.iterdir() if f.suffix == '.png'])
            print(f"\n⚠️  No mapping file found - using sorted image list (may be misaligned)")
        
        if len(image_files) != len(embeddings):
            print(f"❌ ERROR: {class_name} has {len(image_files)} images but {len(embeddings)} embeddings")
            print(f"   This class will be skipped. Re-run script 2 to fix alignment.")
            continue
        
        print(f"\nProcessing: {class_name} ({embeddings.shape[0]} samples)")
        
        # Auto-tune eps if requested
        if args.auto_tune:
            optimal_eps = auto_tune_eps(embeddings, args.min_samples)
            print(f"  Auto-tuned eps: {optimal_eps:.4f}")
            eps_to_use = optimal_eps
        else:
            eps_to_use = args.eps
        
        # Store for cross-class analysis
        all_embeddings.append(embeddings)
        all_labels.extend([len(class_names)] * len(embeddings))
        class_names.append(class_name)
        
        # Reduce dimensions for this class
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(embeddings)
        
        # Cluster within this class
        cluster_labels, n_clusters, n_outliers = cluster_single_class(
            embeddings, class_name, eps_to_use, args.min_samples
        )
        
        # Save statistics
        stats_data.append({
            'class': class_name,
            'n_samples': len(embeddings),
            'n_clusters': n_clusters,
            'n_outliers': n_outliers,
            'outlier_rate': f"{n_outliers/len(embeddings)*100:.1f}%",
            'eps_used': eps_to_use
        })
        
        # Store results
        all_results.append((class_name, X_2d, cluster_labels))
        
        # Visualize this class
        output_path = output_dir / f"{class_name}_clusters.png"
        visualize_intra_class(X_2d, cluster_labels, class_name, output_path)
        
        # Save sample images from each cluster
        save_cluster_samples(image_files, cluster_labels, class_name, output_dir, args.max_samples)
        
        # Create montage if requested
        if args.save_montage:
            montage_path = output_dir / f"{class_name}_montage.png"
            create_cluster_montage(image_files, cluster_labels, class_name, montage_path, args.max_samples)
    
    # Create overview
    if all_results:
        overview_path = output_dir / "all_classes_overview.png"
        visualize_all_classes_overview(all_results, overview_path)
    
    # Save statistics
    if stats_data:
        df = pd.DataFrame(stats_data)
        print("\nSummary:")
        print(df.to_string(index=False))
        
        ## UNCOMMENT IF WANNA SAVE THE CLUSTER STATS
        stats_path = output_dir / "cluster_statistics.csv"
        df.to_csv(stats_path, index=False)
        print(f"\nSaved statistics: {stats_path}")
    

    # CROSS-CLASS VISUALIZATIONS 
    if args.cross_class and all_embeddings:
        X = np.vstack(all_embeddings)
        y = np.array(all_labels)
        cross_path = output_dir / "cross_class_separability.png"
        analyze_cross_class_separability(X, y, class_names, args.eps, args.min_samples, cross_path)
    
    print("\n" + "="*60)
    print("Analysis complete! Check the output directory for:")
    print("  - Scatter plots for each class")
    print("  - Sample images organized by cluster")
    if args.save_montage:
        print("  - Image montages for visual comparison")
    print("  - Statistics CSV file")
    print("="*60)

if __name__ == "__main__":
    main()