"""
Interactive Cluster Viewer using Plotly

Generates interactive HTML visualizations from existing embeddings.
Run AFTER Script 2 (embeddings generated). Independent of Script 3.

Usage:
    python '.\4. interactive_cluster_viewer.py' --root "cropped_imgs_by_class_bmw_7" --output_dir "interactive_clusters_results_bmw_7" --min_samples 3

Features:
    - Hover over points to see filename
    - Click on points to open the image in a new browser tab
    - Zoom/pan to explore dense regions

Note: Click-to-open uses file:// URLs. If your browser blocks this,
      try Firefox or Edge, or copy the path from hover and open manually.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

try:
    import plotly.express as px
except ImportError:
    print("Plotly not installed. Run: pip install plotly")
    exit(1)


def auto_tune_eps(embeddings, min_samples, percentile=90):
    """Automatically find optimal eps using k-nearest neighbors."""
    embeddings_norm = normalize(embeddings, norm='l2')
    neighbors = NearestNeighbors(n_neighbors=min_samples, metric='cosine')
    neighbors.fit(embeddings_norm)
    distances, _ = neighbors.kneighbors(embeddings_norm)
    k_distances = np.sort(distances[:, -1])
    return np.percentile(k_distances, percentile)


def create_interactive_plot(embeddings, image_files, class_name, output_path, eps, min_samples):
    """Create interactive Plotly scatter plot."""

    # UMAP reduction
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_2d = reducer.fit_transform(embeddings)

    # DBSCAN clustering
    embeddings_norm = normalize(embeddings, norm='l2')
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    cluster_labels = dbscan.fit_predict(embeddings_norm)

    # Build dataframe
    df = pd.DataFrame({
        'UMAP1': X_2d[:, 0],
        'UMAP2': X_2d[:, 1],
        'cluster': [f'Outlier' if c == -1 else f'Cluster {c}' for c in cluster_labels],
        'filename': [f.name for f in image_files],
        'filepath': [str(f.absolute()) for f in image_files]
    })

    # Color outliers red
    color_map = {'Outlier': 'red'}

    fig = px.scatter(
        df,
        x='UMAP1',
        y='UMAP2',
        color='cluster',
        hover_data={'filename': True, 'filepath': False, 'UMAP1': ':.2f', 'UMAP2': ':.2f'},
        title=f'Interactive Cluster View: {class_name} ({len(embeddings)} samples)',
        color_discrete_map=color_map
    )

    # Store filepath in customdata for click handler
    fig.update_traces(
        customdata=df[['filepath', 'filename', 'cluster']].values,
        hovertemplate='<b>%{customdata[1]}</b><br>%{customdata[2]}<extra></extra>'
    )

    fig.update_layout(
        hoverlabel=dict(bgcolor="white", font_size=12),
        legend_title_text='Cluster'
    )

    # Write base HTML
    html_content = fig.to_html(include_plotlyjs='cdn', full_html=True)

    # JavaScript for click-to-open functionality
    click_script = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        var plot = document.getElementsByClassName('plotly-graph-div')[0];
        plot.on('plotly_click', function(data) {
            var filepath = data.points[0].customdata[0];
            // Convert Windows path to file:// URL
            var fileUrl = 'file:///' + filepath.replace(/\\\\/g, '/');
            window.open(fileUrl, '_blank');
        });
    });
    </script>
    """

    # Insert script before closing body tag
    html_content = html_content.replace('</body>', click_script + '</body>')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"  Saved: {output_path} (click points to open images)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="cropped_images", help="Root folder with per-class subfolders")
    parser.add_argument("--output_dir", type=str, default="clustering_results", help="Output directory for HTML files")
    parser.add_argument("--min_samples", type=int, default=3, help="DBSCAN min_samples")
    parser.add_argument("--eps", type=float, default=None, help="DBSCAN eps (auto-tuned if not provided)")
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("INTERACTIVE CLUSTER VIEWER")
    print("=" * 60)

    for cls_folder in sorted(root.iterdir()):
        if not cls_folder.is_dir():
            continue

        emb_path = cls_folder / "embeddings_dinov2.npy"
        mapping_path = cls_folder / "embeddings_dinov2_image_list.txt"

        if not emb_path.exists():
            print(f"\nSkipping {cls_folder.name} - no embeddings")
            continue

        embeddings = np.load(str(emb_path))
        class_name = cls_folder.name

        # Load image mapping
        if mapping_path.exists():
            with open(mapping_path, 'r') as f:
                image_names = [line.strip() for line in f.readlines()]
            image_files = [cls_folder / name for name in image_names]
        else:
            image_files = sorted([f for f in cls_folder.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')])

        if len(image_files) != len(embeddings):
            print(f"\nSkipping {class_name} - misaligned ({len(image_files)} images, {len(embeddings)} embeddings)")
            continue

        print(f"\nProcessing: {class_name} ({len(embeddings)} samples)")

        # Auto-tune eps if not provided
        eps = args.eps if args.eps else auto_tune_eps(embeddings, args.min_samples)
        print(f"  Using eps: {eps:.4f}")

        output_path = output_dir / f"{class_name}_interactive.html"
        create_interactive_plot(embeddings, image_files, class_name, output_path, eps, args.min_samples)

    print("\n" + "=" * 60)
    print("Done! Open HTML files in browser to explore.")
    print("=" * 60)


if __name__ == "__main__":
    main()
