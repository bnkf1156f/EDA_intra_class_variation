#!/usr/bin/env python3
"""
Pre-annotation frame quality assessment pipeline.
Analyzes raw frames BEFORE annotation using DINOv2 embeddings, activity clustering,
and quality metrics to identify issues and assess dataset diversity.

Embedding Strategy:
- Frames WITH persons: Scene embedding (768d) weighted 0.7 + lightweight pose features (24d) weighted 0.3 = 792d total
- Frames WITHOUT persons: Scene embeddings (768d) + zero-padded pose features (24d) = 792d total
- Weights are applied to CONCATENATED dimensions (not blended scalars)

Tuning Guide:
    EMBEDDING WEIGHTS (lines 339-340 in compute_multiview_embeddings):
    - More scene-driven clustering (different scenes/objects/environments) => Increase scene_emb weight (0.7 → 0.8+)
    - More pose-driven clustering (different worker activities/postures) => Increase pose_features weight (0.3 → 0.4+)
    - Current: 70% scene weight + 30% pose weight (balanced for both environment and activity diversity)

    CLUSTERING GRANULARITY (in main):
    - Want more fine-grained activity clusters => Increase n_components (8 → 16) OR decrease min_dist in UMAP
    - Want broader activity grouping => Decrease n_components (8 → 4)
    - Min frames per activity => Adjust min_cluster_size (50 → higher for stricter clustering)
    - Cluster strictness (edge frame inclusion) => Adjust min_samples (5 → lower for looser, higher for stricter)

    QUALITY THRESHOLDS (in main):
    - Blur detection sensitivity => anisotropy_threshold (3.6 → lower for stricter, higher for lenient) => >threshold are blur frames

    PDF REPORT OPTIONS (lines 1755-1761 in main):
    - Samples shown per activity => activity_num_samples (20 → adjust based on dataset size)
    - Blurry samples in report => cache_blurry_num_samples (24 → adjust for PDF length)
    - Grid layout for montages => grid_cols_activities (3), grid_cols_blurry (3)
    - PDF file size => pdf_image_quality (70 → lower for smaller files, higher for better quality)
"""

from pathlib import Path
from transformers import AutoImageProcessor, AutoModel
import torch
import numpy as np
import gc
from PIL import Image as PILImage, UnidentifiedImageError, ImageDraw, ImageFont
import sys
from tqdm import tqdm
import cv2
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
import umap
import random
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import shutil
import time
from ultralytics import YOLO
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
matplotlib.use('Agg')  # Non-interactive backend


def scan_frames(root: Path, exts=(".png", ".jpg", ".jpeg")):
    """
    Scans root directory for image files.
    """
    image_paths = []

    for p in sorted(root.rglob('*')):
        if p.is_file() and p.suffix.lower() in exts:
            image_paths.append(p)

    return image_paths


def load_image_safe(path: Path):
    """Load image safely, return None if corrupted."""
    try:
        img = PILImage.open(path).convert("RGB")
        return img
    except (UnidentifiedImageError, OSError) as e:
        print(f"Warning: could not open image {path}: {e}", file=sys.stderr)
        return None


def detect_persons_and_extract_poses(image_paths, pose_model, device, conf_threshold, yolo_batch_size):
    """
    Detect persons in frames and extract pose keypoints using BATCHED inference.
    Returns:
        frame_data: List of dicts with keys:
            - 'path': Path to frame
            - 'persons': List of person detections, each with:
                - 'pose_features': Flattened normalized pose features
            - 'has_persons': Boolean indicating if frame contains persons
            - 'image_shape': (H, W) tuple
    """
    frame_data = []

    # Process in batches
    for batch_start in tqdm(range(0, len(image_paths), yolo_batch_size), desc="Detecting persons (batched)", unit="batch"):
        batch_paths = image_paths[batch_start:batch_start + yolo_batch_size]

        # Load batch of images
        batch_images = []
        batch_paths_valid = []
        batch_image_shapes = []

        for img_path in batch_paths:
            img_cv = cv2.imread(str(img_path))
            if img_cv is not None:
                batch_images.append(img_cv)
                batch_paths_valid.append(img_path)
                batch_image_shapes.append(img_cv.shape[:2])
            else:
                # Store None for corrupted images
                frame_data.append({
                    'path': img_path,
                    'persons': [],
                    'has_persons': False,
                    'image_shape': None
                })

        if len(batch_images) == 0:
            continue

        # Run YOLO pose detection on BATCH
        results_batch = pose_model(batch_images, conf=conf_threshold, verbose=False, device=device)

        # Process each result
        for img_shape, img_path, results in zip(batch_image_shapes, batch_paths_valid, results_batch):
            persons = []
            if results.keypoints is not None and len(results.keypoints) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                keypoints = results.keypoints.data.cpu().numpy()  # (N, 17, 3)

                for bbox, kpts in zip(boxes, keypoints):
                    h, w = img_shape
                    kpts_norm = kpts.copy()
                    kpts_norm[:, 0] /= w
                    kpts_norm[:, 1] /= h

                    pose_features = extract_pose_features(kpts_norm)

                    # Only store pose_features (bbox, keypoints, keypoints_norm are unused downstream)
                    persons.append({
                        'pose_features': pose_features
                    })

            frame_data.append({
                'path': img_path,
                'persons': persons,
                'has_persons': len(persons) > 0,
                'image_shape': img_shape  # (H, W)
            })

        del batch_images, results_batch
        if device == "cuda":
            torch.cuda.empty_cache()

    return frame_data


def extract_pose_features(keypoints_norm):
    """Extract pose-invariant features: limb angles, distances, spatial layout."""
    features = []

    limb_pairs = [
        (5, 7), (7, 9),   # left arm (shoulder-elbow, elbow-wrist)
        (6, 8), (8, 10),  # right arm
        (11, 13), (13, 15),  # left leg (hip-knee, knee-ankle)
        (12, 14), (14, 16),  # right leg
        (5, 11), (6, 12),  # torso (shoulder-hip)
    ]

    for kp1_idx, kp2_idx in limb_pairs:
        kp1, kp2 = keypoints_norm[kp1_idx], keypoints_norm[kp2_idx]
        if kp1[2] > 0.3 and kp2[2] > 0.3:
            dx = kp2[0] - kp1[0]
            dy = kp2[1] - kp1[1]
            angle = np.arctan2(dy, dx)
            distance = np.sqrt(dx**2 + dy**2)
            features.extend([angle, distance])
        else:
            features.extend([0.0, 0.0])

    visible_kpts = keypoints_norm[keypoints_norm[:, 2] > 0.3]
    if len(visible_kpts) > 0:
        center_x = np.mean(visible_kpts[:, 0])
        center_y = np.mean(visible_kpts[:, 1])
        features.extend([center_x, center_y])
    else:
        features.extend([0.5, 0.5])

    if len(visible_kpts) > 0:
        x_range = np.max(visible_kpts[:, 0]) - np.min(visible_kpts[:, 0])
        y_range = np.max(visible_kpts[:, 1]) - np.min(visible_kpts[:, 1])
        features.extend([x_range, y_range])
    else:
        features.extend([0.0, 0.0])

    return np.array(features, dtype=np.float32)


def compute_multiview_embeddings(frame_data, processor, model, device, batch_size, cache_dir=None):
    """
    Compute embeddings with optional pose integration:
    - If person detected: scene + lightweight pose features (weighted)
    - If no person: 100% scene + zero-padded pose/skeleton features (for consistent dimensions)
    """
    # Determine embedding mode FIRST (before checking cache)
    frames_with_persons = sum(1 for f in frame_data if f['has_persons'])
    frames_without_persons = len(frame_data) - frames_with_persons

    # If too few persons detected, use scene-only for ALL frames (avoids UMAP clustering issues)
    min_persons_for_pose = 3
    use_scene_only = frames_with_persons < min_persons_for_pose
    expected_dim = 768 if use_scene_only else 768 + 24  # Scene-only: 768, Scene+pose: 792

    # Check cache with dimension validation
    if cache_dir is not None:
        cache_file = cache_dir / "temp_multiview_emb.npy"
        cache_indices_file = cache_dir / "temp_multiview_emb_indices.npy"

        if cache_file.exists() and cache_indices_file.exists():
            embeddings = np.load(cache_file)
            valid_frame_indices = np.load(cache_indices_file)

            # Validate cache dimension matches current mode
            if embeddings.shape[1] == expected_dim:
                print(f"\n✅ Found cached embeddings: {cache_file}")
                print("   Loading from cache (delete cache files to recompute)...")
                print(f"   Loaded {len(embeddings)} embeddings from cache")
                return embeddings, valid_frame_indices.tolist()
            else:
                print(f"\n⚠️  Cached embeddings dimension mismatch:")
                print(f"   Cached: {embeddings.shape[1]}-dim, Expected: {expected_dim}-dim")
                print(f"   Recomputing embeddings in {'scene-only' if use_scene_only else 'scene+pose'} mode...")

    # Process ALL frames, not just those with persons
    valid_frame_indices = list(range(len(frame_data)))

    if use_scene_only:
        print(f"\n⚠️  Only {frames_with_persons} frames with persons detected (minimum {min_persons_for_pose} needed)")
        print(f"   Switching to SCENE-ONLY embeddings for all {len(frame_data)} frames")
        print(f"   (This ensures sufficient samples for clustering)")
        pose_features_dim = 0  # No pose features in scene-only mode
    else:
        print(f"\nProcessing {len(frame_data)} frames:")
        print(f"  - {frames_with_persons} frames with persons (scene + lightweight pose features)")
        print(f"  - {frames_without_persons} frames without persons (scene only)")
        # Determine pose_features dimension ONCE upfront to avoid mismatch across batches
        # 10 limb pairs × 2 features + 2 center coords + 2 range values = 24
        pose_features_dim = 24

    # Pre-allocate zero arrays for frames without persons (reuse across all frames)
    zero_pose_features = np.zeros(pose_features_dim, dtype=np.float32) if pose_features_dim > 0 else np.array([], dtype=np.float32)

    multiview_embeddings = []
    model.eval()

    with torch.no_grad():
        progress_bar = tqdm(total=len(frame_data), desc="Computing embeddings", unit="frame")

        for batch_start in range(0, len(frame_data), batch_size):
            batch_frames = frame_data[batch_start:batch_start + batch_size]

            # LOAD IMAGES FRESH (avoid holding ALL images in RAM)
            batch_images_pil = []
            batch_has_persons = []
            batch_persons_data = []

            for frame_info in batch_frames:
                img_path = frame_info['path']
                img_cv = cv2.imread(str(img_path))

                if img_cv is None:
                    batch_images_pil.append(None)
                    batch_has_persons.append(False)
                    batch_persons_data.append([])
                    continue

                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                pil_img = PILImage.fromarray(img_rgb)
                batch_images_pil.append(pil_img)
                batch_has_persons.append(frame_info['has_persons'])
                batch_persons_data.append(frame_info['persons'])

            # Prepare scene images (all frames)
            batch_scene_imgs = []
            valid_indices_in_batch = []
            for idx, pil_img in enumerate(batch_images_pil):
                if pil_img is not None:
                    scene_img = pil_img.resize((224, 224), PILImage.Resampling.LANCZOS)
                    batch_scene_imgs.append(scene_img)
                    valid_indices_in_batch.append(idx)

            # Compute scene embeddings (all frames)
            scene_embs = None
            if len(batch_scene_imgs) > 0:
                inputs_scene = processor(images=batch_scene_imgs, return_tensors="pt")
                inputs_scene = {k: v.to(device) for k, v in inputs_scene.items()}

                if device == "cuda":
                    with torch.amp.autocast('cuda'):
                        outputs_scene = model(**inputs_scene)
                        scene_embs = outputs_scene.last_hidden_state.mean(dim=1).cpu().numpy()
                else:
                    outputs_scene = model(**inputs_scene)
                    scene_embs = outputs_scene.last_hidden_state.mean(dim=1).cpu().numpy()

                del outputs_scene, inputs_scene


            # Use ONLY lightweight geometric pose features (24-dim) instead of full DINOv2 skeleton embedding (768-dim)

            # Aggregate pose features per frame (for frames with persons) - SKIP in scene-only mode
            frame_pose_features_aggregates = {}

            if not use_scene_only:
                for valid_idx, batch_idx in enumerate(valid_indices_in_batch):
                    if batch_has_persons[batch_idx]:
                        persons = batch_persons_data[batch_idx]
                        # Average pose features across all persons in frame
                        pose_features_list = [person['pose_features'] for person in persons]
                        frame_pose_features_aggregates[valid_idx] = np.mean(pose_features_list, axis=0)

            # Combine scene embeddings + lightweight pose features
            for valid_idx, batch_idx in enumerate(valid_indices_in_batch):
                scene_emb = scene_embs[valid_idx]

                if use_scene_only:
                    # Scene-only mode: just use DINOv2 scene embeddings (768-dim)
                    multiview_embeddings.append(scene_emb)
                elif batch_has_persons[batch_idx] and valid_idx in frame_pose_features_aggregates:
                    # Frame with person: scene (85%) + lightweight pose features (15%)
                    pose_features_agg = frame_pose_features_aggregates[valid_idx]

                    # Pre-allocate output array to avoid temporary allocations
                    combined_emb = np.empty(768 + pose_features_dim, dtype=np.float32)
                    combined_emb[:768] = scene_emb * 0.7      # Scene dominant
                    combined_emb[768:] = pose_features_agg * 0.3  # Lightweight pose supplement
                    multiview_embeddings.append(combined_emb)
                else:
                    # Frame without person: scene only + zero-padded pose features
                    combined_emb = np.empty(768 + pose_features_dim, dtype=np.float32)
                    combined_emb[:768] = scene_emb
                    combined_emb[768:] = zero_pose_features  # Reuse pre-allocated zeros
                    multiview_embeddings.append(combined_emb)

            # Delete batch tensors and numpy arrays (keep image_cv for quality metrics step)
            del batch_images_pil, batch_scene_imgs, scene_embs
            if device == "cuda":
                torch.cuda.empty_cache()

            progress_bar.update(len(batch_frames))

        progress_bar.close()

        if device == "cuda":
            torch.cuda.empty_cache()

    multiview_embeddings = np.array(multiview_embeddings, dtype=np.float32)

    if cache_dir is not None:
        cache_file = cache_dir / "temp_multiview_emb.npy"
        cache_indices_file = cache_dir / "temp_multiview_emb_indices.npy"

        print(f"\n💾 Saving embeddings to cache: {cache_file}")
        np.save(cache_file, multiview_embeddings)
        np.save(cache_indices_file, np.array(valid_frame_indices))
        print(f"   ✅ Cache saved ({len(multiview_embeddings)} embeddings)")

    return multiview_embeddings, valid_frame_indices


def compute_quality_metrics_batch(frame_data, anisotropy_threshold, cache_blurry_samples):
    """Compute quality metrics by loading images fresh (minimal RAM usage).

    Blurry frames = high anisotropy (motion blur, camera shake).
    """
    metrics = {
        'brightness': [],
        'contrast': [],
        'anisotropy': [],
        'blurry_image_cache': [],
        'corrupted_images': []
    }

    for frame_info in tqdm(frame_data, desc="Computing quality metrics", unit="img"):
        img_path = frame_info['path']

        # LOAD IMAGE FRESH
        img = cv2.imread(str(img_path))

        if img is None:
            # Skip corrupted images
            print(f"\n⚠️  Warning: Corrupted image: {img_path}")
            metrics['corrupted_images'].append(str(img_path))
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        brightness = np.mean(gray)
        contrast = np.std(gray)

        # Directional blur detection (anisotropy - detects motion blur, camera shake)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        energy_x = np.mean(sobelx ** 2)
        energy_y = np.mean(sobely ** 2)
        anisotropy = max(energy_x, energy_y) / (min(energy_x, energy_y) + 1e-6)

        metrics['brightness'].append(brightness)
        metrics['contrast'].append(contrast)
        metrics['anisotropy'].append(anisotropy)

        # Blurry = high anisotropy (motion blur, camera shake)
        is_blurry = (anisotropy > anisotropy_threshold)
        if is_blurry and len(metrics['blurry_image_cache']) < cache_blurry_samples:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(img_rgb)
            metrics['blurry_image_cache'].append({
                'image': pil_img,
                'anisotropy': anisotropy,
                'brightness': brightness,
                'path': img_path
            })

        del img

    metrics['brightness'] = np.array(metrics['brightness'])
    metrics['contrast'] = np.array(metrics['contrast'])
    metrics['anisotropy'] = np.array(metrics['anisotropy'])

    return metrics


def categorize_lighting(brightness_values):
    """Categorize brightness: 0=Dark (<80), 1=Medium (80-175), 2=Bright (>175)."""
    lighting_labels = np.zeros(len(brightness_values), dtype=int)
    lighting_labels[brightness_values < 80] = 0
    lighting_labels[(brightness_values >= 80) & (brightness_values < 175)] = 1
    lighting_labels[brightness_values >= 175] = 2
    return lighting_labels


def cluster_activities_with_umap_hdbscan(embeddings, n_components, min_cluster_size, min_samples):
    """
    Cluster frames by activity using UMAP + HDBSCAN (optimized parameters).
    Pipeline:
    1. Check minimum samples
    2. Standardize features
    3. UMAP for dimensionality reduction (preserves local structure better than PCA)
    4. HDBSCAN for density-based clustering (finds activity patterns + outliers)
    """
    # 1. Check minimum sample requirement for UMAP
    min_samples_required = 3
    if embeddings.shape[0] < min_samples_required:
        print(f"  ⚠️  WARNING: Only {embeddings.shape[0]} samples")
        print(f"      UMAP requires at least {min_samples_required} samples for clustering")
        print(f"      Skipping activity clustering - all frames assigned to single group")
        # Return all frames as single cluster (label 0), no dimensionality reduction
        return np.zeros(embeddings.shape[0], dtype=int), embeddings, {"insufficient_samples": True}

    # 2. Standardize features (important for UMAP)
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # 3. UMAP dimensionality reduction (better manifold preservation than PCA)
    # For spectral initialization, need n_components < n_samples - 1 AND n_components < n_neighbors
    max_components = min(embeddings.shape[0] - 2, embeddings.shape[1])  # -2 for safety margin
    n_components = min(n_components, max_components)
    n_neighbors_umap = min(15, max(2, embeddings.shape[0] - 1))  # Ensure 2 <= n_neighbors < n_samples

    # Ensure n_components <= n_neighbors (UMAP requirement)
    n_components = min(n_components, n_neighbors_umap)

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors_umap,  # Adaptive to dataset size
        min_dist=0.1,        # Minimum distance between points in low-dim space
        metric='euclidean',
        random_state=42
    )

    umap_embeddings = reducer.fit_transform(embeddings_scaled)

    print(f"  UMAP: Reduced {embeddings.shape[1]} dims → {n_components} dims")
    print(f"        (n_neighbors={n_neighbors_umap}, min_dist=0.1)")

    # 4. HDBSCAN clustering (optimized parameters for activity discovery)
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,  # Larger = more stable clusters
        min_samples=min_samples,            # Higher = stricter clustering
        metric='euclidean',
        cluster_selection_method='leaf'      # prefers larger, flatter clusters
    )

    activity_labels = clusterer.fit_predict(umap_embeddings)

    n_clusters = len(set(activity_labels)) - (1 if -1 in activity_labels else 0)
    n_noise = np.sum(activity_labels == -1)

    print(f"  HDBSCAN: Found {n_clusters} activity clusters, {n_noise} noise/outlier frames")

    # 5. Compute cluster quality metrics (cheap, informative)
    cluster_quality = compute_cluster_quality_metrics(umap_embeddings, activity_labels)

    return activity_labels, umap_embeddings, cluster_quality


def compute_cluster_quality_metrics(embeddings, labels):
    """
    Compute cluster quality metrics (fast, no heavy computation).
    Metrics:
    - Silhouette Score: [-1, 1] - higher is better (measures cluster separation)
    - Calinski-Harabasz: [0, inf] - higher is better (ratio of between/within variance)
    - Davies-Bouldin: [0, inf] - LOWER is better (average similarity ratio)
    """
    # Filter out noise points for metrics calculation
    mask = labels != -1
    if np.sum(mask) < 2:
        # Not enough clustered points
        return {
            'silhouette': None,
            'calinski_harabasz': None,
            'davies_bouldin': None,
            'n_clusters': 0,
            'n_noise': len(labels),
            'n_clustered': 0
        }

    embeddings_clustered = embeddings[mask]
    labels_clustered = labels[mask]

    n_unique = len(set(labels_clustered))
    if n_unique < 2:
        # Only one cluster
        return {
            'silhouette': None,
            'calinski_harabasz': None,
            'davies_bouldin': None,
            'n_clusters': n_unique,
            'n_noise': np.sum(~mask),
            'n_clustered': np.sum(mask)
        }

    # Compute metrics
    try:
        # Silhouette: only sample if dataset is large (>10k points)
        n_clustered = len(embeddings_clustered)
        if n_clustered > 10000:
            silhouette = silhouette_score(embeddings_clustered, labels_clustered, sample_size=10000)
        else:
            # No sampling needed for small datasets
            silhouette = silhouette_score(embeddings_clustered, labels_clustered)

        calinski = calinski_harabasz_score(embeddings_clustered, labels_clustered)
        davies = davies_bouldin_score(embeddings_clustered, labels_clustered)

        print(f"  Quality Metrics:")
        print(f"    Silhouette Score: {silhouette:.3f} (higher is better, range [-1, 1])")
        print(f"    Calinski-Harabasz: {calinski:.1f} (higher is better)")
        print(f"    Davies-Bouldin: {davies:.3f} (LOWER is better)")

        return {
            'silhouette': float(silhouette),
            'calinski_harabasz': float(calinski),
            'davies_bouldin': float(davies),
            'n_clusters': n_unique,
            'n_noise': int(np.sum(~mask)),
            'n_clustered': int(np.sum(mask))
        }
    except Exception as e:
        print(f"  ⚠️  Could not compute quality metrics: {e}")
        return {
            'silhouette': None,
            'calinski_harabasz': None,
            'davies_bouldin': None,
            'n_clusters': n_unique,
            'n_noise': int(np.sum(~mask)),
            'n_clustered': int(np.sum(mask))
        }


def create_quality_chart(metrics, output_path, anisotropy_threshold):
    """Generate combined quality distribution charts."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Brightness histogram
    ax = axes[0, 0]
    ax.hist(metrics['brightness'], bins=30, color='#ffc107', edgecolor='black', alpha=0.7)
    ax.axvline(80, color='red', linestyle='--', linewidth=2, label='Low-light threshold')
    ax.axvline(175, color='green', linestyle='--', linewidth=2, label='Bright threshold')
    ax.set_xlabel('Brightness (0-255)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frame Count', fontsize=11, fontweight='bold')
    ax.set_title('Brightness Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Anisotropy histogram (motion blur detection)
    ax = axes[0, 1]
    ax.hist(metrics['anisotropy'], bins=30, color='#2196F3', edgecolor='black', alpha=0.7)
    ax.axvline(anisotropy_threshold, color='red', linestyle='--', linewidth=2,
               label=f'Blurry threshold ({anisotropy_threshold})')
    ax.set_xlabel('Anisotropy (Gradient Imbalance)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frame Count', fontsize=11, fontweight='bold')
    ax.set_title('Motion Blur Detection', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Lighting category pie chart
    ax = axes[1, 0]
    lighting_labels = categorize_lighting(metrics['brightness'])
    dark_count = np.sum(lighting_labels == 0)
    medium_count = np.sum(lighting_labels == 1)
    bright_count = np.sum(lighting_labels == 2)

    colors_pie = ['#424242', '#FFA726', '#FFEB3B']
    labels = [f'Dark (<80)\n{dark_count} frames',
              f'Medium (80-175)\n{medium_count} frames',
              f'Bright (>175)\n{bright_count} frames']
    ax.pie([dark_count, medium_count, bright_count], labels=labels, colors=colors_pie,
           autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10, 'weight': 'bold'})
    ax.set_title('Lighting Distribution', fontsize=12, fontweight='bold')

    # Blur category pie chart (motion blur only)
    ax = axes[1, 1]
    blurry_mask = (metrics['anisotropy'] > anisotropy_threshold)
    blurry_count = np.sum(blurry_mask)
    sharp_count = len(metrics['anisotropy']) - blurry_count

    colors_blur = ['#f44336', '#4caf50']
    labels_blur = [f'Blurry (motion)\n{blurry_count} frames',
                   f'Sharp\n{sharp_count} frames']
    ax.pie([blurry_count, sharp_count], labels=labels_blur, colors=colors_blur,
           autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10, 'weight': 'bold'})
    ax.set_title('Motion Blur Quality', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    return output_path


def create_umap_visualization(embeddings, activity_labels, output_path):
    """Generate UMAP scatter plot colored by activity clusters (handles HDBSCAN noise)."""
    n_neighbors_viz = min(15, embeddings.shape[0] - 1)  # Ensure n_neighbors < n_samples
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors_viz, min_dist=0.1)
    X_2d = reducer.fit_transform(embeddings)

    unique_activities = np.unique(activity_labels)
    noise_mask = activity_labels == -1
    cluster_activities = [a for a in unique_activities if a != -1]
    n_clusters = len(cluster_activities)

    # Adjust figure size and layout based on number of clusters
    if n_clusters <= 15:
        # Few clusters - legend inside plot
        fig, ax = plt.subplots(figsize=(10, 8))
        legend_outside = False
    else:
        # Many clusters - wider figure with legend outside
        fig, ax = plt.subplots(figsize=(14, 8))
        legend_outside = True

    # Plot noise points first (gray, small, transparent)
    if noise_mask.any():
        ax.scatter(X_2d[noise_mask, 0], X_2d[noise_mask, 1],
                  c='lightgray', label='Noise/Outliers',
                  s=20, alpha=0.3, edgecolors='none')

    # Plot clusters
    if n_clusters > 0:
        # Use tab20 for <=20 clusters, generate more colors if needed
        if n_clusters <= 20:
            cmap = matplotlib.colormaps.get_cmap('tab20')
        else:
            cmap = matplotlib.colormaps.get_cmap('hsv').resampled(n_clusters)

        for idx, activity_id in enumerate(cluster_activities):
            mask = activity_labels == activity_id
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                      c=[cmap(idx % 20 if n_clusters <= 20 else idx / n_clusters)],
                      label=f'Activity {activity_id}',
                      s=40, alpha=0.7, edgecolors='black', linewidth=0.5)

    ax.set_xlabel('UMAP Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('UMAP Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title('Frame Diversity - Activity Clustering (UMAP)', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    # Handle legend based on cluster count
    if n_clusters <= 15:
        ax.legend(loc='best', fontsize=9, ncol=2)
    elif n_clusters <= 40:
        # Legend outside to the right
        n_legend_cols = 2 if n_clusters <= 25 else 3
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, ncol=n_legend_cols)
    else:
        # Too many clusters - skip legend, add note
        ax.text(0.98, 0.02, f'{n_clusters} clusters (legend omitted)',
                transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    return output_path


def create_coverage_heatmap(activity_labels, lighting_labels, output_path):
    """Generate heatmap showing Activity × Lighting coverage (excludes noise)."""
    unique_activities = np.unique(activity_labels)
    # Filter out noise (-1)
    cluster_activities = [a for a in unique_activities if a != -1]

    if len(cluster_activities) == 0:
        # No clusters found - create placeholder
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No activity clusters found', ha='center', va='center',
                fontsize=14, color='gray')
        ax.axis('off')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path

    lighting_names = ['Dark (<80)', 'Medium (80-175)', 'Bright (>175)']

    # Build matrix (exclude noise)
    heatmap_data = []
    for activity_id in cluster_activities:
        activity_mask = activity_labels == activity_id
        activity_lighting = lighting_labels[activity_mask]

        row = [
            np.sum(activity_lighting == 0),  # Dark
            np.sum(activity_lighting == 1),  # Medium
            np.sum(activity_lighting == 2)   # Bright
        ]
        heatmap_data.append(row)

    heatmap_data = np.array(heatmap_data)

    # Fixed row height for readability - generate multiple chunks if needed
    n_activities = len(cluster_activities)
    rows_per_page = 20  # Max rows per heatmap chunk

    if n_activities <= rows_per_page:
        # Single heatmap - fits on one page
        fig_height = max(4, n_activities * 0.4)
        fig, ax = plt.subplots(figsize=(8, fig_height))

        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')

        ax.set_xticks(np.arange(len(lighting_names)))
        ax.set_yticks(np.arange(len(cluster_activities)))
        ax.set_xticklabels(lighting_names, fontsize=10, fontweight='bold')
        ax.set_yticklabels([f'Activity {i}' for i in cluster_activities], fontsize=10)

        for i in range(len(cluster_activities)):
            for j in range(len(lighting_names)):
                ax.text(j, i, int(heatmap_data[i, j]),
                       ha="center", va="center", color="black", fontsize=11, weight='bold')

        ax.set_title('Coverage Heatmap: Activity × Lighting Conditions', fontsize=13, fontweight='bold')
        ax.set_xlabel('Lighting Condition', fontsize=11, fontweight='bold')
        ax.set_ylabel('Activity Cluster', fontsize=11, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Frame Count', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        return output_path

    # Multiple chunks needed - split into separate images
    n_chunks = (n_activities + rows_per_page - 1) // rows_per_page
    output_paths = []

    for chunk_idx in range(n_chunks):
        start_row = chunk_idx * rows_per_page
        end_row = min(start_row + rows_per_page, n_activities)
        chunk_activities = cluster_activities[start_row:end_row]
        chunk_data = heatmap_data[start_row:end_row]

        fig_height = max(4, len(chunk_activities) * 0.4)
        fig, ax = plt.subplots(figsize=(8, fig_height))

        im = ax.imshow(chunk_data, cmap='YlOrRd', aspect='auto')

        ax.set_xticks(np.arange(len(lighting_names)))
        ax.set_yticks(np.arange(len(chunk_activities)))
        ax.set_xticklabels(lighting_names, fontsize=10, fontweight='bold')
        ax.set_yticklabels([f'Activity {i}' for i in chunk_activities], fontsize=10)

        for i in range(len(chunk_activities)):
            for j in range(len(lighting_names)):
                ax.text(j, i, int(chunk_data[i, j]),
                       ha="center", va="center", color="black", fontsize=11, weight='bold')

        title = f'Coverage Heatmap: Activity × Lighting (Part {chunk_idx + 1}/{n_chunks})'
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Lighting Condition', fontsize=11, fontweight='bold')
        ax.set_ylabel('Activity Cluster', fontsize=11, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Frame Count', fontsize=10, fontweight='bold')

        plt.tight_layout()

        chunk_path = output_path.parent / f"coverage_heatmap_part{chunk_idx + 1}.png"
        plt.savefig(chunk_path, dpi=100, bbox_inches='tight')
        plt.close()
        output_paths.append(chunk_path)

    # Return list of paths for multi-page rendering
    return output_paths


def create_activity_montage(image_paths, activity_id, max_samples, grid_cols, target_height):
    """
    Create a montage of sample images for an activity cluster.
    """
    # Sample up to max_samples images
    n_samples = min(len(image_paths), max_samples)
    sampled_paths = random.sample(image_paths, n_samples) if len(image_paths) > max_samples else image_paths

    # Load images
    images = []
    for img_path in sampled_paths:
        img = load_image_safe(img_path)
        if img is not None:
            images.append(img)

    if len(images) == 0:
        # Return placeholder
        placeholder = PILImage.new('RGB', (800, 200), color='gray')
        return placeholder

    # Resize all images to same size (maintain aspect ratio)
    resized_images = []
    for img in images:
        aspect_ratio = img.width / img.height
        new_width = int(target_height * aspect_ratio)
        resized = img.resize((new_width, target_height), PILImage.Resampling.LANCZOS)
        resized_images.append(resized)

    # Determine grid dimensions
    n_images = len(resized_images)
    n_rows = (n_images + grid_cols - 1) // grid_cols

    # Calculate montage dimensions
    max_img_width = max(img.width for img in resized_images)
    montage_width = max_img_width * grid_cols + 10 * (grid_cols + 1)  # 10px padding
    montage_height = target_height * n_rows + 10 * (n_rows + 1)

    # Create montage canvas
    montage = PILImage.new('RGB', (montage_width, montage_height), color='white')

    # Paste images
    for idx, img in enumerate(resized_images):
        row = idx // grid_cols
        col = idx % grid_cols
        x = col * (max_img_width + 10) + 10
        y = row * (target_height + 10) + 10
        montage.paste(img, (x, y))

    return montage


def create_blurry_images_montage(blurry_image_cache, grid_cols, target_height):
    """
    Create a montage of blurry image samples using pre-cached images.
    Blurry = low sharpness (focus blur) OR high anisotropy (motion blur).
    """
    if len(blurry_image_cache) == 0:
        return None

    # Use cached images directly (no loading needed)
    images = [item['image'] for item in blurry_image_cache]
    anisotropy_vals = [item['anisotropy'] for item in blurry_image_cache]
    brightness_vals = [item['brightness'] for item in blurry_image_cache]

    if len(images) == 0:
        return None

    # Resize all images to same size (maintain aspect ratio)
    resized_images = []
    for img in images:
        aspect_ratio = img.width / img.height
        new_width = int(target_height * aspect_ratio)
        resized = img.resize((new_width, target_height), PILImage.Resampling.LANCZOS)
        resized_images.append(resized)

    # Determine grid dimensions
    n_images = len(resized_images)
    n_rows = (n_images + grid_cols - 1) // grid_cols

    # Calculate montage dimensions
    max_img_width = max(img.width for img in resized_images)
    montage_width = max_img_width * grid_cols + 10 * (grid_cols + 1)  # 10px padding
    montage_height = (target_height + 45) * n_rows + 10 * (n_rows + 1)  # Extra 45px for 2 lines of text

    # Create montage canvas
    montage = PILImage.new('RGB', (montage_width, montage_height), color='white')

    # Paste images with sharpness, anisotropy, and brightness values
    draw = ImageDraw.Draw(montage)

    try:
        # Try to use a better font
        font = ImageFont.truetype("arial.ttf", 10)
    except:
        # Fallback to default font
        font = ImageFont.load_default()

    for idx, (img, anisotropy, brightness) in enumerate(zip(resized_images, anisotropy_vals, brightness_vals)):
        row = idx // grid_cols
        col = idx % grid_cols
        x = col * (max_img_width + 10) + 10
        y = row * (target_height + 45 + 10) + 10

        # Paste image
        montage.paste(img, (x, y))

        # Add anisotropy and brightness values below image (2 lines)
        text_line1 = f"Aniso: {anisotropy:.2f}"
        text_line2 = f"Bright: {brightness:.1f}"

        text_bbox = draw.textbbox((0, 0), text_line1, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = x + (img.width - text_width) // 2
        text_y = y + target_height + 5

        draw.text((text_x, text_y), text_line1, fill='red', font=font)
        draw.text((text_x, text_y + 15), text_line2, fill='blue', font=font)

    return montage


def compress_image_for_pdf(img_path, quality):
    """Compress PNG to JPEG for smaller PDF size."""
    try:
        img = PILImage.open(img_path)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        jpeg_path = img_path.with_suffix('.jpg')
        img.save(jpeg_path, 'JPEG', quality=quality, optimize=True)
        return jpeg_path
    except Exception as e:
        print(f"\n⚠️  Warning: Failed to compress image {img_path}: {e}")
        # Return original path if compression fails
        return img_path


def generate_pdf_report(analysis_data, output_path, temp_dir, image_quality):
    """
    Generate comprehensive PDF report with compressed images.

    Args:
        image_quality: JPEG quality (1-100). Lower = smaller file. Default 75 is good balance.
    """
    print("\nBuilding PDF document...")
    print(f"  Image compression quality: {image_quality}%")
    doc = SimpleDocTemplate(str(output_path), pagesize=letter,
                           topMargin=0.5*inch, bottomMargin=0.5*inch,
                           leftMargin=0.5*inch, rightMargin=0.5*inch)

    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=10,
        spaceBefore=15,
        fontName='Helvetica-Bold'
    )

    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=8,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )

    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY
    )

    # Add pages
    _add_executive_summary(story, analysis_data, title_style, heading_style, normal_style)
    _add_quality_dashboard(story, analysis_data, temp_dir, heading_style, subheading_style, normal_style, image_quality)
    _add_diversity_analysis(story, analysis_data, temp_dir, heading_style, subheading_style, normal_style, image_quality)
    _add_activity_examples(story, analysis_data, temp_dir, heading_style, subheading_style, normal_style, image_quality)
    _add_cluster_quality(story, analysis_data, heading_style, subheading_style, normal_style)
    _add_coverage_analysis(story, analysis_data, temp_dir, heading_style, normal_style, image_quality)

    # Build PDF
    doc.build(story)

    # Cleanup temp files
    max_retries = 3
    for attempt in range(max_retries):
        try:
            shutil.rmtree(temp_dir)
            break
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(1)
                gc.collect()
            else:
                print(f"\n⚠️  Warning: Could not delete temp directory {temp_dir}")
                print("    You may manually delete it after the process completes.")

    print(f"\n✅ PDF report generated: {output_path}")


def _add_executive_summary(story, analysis_data, title_style, heading_style, normal_style):
    """Page 1: Executive Summary"""
    story.append(Paragraph("PRE-ANNOTATION FRAME QUALITY REPORT", title_style))
    story.append(Paragraph("Engineer Dataset Assessment", heading_style))
    story.append(Spacer(1, 0.2*inch))

    stats = analysis_data['summary_stats']

    summary_data = [
        ['Metric', 'Value', 'Status'],
        ['Total Frames Analyzed', str(stats['total_frames']), ''],
        ['Valid Frames (loaded successfully)', str(stats['valid_frames']), ''],
        ['Activities Detected', str(stats['n_activities']),
         '✅ Good' if stats['n_activities'] >= 3 else '⚠ Limited diversity'],
        ['', '', ''],
        ['Dark Frames (<80 brightness)', f"{stats['dark_count']} ({stats['dark_pct']:.1f}%)",
         '⚠ High' if stats['dark_pct'] > 60 else '✅ OK'],
        ['Medium Frames (80-175)', f"{stats['medium_count']} ({stats['medium_pct']:.1f}%)", ''],
        ['Bright Frames (>175)', f"{stats['bright_count']} ({stats['bright_pct']:.1f}%)", ''],
        ['', '', ''],
        ['Blurry Frames (motion blur, anisotropy >' + str(analysis_data['anisotropy_threshold']) + ')',
         f"{stats['blurry_count']} ({stats['blurry_pct']:.1f}%)",
         '⚠ High' if stats['blurry_pct'] > 40 else '✅ OK'],
        ['Non-Blurry Frames', f"{stats['sharp_count']} ({stats['sharp_pct']:.1f}%)", ''],
    ]

    summary_table = Table(summary_data, colWidths=[3.5*inch, 2*inch, 1.5*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
    ]))

    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))

    # Overall Quality Score
    quality_score = stats['quality_score']
    score_color = colors.green if quality_score >= 7 else (colors.orange if quality_score >= 5 else colors.red)

    story.append(Paragraph(f"<b>Overall Quality Score:</b> <font color='{score_color.hexval()}'>{quality_score:.1f}/10</font>",
                          ParagraphStyle('Score', parent=normal_style, fontSize=13, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.2*inch))

    # Key Recommendations
    story.append(Paragraph("Key Recommendations", ParagraphStyle('RecHeading', parent=normal_style, fontSize=12, fontName='Helvetica-Bold')))

    for rec in analysis_data['recommendations']:
        if rec.startswith('⚠'):
            bullet_style = ParagraphStyle('Warning', parent=normal_style,
                                         leftIndent=20, bulletIndent=10,
                                         textColor=colors.HexColor('#d32f2f'))
        elif rec.startswith('ℹ'):
            bullet_style = ParagraphStyle('Info', parent=normal_style,
                                         leftIndent=20, bulletIndent=10,
                                         textColor=colors.HexColor('#1976d2'))
        else:
            bullet_style = ParagraphStyle('Success', parent=normal_style,
                                         leftIndent=20, bulletIndent=10,
                                         textColor=colors.HexColor('#388e3c'))

        story.append(Paragraph(f"• {rec}", bullet_style))
        story.append(Spacer(1, 0.08*inch))

    story.append(PageBreak())


def _add_quality_dashboard(story, analysis_data, temp_dir, heading_style, subheading_style, normal_style, image_quality):
    """Page 2: Quality Metrics Dashboard"""
    story.append(Paragraph("Quality Metrics Dashboard", heading_style))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph(
        "This section shows the distribution of brightness and sharpness across all frames. "
        "Use these metrics to identify frames that should be dropped before annotation.",
        normal_style
    ))
    story.append(Spacer(1, 0.2*inch))

    quality_chart_path = compress_image_for_pdf(temp_dir / "quality_metrics.png", quality=image_quality)
    img = RLImage(str(quality_chart_path), width=7*inch, height=5.8*inch)
    story.append(img)
    story.append(Spacer(1, 0.2*inch))

    # Frames to drop recommendations
    story.append(Paragraph("Recommended Frames to Drop", subheading_style))

    stats = analysis_data['summary_stats']
    drop_blurry = stats['blurry_count']
    drop_dark = stats['dark_count'] if stats['dark_pct'] > 70 else 0

    if drop_blurry > 0 or drop_dark > 0:
        drop_text = f"Based on quality thresholds, consider removing:<br/>"
        if drop_blurry > 0:
            drop_text += f"• <b>{drop_blurry} blurry frames</b> (motion blur with anisotropy >{analysis_data['anisotropy_threshold']})<br/>"
        if drop_dark > 0:
            drop_text += f"• <b>{drop_dark} extremely dark frames</b> (brightness < 80, representing {stats['dark_pct']:.1f}% of dataset)"

        story.append(Paragraph(drop_text, normal_style))
    else:
        story.append(Paragraph("✅ No frames need to be dropped based on quality metrics.", normal_style))

    story.append(Spacer(1, 0.3*inch))

    # Blurry images montage section
    blurry_montage_path_png = temp_dir / "blurry_images_montage.png"
    if blurry_montage_path_png.exists() and drop_blurry > 0:
        blurry_montage_path = compress_image_for_pdf(blurry_montage_path_png, quality=image_quality)
        story.append(Paragraph("Sample Blurry Frames (Low Sharpness)", subheading_style))
        story.append(Paragraph(
            f"Below are up to {min(20, drop_blurry)} sample blurry frames that should be reviewed for removal. "
            f"These frames have motion blur, out-of-focus issues, or camera shake. "
            f"Sharpness and brightness values are shown below each image.",
            normal_style
        ))
        story.append(Spacer(1, 0.15*inch))

        try:
            blurry_img_obj = PILImage.open(blurry_montage_path)
            img_width, img_height = blurry_img_obj.size

            max_width = 7 * inch
            width_scale = max_width / img_width
            scaled_height = img_height * width_scale

            # Intelligent splitting: if montage too tall, split at row boundaries
            max_page_height = 9.5 * inch  # Safe max height per page

            if scaled_height <= max_page_height:
                # Fits on one page - compress and add directly
                img = RLImage(str(blurry_montage_path), width=max_width, height=scaled_height)
                story.append(img)
            else:
                # Split at row boundaries (use actual grid_cols from blurry montage)
                grid_cols = analysis_data.get('grid_cols_blurry', 4)
                num_images = drop_blurry
                num_rows = (num_images + grid_cols - 1) // grid_cols

                # Calculate height per row
                row_height_px = img_height / num_rows
                row_height_scaled = row_height_px * width_scale

                # Calculate how many rows fit per page
                rows_per_page = max(1, int(max_page_height / row_height_scaled))

                # Split into chunks by whole rows
                num_chunks = (num_rows + rows_per_page - 1) // rows_per_page

                for chunk_idx in range(num_chunks):
                    # Calculate which rows to include
                    start_row = chunk_idx * rows_per_page
                    end_row = min((chunk_idx + 1) * rows_per_page, num_rows)

                    # Calculate pixel coordinates (crop at row boundaries)
                    top = int(start_row * row_height_px)
                    bottom = int(end_row * row_height_px)

                    # Crop the chunk
                    chunk = blurry_img_obj.crop((0, top, img_width, bottom))

                    # Save chunk temporarily as compressed JPEG
                    chunk_path = temp_dir / f"blurry_chunk_{chunk_idx}.jpg"
                    chunk.convert('RGB').save(chunk_path, 'JPEG', quality=90, optimize=True)

                    # Calculate chunk height after scaling
                    chunk_height = (bottom - top) * width_scale

                    # Add chunk to PDF
                    if chunk_idx > 0:
                        story.append(Spacer(1, 0.05*inch))
                        story.append(Paragraph(f"<i>(Continuation - Rows {start_row+1} to {end_row})</i>",
                                             ParagraphStyle('Continuation', parent=normal_style,
                                                          fontSize=9, alignment=TA_CENTER,
                                                          textColor=colors.grey)))
                        story.append(Spacer(1, 0.05*inch))

                    img = RLImage(str(chunk_path), width=max_width, height=chunk_height)
                    story.append(img)

                    # Add page break between chunks (except last one)
                    if chunk_idx < num_chunks - 1:
                        story.append(PageBreak())

                    del chunk  # Free memory

            # Explicit cleanup of blurry montage image
            del blurry_img_obj
            gc.collect()

        except Exception as e:
            story.append(Paragraph(f"⚠ Error loading blurry images montage: {e}", normal_style))
            if 'blurry_img_obj' in locals():
                del blurry_img_obj
                gc.collect()

    story.append(PageBreak())


def _add_diversity_analysis(story, analysis_data, temp_dir, heading_style, subheading_style, normal_style, image_quality):
    """Page 3: Diversity Analysis"""
    story.append(Paragraph("Frame Diversity Analysis", heading_style))
    story.append(Spacer(1, 0.1*inch))

    stats = analysis_data['summary_stats']
    story.append(Paragraph(
        f"Frames were automatically grouped into <b>{stats['n_activities']} activity clusters</b> "
        f"based on semantic similarity (objects, positions, scenarios). "
        f"Each cluster represents a distinct type of work activity or scenario captured in the frames.",
        normal_style
    ))
    story.append(Spacer(1, 0.2*inch))

    umap_chart_path = compress_image_for_pdf(temp_dir / "umap_diversity.png", quality=image_quality)
    img = RLImage(str(umap_chart_path), width=7*inch, height=5.6*inch)
    story.append(img)
    story.append(Spacer(1, 0.2*inch))

    # Activity breakdown
    story.append(Paragraph("Activity Cluster Breakdown", subheading_style))

    activity_data = [['Activity Cluster', 'Frame Count', 'Percentage']]
    unique_activities = np.unique(analysis_data['activity_labels'])

    # Show clusters first, then noise
    cluster_activities = [a for a in unique_activities if a != -1]
    for activity_id in cluster_activities:
        count = np.sum(analysis_data['activity_labels'] == activity_id)
        pct = count / len(analysis_data['activity_labels']) * 100
        activity_data.append([f'Activity {activity_id}', str(count), f'{pct:.1f}%'])

    # Add noise row if present
    if -1 in unique_activities:
        count = np.sum(analysis_data['activity_labels'] == -1)
        pct = count / len(analysis_data['activity_labels']) * 100
        activity_data.append(['Noise/Outliers', str(count), f'{pct:.1f}%'])

    activity_table = Table(activity_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
    activity_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
    ]))

    story.append(activity_table)
    story.append(PageBreak())


def _add_activity_examples(story, analysis_data, temp_dir, heading_style, subheading_style, normal_style, image_quality):
    """Page 4: Activity Visual Examples"""
    story.append(Paragraph("Activity Visual Examples", heading_style))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph(
        "Below are sample frames from each activity cluster to help you visually validate the clustering quality.",
        normal_style
    ))
    story.append(Spacer(1, 0.2*inch))

    unique_activities = np.unique(analysis_data['activity_labels'])
    cluster_activities = [a for a in unique_activities if a != -1]

    # Show regular clusters first
    for activity_id in cluster_activities:
        activity_name = analysis_data.get('activity_names', {}).get(activity_id, f'Activity {activity_id}')
        count = np.sum(analysis_data['activity_labels'] == activity_id)

        story.append(Paragraph(
            f"<b>Activity {activity_id}:</b> {activity_name} ({count} frames)",
            subheading_style
        ))
        story.append(Spacer(1, 0.1*inch))

        montage_path_png = temp_dir / f"activity_{activity_id}_montage.png"
        if montage_path_png.exists():
            montage_path = compress_image_for_pdf(montage_path_png, quality=image_quality)
            try:
                montage_img = PILImage.open(montage_path)
                img_width, img_height = montage_img.size

                max_width = 7 * inch
                width_scale = max_width / img_width
                scaled_height = img_height * width_scale

                # Intelligent splitting: if montage too tall, split at row boundaries
                max_page_height = 9.5 * inch

                if scaled_height <= max_page_height:
                    # Fits on one page
                    img = RLImage(str(montage_path), width=max_width, height=scaled_height)
                    story.append(img)
                else:
                    # Split at row boundaries (use actual grid_cols from activity montage)
                    grid_cols = analysis_data.get('grid_cols_activities', 4)
                    num_images = min(count, 12)  # max_samples for inline display
                    num_rows = (num_images + grid_cols - 1) // grid_cols

                    row_height_px = img_height / num_rows
                    row_height_scaled = row_height_px * width_scale

                    rows_per_page = max(1, int(max_page_height / row_height_scaled))
                    num_chunks = (num_rows + rows_per_page - 1) // rows_per_page

                    for chunk_idx in range(num_chunks):
                        start_row = chunk_idx * rows_per_page
                        end_row = min((chunk_idx + 1) * rows_per_page, num_rows)

                        top = int(start_row * row_height_px)
                        bottom = int(end_row * row_height_px)

                        chunk = montage_img.crop((0, top, img_width, bottom))
                        chunk_path = temp_dir / f"activity_{activity_id}_chunk_{chunk_idx}.jpg"
                        chunk.convert('RGB').save(chunk_path, 'JPEG', quality=90, optimize=True)

                        chunk_height = (bottom - top) * width_scale

                        if chunk_idx > 0:
                            story.append(Spacer(1, 0.05*inch))
                            story.append(Paragraph(f"<i>(Continuation - Rows {start_row+1} to {end_row})</i>",
                                                 ParagraphStyle('Continuation', parent=normal_style,
                                                              fontSize=9, alignment=TA_CENTER,
                                                              textColor=colors.grey)))
                            story.append(Spacer(1, 0.05*inch))

                        img = RLImage(str(chunk_path), width=max_width, height=chunk_height)
                        story.append(img)

                        if chunk_idx < num_chunks - 1:
                            story.append(PageBreak())

                        del chunk

                # Explicit cleanup of montage image
                del montage_img
                gc.collect()

                story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                story.append(Paragraph(f"⚠ Error loading montage: {e}", normal_style))
                story.append(Spacer(1, 0.1*inch))
                if 'montage_img' in locals():
                    del montage_img
                    gc.collect()
        else:
            story.append(Paragraph("⚠ Montage not generated", normal_style))
            story.append(Spacer(1, 0.1*inch))

    # Show noise/outliers if present
    if -1 in unique_activities:
        n_noise = np.sum(analysis_data['activity_labels'] == -1)
        story.append(Paragraph(
            f"<b>Noise/Outliers:</b> Rare or unique frames ({n_noise} frames)",
            subheading_style
        ))
        story.append(Spacer(1, 0.1*inch))

        story.append(Paragraph(
            "These frames did not fit into any activity cluster. They may represent rare activities, "
            "transitions between activities, or frames with poor quality/visibility.",
            normal_style
        ))
        story.append(Spacer(1, 0.1*inch))

        montage_path_png = temp_dir / f"activity_-1_montage.png"
        if montage_path_png.exists():
            montage_path = compress_image_for_pdf(montage_path_png, quality=image_quality)
            try:
                montage_img = PILImage.open(montage_path)
                img_width, img_height = montage_img.size

                max_width = 7 * inch
                width_scale = max_width / img_width
                scaled_height = img_height * width_scale

                # Intelligent splitting: if montage too tall, split at row boundaries
                max_page_height = 9.5 * inch

                if scaled_height <= max_page_height:
                    # Fits on one page
                    img = RLImage(str(montage_path), width=max_width, height=scaled_height)
                    story.append(img)
                else:
                    # Split at row boundaries (use actual grid_cols from activity montage)
                    grid_cols = analysis_data.get('grid_cols_activities', 4)
                    num_images = min(n_noise, 20)  # max_samples for outliers
                    num_rows = (num_images + grid_cols - 1) // grid_cols

                    row_height_px = img_height / num_rows
                    row_height_scaled = row_height_px * width_scale

                    rows_per_page = max(1, int(max_page_height / row_height_scaled))
                    num_chunks = (num_rows + rows_per_page - 1) // rows_per_page

                    for chunk_idx in range(num_chunks):
                        start_row = chunk_idx * rows_per_page
                        end_row = min((chunk_idx + 1) * rows_per_page, num_rows)

                        top = int(start_row * row_height_px)
                        bottom = int(end_row * row_height_px)

                        chunk = montage_img.crop((0, top, img_width, bottom))
                        chunk_path = temp_dir / f"outliers_chunk_{chunk_idx}.jpg"
                        chunk.convert('RGB').save(chunk_path, 'JPEG', quality=90, optimize=True)

                        chunk_height = (bottom - top) * width_scale

                        if chunk_idx > 0:
                            story.append(Spacer(1, 0.05*inch))
                            story.append(Paragraph(f"<i>(Continuation - Rows {start_row+1} to {end_row})</i>",
                                                 ParagraphStyle('Continuation', parent=normal_style,
                                                              fontSize=9, alignment=TA_CENTER,
                                                              textColor=colors.grey)))
                            story.append(Spacer(1, 0.05*inch))

                        img = RLImage(str(chunk_path), width=max_width, height=chunk_height)
                        story.append(img)

                        if chunk_idx < num_chunks - 1:
                            story.append(PageBreak())

                        del chunk

                # Explicit cleanup of montage image
                del montage_img
                gc.collect()

                story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                story.append(Paragraph(f"⚠ Error loading noise montage: {e}", normal_style))
                story.append(Spacer(1, 0.1*inch))
                if 'montage_img' in locals():
                    del montage_img
                    gc.collect()

        # Add filename listing for all noise/outlier frames
        noise_filelist_path = temp_dir / "noise_filenames.txt"
        if noise_filelist_path.exists():
            story.append(Paragraph(
                f"<b>Complete list of all {n_noise} noise/outlier frames:</b>",
                ParagraphStyle('NoiseListHeading', parent=normal_style, fontSize=10, fontName='Helvetica-Bold')
            ))
            story.append(Spacer(1, 0.05*inch))

            try:
                with open(noise_filelist_path, 'r') as f:
                    filenames = f.read().strip().split('\n')

                # Display filenames in a compact format (comma-separated, wrapped)
                filenames_text = ', '.join(filenames)
                story.append(Paragraph(
                    filenames_text,
                    ParagraphStyle('NoiseList', parent=normal_style, fontSize=8, textColor=colors.HexColor('#555555'))
                ))
                story.append(Spacer(1, 0.1*inch))
            except Exception as e:
                story.append(Paragraph(f"⚠ Error loading noise filename list: {e}", normal_style))

    story.append(PageBreak())


def _add_cluster_quality(story, analysis_data, heading_style, subheading_style, normal_style):
    """Page 5: Cluster Quality Metrics"""
    story.append(PageBreak())
    story.append(Paragraph("Cluster Quality Metrics", heading_style))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph(
        "These metrics assess the quality of the activity clustering. Higher Silhouette and Calinski-Harabasz "
        "scores indicate better-defined clusters, while lower Davies-Bouldin scores are better.",
        normal_style
    ))
    story.append(Spacer(1, 0.2*inch))

    cluster_quality = analysis_data['cluster_quality']

    # Metrics explanation table
    silhouette_val = f"{cluster_quality['silhouette']:.3f}" if cluster_quality['silhouette'] is not None else "N/A"
    calinski_val = f"{cluster_quality['calinski_harabasz']:.1f}" if cluster_quality['calinski_harabasz'] is not None else "N/A"
    davies_val = f"{cluster_quality['davies_bouldin']:.3f}" if cluster_quality['davies_bouldin'] is not None else "N/A"

    metrics_data = [
        ['Metric', 'Value', 'Interpretation', 'Range'],
        ['Silhouette Score',
         silhouette_val,
         'Higher is better (cluster separation)',
         '[-1, 1]'],
        ['Calinski-Harabasz',
         calinski_val,
         'Higher is better (between/within variance)',
         '[0, ∞]'],
        ['Davies-Bouldin',
         davies_val,
         'LOWER is better (avg similarity ratio)',
         '[0, ∞]'],
    ]

    metrics_table = Table(metrics_data, colWidths=[1.5*inch, 1.0*inch, 2.5*inch, 1.0*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 0.3*inch))

    # Cluster statistics
    story.append(Paragraph("Cluster Statistics", subheading_style))
    stats_data = [
        ['Statistic', 'Count'],
        ['Number of Clusters', str(cluster_quality['n_clusters'])],
        ['Clustered Frames', str(cluster_quality['n_clustered'])],
        ['Noise/Outlier Frames', str(cluster_quality['n_noise'])],
    ]

    stats_table = Table(stats_data, colWidths=[3*inch, 1.5*inch])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(stats_table)
    story.append(Spacer(1, 0.3*inch))

    # Interpretation
    story.append(Paragraph("Interpretation", subheading_style))

    silhouette = cluster_quality['silhouette']
    davies = cluster_quality['davies_bouldin']

    interpretations = []

    # Silhouette interpretation
    if silhouette is None:
        interpretations.append("⚠ <b>Silhouette Score</b>: Not available (insufficient clusters for calculation).")
    elif silhouette >= 0.5:
        interpretations.append("✅ <b>Silhouette Score</b>: Excellent cluster separation - activities are well-defined.")
    elif silhouette >= 0.3:
        interpretations.append("⚠ <b>Silhouette Score</b>: Moderate cluster separation - some overlap between activities.")
    else:
        interpretations.append("❌ <b>Silhouette Score</b>: Weak cluster separation - activities may be too similar or need different parameters.")

    # Davies-Bouldin interpretation
    if davies is None:
        interpretations.append("⚠ <b>Davies-Bouldin</b>: Not available (insufficient clusters for calculation).")
    elif davies <= 1.0:
        interpretations.append("✅ <b>Davies-Bouldin</b>: Excellent cluster quality - low similarity between different clusters.")
    elif davies <= 1.5:
        interpretations.append("⚠ <b>Davies-Bouldin</b>: Moderate cluster quality - some clusters may be similar.")
    else:
        interpretations.append("❌ <b>Davies-Bouldin</b>: Weak cluster quality - clusters may be poorly separated.")

    # Noise percentage
    noise_pct = cluster_quality['n_noise'] / (cluster_quality['n_clustered'] + cluster_quality['n_noise']) * 100
    if noise_pct <= 10:
        interpretations.append(f"✅ <b>Noise Level</b>: Low ({noise_pct:.1f}%) - most frames fit into clear activity patterns.")
    elif noise_pct <= 25:
        interpretations.append(f"⚠ <b>Noise Level</b>: Moderate ({noise_pct:.1f}%) - some frames don't fit clear patterns.")
    else:
        interpretations.append(f"❌ <b>Noise Level</b>: High ({noise_pct:.1f}%) - many frames are outliers, consider collecting more data or adjusting parameters.")

    for interpretation in interpretations:
        story.append(Paragraph(interpretation, normal_style))
        story.append(Spacer(1, 0.1*inch))

    story.append(Spacer(1, 0.2*inch))

    # Overall quality verdict
    overall_good = (silhouette is not None and silhouette >= 0.3 and
                    davies is not None and davies <= 1.5 and noise_pct <= 25)
    overall_moderate = ((silhouette is not None and silhouette >= 0.2) or
                       (davies is not None and davies <= 2.0 and noise_pct <= 40))

    if overall_good:
        verdict = ("✅ <b>GOOD CLUSTERING QUALITY</b><br/><br/>"
                  "The activity clusters are well-defined and meaningful. This indicates good dataset diversity "
                  "and successful activity differentiation.")
        verdict_color = colors.HexColor('#388e3c')
    elif overall_moderate:
        verdict = ("⚠ <b>MODERATE CLUSTERING QUALITY</b><br/><br/>"
                  "The activity clusters show some structure but could be improved. Consider collecting more "
                  "samples per activity or adjusting clustering parameters.")
        verdict_color = colors.HexColor('#f57c00')
    else:
        verdict = ("❌ <b>WEAK CLUSTERING QUALITY</b><br/><br/>"
                  "The activity clusters are poorly defined. This may indicate insufficient diversity, too few "
                  "samples per activity, or the need for different clustering parameters.")
        verdict_color = colors.HexColor('#d32f2f')

    verdict_style = ParagraphStyle('ClusterVerdict', parent=normal_style, fontSize=11,
                                   textColor=verdict_color, alignment=TA_CENTER,
                                   borderWidth=2, borderColor=verdict_color,
                                   borderPadding=10, backColor=colors.HexColor('#fafafa'))

    story.append(Paragraph(verdict, verdict_style))
    story.append(Spacer(1, 0.2*inch))


def _add_coverage_analysis(story, analysis_data, temp_dir, heading_style, normal_style, image_quality):
    """Page 6+: Coverage Gap Analysis (multi-page if many activities)"""
    story.append(PageBreak())
    story.append(Paragraph("Coverage Gap Analysis", heading_style))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph(
        "This heatmap shows the distribution of lighting conditions across different activities. "
        "Gaps (cells with low/zero counts) indicate missing scenarios that may need additional data collection.",
        normal_style
    ))
    story.append(Spacer(1, 0.2*inch))

    # Check if single heatmap or multiple chunks
    single_heatmap_path = temp_dir / "coverage_heatmap.png"
    chunk_paths = sorted(temp_dir.glob("coverage_heatmap_part*.png"))

    if single_heatmap_path.exists() and not chunk_paths:
        # Single heatmap
        heatmap_chart_path = compress_image_for_pdf(single_heatmap_path, quality=image_quality)
        heatmap_img = PILImage.open(heatmap_chart_path)
        img_width, img_height = heatmap_img.size
        max_width = 6 * inch
        width_scale = max_width / img_width
        scaled_height = min(img_height * width_scale, 8 * inch)
        img = RLImage(str(heatmap_chart_path), width=max_width, height=scaled_height)
        story.append(img)
        story.append(Spacer(1, 0.3*inch))
    elif chunk_paths:
        # Multiple heatmap chunks - each on its own section
        for i, chunk_path in enumerate(chunk_paths):
            if i > 0:
                story.append(PageBreak())
            compressed_path = compress_image_for_pdf(chunk_path, quality=image_quality)
            chunk_img = PILImage.open(compressed_path)
            img_width, img_height = chunk_img.size
            max_width = 6 * inch
            width_scale = max_width / img_width
            scaled_height = min(img_height * width_scale, 8 * inch)
            img = RLImage(str(compressed_path), width=max_width, height=scaled_height)
            story.append(img)
            story.append(Spacer(1, 0.2*inch))
    else:
        story.append(Paragraph("⚠ Heatmap not generated", normal_style))
        story.append(Spacer(1, 0.3*inch))

    # Final verdict
    story.append(Paragraph("Final Assessment", heading_style))

    stats = analysis_data['summary_stats']
    quality_score = stats['quality_score']

    if quality_score >= 7 and stats['n_activities'] >= 3 and stats['blurry_pct'] < 30:
        verdict = ("✅ <b>READY FOR ANNOTATION</b><br/><br/>"
                  "This dataset shows good quality and diversity. The frames are suitable for annotation effort. "
                  "Consider addressing the recommendations above to further improve the dataset.")
        verdict_color = colors.HexColor('#388e3c')
    elif quality_score >= 5:
        verdict = ("⚠ <b>NEEDS IMPROVEMENT</b><br/><br/>"
                  "This dataset has moderate quality but some issues. Review the recommendations carefully. "
                  "Consider dropping low-quality frames and adding more diverse scenarios before annotation.")
        verdict_color = colors.HexColor('#f57c00')
    else:
        verdict = ("❌ <b>NOT READY FOR ANNOTATION</b><br/><br/>"
                  "This dataset has significant quality issues. Annotation effort on these frames may be wasted. "
                  "Strongly recommend improving frame quality and diversity before proceeding.")
        verdict_color = colors.HexColor('#d32f2f')

    verdict_style = ParagraphStyle('Verdict', parent=normal_style, fontSize=11,
                                   textColor=verdict_color, alignment=TA_CENTER,
                                   borderWidth=2, borderColor=verdict_color,
                                   borderPadding=10, backColor=colors.HexColor('#fafafa'))

    story.append(Paragraph(verdict, verdict_style))


## ---------------------------------- MAIN PIPELINE -----------------------------------------
def main():
    # GLOBAL CONFIGURATION
    frames_dir = r"path/to/frames"  # Root directory with extracted frames (replace with your path)
    
    yolo_batch_size = 8   # YOLO pose detection batch size
    yolo_conf_person_thr = 0.3 # YOLO pose detection confidence threshold
    batch_size = 64       # DINOv2 inference batch size

    anisotropy_threshold = 3.6  # Directional gradient anisotropy threshold for motion blur detection. Greater than this are blurry frames

    n_components = 64 # For Clustering, UMAP
    min_cluster_size = 15 # For clustering, An activity must last at least ~n frames to exist
    min_samples = 5 # For clustering, Determines how many neighbors a point needs to be considered a "core point". Edge frames are allowed to stay in cluster

    cache_blurry_num_samples = 24 # Blurry samples to be shown
    activity_num_samples = 20 # Number of samples to be shown per each activity
    outliers_num_samples = 20 # Number of samples to be shown of Outliers
    pdf_image_quality = 70  # JPEG quality for images (1-100). Lower = smaller file.

    grid_cols_activities = 4  # Grid columns for activity montages
    grid_cols_blurry = 4  # Grid columns for blurry sample montage 

    output_dir = "frame_analysis_results_dinov2_pose_emb"  # Output directory for results
    pdf_name = "PreAnnotation_Quality_Report.pdf"  # Output PDF filename
    use_embedding_cache = True  # Set to True to cache embeddings (recommended for large datasets >5000 frames)

    model_name = "facebook/dinov2-base"
    # facebook/dinov2-base --- BEST results
    # facebook/dinov2-with-registers-base
    # facebook/dinov3-vitl16-pretrain-lvd1689m --- requires permission

    # Setup
    frames_root = Path(frames_dir)
    if not frames_root.exists():
        raise RuntimeError(f"Frames directory does not exist: {frames_root}")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # GPU setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"Using GPU: {gpu_props.name}")
        print(f"GPU Memory: {gpu_props.total_memory/1e9:.1f}GB")
        torch.cuda.set_per_process_memory_fraction(0.95)
    else:
        print("Using CPU - GPU not available")

    # Load DINOv2 model from Hugging Face (cached to models/ folder)
    print(f"\nLoading model from Hugging Face: {model_name} ...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    # STEP 1: SCAN FRAMES
    print("\n" + "="*60)
    print("SCANNING FRAMES")
    print("="*60)
    image_paths = scan_frames(frames_root)

    if not image_paths:
        print("No frames found. Exiting.")
        return

    print(f"\nTotal frames found: {len(image_paths)}")

    # Load YOLOv11 Pose model
    print(f"\nLoading YOLO11 Pose model...")
    pose_model = YOLO('models/yolo11n-pose.pt')
    print("  ✅ YOLOv11 Pose model loaded")

    # STEP 2: PERSON DETECTION & POSE EXTRACTION 
    print("\n" + "="*60)
    print("STEP 2: PERSON DETECTION & POSE EXTRACTION")
    print("="*60)
    print(f"✅ Using batch size {yolo_batch_size} for YOLO inference.")
    frame_data = detect_persons_and_extract_poses(image_paths, pose_model, device, conf_threshold=yolo_conf_person_thr, yolo_batch_size=yolo_batch_size)

    # Count frames with persons
    frames_with_persons = sum(1 for f in frame_data if f['has_persons'])
    frames_without_persons = len(image_paths) - frames_with_persons
    print(f"\n  ✅ Found persons in {frames_with_persons}/{len(image_paths)} frames")
    if frames_without_persons > 0:
        print(f"  ℹ️  {frames_without_persons} frames without persons will use scene-only embeddings")

    # STEP 3: COMPUTE EMBEDDINGS (pose-aware when persons detected, scene-only otherwise)
    print("\n" + "="*60)
    print("STEP 3: COMPUTE EMBEDDINGS")
    print("="*60)
    cache_dir = output_dir if use_embedding_cache else None
    embeddings, valid_frame_indices = compute_multiview_embeddings(frame_data, processor, model, device, batch_size, cache_dir=cache_dir)

    if len(embeddings) == 0:
        print("No valid multiview embeddings. Exiting.")
        return

    print(f"  Embeddings shape: {embeddings.shape}")

    # Map valid_frame_indices to valid frame_data for quality metrics
    valid_frame_data = [frame_data[i] for i in valid_frame_indices]
    valid_paths = [f['path'] for f in valid_frame_data]  # Extract paths for later use

    # STEP 4: COMPUTE QUALITY METRICS (reuses pre-loaded images from frame_data)
    print("\n" + "="*60)
    print("STEP 4: COMPUTE QUALITY METRICS")
    print("="*60)
    quality_metrics = compute_quality_metrics_batch(valid_frame_data, anisotropy_threshold=anisotropy_threshold, cache_blurry_samples=cache_blurry_num_samples)

    # Report corrupted images
    if len(quality_metrics['corrupted_images']) > 0:
        print(f"\n⚠️  Warning: {len(quality_metrics['corrupted_images'])} corrupted/unreadable images skipped")
        print(f"   These images will be excluded from quality metrics analysis")

    # STEP 5: UMAP + HDBSCAN CLUSTERING
    print("\n" + "="*60)
    print("STEP 5: UMAP + HDBSCAN CLUSTERING")
    print("="*60)
    activity_labels, umap_embeddings, cluster_quality = cluster_activities_with_umap_hdbscan(embeddings, n_components=n_components, min_cluster_size=min_cluster_size, min_samples=min_samples)

    # Count activities (excluding noise)
    unique_activities = np.unique(activity_labels)
    n_activities = len([a for a in unique_activities if a != -1])
    n_noise = np.sum(activity_labels == -1)

    print(f"  ✅ Found {n_activities} distinct activity clusters")
    if n_noise > 0:
        print(f"  ℹ️  {n_noise} frames marked as noise/outliers (not enough similarity to form clusters)")

    # STEP 6: CATEGORIZE LIGHTING
    print("\n" + "="*60)
    print("STEP 6: CATEGORIZE LIGHTING")
    print("="*60)
    lighting_labels = categorize_lighting(quality_metrics['brightness'])

    dark_count = np.sum(lighting_labels == 0)
    medium_count = np.sum(lighting_labels == 1)
    bright_count = np.sum(lighting_labels == 2)

    print(f"  Dark frames (<80): {dark_count}")
    print(f"  Medium frames (80-175): {medium_count}")
    print(f"  Bright frames (>175): {bright_count}")

    # STEP 7: ANALYZE BLUR (motion blur via anisotropy)
    print("\n" + "="*60)
    print("STEP 7: ANALYZE BLUR (Motion Blur)")
    print("="*60)

    # Blurry = high anisotropy (motion blur, camera shake)
    blurry_mask = (quality_metrics['anisotropy'] > anisotropy_threshold)
    blurry_count = np.sum(blurry_mask)
    sharp_count = len(quality_metrics['anisotropy']) - blurry_count

    print(f"  Blurry frames (anisotropy >{anisotropy_threshold}): {blurry_count}")
    print(f"  Non-blurry frames: {sharp_count}")

    # STEP 8: SUMMARIZE ACTIVITIES
    print("\n" + "="*60)
    print("STEP 8: ACTIVITY SUMMARY")
    print("="*60)
    activity_names = {}
    unique_activities = np.unique(activity_labels)

    # Skip noise (-1)
    cluster_activities = [a for a in unique_activities if a != -1]

    for activity_id in cluster_activities:
        activity_mask = activity_labels == activity_id
        n_frames = np.sum(activity_mask)
        activity_names[activity_id] = f"Activity {activity_id}"
        print(f"  Activity {activity_id}: {n_frames} frames")

    # Add noise label
    if -1 in unique_activities:
        activity_names[-1] = "Noise/Outliers"
        print(f"  Noise/Outliers: {n_noise} frames")

    # STEP 9: COMPUTE QUALITY SCORE
    if len(quality_metrics['anisotropy']) == 0:
        # No valid frames for quality metrics
        quality_score = 0
        print("  ⚠️  No valid quality metrics - all images corrupted or failed to load")
    else:
        median_brightness = np.median(quality_metrics['brightness'])
        median_contrast = np.median(quality_metrics['contrast'])
        median_anisotropy = np.median(quality_metrics['anisotropy'])

        # Anisotropy score (lower is better - inverted)
        anisotropy_score = max(0, 10 - (median_anisotropy / 5) * 10)  # Lower anisotropy = sharper
        brightness_score = 10 - abs(median_brightness - 127) / 127 * 10
        contrast_score = min(10, median_contrast / 50 * 10)

        quality_score = (anisotropy_score + brightness_score + contrast_score) / 3

    # STEP 10: GENERATE RECOMMENDATIONS
    recommendations = []

    if len(valid_paths) == 0:
        recommendations.append("❌ CRITICAL: No valid frames found! All images are corrupted or unreadable.")
        dark_pct = 0
        blurry_pct = 0
        noise_pct = 0
    else:
        dark_pct = dark_count / len(valid_paths) * 100
        blurry_pct = blurry_count / len(valid_paths) * 100
        noise_pct = n_noise / len(valid_paths) * 100

    if dark_pct > 60:
        recommendations.append(f"⚠ {dark_pct:.0f}% frames are low-light. Consider adding well-lit frames or dropping extreme dark frames.")
    if blurry_pct > 40:
        recommendations.append(f"⚠ {blurry_pct:.0f}% frames are blurry (motion blur). Use higher shutter speed, stabilize camera, or reduce camera movement. Consider dropping these frames.")
    if n_activities < 3:
        recommendations.append(f"⚠ Only {n_activities} activity patterns detected. Capture more diverse work activities/scenarios.")
    if len(embeddings) < 50:
        recommendations.append(f"⚠ Only {len(embeddings)} valid frames. Minimum 100-200 recommended for robust YOLO training.")
    if noise_pct > 30:
        recommendations.append(f"⚠ {noise_pct:.0f}% frames marked as noise/outliers. Review noise montage - may indicate inconsistent capture conditions or rare activities.")
    elif noise_pct > 0:
        recommendations.append(f"ℹ️ {noise_pct:.1f}% frames are noise/outliers (rare activities or transitions). Review noise montage to verify quality.")

    if not recommendations:
        recommendations.append("✅ Dataset quality looks good! Proceed with annotation.")

    # STEP 11: GENERATE PDF REPORT
    print("\n" + "="*70)
    print("STEP 11: GENERATING PDF REPORT")
    print("="*70)

    # Create temp directory for charts
    temp_dir = Path("temp_preannotation_charts")
    temp_dir.mkdir(exist_ok=True)

    # Generate visualizations
    print("\nGenerating visualizations...")
    quality_chart_path = temp_dir / "quality_metrics.png"
    umap_chart_path = temp_dir / "umap_diversity.png"
    heatmap_chart_path = temp_dir / "coverage_heatmap.png"

    create_quality_chart(quality_metrics, quality_chart_path, anisotropy_threshold=anisotropy_threshold)
    plt.close('all') 
    gc.collect()

    create_umap_visualization(umap_embeddings, activity_labels, umap_chart_path)  # Use UMAP embeddings for visualization
    plt.close('all')
    gc.collect()

    create_coverage_heatmap(activity_labels, lighting_labels, heatmap_chart_path)
    plt.close('all')
    gc.collect()

    # Generate activity montages
    print("\nGenerating activity montages...")
    for activity_id in cluster_activities:
        activity_mask = activity_labels == activity_id
        activity_paths = [valid_paths[i] for i, mask_val in enumerate(activity_mask) if mask_val]

        montage = create_activity_montage(activity_paths, activity_id, max_samples=activity_num_samples, grid_cols=grid_cols_activities, target_height=250)
        montage_path = temp_dir / f"activity_{activity_id}_montage.png"
        montage.save(montage_path)
        del montage  # FREE MEMORY
        gc.collect()
        print(f"  ✅ Activity {activity_id} montage created")

    # Generate noise/outlier montage if present
    if -1 in unique_activities and n_noise > 0:
        noise_mask = activity_labels == -1
        noise_paths = [valid_paths[i] for i, mask_val in enumerate(noise_mask) if mask_val]

        # Show 20 samples in montage (larger thumbnails for better visibility)
        montage = create_activity_montage(noise_paths, -1, max_samples=outliers_num_samples, grid_cols=grid_cols_activities, target_height=250)
        montage_path = temp_dir / f"activity_-1_montage.png"
        montage.save(montage_path)
        del montage  # FREE MEMORY
        gc.collect()
        print(f"  ✅ Noise/Outliers montage created ({n_noise} frames)")

        # Save list of all noise/outlier filenames for PDF
        noise_filenames = [str(p.name) for p in noise_paths]
        noise_filelist_path = temp_dir / "noise_filenames.txt"
        with open(noise_filelist_path, 'w') as f:
            for fname in noise_filenames:
                f.write(fname + '\n')
        print(f"  ✅ Noise/Outliers filename list saved ({len(noise_filenames)} files)")

    # Generate blurry images montage
    print("\nGenerating blurry images montage...")
    blurry_montage = create_blurry_images_montage(quality_metrics['blurry_image_cache'], grid_cols=grid_cols_blurry, target_height=250)
    if blurry_montage is not None:
        blurry_montage_path = temp_dir / "blurry_images_montage.png"
        blurry_montage.save(blurry_montage_path)
        del blurry_montage  # FREE MEMORY
        gc.collect()
        print("  ✅ Blurry images montage created")
    else:
        print("  ℹ️  No blurry images found - skipping montage")

    print("  ✅ All visualizations generated")

    # Prepare analysis data for PDF
    analysis_data = {
        'quality_metrics': quality_metrics,
        'activity_labels': activity_labels,
        'lighting_labels': lighting_labels,
        'activity_names': activity_names,
        'valid_paths': valid_paths,
        'anisotropy_threshold': anisotropy_threshold,
        'grid_cols_activities': grid_cols_activities,
        'grid_cols_blurry': grid_cols_blurry,
        'recommendations': recommendations,
        'cluster_quality': cluster_quality,
        'summary_stats': {
            'total_frames': len(image_paths),
            'valid_frames': len(valid_paths),
            'n_activities': n_activities,
            'dark_count': dark_count,
            'dark_pct': dark_pct,
            'medium_count': medium_count,
            'medium_pct': (medium_count / len(valid_paths) * 100) if len(valid_paths) > 0 else 0,
            'bright_count': bright_count,
            'bright_pct': (bright_count / len(valid_paths) * 100) if len(valid_paths) > 0 else 0,
            'blurry_count': blurry_count,
            'blurry_pct': blurry_pct,
            'sharp_count': sharp_count,
            'sharp_pct': (sharp_count / len(valid_paths) * 100) if len(valid_paths) > 0 else 0,
            'quality_score': quality_score
        }
    }

    # Generate PDF report
    pdf_path = output_dir / pdf_name
    generate_pdf_report(analysis_data, pdf_path, temp_dir, image_quality=pdf_image_quality)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Output: {pdf_path}")
    print("="*60)

    # Clean up cache files (optional - user can keep them for faster re-runs)
    if use_embedding_cache:
        cache_file = output_dir / "temp_multiview_emb.npy"
        cache_indices_file = output_dir / "temp_multiview_emb_indices.npy"
        if cache_file.exists() or cache_indices_file.exists():
            print(f"\nℹ️  Embedding cache files saved in {output_dir}")
            print("   Delete temp_multiview_emb.* files to force recomputation on next run")

    # Clean up GPU memory
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
