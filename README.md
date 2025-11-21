# EDA INTRA-CLASS VARIATION Scripts Documentation

## Pipeline Overview --- PRE-TRAINING

Analyzes intra-class variations in object detection datasets through:
1. **Cropped Images Extraction**: From annotated images (pre-training)
2. **Embedding Generation using DINOv2**: Using DINOv2 to create 768D semantic representations
3. **Clustering Analysis**: DBSCAN to discover sub-groups and outliers within each class

## Pipeline Overview --- POST-TRAINING
1. **Cropped Images Extraction**: Using YOLOv8 trained model, extract smart class frames and crop accordingly
2. **Embedding Generation using DINOv2**: Using DINOv2 to create 768D semantic representations
3. **Clustering Analysis**: DBSCAN to discover sub-groups and outliers within each class

**Purpose**: Understand data patterns, assess dataset and model quality, and identify class diversity before/after training.

---

## Script: 1. ann_txt_files_crop_bbox.py
**Stage: PRE-TRAINING** - Analyze annotated data before model training

### Usage
```bash
python `1. ann_txt_files_crop_bbox.py` \
    --imgs_label_path "path/to/LabelledData" \
    --classes 0 1 2 3 4 \
    --class_ids_to_names 0 board 1 screw 2 screw_holder 3 tape 4 case \
    --output_dir cropped_imgs_by_class
```

### Parameters
| Parameter | Description |
|-----------|-------------|
| `--imgs_label_path` | Folder with image-label pairs (.png/.jpg + .txt) |
| `--classes` | Space-separated class IDs to extract |
| `--class_ids_to_names` | ID-name mapping: `id1 name1 id2 name2 ...` |
| `--output_dir` | Output directory (default: `cropped_imgs_by_class`) |

### What It Does
1. Validates dataset (checks image-label pairing)
2. Verifies class mapping completeness
3. Converts YOLO normalized coords → pixel coords
4. Crops bounding boxes per class
5. Saves to class-specific folders

### Output Structure
```
cropped_imgs_by_class/
├── board/
│   ├── frame_001_crop_0.png
│   └── ...
├── screw/
└── ...
```

### Common Issues
- **"Class ID(s) not in mapping"**: Add missing IDs to `--class_ids_to_names`
- **"TXT files missing images"**: Fix dataset consistency
- **Empty crops**: Check annotation normalization in .txt files

---

## Script: 1. yolo_model_crop_bbox_per_class.py
**Stage: POST-TRAINING** - Verify trained model detections (optional)

### Usage
```bash
python `1. yolo_model_crop_bbox.py` \
    --model "path/to/model.pt" \
    --video "path/to/video.mp4" \
    --classes class1 class2 class3 \
    --frame_stride 3 \
    --num_frames 100 \
    --conf_thresh 0.4 \
    --output "cropped_images"
```

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | - | Path to YOLOv8 .pt model |
| `--video` | - | Input video file |
| `--classes` | - | Class names to detect and crop |
| `--num_frames` | 100 | Samples per class to extract uniformly |
| `--frame_stride` | 3 | Process every Nth frame (speed optimization as per frame minor changes assumed) |
| `--conf_thresh` | 0.4 | Detection confidence threshold |
| `--output` | Output Directory to save the cropped object bboxes resultant |

### Frame Stride Guide (30fps video)
- `stride=2`: 15 fps (every other frame)
- `stride=3`: 10 fps (every 3rd frame) - **recommended**
- `stride=5`: 6 fps (every 5th frame)

### What It Does
1. Processes video with frame stride (e.g., every 3rd frame)
2. Runs YOLO detection on selected frames
3. Collects annotations for target classes
4. Smart sampling: uniformly extract cropped object bboxes for each class with class_frame_stride (e.g., 1000 annotations of X class then for 100 samples to collect for this class, 1000/100 so 10th frame stride of X class)
5. Saves cropped detections per class

### Output Structure
```
cropped_images/
├── class1/
│   ├── frame_001_1.png
│   └── ...
└── class2/
```

---

## Script: 2. save_dinov2_embeddings_per_class.py
**Generates semantic embeddings from cropped images**
This script processes cropped object images, and uses DINOv2 model to generate embeddings for each class. It's optimized for GPU usage with memory management and mixed precision support.

### Usage
```bash
python `2. save_dinov2_embeddings_per_class.py` --root ./cropped_images --batch 32
```

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--root` | `cropped_images` | Root folder with class subfolders |
| `--batch` | 32 | Batch size for inference |
| `--save_suffix` | `embeddings_dinov2.npy` | Filename for embeddings saved in each class folder (default: embeddings_dinov2.npy) | Saved as <class>/<save_suffix>

### Features
1. **GPU Optimization**:
   - Automatic mixed precision (FP16) for faster processing
   - Smart GPU memory management with garbage collection
   - Real-time GPU memory usage monitoring
   - Uses up to 95% of available GPU memory (can adjust for lower-memory GPUs)
   - Automatic cache clearing when memory usage is high

2. **Batch Processing**:
   - Processes images in configurable batch sizes
   - Efficient memory usage with immediate CPU offloading
   - Handles corrupted or unreadable images gracefully
   - Shows progress with detailed batch statistics

3. **Embedding Generation**:
   - Uses facebook/dinov2-base model
   - Generates 768-dimensional feature vectors
   - Applies mean pooling over sequence dimension
   - Saves embeddings in numpy format for easy loading

4. **Progress Tracking**:
   - Shows GPU information at startup
   - Displays real-time batch processing progress
   - Reports GPU memory usage
   - Shows embedding statistics per class

### Directory Structure
```
--root/
    class1/
        frame_001.png
        frame_002.png
        ...
        embeddings_dinov2.npy           # Contains N×768 embedding matrix
        embeddings_dinov2_image_list.txt # Maps each embedding to its image file
    class2/
        ...
```

### Technical Details
- Model: facebook/dinov2-base
- Embedding dimension: 768
- Input format: RGB images
- Output format: numpy array (.npy)
- GPU memory limit: 95% of available memory
- Error handling: Skips corrupted images and tracks valid images
- **Image-Embedding Alignment**: Saves `embeddings_dinov2_image_list.txt` to preserve correct mapping
- File overwrite protection: Asks before overwriting existing embeddings

---

## Script: 3. clustering_of_classes_embeddings.py
**Discovers sub-groups and outliers using DBSCAN**

### Basic Usage
```bash
# Auto-tuning (recommended)
python `3. clustering_of_classes_embeddings.py` --root ./cropped_images --auto_tune --min_samples 3

# Manual epsilon
python `3. clustering_of_classes_embeddings.py` --root ./cropped_images --eps 0.15 --min_samples 3

# Full analysis with visualizations
python `3. clustering_of_classes_embeddings.py` --root ./cropped_images --auto_tune --min_samples 3 --save_montage --cross_class
```

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--root` | `cropped_images` | Root folder with embeddings |
| `--eps` | 0.15 | Max distance for neighbors (cosine) |
| `--min_samples` | 3 | Min points to form cluster |
| `--auto_tune` | False | Auto-find optimal eps per class |
| `--output_dir` | `clustering_results` | Output directory |
| `--max_samples` | 5 | Sample images per cluster |
| `--cross_class` | False | Cross-class separability analysis |
| `--save_montage` | False | Create image grid montages |

### Parameter Guide
**Epsilon (eps)**:
- `0.1`: Strict (tight clusters, many outliers)
- `0.15`: Balanced - **recommended starting point**
- `0.2-0.3`: Lenient (broader groups, fewer outliers)
- `--auto_tune`: Uses 90th percentile k-NN distance

**Min Samples**:
- `2`: Very sensitive (small groups)
- `3`: Balanced - **recommended**
- `5+`: Conservative (well-defined clusters only)

### Working Process

1. **Loading Phase**:
   - Loads embeddings for each class
   - Loads image mapping file (`embeddings_dinov2_image_list.txt`) for correct alignment
   - Falls back to sorted image list if mapping file is missing (with warning)
   - Validates data consistency (images count = embeddings count)

2. **Auto-Tuning (Optional)**:
   - Uses k-Nearest Neighbors to compute pairwise distances
   - Selects 90th percentile of k-th nearest neighbor distances
   - Provides class-specific optimal eps values

3. **Dimensionality Reduction**:
   - Applies UMAP to reduce 768D embeddings to 2D for visualization
   - Uses cosine distance metric (better for normalized embeddings)
   - Random state = 42 for reproducibility

4. **Clustering**:
   - Normalizes embeddings using L2 norm
   - Applies DBSCAN with cosine distance metric
   - Identifies clusters (dense regions) and outliers (label = -1)
   - Reports cluster counts and outlier rates

5. **Visualization**:
   - Creates scatter plots with UMAP projections
   - Generates cluster size distribution bar charts
   - Produces overview grid comparing all classes
   - Optional: Cross-class separability analysis

6. **Sample Extraction**:
   - Saves sample images organized by cluster
   - Creates montages showing representative samples
   - Preserves original image filenames for traceability

### Output Files
```
clustering_results/
├── {class}_clusters.png         # UMAP scatter + bar chart
├── {class}_montage.png          # Image grid (optional)
├── {class}_samples/             # Organized by cluster
│   ├── cluster_0/
│   ├── cluster_1/
│   └── outliers/
├── all_classes_overview.png     # Compare all classes
├── cross_class_separability.png # (optional)
└── cluster_statistics.csv       # Numerical summary
```

### Interpreting Results
**Cluster Counts**:
- 0 clusters → eps too strict
- 1 cluster → Homogeneous class
- 2-4 clusters → Normal variation
- 5+ clusters → High diversity (consider sub-classes)

**Outlier Rates**:
- <5% → Good data quality
- 5-10% → Moderate variation
- 10-20% → High variation or annotation issues
- >20% → Review data quality or adjust eps

**Example Output**:
```
board:        2 clusters,  4 outliers (2.0%)   → Consistent appearance
screw:        4 clusters, 21 outliers (10.5%)  → Multiple orientations
screw_holder: 7 clusters, 12 outliers (6.0%)   → High diversity (empty/full states)
```

### Common Issues
- **All outliers (0 clusters)**: Increase `--eps` or use `--auto_tune`
- **Only 1 cluster**: Decrease `--eps` (try 0.1)
- **Too many small clusters**: Increase `--min_samples` or `--eps`
- **Image count mismatch**: Re-run embedding generation (Script 2)
- **"No mapping file found" warning**: Re-run Script 2 to generate alignment file

---

## Complete Workflows

### Pre-Training Analysis
```bash
# 1. Extract crops from annotated data
python 1. ann_txt_files_crop_bbox.py \
    --imgs_label_path "./LabelledData" \
    --classes 0 1 2 \
    --class_ids_to_names 0 board 1 screw 2 screw_holder

# 2. Generate embeddings
python 2. save_dinov2_embeddings_per_class.py --root ./cropped_imgs_by_class --batch 32

# 3. Analyze clusters
python 3. clustering_of_classes_embeddings.py --root ./cropped_imgs_by_class --auto_tune --min_samples 3 --save_montage
```

### Post-Training Analysis (Optional)
```bash
# 1. Extract crops using trained model
python 1. yolo_model_crop_bbox.py \
    --model "model.pt" \
    --video "video.mp4" \
    --classes board screw screw_holder \
    --num_frames 200

# 2-3. Same as above (use --root ./cropped_images)
```

---

## Pipeline Summary

| Script | Input | Output | Purpose |
|--------|-------|--------|---------|
| 1. ann_txt_files_crop_bbox.py | Annotated images + TXT | Cropped images | **Pre-training** data inspection |
| 1. yolo_model_crop_bbox_per_class.py | Video + YOLO model | Cropped images | **Post-training** detection verification |
| 2. save_dinov2_embeddings_per_class.py | Cropped images | 768D embeddings | Semantic representations |
| 3. clustering_of_classes_embeddings.py | Embeddings + images list | Clusters + insights | Intra-class variation analysis |

---

## Key Takeaways
- **Pre-training**: Use Script 1a to validate annotation quality and understand labeled patterns
- **Post-training**: Use Script 1b to verify model detections match expectations
- **Clustering insights**: Identify sub-groups, outliers, and data quality issues
- **Actionable**: Adjust augmentation strategy, split classes, or fix annotations based on results

## Dependencies
- require pytorch >= 2.1
- refer to file: <a href=""https://github.com/bnkf1156f/EDA_intra_class_variation/blob/main/requirements.txt>requirements.txt<>

# REFERENCES
* DINOv2 Clustering: https://medium.com/@EnginDenizTangut/%EF%B8%8F-image-clustering-with-dinov2-and-hdbscan-a-hands-on-guide-35c6e29036f2
* DB SCAN Clustering Understanding: https://medium.com/@sachinsoni600517/clustering-like-a-pro-a-beginners-guide-to-dbscan-6c8274c362c4 
* Yolov8 embeddings: https://docs.ultralytics.com/modes/predict/#inference-arguments
* Post-training pipeline inspo: https://medium.com/@albertferrevidal/facings-product-identifier-using-yolov8-and-image-embeddings-d3ca34463022
