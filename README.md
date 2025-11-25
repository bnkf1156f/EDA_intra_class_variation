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

## 🚀 Master Scripts (Automated Pipeline)

Two master scripts are provided to run the complete pipeline automatically with memory management and GPU cooling.

### Script: 4. master_script_pretrain.py
**Automated PRE-TRAINING pipeline** - Analyzes annotated dataset before model training

#### Features
- ✅ **Automatic Step Execution**: Runs all 3 pipeline scripts in sequence
- 🧹 **Memory Management**: Aggressive cleanup between steps (GPU + RAM)
- ❄️ **Cooling Breaks**: 10-second pauses to prevent laptop GPU thermal throttling
- 📊 **Resource Monitoring**: Real-time GPU/RAM usage tracking
- ⚠️ **Safety Checks**: Warns if memory usage >90% before continuing
- 🔄 **Error Handling**: Graceful cleanup on failures or user interruption

#### Quick Start
```bash
# Edit global variables in script, then run:
python "4. master_script_pretrain.py"
```

#### Configuration Variables
Edit these at the top of `main()` function:

```python
## GLOBAL VARIABLES PRE-TRAINING ##
imgs_label_path = "path/to/LabelledData"
class_names = ["class1", "class2", "class3"]
cropped_bbox_dir = "cropped_imgs_by_class"
batch_size = "32"
epsilon = "0.15"  # Ignored if auto_tune is used
min_pts = "3"
output_cluster_dir = "clustering_results"
max_cluster_samples = "15"
```

#### Pipeline Steps
1. **Crop Extraction** (`1. ann_txt_files_crop_bbox.py`)
   - Extracts bounding boxes from annotated images
   - Validates dataset integrity
   - Organizes crops by class

2. **Embedding Generation** (`2. save_dinov2_embeddings_per_class.py`)
   - Generates 768D DINOv2 embeddings
   - GPU-accelerated with mixed precision
   - ~5-10 minutes for 1000 images on RTX 4060

3. **Clustering Analysis** (`3. clustering_of_classes_embeddings.py`)
   - Auto-tunes DBSCAN epsilon per class
   - Generates visualizations and montages
   - Cross-class separability analysis

#### Typical Runtime
- **Small Dataset** (500-1000 images): 5-10 minutes
- **Medium Dataset** (2000-5000 images): 15-30 minutes
- **Large Dataset** (10000+ images): 45+ minutes

*Times based on RTX 4060 Laptop GPU*

---

### Script: 4. master_script_dinov2_posttrain.py
**Automated POST-TRAINING pipeline** - Analyzes trained model detections

#### Features
Same as pre-training script, plus:
- 🎥 **Video Processing**: Extracts detections from video files
- 🎯 **Smart Sampling**: Uniform frame selection per class
- ⚡ **Frame Stride**: Process every Nth frame for efficiency

#### Quick Start
```bash
# Edit global variables in script, then run:
python "4. master_script_dinov2_posttrain.py"
```

#### Configuration Variables
```python
## GLOBAL VARIABLES POST-TRAINING ##
model_path = "path/to/model.pt"
video_path = "path/to/video.mp4"
classes_space_separated = ["board", "screw", "screw_holder"]
per_class_num_frames = "1000"  # Target samples per class
conf_thresh = "0.4"
frame_stride_per_video = "3"  # Process every 3rd frame
cropped_bbox_dir = "cropped_images"
batch_size = "32"
epsilon = "0.15"
min_pts = "3"
output_cluster_dir = "clustering_results"
max_cluster_samples = "5"
```

#### Pipeline Steps
1. **Detection & Cropping** (`1. yolo_model_crop_bbox_per_class.py`)
   - Runs YOLOv8 inference on video
   - Applies frame stride (e.g., every 3rd frame)
   - Smart sampling: uniformly extracts target number of crops per class
   - **⚠️ Most time-consuming step** (10-15 minutes for 20-min video)

2. **Embedding Generation** (same as pre-training)
3. **Clustering Analysis** (same as pre-training)

#### Typical Runtime
**Example**: 22-minute video, 3 classes, 1000 samples/class, frame_stride=3
- Step 1 (Detection): ~12-15 minutes
- Step 2 (Embeddings): ~3-5 minutes
- Step 3 (Clustering): ~1-2 minutes
- **Total: ~18-20 minutes** on RTX 4060 Laptop GPU

#### Frame Stride Recommendations
| Video FPS | Stride | Effective FPS | Use Case |
|-----------|--------|---------------|----------|
| 30 | 2 | 15 | High variation scenes |
| 30 | 3 | 10 | **Recommended default** |
| 30 | 5 | 6 | Slow-moving objects |
| 60 | 6 | 10 | High FPS cameras |

---

### Master Script Output

Both scripts provide detailed progress tracking:

```
==============================================================
INTRA-CLASS VARIATION EDA PIPELINE
==============================================================
  GPU: NVIDIA GeForce RTX 4060 Laptop GPU
  VRAM: 8.0GB
==============================================================

Pipeline Configuration:
  Video: till_20_oct.mp4 (22 mins)
  Classes: board, screw, screw_holder
  Frames per class: 1000
  Total estimated frames: 3000

==============================================================
STEP 1/3: EXTRACTING BOUNDING BOXES FROM VIDEO
⚠️  This step is time-consuming
==============================================================
Running: 1. yolo_model_crop_bbox_per_class.py
...
✅ 1. yolo_model_crop_bbox_per_class.py completed successfully

🧹 Clearing memory...
   GPU Memory: 0.23GB allocated, 0.50GB reserved
   RAM Usage: 45.2% (7.2GB / 16.0GB)
   ✅ Memory cleared

❄️  Cooling break: 10 seconds...
   This prevents thermal throttling on your laptop GPU
   ✅ Cooling break complete

==============================================================
STEP 2/3: GENERATING DINOV2 EMBEDDINGS
==============================================================
⚠️  This step is GPU-intensive
   Processing ~3000 images
...
```

### Memory Management Features

#### Automatic Cleanup
- **GPU Cache**: `torch.cuda.empty_cache()` after each step
- **RAM**: Python garbage collection via `gc.collect()`
- **Monitoring**: Real-time memory usage display

#### Safety Checks
```
⚠️  WARNING: RAM usage is very high (>90%)
   Consider closing other applications
   Continue anyway? (y/n):
```

#### Cooling Breaks
- **Duration**: 10 seconds between steps (configurable)
- **Purpose**: Prevent thermal throttling on laptop GPUs
- **When**: After Steps 1 and 2 (not needed after final step)

---

### Troubleshooting Master Scripts

#### Out of Memory (GPU)
1. **Reduce batch size**: Change `batch_size = "16"` or `"8"`
2. **Reduce samples**: Lower `per_class_num_frames` (post-training)
3. **Check GPU usage**: Close other GPU applications (browsers, games)

#### Out of Memory (RAM)
1. **Close applications**: Browsers, IDEs, etc.
2. **Process smaller batches**: Run pipeline on subset of classes
3. **Increase swap space**: Add virtual memory (Windows/Linux)

#### Script Hangs or Crashes
1. **Increase cooling breaks**: Edit `cooling_break(10)` to `cooling_break(30)`
2. **Monitor temperatures**: Use `nvidia-smi` or GPU-Z
3. **Check file paths**: Ensure all paths in global variables exist

#### Slow Performance
- **Video too long?**: Increase `frame_stride` to 5 or 10
- **Too many classes?**: Process classes in batches (edit `classes` list)
- **GPU throttling?**: Increase cooling break duration
- **Disk I/O bottleneck?**: Use SSD instead of HDD for output directories

---

## Individual Scripts Documentation

### Script: 1. ann_txt_files_crop_bbox.py
**Stage: PRE-TRAINING** - Analyze annotated data before model training

#### Usage
```bash
python "1. ann_txt_files_crop_bbox.py" \
    --imgs_label_path "path/to/LabelledData" \
    --classes 0 1 2 3 4 \
    --class_ids_to_names 0 board 1 screw 2 screw_holder 3 tape 4 case \
    --output_dir cropped_imgs_by_class
```

#### Parameters
| Parameter | Description |
|-----------|-------------|
| `--imgs_label_path` | Folder with image-label pairs (.png/.jpg + .txt) |
| `--classes` | Space-separated class IDs to extract |
| `--class_ids_to_names` | ID-name mapping: `id1 name1 id2 name2 ...` |
| `--output_dir` | Output directory (default: `cropped_imgs_by_class`) |

#### What It Does
1. Validates dataset (checks image-label pairing)
2. Verifies class mapping completeness
3. Converts YOLO normalized coords → pixel coords
4. Crops bounding boxes per class
5. Saves to class-specific folders

#### Output Structure
```
cropped_imgs_by_class/
├── board/
│   ├── frame_001_crop_0.png
│   └── ...
├── screw/
└── ...
```

#### Common Issues
- **"Class ID(s) not in mapping"**: Add missing IDs to `--class_ids_to_names`
- **"TXT files missing images"**: Fix dataset consistency
- **Empty crops**: Check annotation normalization in .txt files

---

### Script: 1. yolo_model_crop_bbox_per_class.py
**Stage: POST-TRAINING** - Verify trained model detections

#### Usage
```bash
python "1. yolo_model_crop_bbox_per_class.py" \
    --model "path/to/model.pt" \
    --video "path/to/video.mp4" \
    --classes class1 class2 class3 \
    --frame_stride 3 \
    --num_frames 100 \
    --conf_thresh 0.4 \
    --output "cropped_images"
```

#### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | - | Path to YOLOv8 .pt model |
| `--video` | - | Input video file |
| `--classes` | - | Class names to detect and crop |
| `--num_frames` | 100 | Samples per class to extract uniformly |
| `--frame_stride` | 3 | Process every Nth frame (speed optimization) |
| `--conf_thresh` | 0.4 | Detection confidence threshold |
| `--output` | `cropped_images` | Output directory |

#### Frame Stride Guide (30fps video)
- `stride=2`: 15 fps (every other frame)
- `stride=3`: 10 fps (every 3rd frame) - **recommended**
- `stride=5`: 6 fps (every 5th frame)

#### What It Does
1. Processes video with frame stride (e.g., every 3rd frame)
2. Runs YOLO detection on selected frames
3. Collects annotations for target classes
4. Smart sampling: uniformly extracts cropped bboxes per class
   - Example: 1000 detections of class X, target 100 samples → samples every 10th detection
5. Saves cropped detections per class

#### Output Structure
```
cropped_images/
├── class1/
│   ├── frame_001_1.png
│   └── ...
└── class2/
```

---

### Script: 2. save_dinov2_embeddings_per_class.py
**Generates semantic embeddings from cropped images**

This script processes cropped object images and uses DINOv2 model to generate embeddings for each class. Optimized for GPU usage with memory management and mixed precision support.

#### Usage
```bash
python "2. save_dinov2_embeddings_per_class.py" --root ./cropped_images --batch 32
```

#### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--root` | `cropped_images` | Root folder with class subfolders |
| `--batch` | 32 | Batch size for inference |
| `--save_suffix` | `embeddings_dinov2.npy` | Filename for embeddings |

#### Features
1. **GPU Optimization**:
   - Automatic mixed precision (FP16) for faster processing
   - Smart GPU memory management with garbage collection
   - Real-time GPU memory usage monitoring
   - Uses up to 95% of available GPU memory
   - Automatic cache clearing when memory usage is high

2. **Batch Processing**:
   - Processes images in configurable batch sizes
   - Efficient memory usage with immediate CPU offloading
   - Handles corrupted/unreadable images gracefully
   - Progress tracking with detailed batch statistics

3. **Embedding Generation**:
   - Uses facebook/dinov2-base model
   - Generates 768-dimensional feature vectors
   - Applies mean pooling over sequence dimension
   - Saves embeddings in numpy format

4. **Progress Tracking**:
   - Shows GPU information at startup
   - Displays real-time batch processing progress
   - Reports GPU memory usage
   - Shows embedding statistics per class

#### Directory Structure
```
--root/
    class1/
        frame_001.png
        frame_002.png
        ...
        embeddings_dinov2.npy           # N×768 embedding matrix
        embeddings_dinov2_image_list.txt # Maps embeddings to images
    class2/
        ...
```

#### Technical Details
- **Model**: facebook/dinov2-base
- **Embedding dimension**: 768
- **Input format**: RGB images
- **Output format**: numpy array (.npy)
- **GPU memory limit**: 95% of available memory
- **Error handling**: Skips corrupted images, tracks valid images
- **Image-Embedding Alignment**: Saves `embeddings_dinov2_image_list.txt` for correct mapping
- **File overwrite protection**: Asks before overwriting existing embeddings

---

### Script: 3. clustering_of_classes_embeddings.py
**Discovers sub-groups and outliers using DBSCAN**

#### Basic Usage
```bash
# Auto-tuning (recommended)
python "3. clustering_of_classes_embeddings.py" --root ./cropped_images --auto_tune --min_samples 3

# Manual epsilon
python "3. clustering_of_classes_embeddings.py" --root ./cropped_images --eps 0.15 --min_samples 3

# Full analysis with visualizations
python "3. clustering_of_classes_embeddings.py" --root ./cropped_images --auto_tune --min_samples 3 --save_montage --cross_class
```

#### Parameters
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

#### Parameter Guide
**Epsilon (eps)**:
- `0.1`: Strict (tight clusters, many outliers)
- `0.15`: Balanced - **recommended starting point**
- `0.2-0.3`: Lenient (broader groups, fewer outliers)
- `--auto_tune`: Uses 90th percentile k-NN distance

**Min Samples**:
- `2`: Very sensitive (small groups)
- `3`: Balanced - **recommended**
- `5+`: Conservative (well-defined clusters only)

#### Working Process

1. **Loading Phase**:
   - Loads embeddings for each class
   - Loads image mapping file (`embeddings_dinov2_image_list.txt`)
   - Falls back to sorted image list if mapping missing (with warning)
   - Validates data consistency

2. **Auto-Tuning (Optional)**:
   - Uses k-Nearest Neighbors for pairwise distances
   - Selects 90th percentile of k-th nearest neighbor distances
   - Provides class-specific optimal eps values

3. **Dimensionality Reduction**:
   - Applies UMAP to reduce 768D → 2D for visualization
   - Uses cosine distance metric
   - Random state = 42 for reproducibility

4. **Clustering**:
   - Normalizes embeddings using L2 norm
   - Applies DBSCAN with cosine distance
   - Identifies clusters and outliers (label = -1)
   - Reports cluster counts and outlier rates

5. **Visualization**:
   - Scatter plots with UMAP projections
   - Cluster size distribution bar charts
   - Overview grid comparing all classes
   - Optional: Cross-class separability analysis

6. **Sample Extraction**:
   - Saves sample images organized by cluster
   - Creates montages showing representatives
   - Preserves original filenames for traceability

#### Output Files
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

#### Interpreting Results
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
screw_holder: 7 clusters, 12 outliers (6.0%)   → High diversity
```

#### Common Issues
- **All outliers (0 clusters)**: Increase `--eps` or use `--auto_tune`
- **Only 1 cluster**: Decrease `--eps` (try 0.1)
- **Too many small clusters**: Increase `--min_samples` or `--eps`
- **Image count mismatch**: Re-run embedding generation
- **"No mapping file found" warning**: Re-run Script 2

---

## Complete Workflows

### Pre-Training Analysis (Automated)
```bash
# Recommended: Use master script
python "4. master_script_pretrain.py"

# Edit these variables in the script before running:
# - imgs_label_path
# - class_names
# - cropped_bbox_dir
# - output_cluster_dir
```

### Pre-Training Analysis (Manual)
```bash
# 1. Extract crops from annotated data
python "1. ann_txt_files_crop_bbox.py" \
    --imgs_label_path "./LabelledData" \
    --classes 0 1 2 \
    --class_ids_to_names 0 board 1 screw 2 screw_holder

# 2. Generate embeddings
python "2. save_dinov2_embeddings_per_class.py" \
    --root ./cropped_imgs_by_class --batch 32

# 3. Analyze clusters
python "3. clustering_of_classes_embeddings.py" \
    --root ./cropped_imgs_by_class \
    --auto_tune --min_samples 3 --save_montage
```

### Post-Training Analysis (Automated)
```bash
# Recommended: Use master script
python "4. master_script_dinov2_posttrain.py"

# Edit these variables in the script before running:
# - model_path
# - video_path
# - classes_space_separated
# - per_class_num_frames
```

### Post-Training Analysis (Manual)
```bash
# 1. Extract crops using trained model
python "1. yolo_model_crop_bbox_per_class.py" \
    --model "model.pt" \
    --video "video.mp4" \
    --classes board screw screw_holder \
    --num_frames 200

# 2-3. Same as pre-training (use --root ./cropped_images)
```

---

## Pipeline Summary

| Script | Input | Output | Purpose |
|--------|-------|--------|---------|
| **4. master_script_pretrain.py** | Config variables | Complete analysis | **Automated pre-training pipeline** |
| **4. master_script_dinov2_posttrain.py** | Config variables | Complete analysis | **Automated post-training pipeline** |
| 1. ann_txt_files_crop_bbox.py | Annotated images + TXT | Cropped images | Pre-training data inspection |
| 1. yolo_model_crop_bbox_per_class.py | Video + YOLO model | Cropped images | Post-training detection verification |
| 2. save_dinov2_embeddings_per_class.py | Cropped images | 768D embeddings | Semantic representations |
| 3. clustering_of_classes_embeddings.py | Embeddings + images | Clusters + insights | Intra-class variation analysis |

---

## Key Takeaways
- 🚀 **Use master scripts** for automated pipeline execution with memory management
- **Pre-training**: Validate annotation quality and understand labeled patterns
- **Post-training**: Verify model detections match expectations
- **Clustering insights**: Identify sub-groups, outliers, and data quality issues
- **Actionable**: Adjust augmentation, split classes, or fix annotations based on results
- ❄️ **Laptop GPU users**: Master scripts include cooling breaks to prevent throttling

## Dependencies
- Python >= 3.8
- PyTorch >= 2.1
- CUDA-compatible GPU (recommended)
- See [requirements.txt](https://github.com/bnkf1156f/EDA_intra_class_variation/blob/main/requirements.txt)

## References
* DINOv2 Clustering: https://medium.com/@EnginDenizTangut/%EF%B8%8F-image-clustering-with-dinov2-and-hdbscan-a-hands-on-guide-35c6e29036f2
* DBSCAN Understanding: https://medium.com/@sachinsoni600517/clustering-like-a-pro-a-beginners-guide-to-dbscan-6c8274c362c4
* YOLOv8 Embeddings: https://docs.ultralytics.com/modes/predict/#inference-arguments
* Post-training Pipeline: https://medium.com/@albertferrevidal/facings-product-identifier-using-yolov8-and-image-embeddings-d3ca34463022
