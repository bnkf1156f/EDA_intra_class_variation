# EDA INTRA-CLASS VARIATION Scripts Documentation

🔒 **Data Privacy**: All processing is 100% local - your proprietary frames, annotations, and models never leave your machine. No API calls, no telemetry, fully air-gappable after initial model downloads.

---

## Pipeline Overview --- PRE-ANNOTATION (Frame Quality Assessment)

**NEW**: Analyze frame quality **BEFORE sending to annotators** to ensure annotation effort is worthwhile.

1. **Scan Frames**: Load raw frame images (no annotations needed)
2. **DINOv2 Embeddings**: Generate semantic representations
3. **Activity Clustering**: Discover different scenarios/activities in frames
4. **Quality Metrics**: Analyze brightness, contrast, motion blur
5. **PDF Report**: Generate comprehensive quality assessment with visual montages

**Purpose**: Validate that extracted frames are diverse and high-quality before wasting annotation time on redundant or blurry frames.

---

## Pipeline Overview --- PRE-TRAINING (After Annotation)

Analyzes intra-class variations in annotated object detection datasets through:
1. **Cropped Images Extraction**: From annotated images (after annotation is complete)
2. **Embedding Generation using DINOv2**: Using DINOv2 to create 768D semantic representations
3. **Clustering Analysis**: DBSCAN to discover sub-groups and outliers within each class

## Pipeline Overview --- POST-TRAINING (After Model Training)
1. **Cropped Images Extraction**: Using YOLOv8 trained model, extract smart class frames and crop accordingly
2. **Embedding Generation using DINOv2**: Using DINOv2 to create 768D semantic representations
3. **Clustering Analysis**: DBSCAN to discover sub-groups and outliers within each class

**Purpose**: Understand data patterns, assess dataset and model quality, and identify class diversity before/after training.

---

## 🚀 Master Scripts (Automated Pipeline)

Three master scripts are provided: one for **pre-annotation** frame quality assessment, and two for **post-annotation** intra-class variation analysis.

---

### Script: 1. master_script_dinov2_PreAnnotate.py
**PRE-ANNOTATION Frame Quality Assessment** - Validate frames BEFORE sending to annotators

#### Purpose
Helps engineers assess whether extracted frames are worthy of annotation effort by analyzing:
- **Lighting diversity** (dark/medium/bright distribution)
- **Motion blur detection** (identify frames with camera shake or motion blur)
- **Activity/scenario diversity** (detect different work scenarios, object positions, camera angles)
- **Coverage gaps** (missing scenarios, imbalanced lighting)

**Use this when**: You have extracted frames or have images, and want to validate quality/diversity BEFORE annotation.

#### Features
- 🎯 **Activity Clustering**: Automatically groups frames by semantic similarity (different scenarios)
- 📊 **Quality Metrics**: Brightness, contrast, motion blur analysis
- 🖼️ **Visual Montages**: See sample images per activity cluster in PDF
- 🧠 **Adaptive Embeddings**: Uses pose features when persons detected, scene-only otherwise
- 📄 **PDF Report**: n-page comprehensive quality assessment
- ⚠️ **Drop Recommendations**: Which frames to remove before annotation

#### Quick Start
```bash
# Edit global variables in script, then run:
python "1. master_script_dinov2_PreAnnotate.py"
```

#### Configuration Variables
Edit these at the top of `main()` function:

```python
frames_dir = r"D:\Your\Frames\Folder"  # Root directory with extracted frames
batch_size = 64  # DINOv2 batch size
anisotropy_threshold = 3.6  # Motion blur detection threshold (gradient anisotropy)
output_dir = "frame_analysis_results"
pdf_name = "PreAnnotation_Quality_Report.pdf"
use_embedding_cache = True  # Cache embeddings for faster re-runs
```

#### Parameter Guide

**anisotropy_threshold** (Motion Blur Detection):
- `2.5`: Very strict - Only perfectly sharp frames (no motion blur or camera shake)
- `3.6`: **Recommended** - Balanced motion blur detection
- `5.0`: Lenient - Allow more motion blur

**use_embedding_cache**:
- `True`: **Recommended** - Saves embeddings to disk for faster re-runs
- `False`: Recompute embeddings every time (slower)

#### PDF Report Structure (5 Pages)

**Page 1: Executive Summary**
- Total frames, activities detected, quality score (0-10)
- Breakdown: dark/medium/bright and blurry/sharp frames
- Key recommendations (frames to drop, scenarios to add)
- Overall verdict: "Ready for annotation" / "Needs improvement" / "Not ready"

**Page 2: Quality Metrics Dashboard**
- Brightness histogram (shows low-light threshold at 80)
- Anisotropy histogram (shows motion blur threshold)
- Lighting distribution pie chart (Dark/Medium/Bright %)
- Motion blur distribution pie chart (Blurry/Sharp %)
- **Specific recommendations**: "Drop 45 blurry frames"

**Page 3: Diversity Analysis**
- UMAP scatter plot (frames colored by activity cluster)
- Activity breakdown table with frame counts
- Assessment of scenario diversity

**Page 4-m: Activity Visual Examples** ⭐
- **Montage for each activity** (4×3 grid of 12 sample images)
- Visual validation of clustering quality
- Labeled by Activity ID

**Page m+1: Coverage Gap Analysis**
- Heatmap: Activity × Lighting conditions
- Identifies missing combinations
- Final assessment with actionable verdict

#### Interpreting Results

**Quality Score**:
- `7-10`: ✅ Good quality, ready for annotation
- `5-7`: ⚠️ Moderate quality, review recommendations
- `<5`: ❌ Poor quality, improve before annotation

**Activity Count**:
- `1-2 activities`: ⚠️ Low diversity (frames from same video/angle)
- `3-6 activities`: ✅ Good diversity
- `7+ activities`: ✅ Excellent diversity

**Example Recommendations**:
```
⚠️ 60% frames are low-light. Consider adding well-lit frames or dropping extreme dark frames.
⚠️ 45% frames are blurry (motion blur). Use higher shutter speed, stabilize camera, or reduce camera movement. Consider dropping blurry frames.
ℹ️ Only 2 activity patterns detected. Capture more diverse work scenarios.
✅ Dataset quality looks good! Proceed with annotation.
```

#### Activity Clustering Explained

**What is an "Activity"?**
A semantically distinct group of frames representing:
- Different work scenarios (installation vs inspection vs welding)
- Different object arrangements (left/center/right)
- Different camera angles (close-up vs wide-angle)
- Different environmental contexts

**How to Validate**:
1. Check Page 4 montages
2. **Good clustering**: All 12 images in Activity 0 montage show similar scenario
3. **Poor clustering**: Montage shows mixed unrelated scenarios
4. **Fix**: Adjust `distance_threshold` (lower for stricter grouping)

**Visual Example**:
```
Activity 0 montage: [12 images all showing person screwing from left angle] ✅
Activity 1 montage: [6 welding, 3 drilling, 3 inspection mixed together] ❌
```

#### Validation Workflow

1. **Run script** with default settings
2. **Review PDF Page 1**: Check quality score and recommendations
3. **Review PDF Page 2**: Identify motion-blurred/dark frames to drop
4. **Review PDF Page 4**: Validate activity montages look correct
5. **Review PDF Page 5**: Check for coverage gaps
6. **Take action**:
   - Drop blurry frames (anisotropy > threshold)
   - Extract more frames from diverse videos if only 1-2 activities
   - Balance lighting if heavily skewed to dark/bright
7. **Re-run** if needed after improvements

#### Example Use Case

**Scenario**: Extracted 1000 frames from 3 videos for YOLO training

**Result**:
```
Activities detected: 8
Quality score: 7.2/10
Blurry frames (motion blur): 12% (120 frames)
Dark frames: 45% (450 frames)
Coverage gaps: Activity 3 has no bright lighting frames
```

**Action**:
1. Drop 120 motion-blurred frames
2. Extract more frames from bright lighting conditions
3. Add frames showing Activity 3 in bright light
4. Final dataset: 900 high-quality, diverse frames ready for annotation ✅

#### Output Structure
```
frame_analysis_results/
└── PreAnnotation_Quality_Report.pdf
```

---

### Script: 1. master_script_pretrain.py
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
python "1. master_script_dinov2_pretrain.py"
```

#### Configuration Variables
Edit these at the top of `main()` function:

```python
## GLOBAL VARIABLES PRE-TRAINING ##
imgs_label_path = "path/to/LabelledData"
class_names = ["class0", "class1", "class2"]
cropped_bbox_dir = "cropped_imgs_by_class"
batch_size = "64"
epsilon = "0.15"  # Ignored if auto_tune is used
min_pts = "3"
output_cluster_dir = "clustering_results"
max_cluster_samples = "20"

# PDF Generation (optional)
temp_file = "temp_ann_file.txt"
pdf_generate = True  # Set to False to skip PDF generation
pdf_name = "PDF_REPORT"
```

#### Pipeline Steps
1. **Crop Extraction** (`postannotation_scripts/1. ann_txt_files_crop_bbox.py`)
   - Extracts bounding boxes from annotated images
   - Validates dataset integrity
   - Organizes crops by class
   - Saves statistics to temp txt file

2. **Embedding Generation** (`postannotation_scripts/2. save_dinov2_embeddings_per_class.py`)
   - Generates 768D DINOv2 embeddings
   - GPU-accelerated with mixed precision

3. **Clustering Analysis** (`postannotation_scripts/3. clustering_of_classes_embeddings.py`)
   - Auto-tunes DBSCAN epsilon per class if selected
   - Generates visualizations and montages
   - Cross-class separability analysis if selected

4. **PDF Report Generation** (Optional - `postannotation_scripts/4. generate_pdf.py`)
   - Combines dataset statistics and clustering visualizations
   - Creates professional multi-page PDF report
   - Only runs if `pdf_generate = True` 

---

### Script: 1. master_script_dinov2_posttrain.py
**Automated POST-TRAINING pipeline** - Analyzes trained model detections on a test video

#### Features
Same as pre-training script, plus:
- 🎥 **Video Processing**: Extracts detections from video files
- 🎯 **Smart Sampling**: Uniform frame selection per class
- ⚡ **Frame Stride**: Process every Nth frame for efficiency

#### Quick Start
```bash
# Edit global variables in script, then run:
python "1. master_script_dinov2_posttrain.py"
```

#### Configuration Variables
```python
## GLOBAL VARIABLES POST-TRAINING ##
model_path = "path/to/model.pt"
video_path = "path/to/video.mp4"
classes_space_separated = ["class0", "class1", "class2"]
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
1. **Detection & Cropping** (`postannotation_scripts/1. yolo_model_crop_bbox_per_class.py`)
   - Runs YOLOv8 inference on video
   - Applies frame stride (e.g., every 3rd frame)
   - Smart sampling: uniformly extracts target number of crops per class
   - **⚠️ Most time-consuming step** (10-15 minutes for 20-min video)

2. **Embedding Generation** (same as pre-training)
3. **Clustering Analysis** (same as pre-training)

#### Frame Stride Recommendations
| Video FPS | Stride | Effective FPS | Use Case |
|-----------|--------|---------------|----------|
| 30 | 2 | 15 | High variation scenes |
| 30 | 3 | 10 | **Recommended default** |
| 30 | 5 | 6 | Slow-moving objects |
| 60 | 6 | 10 | High FPS cameras |

---

### Master Script Output

Both scripts provide detailed progress tracking.


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

### Script: postannotation_scripts/1. ann_txt_files_crop_bbox.py
**Stage: PRE-TRAINING** - Analyze annotated data before model training

#### Usage
```bash
python "postannotation_scripts/1. ann_txt_files_crop_bbox.py" \
    --imgs_label_path "path/to/LabelledData" \
    --classes 0 1 2 \
    --class_ids_to_names 0 class0 1 class1 2 class2 \
    --output_dir cropped_imgs_by_class
```

#### Parameters
| Parameter | Description |
|-----------|-------------|
| `--imgs_label_path` | Folder with image-label pairs (.png/.jpg + .txt) |
| `--classes` | Space-separated class IDs to extract |
| `--class_ids_to_names` | ID-name mapping: `id1 name1 id2 name2 ...` |
| `--output_dir` | Output directory (default: `cropped_imgs_by_class`) |
| `--output_txt_file` | Optional: Path to save statistics txt file for PDF generation |

#### What It Does
1. Validates dataset (checks image-label pairing)
2. Verifies class mapping completeness
3. **Scans for unexpected class IDs** (annotation errors)
4. **Reports class distribution** with imbalance warnings
5. Converts YOLO normalized coords → pixel coords
6. Crops bounding boxes per class
7. Saves to class-specific folders

#### Dataset Validation Output
The script performs comprehensive validation before cropping:

```
------------------------------------------------------------
|                 VALIDATING ANNOTATIONS                   |
------------------------------------------------------------

❌  UNEXPECTED CLASS IDs FOUND (not in class_ids_to_names):
    Expected class IDs: ['0', '1', '2']
    -----------------------------------------------
    Class ID '3': 12 annotations in 3 file(s)
        - frame_001.txt
        - frame_042.txt
        - frame_089.txt

    ⚠️  FIX THESE ANNOTATIONS BEFORE YOLO TRAINING!

📊  CLASS DISTRIBUTION (all annotations in dataset):
    Class 0 (class0): 450 annotations (42.5%)
    Class 1 (class1): 520 annotations (49.1%)
    Class 2 (class2): 90 annotations (8.5%)
⚠️  Class ID '3': 12 annotations (UNKNOWN)

⚠️  CLASS IMBALANCE WARNING: Ratio 520:90 = 5.8x
    Consider balancing your dataset for better YOLO training.
```

This catches:
- **Annotation typos**: Wrong class IDs from human error
- **Extra classes**: Accidentally created classes during labeling
- **Class imbalance**: >10x ratio between largest/smallest class
- **Empty annotation files**: Images with no objects labeled

#### Output Structure
```
cropped_imgs_by_class/
├── class0/
│   ├── frame_001_crop_0.png
│   └── ...
├── class1/
└── ...
```

#### Common Issues
- **"TXT files missing images"**: Fix dataset consistency
- **Empty crops**: Check annotation normalization in .txt files

---

### Script: postannotation_scripts/1. yolo_model_crop_bbox_per_class.py
**Stage: POST-TRAINING** - Verify trained model detections

#### Usage
```bash
python "postannotation_scripts/1. yolo_model_crop_bbox_per_class.py" \
    --model "path/to/model.pt" \
    --video "path/to/video.mp4" \
    --classes class0 class1 class2 \
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

### Script: postannotation_scripts/2. save_dinov2_embeddings_per_class.py
**Generates semantic embeddings from cropped images**

This script processes cropped object images and uses DINOv2 model to generate embeddings for each class. Optimized for GPU usage with memory management and mixed precision support.

#### Usage
```bash
python "postannotation_scripts/2. save_dinov2_embeddings_per_class.py" --root ./cropped_images --batch 32
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

### Script: postannotation_scripts/3. clustering_of_classes_embeddings.py
**Discovers sub-groups and outliers using DBSCAN**

#### Basic Usage
```bash
# Auto-tuning (recommended)
python "postannotation_scripts/3. clustering_of_classes_embeddings.py" --root ./cropped_images --auto_tune --min_samples 3

# Manual epsilon
python "postannotation_scripts/3. clustering_of_classes_embeddings.py" --root ./cropped_images --eps 0.15 --min_samples 3

# Full analysis with visualizations
python "postannotation_scripts/3. clustering_of_classes_embeddings.py" --root ./cropped_images --auto_tune --min_samples 3 --save_montage --cross_class
```

#### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--root` | `cropped_images` | Root folder with embeddings |
| `--eps` | 0.15 | Max distance for neighbors (cosine) |
| `--min_samples` | 3 | Min points to form cluster |
| `--auto_tune` | False | Auto-find optimal eps per class |
| `--output_dir` | `clustering_results` | Output directory |
| `--max_samples` | 5 | Sample images per cluster (outliers: all saved) |
| `--save_suffix` | `embeddings_dinov2.npy` | Embedding filename (must match Script 2 output) |
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
   - Loads embeddings for each class from `--save_suffix` filename
   - Dynamically derives mapping filename (e.g., `embeddings_dinov2.npy` → `embeddings_dinov2_image_list.txt`)
   - Falls back to sorted image list if mapping missing (with warning)
   - Validates data consistency (alignment between images and embeddings)

2. **Auto-Tuning (Optional)**:
   - Uses k-Nearest Neighbors for pairwise distances
   - Selects 90th percentile of k-th nearest neighbor distances
   - Provides class-specific optimal eps values
   - **Cross-class analysis**: Uses median of per-class eps values when auto-tuned

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
   - **Regular clusters**: Saves up to `--max_samples` random images per cluster
   - **Outliers**: Saves ALL outlier images (not sampled) for thorough review
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
class0:  2 clusters,  4 outliers (2.0%)   → Consistent appearance
class1:  4 clusters, 21 outliers (10.5%)  → Multiple orientations
class2:  7 clusters, 12 outliers (6.0%)   → High diversity
```

#### Common Issues
- **All outliers (0 clusters)**: Increase `--eps` or use `--auto_tune`
- **Only 1 cluster**: Decrease `--eps` (try 0.1)
- **Too many small clusters**: Increase `--min_samples` or `--eps`
- **Image count mismatch**: Re-run embedding generation
- **"No mapping file found" warning**: Re-run Script 2

---

### Script: postannotation_scripts/4. generate_pdf.py
**Generate comprehensive PDF reports combining dataset statistics and clustering visualizations**

Creates a multi-page PDF report that combines Script 1 dataset quality statistics with Script 3 clustering montages for professional presentation and documentation.

#### Usage
```bash
python "postannotation_scripts/4. generate_pdf.py" \
    --temp_txt_file temp_ann_file.txt \
    --clustering_dir clustering_results \
    --pdf_name "REPORT"
```

#### Parameters
| Parameter | Description |
|-----------|-------------|
| `--temp_txt_file` | Path to Script 1's temporary statistics txt file |
| `--clustering_dir` | Path to Script 3's clustering output directory |
| `--pdf_name` | Output PDF filename (without .pdf extension) |

#### What It Does
1. **Parses Script 1 statistics**: Dataset quality, class distribution, imbalance warnings
2. **Loads Script 3 cluster data**: Reads `cluster_statistics.csv` for per-class metrics
3. **Generates visualizations**: Bar charts for class distribution and imbalance ratios
4. **Combines montages**: Embeds clustering montages for each class
5. **Smart image handling**:
   - Compresses images to JPEG quality 90 for smaller file sizes
   - Splits tall montages at cluster row boundaries (no clusters cut in half)
   - Adds continuation labels when montages span multiple pages
6. **Creates PDF**: Professional multi-page report with proper formatting

#### Features
- **Automatic image compression**: Reduces PDF size by ~80% using JPEG quality 90
- **Intelligent splitting**: Montages split at cluster boundaries, never mid-cluster
- **Page-aware layout**: Fits multiple cluster rows per page, adds continuation labels
- **Error handling**: Creates placeholder images for missing montages
- **PIL decompression bomb protection disabled**: Handles large montages (25+ clusters)

#### Output Structure
```
Page 1: Dataset Quality Report
  - Dataset overview statistics
  - Class distribution bar chart
  - Class imbalance analysis chart

Pages 2-N: Per-Class Clustering Montages
  - Class title
  - Montage image (possibly split across pages)
  - Continuation labels (e.g., "Continuation - Clusters 3 to 5")
  - Statistics: Samples | Clusters | Outliers | Epsilon
```

#### Technical Details
- **Image format**: Converts PNG montages → JPEG quality 90 (80-90% smaller)
- **Split logic**: Calculates cluster rows from `cluster_statistics.csv`, fits whole rows per page
- **Max page height**: 8.5 inches for montage content
- **Max page width**: 7 inches for images
- **Temporary files**: Auto-cleaned after PDF generation

#### Example Output
For a class with 25 clusters:
- **Page 1**: Title + Clusters 0-2 (3 rows fit)
- **Page 2**: "(Continuation - Clusters 3 to 5)" + Clusters 3-5
- **Page 3**: "(Continuation - Clusters 6 to 8)" + Clusters 6-8
- ... (continues until all clusters shown)
- **Final page**: Last clusters + outliers + statistics caption

#### Common Issues
- **"Error loading montage image"**: Image too large - now fixed with PIL limits disabled
- **PDF too large (>500KB)**: Already optimized with JPEG compression at quality 90
- **Missing montages**: Ensure Script 3 was run with `--save_montage` flag

#### Requirements
```bash
pip install reportlab pillow matplotlib pandas
```

---

### Script: 1. interactive_cluster_viewer.py (Optional)
**Interactive HTML visualization using Plotly**

Standalone optional tool for exploring clusters interactively. Run after Script 2 (embeddings). Independent of Script 3.

#### Usage
```bash
python "1. interactive_cluster_viewer.py" --root cropped_imgs_by_class
```

#### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--root` | `cropped_images` | Root folder with embeddings |
| `--output_dir` | `clustering_results` | Output directory for HTML files |
| `--min_samples` | 3 | DBSCAN min_samples |
| `--eps` | auto | DBSCAN eps (auto-tuned if not provided) |

#### Features
- **Hover** over points to see filename
- **Zoom/pan** with mouse
- **Lasso select** to isolate regions
- Generates one HTML file per class

#### Output
```
clustering_results/
├── person_interactive.html
├── hands_interactive.html
└── ...
```

#### Requirements
```bash
pip install plotly
```

---

## Complete Workflows

### Pre-Training Analysis (Automated)
```bash
# Recommended: Use master script
python "1. master_script_pretrain.py"

# Edit these variables in the script before running:
# - imgs_label_path
# - class_names
# - cropped_bbox_dir
# - output_cluster_dir
```

### Pre-Training Analysis (Manual)
```bash
# 1. Extract crops from annotated data
python "postannotation_scripts/1. ann_txt_files_crop_bbox.py" \
    --imgs_label_path "./LabelledData" \
    --classes 0 1 2 \
    --class_ids_to_names 0 class0 1 class1 2 class2

# 2. Generate embeddings
python "postannotation_scripts/2. save_dinov2_embeddings_per_class.py" \
    --root ./cropped_imgs_by_class --batch 32

# 3. Analyze clusters
python "postannotation_scripts/3. clustering_of_classes_embeddings.py" \
    --root ./cropped_imgs_by_class \
    --auto_tune --min_samples 3 --save_montage
```

### Post-Training Analysis (Automated)
```bash
# Recommended: Use master script
python "1. master_script_dinov2_posttrain.py"

# Edit these variables in the script before running:
# - model_path
# - video_path
# - classes_space_separated
# - per_class_num_frames
```

### Post-Training Analysis (Manual)
```bash
# 1. Extract crops using trained model
python "postannotation_scripts/1. yolo_model_crop_bbox_per_class.py" \
    --model "model.pt" \
    --video "video.mp4" \
    --classes class0 class1 class2 \
    --num_frames 200

# 2-3. Same as pre-training (use --root ./cropped_images)
```

---

## Pipeline Summary

| Script | Input | Output | Purpose | When to Use |
|--------|-------|--------|---------|-------------|
| **1. master_script_dinov2_PreAnnotate.py** | Raw frames | PDF quality report | **Pre-annotation frame assessment** | BEFORE annotation |
| **1. master_script_dinov2_pretrain.py** | Config variables | Complete analysis | **Automated pre-training pipeline** | AFTER annotation |
| **1. master_script_dinov2_posttrain.py** | Config variables | Complete analysis | **Automated post-training pipeline** | AFTER model training |
| postannotation_scripts/1. ann_txt_files_crop_bbox.py | Annotated images + TXT | Cropped images | Pre-training data inspection | After annotation |
| postannotation_scripts/1. yolo_model_crop_bbox_per_class.py | Video + YOLO model | Cropped images | Post-training detection verification | After training |
| postannotation_scripts/2. save_dinov2_embeddings_per_class.py | Cropped images | 768D embeddings | Semantic representations | Part of pipeline |
| postannotation_scripts/3. clustering_of_classes_embeddings.py | Embeddings + images | Clusters + insights | Intra-class variation analysis | Part of pipeline |
| postannotation_scripts/4. generate_pdf.py | Script 1 stats + Script 3 results | PDF report | Professional documentation | Part of pipeline |
| 1. interactive_cluster_viewer.py | Embeddings | Interactive HTML | Optional: Plotly exploration | Optional addon |

---

## Key Takeaways
- 🚀 **Use master scripts** for automated pipeline execution with memory management
- **Pre-annotation** (NEW): Validate frame quality/diversity BEFORE annotation effort
- **Pre-training**: Validate annotation quality and understand labeled patterns AFTER annotation
- **Post-training**: Verify model detections match expectations AFTER model training
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
