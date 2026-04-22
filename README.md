# EDA Intra-Class Variation Scripts

Personal pipeline I use to check dataset quality before/after annotation. Runs fully local — no API calls, no cloud, everything stays on your machine (needs internet once to download model weights from HuggingFace).

Three modes depending on where you are in the pipeline:
- **Pre-annotation** — check if your frames are worth annotating (diversity, blur, lighting)
- **Pre-training** — check your annotated dataset for intra-class variation and outliers
- **Post-training** — check if your trained YOLO model detections make sense

---

## Quick Start

Always run from project root. The launcher shows a menu: option 1 = pre-annotation, option 2 = pre-training. Post-training isn't in the menu — run it directly.

```bat
exec.bat        # Windows
./exec.sh       # Linux/Mac
```

Or run directly:
```bash
python "master_scripts/1. master_script_Dinov2_PaCMAP_PreAnn.py"    # pre-annotation
python "master_scripts/1. master_script_dinov2_PostAnn_PreTrain.py"  # pre-training
python "master_scripts/1. master_script_dinov2_PostTrain.py"         # post-training (edit config vars at top of script)
```

Pre-annotation and pre-training use interactive prompts. Post-training requires editing config variables at the top of the script.

---

## Pre-Annotation: Frame Quality Assessment

**Script**: `master_scripts/1. master_script_Dinov2_PaCMAP_PreAnn.py`

Use this when you have raw frames and want to know if they're worth annotating. Catches: redundant frames, motion blur, bad lighting, missing scenarios.

**What it does**:
- DINOv2 [CLS||avg_patches] embeddings (1536d) — adaptive: adds pose features when persons detected
- PCA whitening → PaCMAP dimensionality reduction → HDBSCAN clustering
- Motion blur detection via directional gradient anisotropy
- Generates a 5-page PDF report

**Output**: `preann_results/PreAnnotation_Quality_Report_{FolderName}.pdf`

**Prompts**:
1. Path to frames folder
2. "Change default parameters?" — No = sensible defaults, Yes = exposes everything
3. Clustering params (always asked):
   - `n_components`: PCA dims before HDBSCAN (default: 128)
   - `min_cluster_size`: Minimum frames per activity group (default: 10)
   - `min_samples`: HDBSCAN core point strictness (default: 3)
4. Use embedding cache? (Yes = faster re-runs)

**Key parameters**:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `n_components` | 128 | Higher = finer activity separation |
| `min_cluster_size` | 10 | Lower = more small groups detected |
| `min_samples` | 3 | Lower = more frames included in clusters |
| `anisotropy_threshold` | 3.6 | Motion blur cutoff (2.5 strict, 5.0 lenient) |
| `batch_size` | 64 | DINOv2 batch size |

**PDF structure**:
- Page 1: Executive summary — quality score, frame breakdown, recommendations
- Page 2: Quality metrics dashboard (brightness, blur histograms, lighting distribution)
- Page 3: PaCMAP scatter plot + activity breakdown table
- Pages 4–M: Activity montages (4 columns × up to 5 rows per activity)
- Page M+1: Coverage gap heatmap (activity × lighting)

**Interpreting results**:

Quality score (0–10): average of three sub-scores — sharpness (`max(0, 10 - median_anisotropy/5*10)`), brightness centering (`10 - |median_brightness - 127|/127*10`), contrast (`min(10, median_contrast/50*10)`). 7–10 = good, 5–7 = review needed, <5 = fix before annotating.

Activity count: 1–2 = low diversity (same angle/video), 3–6 = good, 7+ = excellent

> Note: Always validate montages visually. Clusters can form on viewpoint/lighting rather than semantic content.

**Output files**:
```
preann_results/
├── PreAnnotation_Quality_Report_{FolderName}.pdf
└── outliers_{FolderName}/   (optional — prompted at end)
```

---

## Pre-Training: Annotated Dataset Analysis

**Script**: `master_scripts/1. master_script_dinov2_PostAnn_PreTrain.py`

Use after annotation to understand intra-class variation — are your classes homogeneous or do they have distinct sub-groups? Finds outliers and annotation errors.

**What it does**: Crops bboxes from YOLO annotations → DINOv2 768d embeddings → DBSCAN clustering → PDF report

**Output**: `postann_pretrain_results/clustering_results_{FolderName}/PDF_REPORT_{FolderName}.pdf`

**Prompts**:
1. Images folder, labels folder, classes.txt
2. Output subfolder names (smart defaults from your dataset folder name)
3. "Change default parameters?" gate
4. Clustering flags (always asked): auto-tune eps, cross-class analysis
5. k-NN percentile (if auto-tune on) or manual epsilon (if off)
6. Generate PDF?

**Default values** (master script defaults — override the individual script's own defaults):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 64 | DINOv2 batch size |
| `auto_tune` | True | Auto-calculate DBSCAN epsilon per class |
| `auto_tune_percentile` | 95 | k-NN percentile (90=tight, 95=balanced, 98=loose) |
| `epsilon` | 0.15 | Manual eps fallback (only if auto-tune off) |
| `cross_class` | True | Cross-class separability analysis |
| `min_pts` | 3 | Min points per cluster |
| `umap_min_dist` | 0.05 | UMAP min_dist |
| `max_cluster_samples` | 20 | Max sample images per cluster |
| `use_embedding_cache` | True | Skip re-generation if embeddings exist |

**Pipeline steps** (run automatically):
1. `postannotation_scripts/1. ann_txt_files_crop_bbox.py` — crop + validate dataset
2. `postannotation_scripts/2. save_dinov2_embeddings_per_class.py` — generate embeddings
3. `postannotation_scripts/3. clustering_of_classes_embeddings.py` — DBSCAN clustering
4. `postannotation_scripts/4. generate_pdf.py` — PDF report (optional)

---

## Post-Training: Model Detection Analysis

**Script**: `master_scripts/1. master_script_dinov2_PostTrain.py`

Use after training to check if model detections are sensible. Not in `./exec.bat` — run directly.

```bash
python "master_scripts/1. master_script_dinov2_PostTrain.py"
```

Same pipeline as pre-training but Step 1 runs YOLO inference on video instead of reading annotation files.

**Configuration variables** (at top of script):
```python
RESULTS_DIR = "posttrain_results"
model_path = "path/to/model.pt"
video_path = "path/to/video.mp4"
classes_space_separated = ["class0", "class1", "class2"]
per_class_num_frames = 1000
conf_thresh = 0.4
frame_stride_per_video = 3        # every 3rd frame = 10fps on 30fps video
```

**Frame stride guide** (30fps video):

| Stride | Effective FPS | Use when |
|--------|---------------|----------|
| 2 | 15 | High variation scenes |
| 3 | 10 | Recommended default |
| 5 | 6 | Slow-moving objects |

---

## Individual Scripts

### postannotation_scripts/1. ann_txt_files_crop_bbox.py

```bash
python "postannotation_scripts/1. ann_txt_files_crop_bbox.py" \
    --imgs_label_path "path/to/LabelledData" \
    --classes 0 1 2 \
    --class_ids_to_names 0 class0 1 class1 2 class2 \
    --output_dir cropped_imgs_by_class
```

Validates dataset before cropping — catches unexpected class IDs, empty labels, imbalance ratio >10x. Output: `cropped_imgs_by_class/{class}/*.png`

### postannotation_scripts/1. yolo_model_crop_bbox_per_class.py

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

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_frames` | 100 | Target samples per class (uniform sampling) |
| `--frame_stride` | 3 | Process every Nth frame |
| `--conf_thresh` | 0.4 | Detection confidence threshold |

### postannotation_scripts/2. save_dinov2_embeddings_per_class.py

```bash
python "postannotation_scripts/2. save_dinov2_embeddings_per_class.py" \
    --root ./cropped_imgs_by_class --batch 32
```

Generates 768d DINOv2 embeddings per class. Saves `embeddings_dinov2.npy` + `embeddings_dinov2_image_list.txt` (image-embedding alignment) per class folder.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch` | 32 | Batch size |
| `--save_suffix` | `embeddings_dinov2.npy` | Output filename |
| `--use_cache` | False | Skip if embeddings already exist |

### postannotation_scripts/3. clustering_of_classes_embeddings.py

```bash
# Auto-tune (recommended)
python "postannotation_scripts/3. clustering_of_classes_embeddings.py" \
    --root ./cropped_imgs_by_class --auto_tune --min_samples 3 --save_montage --cross_class

# Manual epsilon
python "postannotation_scripts/3. clustering_of_classes_embeddings.py" \
    --root ./cropped_imgs_by_class --eps 0.15 --min_samples 3
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--eps` | 0.15 | DBSCAN epsilon (0.1 strict, 0.3 lenient) |
| `--min_samples` | 3 | Min points to form cluster |
| `--auto_tune` | False | k-NN percentile auto-tune per class |
| `--auto_tune_percentile` | 90 | 90=tight, 95=balanced, 98=loose |
| `--umap_min_dist` | 0.05 | UMAP min_dist (0.0=tight, 0.1=loose) |
| `--max_samples` | 5 | Sample images per cluster (outliers: all saved) |
| `--cross_class` | False | Cross-class separability analysis |
| `--save_montage` | False | Generate image grid montages |
| `--uniform_eps_threshold` | 0.10 | Below this eps = class treated as uniform |
| `--uniform_downsample_target` | 5000 | Downsample target for uniform classes |
| `--uniform_min_samples` | 10000 | Only downsample if class has more than this |

Output:
```
clustering_results/
├── {class}_clusters.png
├── {class}_montage.png
├── {class}_samples/
│   ├── cluster_0/
│   └── outliers/
├── all_classes_overview.png
├── cross_class_separability.png
└── cluster_statistics.csv
```

Interpreting cluster counts: 1 = homogeneous class, 2–4 = normal variation, 5+ = high diversity (check montages before splitting classes — clusters can reflect viewpoint/lighting, not always semantic sub-classes).

Outlier rates: <5% = good, 5–10% = moderate variation, >20% = check data quality or loosen eps.

### postannotation_scripts/4. generate_pdf.py

Usually invoked automatically by the pre-training master script. Standalone use:

```bash
python "postannotation_scripts/4. generate_pdf.py" \
    --temp_txt_file postann_pretrain_results/temp_ann_file.txt \
    --clustering_dir postann_pretrain_results/clustering_results_MyDataset \
    --pdf_name "postann_pretrain_results/clustering_results_MyDataset/PDF_REPORT_MyDataset" \
    --imgs_path path/to/images \
    --label_path path/to/labels \
    --classes_txt path/to/classes.txt \
    --auto_tune --auto_tune_percentile 95 --cross_class
```

PDF structure: Page 1 = dataset overview + pipeline config + per-class clustering summary tables. Page 2 = cross-class separability or all-classes overview graph. Pages 3–N = per-class montages (split at cluster row boundaries, never mid-cluster).

### master_scripts/1. interactive_cluster_viewer.py (optional)

Interactive Plotly HTML explorer. Run after Script 2, independent of Script 3.

```bash
python "master_scripts/1. interactive_cluster_viewer.py" \
    --root postann_pretrain_results/cropped_imgs_by_class
```

---

## Decision Guide

**Pre-annotation output → what to do:**

| Result | Action |
|--------|--------|
| Quality score < 5 | Fix data before annotating — re-shoot or filter frames |
| >30% blurry frames | Drop them or re-extract with higher shutter speed |
| >40% dark frames | Add well-lit frames or adjust camera settings |
| 1–2 activities | Low diversity — capture from more angles/scenarios |
| Activity montage shows mixed unrelated content | Decrease `min_cluster_size` (allow smaller tighter groups) or increase `n_components` (finer PCA detail) and re-run |

**Pre/post-training clustering output → what to do:**

| Result | Action |
|--------|--------|
| 0 clusters (all outliers) | eps too strict — increase `--eps` or use `--auto_tune` |
| 1 cluster | Class is visually homogeneous — nothing to split |
| 5+ clusters | Check montages first — if clusters are semantic, consider sub-classes; if viewpoint/lighting artifacts, ignore |
| >20% outliers | Check outlier folder — annotation errors or genuinely rare appearances |
| Class imbalance >10x flagged | Add more data for underrepresented class before training |

---

## Pipeline Summary

| Script | When to use |
|--------|-------------|
| `master_scripts/1. master_script_Dinov2_PaCMAP_PreAnn.py` | Before annotation — frame quality check |
| `master_scripts/1. master_script_dinov2_PostAnn_PreTrain.py` | After annotation — intra-class variation |
| `master_scripts/1. master_script_dinov2_PostTrain.py` | After training — model detection check |
| `postannotation_scripts/1. ann_txt_files_crop_bbox.py` | Manual: crop from YOLO annotations |
| `postannotation_scripts/1. yolo_model_crop_bbox_per_class.py` | Manual: crop from model detections |
| `postannotation_scripts/2. save_dinov2_embeddings_per_class.py` | Manual: generate embeddings |
| `postannotation_scripts/3. clustering_of_classes_embeddings.py` | Manual: DBSCAN clustering |
| `postannotation_scripts/4. generate_pdf.py` | Manual: generate PDF report |

---

## Dependencies

```bash
pip install -r requirements.txt
```

Core: PyTorch >= 2.1 with CUDA, transformers (DINOv2), ultralytics (YOLO), scikit-learn, pacmap, hdbscan, opencv-python, pillow, numpy, pandas, matplotlib, seaborn, reportlab, tqdm, questionary

---

## References

- DINOv2 clustering: https://medium.com/@EnginDenizTangut/%EF%B8%8F-image-clustering-with-dinov2-and-hdbscan-a-hands-on-guide-35c6e29036f2
- DBSCAN: https://medium.com/@sachinsoni600517/clustering-like-a-pro-a-beginners-guide-to-dbscan-6c8274c362c4
- Post-training pipeline inspiration: https://medium.com/@albertferrevidal/facings-product-identifier-using-yolov8-and-image-embeddings-d3ca34463022
