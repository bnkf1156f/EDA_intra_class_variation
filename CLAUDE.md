# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **intra-class variation analysis pipeline** for object detection datasets. It analyzes class heterogeneity using DINOv2 embeddings and DBSCAN clustering to discover sub-groups and outliers within each class.

**Two pipeline modes:**
- **Pre-training**: Analyzes annotated dataset before model training (uses YOLO annotation .txt files)
- **Post-training**: Analyzes trained YOLOv8 model detections on video (validates model quality)

## Core Principles
- **Keep it simple** - This is exploratory EDA, not production code
- **No overcomplications** - Engineers just need to see cluster patterns and outliers
- **Consistency first** - README must match actual script behavior exactly

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/1. ann_txt_files_crop_bbox.py` | Pre-training: Extract crops from YOLO annotations |
| `scripts/1. yolo_model_crop_bbox_per_class.py` | Post-training: Extract crops from model detections |
| `scripts/2. save_dinov2_embeddings_per_class.py` | Generate DINOv2 embeddings |
| `scripts/3. clustering_of_classes_embeddings.py` | DBSCAN clustering analysis |
| `scripts/4. generate_pdf.py` | Generate PDF reports combining dataset stats + clustering results |
| `1. interactive_cluster_viewer.py` | Optional: Interactive Plotly HTML viewer |
| `1. master_script_dinov2_pretrain.py` | Automated pre-training pipeline |
| `1. master_script_dinov2_posttrain.py` | Automated post-training pipeline |

## Running the Pipeline

### Automated (Recommended)
```bash
# Pre-training analysis (annotated images)
python "1. master_script_dinov2_pretrain.py"

# Post-training analysis (video + trained model)
python "1. master_script_dinov2_posttrain.py"
```

Configure global variables at the top of `main()` in each master script before running.

### Manual Step-by-Step
```bash
# Step 1a: Pre-training - crop from annotations
python "scripts/1. ann_txt_files_crop_bbox.py" --imgs_label_path "path/to/LabelledData" --classes 0 1 2 --class_ids_to_names 0 board 1 screw 2 holder --output_dir cropped_imgs

# Step 1b: Post-training - crop from video using model
python "scripts/1. yolo_model_crop_bbox_per_class.py" --model model.pt --video video.mp4 --classes board screw holder --num_frames 100 --frame_stride 3

# Step 2: Generate DINOv2 embeddings
python "scripts/2. save_dinov2_embeddings_per_class.py" --root ./cropped_imgs --batch 32

# Step 3: DBSCAN clustering analysis
python "scripts/3. clustering_of_classes_embeddings.py" --root ./cropped_imgs --auto_tune --min_samples 3 --save_montage --cross_class
```

## Architecture

### Pipeline Flow
```
Input (annotated images OR video+model)
    |
    v
[Script 1] Crop bounding boxes per class --> cropped_imgs/<class>/*.png
                                             temp_ann_file.txt (statistics)
    |
    v
[Script 2] DINOv2 embeddings --> cropped_imgs/<class>/embeddings_dinov2.npy
                                 cropped_imgs/<class>/embeddings_dinov2_image_list.txt
    |
    v
[Script 3] DBSCAN clustering --> clustering_results/<class>_clusters.png
                                 clustering_results/<class>_montage.png
                                 clustering_results/<class>_samples/cluster_N/
                                 clustering_results/cluster_statistics.csv
    |
    v (optional)
[Script 4] PDF report generator --> output.pdf (combines Script 1 + 3 results)
[Script 5] Interactive viewer --> clustering_results/<class>_interactive.html
```

### Key Data Structures
- **Embeddings**: 768-dimensional vectors from `facebook/dinov2-base`
- **Image mapping**: `embeddings_dinov2_image_list.txt` maintains alignment between embeddings and source images
- **YOLO format**: `class_id x_center y_center width height` (normalized coordinates)

### Script Interdependencies
- Script 1a/1b → Creates class folders with cropped images + `temp_ann_file.txt` statistics
- Script 2 → Reads class folders, creates `.npy` + `_image_list.txt`
- Script 3 → Reads embeddings + mapping file, creates visualizations + `cluster_statistics.csv`
- Script 4 → Reads `temp_ann_file.txt` (Script 1) + clustering results (Script 3), generates PDF
- Script 5 → (Optional) Reads embeddings, creates interactive HTML (independent of Scripts 3 & 4)

**Critical**:
- If Script 2 changes output format, Script 3 must be updated to match!
- Script 4 depends on Script 1's txt output and Script 3's montage PNG files

### Master Script Features
- Memory management: GPU cache clearing, garbage collection between steps
- Cooling breaks: 10-second pauses to prevent laptop GPU thermal throttling
- Resource monitoring: Warns if RAM >90% or GPU >85%
- Error handling: Cleanup on failure or keyboard interrupt

## Key Parameters

### Clustering (Script 3)
- `--eps`: DBSCAN epsilon (0.1 strict, 0.15 balanced, 0.2-0.3 lenient)
- `--min_samples`: Minimum points to form cluster (2-3 typical)
- `--auto_tune`: Recommended - uses 90th percentile k-NN distance per class
- `--save_suffix`: Embedding filename to load (default: `embeddings_dinov2.npy`, must match Script 2 output)
- `--max_samples`: Sample images per cluster (default: 5, outliers are all saved)

### Video Processing (Script 1 post-training)
- `--frame_stride`: Process every Nth frame (3 = 10fps for 30fps video)
- `--num_frames`: Target samples per class (uniform sampling)
- `--conf_thresh`: Detection confidence threshold (default 0.4)

## Known Quirks
1. **Image-Embedding Alignment**: Script 2 creates a mapping file with name derived from `--save_suffix` parameter (e.g., `embeddings_dinov2.npy` → `embeddings_dinov2_image_list.txt`). Script 3 must receive the same `--save_suffix` value or alignment will fail. The mapping file ensures correct correspondence even when corrupted images are skipped.
2. **Sorted Order**: Both scripts use `sorted()` on file paths to maintain consistent ordering
3. **CSV Saving**: Script 3 has CSV saving enabled (not commented out)
4. **Cross-Class Epsilon**: When `--auto_tune` is enabled with `--cross_class`, Script 3 uses the median of per-class eps values for cross-class DBSCAN clustering
5. **Outlier Sampling**: Script 3 saves ALL outlier images (not sampled), while regular clusters are sampled up to `--max_samples`
6. **PDF Image Splitting**: Script 4 splits tall montages at cluster row boundaries (never mid-cluster) by reading `cluster_statistics.csv` to determine number of rows
7. **PDF Image Compression**: Script 4 converts all PNG montages to JPEG quality 90 for ~80% file size reduction
8. **PIL Decompression Bomb**: Script 4 disables PIL's `MAX_IMAGE_PIXELS` limit to handle large montages (25+ clusters)

## File Naming Patterns
- Script 1a outputs: `{basename}_crop_{idx}.png` + `temp_ann_file.txt`
- Script 1b outputs: `frame_{frame_idx:06d}.png`
- Script 2 outputs: `{save_suffix}.npy` + `{save_suffix_without_.npy}_image_list.txt`
  - Default: `embeddings_dinov2.npy` + `embeddings_dinov2_image_list.txt`
  - Custom: `custom_emb.npy` + `custom_emb_image_list.txt`
- Script 3 outputs: `{class}_clusters.png`, `{class}_montage.png`, `cluster_statistics.csv`
- Script 4 outputs: `{pdf_name}.pdf` + temp JPEG chunks (auto-cleaned)

## Dependencies

- PyTorch >= 2.1 with CUDA
- ultralytics (YOLOv8)
- transformers (DINOv2)
- scikit-learn (DBSCAN)
- umap-learn
- opencv-python, pillow, numpy, pandas, matplotlib, seaborn, psutil, tqdm
- reportlab (for PDF generation in Script 4)

---

## Guidelines for Claude Code

### When Reviewing/Updating Code

**Always check these files together:**
1. Read the specific script being modified
2. Read `README.md` sections for that script
3. Verify parameter tables, usage examples, and output descriptions match

**Critical consistency points:**
- **Script names**: Use full filenames exactly (e.g., `scripts/1. yolo_model_crop_bbox_per_class.py`)
- **Parameter defaults**: Must match argparse defaults in scripts
- **Output file structures**: Directory trees in README must match actual script output

### Code Review Checklist
- [ ] Read all scripts completely
- [ ] Read entire README.md
- [ ] Verify parameter tables match argparse definitions
- [ ] Confirm output structures match actual `os.makedirs()` and file writes
- [ ] Check usage examples have correct script names and parameters

### README Update Protocol

**When script parameters change:**
1. Find the parameter table in README under that script's section
2. Update the exact row with new default/description
3. Check if the change affects workflow examples

**When output format changes:**
1. Update the "Output Structure" code block
2. Update directory structure sections
3. Update "Output Files" descriptions

### Common User Requests

**"Update README after code change"**
1. Ask which script was changed
2. Read that script completely
3. Find corresponding README section
4. Update only the affected parts

**"Check consistency"**
1. Read all scripts
2. Read full README
3. Report mismatches with file:line references
4. Focus on functional issues, not stylistic preferences

**"Fix bug/improve script"**
1. First ask: "Is this for simplicity or production robustness?"
2. If simplicity: minimal change only
3. If robustness: verify it doesn't break the simple workflow
4. Always update README after code changes

### Prohibited Actions
- Don't add complex logging frameworks
- Don't add database persistence
- Don't add web dashboards
- Don't add multi-threaded/async processing (unless explicitly requested)
- Don't create new documentation files unless asked

### Encouraged Actions
- Fix actual bugs
- Improve error messages for clarity
- Add inline comments explaining complex logic
- Keep README synchronized with code
- Validate assumptions (like sorted order preservation)

### Response Style
- Keep answers concise and direct
- For simple questions: 1-2 sentences max
- For consistency checks: bullet list of issues
- For code changes: show exact edits, then update README
- Don't write lengthy explanations unless asked

## File Paths
- Main directory: `D:\VkRetro\YoloDetectExtractFrames\EDA_intra_class_variation_scripts\`
- Core scripts (1, 2, 3): `D:\VkRetro\YoloDetectExtractFrames\EDA_intra_class_variation_scripts\scripts\`
- Master scripts (1.*): Main directory
- README: `D:\VkRetro\YoloDetectExtractFrames\EDA_intra_class_variation_scripts\README.md`
