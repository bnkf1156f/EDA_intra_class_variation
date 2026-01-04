# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **intra-class variation analysis pipeline** for object detection datasets. It analyzes class heterogeneity using DINOv2 embeddings and clustering to discover sub-groups and outliers.

**Three pipeline modes:**
- **Pre-annotation**: Analyzes raw frames BEFORE annotation to assess quality/diversity (prevents wasting annotation effort)
- **Pre-training**: Analyzes annotated dataset AFTER annotation, before model training (uses YOLO annotation .txt files)
- **Post-training**: Analyzes trained YOLOv8 model detections on video (validates model quality)

## Core Principles
- **Keep it simple** - This is exploratory EDA, not production code
- **No overcomplications** - Engineers just need to see cluster patterns and outliers
- **Consistency first** - README must match actual script behavior exactly

## Scripts

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `1. master_script_dinov2_PreAnnotate.py` | **Pre-annotation**: Frame quality assessment with PDF report | BEFORE annotation |
| `1. master_script_dinov2_pretrain.py` | Automated pre-training pipeline | AFTER annotation |
| `1. master_script_dinov2_posttrain.py` | Automated post-training pipeline | AFTER model training |
| `postannotation_scripts/1. ann_txt_files_crop_bbox.py` | Pre-training: Extract crops from YOLO annotations | Part of pretrain pipeline |
| `postannotation_scripts/1. yolo_model_crop_bbox_per_class.py` | Post-training: Extract crops from model detections | Part of posttrain pipeline |
| `postannotation_scripts/2. save_dinov2_embeddings_per_class.py` | Generate DINOv2 embeddings | Part of both pipelines |
| `postannotation_scripts/3. clustering_of_classes_embeddings.py` | DBSCAN clustering analysis | Part of both pipelines |
| `postannotation_scripts/4. generate_pdf.py` | Generate PDF reports combining dataset stats + clustering results | Part of both pipelines |
| `1. interactive_cluster_viewer.py` | Optional: Interactive Plotly HTML viewer | Optional addon |

## Running the Pipeline

### Automated (Recommended)
```bash
# PRE-ANNOTATION: Frame quality assessment (BEFORE annotation)
python "1. master_script_dinov2_PreAnnotate.py"

# PRE-TRAINING: Intra-class analysis (AFTER annotation, before training)
python "1. master_script_dinov2_pretrain.py"

# POST-TRAINING: Model detection analysis (AFTER model training)
python "1. master_script_dinov2_posttrain.py"
```

Configure global variables at the top of `main()` in each master script before running.

### Manual Step-by-Step

**Pre-annotation workflow** (raw frames → quality PDF):
```bash
# Single script, no steps - directly outputs PDF
python "1. master_script_dinov2_PreAnnotate.py"
```

**Pre-training workflow** (annotated images → clustering):
```bash
# Step 1: Pre-training - crop from annotations
python "postannotation_scripts/1. ann_txt_files_crop_bbox.py" --imgs_label_path "path/to/LabelledData" --classes 0 1 2 --class_ids_to_names 0 board 1 screw 2 holder --output_dir cropped_imgs

# Step 2: Generate DINOv2 embeddings
python "postannotation_scripts/2. save_dinov2_embeddings_per_class.py" --root ./cropped_imgs --batch 32

# Step 3: DBSCAN clustering analysis
python "postannotation_scripts/3. clustering_of_classes_embeddings.py" --root ./cropped_imgs --auto_tune --min_samples 3 --save_montage --cross_class
```

**Post-training workflow** (video + model → clustering):
```bash
# Step 1: Post-training - crop from video using model
python "postannotation_scripts/1. yolo_model_crop_bbox_per_class.py" --model model.pt --video video.mp4 --classes board screw holder --num_frames 100 --frame_stride 3

# Steps 2-3: Same as pre-training (use --root ./cropped_images)
```

## Architecture

### Pipeline Flow

**Pre-annotation workflow**:
```
Input: Raw frames (.png, .jpg, .jpeg)
    |
    v
[1. master_script_dinov2_PreAnnotate.py]
    - Person detection (YOLOv8-pose)
    - DINOv2 embeddings (adaptive: pose+scene OR scene-only)
    - Activity clustering (UMAP + HDBSCAN)
    - Quality metrics (brightness, sharpness, contrast, anisotropy)
    - Lighting categorization
    - Visual montages per activity
    |
    v
Output: PreAnnotation_Quality_Report.pdf (5 pages)
    - Executive summary with quality score
    - Quality metrics dashboard
    - Activity diversity analysis with UMAP
    - Visual montages (12 samples per activity)
    - Coverage gap heatmap
```

**Pre-training/Post-training workflow**:
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

### Pre-Annotation Script (1. master_script_dinov2_PreAnnotate.py)
- `frames_dir`: Root directory with extracted frame images
- `anisotropy_threshold`: Motion blur detection threshold (2.5 strict, 3.6 balanced, 5.0 lenient)
- `use_embedding_cache`: Cache embeddings to disk for faster re-runs (True recommended)
- `batch_size`: DINOv2 inference batch size (default 64)
- `cache_blurry_num_samples`: Number of blurry sample images to show in PDF (default 24)
- `grid_cols_activities`: Grid columns for activity montages (default 3)
- `grid_cols_blurry`: Grid columns for blurry sample montage (default 3)

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

### Pre-Annotation Script
1. **Activity Clustering**: Uses UMAP (n_components dims) + HDBSCAN (density-based clustering), NOT hierarchical clustering or PCA
2. **Adaptive Embeddings**:
   - Frames WITH persons: 70% scene (768d) + 30% lightweight pose features (24d) = 792 total dims
   - Frames WITHOUT persons: 100% scene embeddings (768d) + zero-padded pose (24d) = 792 total dims
3. **Visual Montages**: Creates configurable grid (default 3 columns) with variable sample counts per activity
4. **Lighting Thresholds**: Fixed thresholds (Dark <80, Medium 80-175, Bright >175) based on grayscale mean
5. **Quality Score**: Computed from median brightness/anisotropy/contrast, scale 0-10 (lower anisotropy = sharper)
6. **Motion Blur Detection**: Uses directional gradient anisotropy (Sobel X/Y energy ratio) instead of Laplacian variance
7. **Person Detection**: Uses YOLOv11-pose (conf=0.3 default) to extract pose features when persons visible

### Pre-Training/Post-Training Scripts
1. **Image-Embedding Alignment**: Script 2 creates a mapping file with name derived from `--save_suffix` parameter (e.g., `embeddings_dinov2.npy` → `embeddings_dinov2_image_list.txt`). Script 3 must receive the same `--save_suffix` value or alignment will fail. The mapping file ensures correct correspondence even when corrupted images are skipped.
2. **Sorted Order**: Both scripts use `sorted()` on file paths to maintain consistent ordering
3. **CSV Saving**: Script 3 has CSV saving enabled (not commented out)
4. **Cross-Class Epsilon**: When `--auto_tune` is enabled with `--cross_class`, Script 3 uses the median of per-class eps values for cross-class DBSCAN clustering
5. **Outlier Sampling**: Script 3 saves ALL outlier images (not sampled), while regular clusters are sampled up to `--max_samples`
6. **PDF Image Splitting**: Script 4 splits tall montages at cluster row boundaries (never mid-cluster) by reading `cluster_statistics.csv` to determine number of rows
7. **PDF Image Compression**: Script 4 converts all PNG montages to JPEG quality 90 for ~80% file size reduction
8. **PIL Decompression Bomb**: Script 4 disables PIL's `MAX_IMAGE_PIXELS` limit to handle large montages (25+ clusters)

## File Naming Patterns

### Pre-Annotation Script
- Output: `frame_analysis_results/PreAnnotation_Quality_Report.pdf`
- Temp files: `temp_preannotation_charts/` (auto-cleaned after PDF generation)
- Embedding files: `temp_multiview_emb_indices.npy` and `temp_multiview_emb.npy` in output_dir/ which should be deleted before next run!

### Pre-Training/Post-Training Scripts
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

### 🚨 CRITICAL LESSON: Understand the Request Type

**The fundamental mistake (2026-01-02 - Review vs Implementation):**
- User asked to **"REVIEW for logic and clean code"**
- Claude immediately started **IMPLEMENTING FIXES** without permission
- Claude made multiple code changes (grid_cols, keypoint storage, montage cleanup) without asking
- User expected a **REPORT ONLY**, not automatic changes

**MANDATORY PROTOCOL - Distinguish Request Types:**

**"Review" / "Check" / "Analyze" / "Find" = REPORT ONLY**
- ✅ Generate findings report
- ✅ List issues with line numbers and impact
- ✅ Explain what's wrong and why
- ❌ **DO NOT make ANY code changes**
- ❌ **DO NOT implement fixes**
- ✅ STOP after delivering the report
- ✅ WAIT for user to tell you which fixes they want

**"Fix" / "Implement" / "Change" / "Update" = ASK FIRST, THEN EDIT**
- ✅ Ask which specific issues to fix
- ✅ Discuss tradeoffs and options
- ✅ Get explicit approval
- ✅ Then make changes
- ❌ **DO NOT assume which fixes user wants**

**Examples:**

❌ **BAD - What Claude did:**
```
User: "Review for logic and inefficiencies"
Claude: *immediately makes 10 code changes*
User: "I NEVER ASKED YOU TO CHANGE THINGS!!!"
```

✅ **GOOD - What Claude should do:**
```
User: "Review for logic and inefficiencies"
Claude: *generates detailed report of 5 issues found*
Claude: "I found these issues. Which ones would you like me to fix?"
User: "Fix issues 1 and 3"
Claude: *fixes only issues 1 and 3*
```

**Key principle: Do ONLY what was asked. Don't add extra work without permission. STOP after completing the request.**

---

### 🚨 CRITICAL: ASK FIRST, EDIT SECOND

**NEVER jump straight to implementing fixes without discussion!**

**The fundamental mistake (2026-01-02):**
- User reported: `TypeError: create_activity_montage() missing 1 required positional argument: 'target_height'`
- Claude immediately assumed `target_height=180` (matching internal default at line 874)
- Claude applied the fix WITHOUT asking the user about intent or desired thumbnail size
- User had actually set BOTH activities and outliers to `target_height=280` for better PDF visibility
- Claude's assumption was WRONG and showed lack of understanding of user's design goals

**MANDATORY PROTOCOL - When encountering bugs, missing parameters, or requested changes:**

1. **STOP and READ the context**
   - What is the user trying to accomplish?
   - Are there existing patterns in the code that reveal intent?
   - What are the tradeoffs involved?

2. **ASK BEFORE ACTING**
   - "I see `target_height` is missing. The function has an internal default of 180, but the outlier montage uses 280. Which value should I use for regular activities, and why?"
   - "Do you want uniform thumbnail sizes across all montages, or different sizes for different purposes?"
   - "Should this be a configurable parameter instead of hardcoded?"

3. **DISCUSS TRADEOFFS**
   - Explain what each option means (e.g., "280 = larger thumbnails = better visibility but fewer fit per page")
   - Present options and let the user choose
   - Don't assume you know the "correct" answer

4. **PROPOSE, DON'T PRESUME**
   - "I can set it to 180 (matches internal default) or 280 (matches outliers). Which do you prefer?"
   - Wait for approval before making changes
   - If unclear, default to asking rather than guessing

**Examples of GOOD vs BAD responses:**

❌ **BAD (What Claude did):**
```
Fixed! Added target_height=180 parameter. You can re-run now.
```

✅ **GOOD (What Claude should have done):**
```
I found the issue - line 1973 is missing the target_height parameter.

I see two different values in the code:
- Internal default (line 874): 180
- Outlier montage (line 1986): 280

Question: Should regular activity montages use 180 or 280?
(280 = larger thumbnails for better visibility, 180 = more compact)

Or should I make this a configurable parameter at the top of main()?
```

**Key principle: The user has REASONS for their choices. Respect them by asking first.**

### 🚨 CRITICAL: Memory Management at Scale

**NEVER optimize without understanding scale and hardware constraints!**

**The catastrophic mistake (2025-12-31):**
- Assumed "load-once, reuse" was optimal
- With 10K images @ 2-4MB each = **20-40GB RAM** (instant laptop crash on 8GB system)
- User's laptop hit 97% RAM before shutdown

**GOLDEN RULES:**
1. **ASK ABOUT SCALE FIRST** - 100 images vs 10K images = completely different architecture
2. **ASK ABOUT HARDWARE** - 8GB vs 32GB RAM changes everything
3. **RAM >> Speed** - Disk I/O (1-2 sec) is FREE compared to RAM exhaustion (system crash)
4. **Load-process-delete pattern** - For large datasets (5K+ items), ALWAYS reload data in batches rather than holding in RAM
5. **Test assumptions** - "Avoid redundant I/O" is WRONG when it causes memory bloat

**Memory optimization checklist for large-scale scripts:**
- [ ] Load images in batches, delete immediately after processing
- [ ] Delete matplotlib figures after saving (`plt.close('all')` + `gc.collect()`)
- [ ] Delete PIL montages after saving to disk
- [ ] Remove unused data from analysis dicts (e.g., embeddings not needed in PDF)
- [ ] Free GPU memory after each batch (`torch.cuda.empty_cache()`)
- [ ] Use `del` + `gc.collect()` aggressively between processing steps

**When in doubt:**
- Ask user about dataset size
- Ask user about RAM constraints
- Propose tradeoffs (speed vs memory) and let USER decide

### When Reviewing/Updating Code

**Always check these files together:**
1. Read the specific script being modified
2. Read `README.md` sections for that script
3. Verify parameter tables, usage examples, and output descriptions match

**Critical consistency points:**
- **Script names**: Use full filenames exactly (e.g., `postannotation_scripts/1. yolo_model_crop_bbox_per_class.py`)
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
- **NEVER create new documentation files** - Only update README.md and CLAUDE.md
- Don't create separate guide files (e.g., ACTIVITY_VALIDATION_GUIDE.md) - integrate into existing docs

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
- Main directory: `<project_root>/`
- Core scripts (1, 2, 3) for Post-annotation: `<project_root>/postannotation_scripts/`
- Master scripts (1.*): Main directory
- README: `<project_root>/README.md`
