# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **intra-class variation analysis pipeline** for object detection datasets. It analyzes class heterogeneity using DINOv2 embeddings and clustering to discover sub-groups and outliers.

**Three pipeline modes:**
- **Pre-annotation**: Analyzes raw frames BEFORE annotation to assess quality/diversity (uses DINOv2 + PaCMAP)
- **Pre-training**: Analyzes annotated dataset AFTER annotation, before model training (uses DINOv2 embeddings + YOLO annotation .txt files)
- **Post-training**: Analyzes trained YOLOv8 model detections on video (uses DINOv2 embeddings, validates model quality)

## Core Principles
- **Keep it simple** - This is exploratory EDA, not production code
- **No overcomplications** - Engineers just need to see cluster patterns and outliers
- **Consistency first** - README must match actual script behavior exactly

## Scripts

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `master_scripts/master_script_dinov2_PreAnn.py` | **Pre-annotation**: Frame quality assessment with PDF report (uses DINOv2 + PaCMAP) | BEFORE annotation |
| `master_scripts/master_script_dinov2_PostAnn_PreTrain.py` | Automated pre-training pipeline (uses DINOv2) | AFTER annotation |
| `master_scripts/master_script_dinov2_PostTrain.py` | Automated post-training pipeline (uses DINOv2) | AFTER model training |
| `postannotation_scripts/1. ann_txt_files_crop_bbox.py` | Pre-training: Extract crops from YOLO annotations | Part of pretrain pipeline |
| `postannotation_scripts/1. yolo_model_crop_bbox_per_class.py` | Post-training: Extract crops from model detections | Part of posttrain pipeline |
| `postannotation_scripts/2. save_dinov2_embeddings_per_class.py` | Generate DINOv2 embeddings | Part of both pipelines |
| `postannotation_scripts/3. clustering_of_classes_embeddings.py` | DBSCAN clustering analysis | Part of both pipelines |
| `postannotation_scripts/4. generate_pdf.py` | Generate PDF reports combining dataset stats + clustering results | Part of both pipelines |
| `utils/preann_pdf_generate.py` | PDF/montage generation for pre-annotation pipeline | Imported by DINOv2 PreAnn master script |
| `master_scripts/1. interactive_cluster_viewer.py` | Optional: Interactive Plotly HTML viewer | Optional addon |

## Running the Pipeline

**IMPORTANT**: Always run from the **project root** directory (where `exec.bat` lives). Running from `master_scripts/` will cause path errors.

### Automated (Recommended)
```bash
# Use the launcher — Pre-Annotation and Pre-Training available via menu:
./exec.bat

# Or run directly from project root:
python "master_scripts/master_script_dinov2_PreAnn.py"    # Pre-annotation
python "master_scripts/master_script_dinov2_PostAnn_PreTrain.py"  # Pre-training
python "master_scripts/master_script_dinov2_PostTrain.py"         # Post-training (not in exec.bat)
```

All master scripts use **interactive questionary prompts** — paths use autocomplete with validation, booleans use Yes/No confirms, and niche parameters are hidden behind a "Change default parameters?" gate.

### Manual Step-by-Step

**Pre-annotation workflow** (raw frames → quality PDF):
```bash
python "master_scripts/master_script_dinov2_PreAnn.py"
```

**Pre-training workflow** (annotated images → clustering):
```bash
# Step 1: Pre-training - crop from annotations
python "postannotation_scripts/1. ann_txt_files_crop_bbox.py" --imgs_label_path "path/to/LabelledData" --classes 0 1 2 --class_ids_to_names 0 board 1 screw 2 holder --output_dir postann_pretrain_results/cropped_imgs_by_class

# Step 2: Generate DINOv2 embeddings
python "postannotation_scripts/2. save_dinov2_embeddings_per_class.py" --root ./postann_pretrain_results/cropped_imgs_by_class --batch 32

# Step 3: DBSCAN clustering analysis
python "postannotation_scripts/3. clustering_of_classes_embeddings.py" --root ./postann_pretrain_results/cropped_imgs_by_class --auto_tune --min_samples 3 --save_montage
```

**Post-training workflow** (video + model → clustering):
```bash
# Step 1: Post-training - crop from video using model
python "postannotation_scripts/1. yolo_model_crop_bbox_per_class.py" --model model.pt --video video.mp4 --classes board screw holder --num_frames 100 --frame_stride 3 --output posttrain_results/cropped_images

# Steps 2-3: Same as pre-training (use --root ./posttrain_results/cropped_images)
```

## Architecture

### Pipeline Flow

**Pre-annotation workflow**:
```
Input: Raw frames (.png, .jpg, .jpeg)
    |
    v
[master_scripts/master_script_dinov2_PreAnn.py]
    - Person detection (YOLOv11-pose)
    - DINOv2 [CLS||avg_patches] embeddings (1536d, adaptive: scene weight + pose weight when persons detected)
    - PCA whitening → PaCMAP dimensionality reduction + HDBSCAN clustering
    - Quality metrics (brightness, sharpness, contrast, anisotropy)
    - Lighting categorization
    - Visual montages per activity
    |
    v
Output: preann_results/PreAnnotation_Quality_Report_{FolderName}.pdf (5 pages)
    - Executive summary with quality score (includes embedding model + scene/pose weights + outlier count)
    - Quality metrics dashboard
    - Activity diversity analysis with PaCMAP
    - Visual montages (N samples per activity)
    - Coverage gap heatmap
    - Optional: preann_results/outliers_{FolderName}/ (user prompted at end of run)
```

**Pre-training/Post-training workflow**:

Each master script defines its own `RESULTS_DIR` (`postann_pretrain_results/` for pre-training, `posttrain_results/` for post-training). All outputs go under that directory. Below uses `<results>/` as placeholder:

```
Input (annotated images OR video+model)
    |
    v
[Script 1] Crop bounding boxes per class --> <results>/cropped_imgs_by_class/<class>/*.png
                                             <results>/temp_ann_file.txt (statistics, pre-training only)
    |
    v
[Script 2] DINOv2 embeddings --> <results>/cropped_imgs_by_class/<class>/embeddings_dinov2.npy
                                 <results>/cropped_imgs_by_class/<class>/embeddings_dinov2_image_list.txt
    |
    v
[Script 3] DBSCAN clustering --> <results>/clustering_results/<class>_clusters.png
                                 <results>/clustering_results/<class>_montage.png
                                 <results>/clustering_results/<class>_samples/cluster_N/
                                 <results>/clustering_results/cluster_statistics.csv
    |
    v (optional)
[Script 4] PDF report generator --> <results>/clustering_results/output.pdf (combines Script 1 + 3 results)
[Script 5] Interactive viewer --> <results>/clustering_results/<class>_interactive.html
```

### Key Data Structures
- **Embeddings**:
  - Pre-annotation: 1536-dimensional vectors from `facebook/dinov2-base` (CLS + avg_patches concatenated), with PCA whitening before PaCMAP
  - Pre/Post-training: 768-dimensional vectors from `facebook/dinov2-base`
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
- Interactive questionary prompts (path autocomplete, Yes/No confirms, "Change defaults?" gate)
- Cancel handling: Ctrl+C exits cleanly at any prompt (handles prompt_toolkit RuntimeError on Windows)
- Memory management: GPU cache clearing, garbage collection between steps
- Cooling breaks: 10-second pauses to prevent laptop GPU thermal throttling
- Resource monitoring: Warns if RAM >90% or GPU >85%
- Error handling: Cleanup on failure or keyboard interrupt
- Launcher: `exec.bat` provides menu to select which pipeline to run

## Key Parameters

### Pre-Annotation Script (master_script_dinov2_PreAnn.py)
All parameters are configured via interactive prompts. Clustering params are always asked; niche params only shown if "Change default parameters?" = Yes.

**Always prompted (clustering):**
- `min_cluster_size`: Smallest group of frames that counts as an activity (default 25)
- `min_samples`: How strict a frame must match neighbors to join a cluster (default 3, lower=more frames included)
- `use_embedding_cache`: Cache embeddings to disk for faster re-runs (default True)

**Behind "Change defaults?" gate:**
- `batch_size`: DINOv2 inference batch size (default 64)
- `anisotropy_threshold`: Motion blur detection threshold (2.5 strict, 3.6 balanced, 5.0 lenient)
- `cache_blurry_num_samples`: Blurry sample images in PDF (default 24)
- `grid_cols_activities`/`grid_cols_blurry`: Grid columns for montages (default 4)

### Clustering (Script 3)
- `--eps`: DBSCAN epsilon (0.1 strict, 0.15 balanced, 0.2-0.3 lenient)
- `--min_samples`: Minimum points to form cluster (2-3 typical)
- `--auto_tune`: Recommended - uses k-NN distance percentile per class
- `--auto_tune_percentile`: k-NN percentile for auto-tune (default: 90; 90=tight, 95=balanced, 98=loose)
- `--umap_min_dist`: UMAP min_dist parameter (default: 0.05; 0.0=tight, 0.1=loose)
- `--save_suffix`: Embedding filename to load (default: `embeddings_dinov2.npy`, must match Script 2 output)
- `--max_samples`: Sample images per cluster (default: 5, outliers are all saved) — master script overrides to 20
- `--pca_components`: PCA dims before DBSCAN (default: 128; 0=disabled). Reduces 768d→Nd before clustering — improves DBSCAN on large classes (curse of dimensionality). Applied per-class.
- `--uniform_eps_threshold`: If auto-tuned eps < this value, class is considered uniform (default: 0.10)
- `--uniform_downsample_target`: Target sample count when downsampling uniform classes (default: 5000) — master script overrides to 4000
- `--uniform_min_samples`: Only downsample if class has more than this many samples (default: 10000) — master script overrides to 12000

### Video Processing (Script 1 post-training)
- `--frame_stride`: Process every Nth frame (3 = 10fps for 30fps video)
- `--num_frames`: Target samples per class (uniform sampling)
- `--conf_thresh`: Detection confidence threshold (default 0.4)

## Known Quirks

### Pre-Annotation Script
1. **Embedding Model**: Uses DINOv2 (`facebook/dinov2-base`) with CLS token + avg_patches concatenated = 1536d
2. **Activity Clustering**: PCA whitening → PaCMAP (dimensionality reduction) → HDBSCAN (density-based clustering)
3. **Adaptive Embeddings**:
   - Frames WITH persons: scene (1536d) + pose features, weighted and concatenated
   - Frames WITHOUT persons: scene embeddings (1536d) + zero-padded pose features
   - Weights applied to CONCATENATED dimensions (not blended scalars)
4. **Visual Montages**: Creates configurable grid (default 4 columns) with variable sample counts per activity
5. **Lighting Thresholds**: Fixed thresholds (Dark <80, Medium 80-175, Bright >175) based on grayscale mean
6. **Quality Score**: Computed from median brightness/anisotropy/contrast, scale 0-10 (lower anisotropy = sharper)
7. **Motion Blur Detection**: Uses directional gradient anisotropy (Sobel X/Y energy ratio) instead of Laplacian variance
8. **Person Detection**: Uses YOLOv11-pose (conf=0.3 default) to extract pose features when persons visible

### Pre-Training/Post-Training Scripts
1. **Image-Embedding Alignment**: Script 2 creates a mapping file with name derived from `--save_suffix` parameter (e.g., `embeddings_dinov2.npy` → `embeddings_dinov2_image_list.txt`). Script 3 must receive the same `--save_suffix` value or alignment will fail. The mapping file ensures correct correspondence even when corrupted images are skipped.
2. **Sorted Order**: Both scripts use `sorted()` on file paths to maintain consistent ordering
3. **CSV Saving**: Script 3 has CSV saving enabled (not commented out)
4. **Outlier Sampling**: Script 3 saves ALL outlier images (not sampled), while regular clusters are sampled up to `--max_samples`
5. **Montage rendering**: Script 3 uses matplotlib for montages (layout: `n_cols × n_rows` grid at 200 DPI). Images are pre-decoded and resized via PIL before passing to `imshow` for speed. Layout and visual output identical to original.
6. **PDF Image Splitting**: Script 4 splits tall montages at cluster row boundaries (never mid-cluster) by reading `cluster_statistics.csv` to determine number of rows
7. **PDF Image Compression**: Script 4 converts all PNG montages to JPEG quality 75 for ~80% file size reduction (configurable via `--pdf_quality`)
8. **PIL Decompression Bomb**: Script 4 disables PIL's `MAX_IMAGE_PIXELS` limit to handle large montages (25+ clusters)
9. **PDF Page 1 layout**: Dataset overview table → Pipeline config table → Annotation distribution bar chart (always) → Contrastive group chart (only if `--contrastive_groups_json` provided, shown after plain chart) → Per-class clustering summary table. Annotation issue files detail section rendered if any issues found.
10. **PDF Page 2+ — centroid overview**: Pages 2+ show `centroid_overview_N.png` plots (one per page). `--cross_class` flag removed; replaced by `--save_class_scatter` for per-class UMAP scatter (off by default).
11. **PDF args from master script**: Script 4 accepts `--imgs_path`, `--label_path`, `--classes_txt`, `--auto_tune`, `--auto_tune_percentile`, `--epsilon`, `--pdf_quality`, `--contrastive_groups_json` — all passed automatically by the pre-training master script.
12. **Reuse crops/embeddings**: Master script pre-flight detects existing `.npy` embeddings in cropped folder — offers to skip Steps 1 & 2 and jump straight to clustering. Folder is `shutil.rmtree`-deleted before overwrite (not just warned).
13. **Contrastive distribution chart**: When `--contrastive_groups_json` is provided (JSON `{group_name: [class_a, class_b]}`), Script 4 renders a second grouped horizontal bar chart **after** the plain annotation chart (plain chart always shown). Only grouped classes appear in this chart — ungrouped classes are not appended. One row per group, sub-bars per variant, palette `['#3a7fc1', '#d95f49', ...]`. Legend labels derived from last `_`-token of class name at each variant position (if all groups agree on token); falls back to `Variant N`. Master script prompts via `questionary.checkbox` — user ticks class names to form groups, auto-names as `"A vs B"` (editable). Backward compatible — no arg = plain chart only.
14. **PDF Insights & Recommendations page**: Script 4 appends a final page after montages — color-coded flag table per class (🔴 Critical / 🟡 Warning / 🟢 OK / ℹ Info). 17 rules evaluated: outlier rate thresholds, cluster count, all-noise, tight-core+noise, homogeneous OK, sample count, systematic noise, eps spread, over-fragmentation, over-clustering, zero outliers, class imbalance. Dataset-level imbalance (max/min ratio) added as cross-class row. Classes with no flags get green "✓ No issues". `n_samples` parsed as both plain int and `"clustered/total"` string.

## File Naming Patterns

### Pre-Annotation Script
- Output: `preann_results/PreAnnotation_Quality_Report_{FolderName}.pdf`
- Outliers: `preann_results/outliers_{FolderName}/` (user prompted at end of run; warns on overwrite)
- Temp files: `temp_preannotation_charts/` (auto-cleaned after PDF generation)
- Embedding cache files: `temp_multiview_emb_indices.npy` and `temp_multiview_emb.npy` in output_dir/ (DINOv2 embeddings, should be deleted before next run if inputs changed!)

### Pre-Training/Post-Training Scripts (under `postann_pretrain_results/` or `posttrain_results/` respectively)
- Script 1a outputs: `{basename}_crop_{idx}.jpg` (JPEG quality 95) + `temp_ann_file.txt`
- Script 1b outputs: `frame_{frame_idx:06d}.png`
- Script 2 outputs: `{save_suffix}.npy` + `{save_suffix_without_.npy}_image_list.txt`
  - Default: `embeddings_dinov2.npy` + `embeddings_dinov2_image_list.txt`
  - Custom: `custom_emb.npy` + `custom_emb_image_list.txt`
- Script 3 outputs: `{class}_clusters.png` (if `--save_class_scatter`), `{class}_montage.png` (if `--save_montage`), `cluster_statistics.csv`, `centroid_overview_N.png`
- Script 4 outputs: `{pdf_name}.pdf` + `temp_pdf_charts/` directory with temp JPEG chunks (auto-cleaned)

### Pre-Training Master Script Output Naming (folder-name-derived)
The pre-training master script derives all output names from the images folder name (`{FolderName}`) at prompt time:
- Cropped images: `postann_pretrain_results/cropped_imgs_by_class_{FolderName}/`
- Clustering results: `postann_pretrain_results/clustering_results_{FolderName}/`
- PDF report: `postann_pretrain_results/clustering_results_{FolderName}/PDF_REPORT_{FolderName}.pdf`
These are always prompted with the derived default shown — user can accept or override.

## Dependencies

- PyTorch >= 2.1 with CUDA
- ultralytics (YOLOv8/YOLOv11)
- transformers (DINOv2 for all pipelines)
- scikit-learn (DBSCAN, PCA)
- pacmap (pre-annotation PaCMAP dimensionality reduction)
- hdbscan (pre-annotation clustering)
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

### Change Log Protocol

The `change_log/` directory must stay current:
- `change_log/changes.md` — completed changes (what was done)
- `change_log/plans.md` — upcoming work (bucket list)

**When user says "I'm ready to commit" (or similar):**
1. Ask for a brief description of what was done if not already clear
2. Update `changes.md`: add a new bullet at the top under "Completed Changes" summarizing the work
3. Update `plans.md`: if the work completes a planned item, remove or strike it from the list
4. Stop — the user handles the actual git commit

**Ongoing maintenance:**
- When user describes new plans or ideas → add to `plans.md`
- When work on a plan item starts → note "in progress" on that item
- Never let the log go stale: every "ready to commit" = a `changes.md` update

**Fully done vs partially done — CRITICAL:**
- **Fully done** → remove item from `plans.md` entirely, add a bullet to `changes.md`
- **Partially done** → keep in `plans.md` with updated status notes (e.g. `✅ X done, remaining: Y`)
- **NEVER** leave a `✅ DONE` item sitting in `plans.md` — it belongs in `changes.md`
- After removing items, renumber the remaining `plans.md` entries

---

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
- Master scripts: `<project_root>/master_scripts/`
- Core scripts (1, 2, 3) for Post-annotation: `<project_root>/postannotation_scripts/`
- Utility modules: `<project_root>/utils/` (e.g., `preann_pdf_generate.py`)
- README: `<project_root>/README.md`
