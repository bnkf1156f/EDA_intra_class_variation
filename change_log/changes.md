# Change Log

## Completed Changes (from git history)

- **Initial pipeline** — First working version of intra-class variation EDA pipeline (DBSCAN clustering on DINOv2 embeddings)
- **Requirements** — Added `requirements.txt` with dependencies
- **Outlier sampling** — Save ALL outliers in samples folder; decrease cooling break to 10s; ignore `classes.txt` if found
- **README / docs** — Multiple rounds of README updates, master script documentation, clarifications
- **Interactive clustering** — Added optional interactive cluster viewer (Plotly HTML) after Script 2; moved scripts to separate folder; fixed CPU bug; fixed cluster number mismatch bugs
- **PDF generation (Script 4)** — Added PDF generation combining Script 1 + Script 3 results; optimized and compressed output (~80% size reduction via JPEG quality 75)
- **Tkinter fix** — Fixed threading issue during plots in clustering script
- **Pre-annotation explorer** — Added new script to explore frames/images before annotation
- **YOLO pose model** — Added YOLOv26n as default pose model for pre-annotation
- **SigLIP embeddings** — Replaced DINOv2 with SigLIP (`google/siglip-base-patch16-224`) for pre-annotation; adaptive 90/10 scene-pose weighting; added `utils/test_embedding_speed.py` benchmark
- **Consistent documentation** — Aligned README, CLAUDE.md, and script docstrings
- **PDF improvements** — Added person detection statistics, improved layout, fixed montage rendering bugs, added filenames below thumbnails, resolved PDF height/row-splitting issue
- **Optimal clustering params** — Fixed default parameters for PreAnn script (lower outliers); documented tuning guide in CLAUDE.md (optimal: `min_dist_umap=0.0`)
- **Code consistency fixes** — Fixed consistency issues across scripts; fixed `--save_suffix` parameter in Script 3; fixed cross-class analysis to use median eps when auto-tuned
- **Dataset analysis extras** — Added invalid annotation and class imbalance detection options
- **Interactive prompts** — Replaced hardcoded config with `questionary`-based prompts in both master scripts; pre-training script now supports labels and images in separate folders; uniform-class detection and handling for large/low-variance classes
- **Documentation update** — Updated README and CLAUDE.md
- **Error handling** — Improved handling for missing image files; skip classes with insufficient samples for clustering; accept string inputs for paths
- **Launcher** — Added `exec.bat` launcher for pipeline selection; added interactive prompts across all scripts
- **Project reorganization** — Moved master scripts to `master_scripts/`; extracted PDF module to `utils/`; dropped Post-Training from `exec.bat` (still runnable directly); added CWD warning; V-JEPA 2.1 noted as next embedding experiment
- **Output directory refactor** — Refactored output directory structure
- **requirements.txt versioning** — Pinned package versions with CUDA compatibility
- **Auto-detect splits** — Auto-detect train/val splits in annotation scripts
- **Linux support** — Added `exec.sh` launcher for Linux systems
- **sentencepiece / protobuf** — Added to `requirements.txt`
- **Delete cropped folder prompt** — Added user prompt to delete cropped images folder after pipeline run
- **DINOv2 pre-annotation script** — Replaced SigLIP pre-annotation script with new `master_script_Dinov2_PaCMAP_PreAnn.py`; uses DINOv2 [CLS||avg_patches] (1536d) + PCA whitening + PaCMAP + HDBSCAN; SigLIP script removed
- **PDF report enhancements (DINOv2 PreAnn)** — Added config rows to PDF summary table (embedding model, scene/pose weights); added Noise/Outlier Frames row after Activities Detected; PDF default name now includes frames folder name (`PreAnnotation_Quality_Report_{FolderName}.pdf`); replaced verbose outlier filename list with count + save-to-folder prompt at end of run (copies all outliers to `outliers_{FolderName}/`, warns on overwrite)
- **Smart output naming + PDF improvements (PreTrain)** — Pre-training master script now derives all output folder/PDF name defaults from the images folder name (`{FolderName}`); `cropped_bbox_dir`, `output_cluster_dir`, and `pdf_name` are always prompted (moved out of "Change defaults?" gate) with clear defaults shown. Script 4 (`generate_pdf.py`) fully redesigned: Page 1 replaced with three clean tables (dataset overview, pipeline config, per-class clustering summary — no bar/imbalance charts); Page 2 shows cross-class separability graph if enabled, else all-classes overview; new CLI args added (`--imgs_path`, `--label_path`, `--classes_txt`, `--auto_tune`, `--auto_tune_percentile`, `--epsilon`, `--cross_class`) all wired through master script automatically.
