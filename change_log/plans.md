# Plans / Bucket List

Planned changes in rough priority order. Move items to `changes.md` once done and tested.

---

## 1. Missing Labels Handling
- Detect and report frames that are missing annotation label files
- Surface this info clearly in the PDF report or terminal output
- Decide: hard error vs warning vs skip-and-count

## 2. Efficient Embedding Cache Usage
- Improve how embedding cache is read/written — avoid redundant recomputation
- Possibly: hash input folder contents to invalidate cache automatically when images change
- Keep memory-safe (load-process-delete pattern, no full-dataset RAM load)

## 3. Smarter Output Folder Naming
- Auto-generate meaningful output folder names based on input dataset name / timestamp
- Reduce burden on user to manually specify output paths at every run
- e.g., `postann_pretrain_results/board_dataset_2026-04-08/` instead of a flat default

## 4. Integrate Interactive Cluster Viewer into Master Scripts
- Fold `interactive_cluster_viewer.py` as an optional step inside the master script prompts
- User sees: "Generate interactive HTML viewer? [Y/n]"
- Avoids needing to run it as a separate standalone script

## 5. PDF Report Improvements (in progress)
- ✅ DINOv2 PreAnn: embedding model, scene/pose weights, outlier count added to summary table
- Remaining (PostTrain + PreTrain scripts): add run config params (eps, min_samples, UMAP/PaCMAP params, input path, class names)
- Consider a dedicated "Run Config" page for reproducibility
