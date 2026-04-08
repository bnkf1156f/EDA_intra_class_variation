# Plans / Bucket List

Planned changes in rough priority order. Move items to `changes.md` once done and tested.

---

## 1. DINOv2 for Pre-Annotation (in progress)
- Replace SigLIP with DINOv2 for the pre-annotation pipeline
- Adapt embedding logic (pose weighting, scene weighting) to work with DINOv2 output
- Test and validate clustering quality vs current SigLIP baseline
- Commit only after tested and confirmed working
- *Status: in progress — do NOT commit until ready*

## 2. Missing Labels Handling
- Detect and report frames that are missing annotation label files
- Surface this info clearly in the PDF report or terminal output
- Decide: hard error vs warning vs skip-and-count

## 3. Efficient Embedding Cache Usage
- Improve how embedding cache is read/written — avoid redundant recomputation
- Possibly: hash input folder contents to invalidate cache automatically when images change
- Keep memory-safe (load-process-delete pattern, no full-dataset RAM load)

## 4. Smarter Output Folder Naming
- Auto-generate meaningful output folder names based on input dataset name / timestamp
- Reduce burden on user to manually specify output paths at every run
- e.g., `postann_pretrain_results/board_dataset_2026-04-08/` instead of a flat default

## 5. Integrate Interactive Cluster Viewer into Master Scripts
- Fold `interactive_cluster_viewer.py` as an optional step inside the master script prompts
- User sees: "Generate interactive HTML viewer? [Y/n]"
- Avoids needing to run it as a separate standalone script

## 6. PDF Report Improvements
- Add user-configured parameters to the PDF (what settings were used: eps, min_samples, UMAP params, etc.)
- Add model weights / embedding model name used
- Add dataset input path and class names
- Make it a reproducibility record, not just a results dump
- Consider a "Run Config" page or summary table at the start of the PDF
