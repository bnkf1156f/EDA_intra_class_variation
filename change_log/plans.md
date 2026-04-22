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

## 3. Integrate Interactive Cluster Viewer into Master Scripts
- Fold `interactive_cluster_viewer.py` as an optional step inside the master script prompts
- User sees: "Generate interactive HTML viewer? [Y/n]"
- Avoids needing to run it as a separate standalone script