# Plans / Bucket List

Planned changes in rough priority order. Move items to `changes.md` once done and tested.

---

## 1. Degenerate Crop Detection + PDF Surface [PRIORITY 1]
- Script 2: detect tiny/near-zero crops (e.g. `min(W,H) < 8px`) during embedding, log to `degenerate_crops.txt` per class folder
- Script 4 PDF: read that file → warning box or section "X degenerate crops in class Y" + optional thumbnail montage
- Note: DBSCAN likely catches these as outliers anyway — PDF callout makes it explicit for engineers
- Decision pending: full tracking in PDF vs trust DBSCAN to surface naturally

## 2. Missing Labels Handling
- Detect and report frames that are missing annotation label files
- Surface this info clearly in the PDF report or terminal output
- Decide: hard error vs warning vs skip-and-count

## 3. Efficient Embedding Cache Usage
- Improve how embedding cache is read/written — avoid redundant recomputation
- Possibly: hash input folder contents to invalidate cache automatically when images change
- Keep memory-safe (load-process-delete pattern, no full-dataset RAM load)

## 4. Integrate Interactive Cluster Viewer into Master Scripts
- Fold `interactive_cluster_viewer.py` as an optional step inside the master script prompts
- User sees: "Generate interactive HTML viewer? [Y/n]"
- Avoids needing to run it as a separate standalone script

---

## LOW PRIORITY — Architecture Robustness

## 6. Reduce Script 4 Fragility (inter-script coupling)
- Script 4 silently breaks if Script 1's `temp_ann_file.txt` format changes or Script 3's file naming changes
- Options: structured JSON for inter-script data, explicit format versioning, or validation on load
- Not urgent — works fine until a naming refactor breaks it

## 7. Break Up Master Script Monoliths
- Master scripts are long and hard to unit test in isolation
- Refactor into smaller functions/modules that can be tested independently
- Low risk to leave as-is for internal EDA use

## 8. Embedding Cache Auto-Invalidation
- Current: manual delete required when input images change — silent footgun
- Option: hash input folder contents → auto-invalidate cache on change
- Already tracked in item 3 (cache efficiency) but this is the safety angle