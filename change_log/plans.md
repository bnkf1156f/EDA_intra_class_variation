# Plans / Bucket List

Planned changes in rough priority order. Move items to `changes.md` once done and tested.

---

## 🔥 HIGH PRIORITY — Slack EDA Service

Turn the EDA pipeline into a Slack-triggered service:
- Slack slash command or bot message: `/eda /path/to/dataset`
- Bot server (Python, runs locally on machine) spawns master script as subprocess
- Streams stdout progress back to Slack thread
- Uploads final PDF to Slack thread when done
- Single user / local machine scope (Option A — no infra needed)
- Need to clarify: stream progress vs just "done + PDF", and TS4 integration angle
- S3 bucket -> pdf 

## PDF Insights & Recommendations Section (last page of PDF)

Auto-generated from `cluster_statistics.csv`. Summary table format (class | issue | recommendation).

**Rules confirmed:**
- `outlier_rate > 8%` → "High outlier rate — review for annotation errors"
- `n_clusters > 15` → "High intra-class variation — verify class definition is consistent"
- `n_clusters <= 2 AND n_samples > 1000` → "Visually homogeneous — OK"
- Severe class imbalance (min/max sample ratio) → "Risk of model bias — consider resampling or loss weighting"
- Uniform + high outliers combo (homogeneous class BUT >8% outliers) → "Possible systematic annotation noise"
- `n_samples < 50` → "Insufficient data — interpret clustering results cautiously"
- High auto-tuned eps (>0.5) → "Embeddings very spread — class may contain fundamentally different visual concepts"
- Downsampled classes → note in table

**Not included (needs model output):**
- mAP before/after outlier removal
- Cross-class embedding overlap

**Implementation:** goes in `4. generate_pdf.py` as final PDF page, reads `cluster_statistics.csv` directly.

## Dashboard instead of PDF
- can be viewed over a dashboard or something with images saved / persisted under assets and if that saved us the hassle of generating documents
- https://diffprog.slack.com/archives/C06Q7DXJDUJ/p1777549467871949

## Missing Labels Handling
- Detect and report frames that are missing annotation label files
- Surface this info clearly in the PDF report or terminal output
- Decide: hard error vs warning vs skip-and-count

## Efficient Embedding Cache Usage
- Improve how embedding cache is read/written — avoid redundant recomputation
- Possibly: hash input folder contents to invalidate cache automatically when images change
- Keep memory-safe (load-process-delete pattern, no full-dataset RAM load)

## Integrate Interactive Cluster Viewer into Master Scripts
- Fold `interactive_cluster_viewer.py` as an optional step inside the master script prompts
- User sees: "Generate interactive HTML viewer? [Y/n]"
- Avoids needing to run it as a separate standalone script

---

## LOW PRIORITY — Architecture Robustness

## Reduce Script 4 Fragility (inter-script coupling)
- Script 4 silently breaks if Script 1's `temp_ann_file.txt` format changes or Script 3's file naming changes
- Options: structured JSON for inter-script data, explicit format versioning, or validation on load
- Not urgent — works fine until a naming refactor breaks it

## Break Up Master Script Monoliths
- Master scripts are long and hard to unit test in isolation
- Refactor into smaller functions/modules that can be tested independently
- Low risk to leave as-is for internal EDA use

## Embedding Cache Auto-Invalidation
- Current: manual delete required when input images change — silent footgun
- Option: hash input folder contents → auto-invalidate cache on change
- Already tracked in item 3 (cache efficiency) but this is the safety angle