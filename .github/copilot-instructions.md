# GitHub Copilot Instructions for EDA Intra-Class Variation Pipeline

## Project Context
This is a 3-script pipeline for analyzing intra-class variations in object detection datasets:
1. **Script 1a** (Pre-training): `1. ann_txt_files_crop_bbox.py` - Extract crops from YOLO annotations
2. **Script 1b** (Post-training): `1. yolo_model_crop_bbox_per_class.py` - Extract crops from YOLO model detections
3. **Script 2**: `2. save_dinov2_embeddings_per_class.py` - Generate DINOv2 embeddings
4. **Script 3**: `3. clustering_of_classes_embeddings.py` - DBSCAN clustering analysis

## Core Principles
- **Keep it simple** - This is exploratory EDA, not production code
- **No overcomplications** - Engineers just need to see cluster patterns and outliers
- **Consistency first** - README must match actual script behavior exactly

## When Reviewing/Updating Code

### Always Check These Files Together:
1. Read the specific script being modified
2. Read `README.md` sections for that script
3. Verify parameter tables, usage examples, and output descriptions match

### Critical Consistency Points:
- **Script names**: Use full filenames exactly (e.g., `1. yolo_model_crop_bbox_per_class.py`, not `1. yolo_model_crop_bbox.py`)
- **Parameter defaults**: Must match argparse defaults in scripts
- **Output file structures**: Directory trees in README must match actual script output
- **File naming patterns**: 
  - Script 1a outputs: `{basename}_crop_{idx}.png`
  - Script 1b outputs: `frame_{frame_idx:06d}.png`
  - Script 2 outputs: `embeddings_dinov2.npy` + `embeddings_dinov2_image_list.txt`

### Known Quirks:
1. **Image-Embedding Alignment**: Script 2 creates a mapping file (`embeddings_dinov2_image_list.txt`) because corrupted images are skipped. Script 3 reads this to ensure correct correspondence.
2. **Sorted Order**: Both scripts use `sorted()` on file paths to maintain consistent ordering
3. **CSV Saving**: Script 3 has CSV saving enabled (not commented out)

## README Update Protocol

### When Script Parameters Change:
1. Find the parameter table in README under that script's section
2. Update the exact row with new default/description
3. Check if the change affects workflow examples at the bottom

### When Output Format Changes:
1. Update the "Output Structure" code block
2. Update the "Directory Structure" if Script 2
3. Update "Output Files" if Script 3

### When Adding Features:
1. Add to "Features" or "What It Does" list
2. Update "Technical Details" if relevant
3. Add to "Common Issues" if it might confuse users

## Code Review Checklist

When asked to review consistency:
- [ ] Read all 3-4 scripts completely
- [ ] Read entire README.md
- [ ] Check script filenames in README table (line ~329)
- [ ] Verify parameter tables match argparse definitions
- [ ] Confirm output structures match actual `os.makedirs()` and file writes
- [ ] Check usage examples have correct script names and parameters

## Common User Requests

### "Update README after code change"
1. Ask which script was changed
2. Read that script completely
3. Find corresponding README section
4. Update only the affected parts (parameters/output/features)

### "Check consistency"
1. Read all scripts
2. Read full README
3. Report mismatches with line numbers
4. Focus on functional issues, not stylistic preferences

### "Fix bug/improve script"
1. **First ask**: "Is this for simplicity or production robustness?"
2. If simplicity: minimal change only
3. If robustness: verify it doesn't break the simple workflow
4. Always update README after code changes

## README Structure Reference
```
Line ~1-15:    Pipeline overviews (PRE-TRAINING, POST-TRAINING)
Line ~17-60:   Script 1a (ann_txt_files_crop_bbox.py)
Line ~62-110:  Script 1b (yolo_model_crop_bbox_per_class.py)  
Line ~112-180: Script 2 (save_dinov2_embeddings_per_class.py)
Line ~182-280: Script 3 (clustering_of_classes_embeddings.py)
Line ~282-310: Complete Workflows
Line ~312-338: Pipeline Summary + Key Takeaways
```

## Script Interdependencies
- Script 1a/1b → Creates class folders with cropped images
- Script 2 → Reads class folders, creates `.npy` + `_image_list.txt`
- Script 3 → Reads embeddings + mapping file, creates visualizations

**Critical**: If Script 2 changes output format, Script 3 must be updated to match!

## Prohibited Actions
- ❌ Don't add complex logging frameworks
- ❌ Don't add database persistence
- ❌ Don't add web dashboards
- ❌ Don't add multi-threaded/async processing (unless explicitly requested)
- ❌ Don't create new documentation files unless asked

## Encouraged Actions
- ✅ Fix actual bugs (like the image-embedding alignment issue)
- ✅ Improve error messages for clarity
- ✅ Add inline comments explaining complex logic
- ✅ Keep README synchronized with code
- ✅ Validate assumptions (like sorted order preservation)

## Quick Reference: File Paths
- Scripts: `c:\VkRetro\YoloDetectExtractFrames\EDA_intra_class_variation_scripts\<script_name>.py`
- README: `c:\VkRetro\YoloDetectExtractFrames\EDA_intra_class_variation_scripts\README.md`

## Response Style
- Keep answers concise and direct
- For simple questions: 1-2 sentences max
- For consistency checks: bullet list of issues
- For code changes: show exact edits, then update README
- Don't write lengthy explanations unless asked

---

**Last Updated**: After fixing image-embedding alignment bug (Nov 2025)
