"""
Annotation TXT Files → Cropped Bounding Box Images by Class

This script processes YOLO-format labelled images (.png/.jpg and .txt pairs), and extracts
cropped object images from bounding boxes for specified class IDs.

Images and labels can be in the same folder or separate folders.
Supports train/val/test split directories automatically.

Usage (same folder):
    python "ann_txt_files_crop_bbox.py" \
        --imgs_label_path "path/to/LabelledData" \
        --classes 0 1 2 \
        --class_ids_to_names 0 class0 1 class1 2 class2 \
        --output_dir cropped_imgs_by_class

Usage (separate folders):
    python "ann_txt_files_crop_bbox.py" \
        --imgs_path "path/to/images" \
        --label_path "path/to/labels" \
        --classes 0 1 2 \
        --class_ids_to_names 0 class0 1 class1 2 class2 \
        --output_dir cropped_imgs_by_class

Usage (split dataset - auto-detected):
    python "ann_txt_files_crop_bbox.py" \
        --imgs_path "path/to/images"   \
        --label_path "path/to/labels"  \
        --classes 0 1 2 \
        --class_ids_to_names 0 class0 1 class1 2 class2 \
        --output_dir cropped_imgs_by_class
    # If images/train/, images/val/, labels/train/, labels/val/ exist,
    # both splits are processed automatically.

This script:
 - Validates dataset consistency (ensures each image has a label file and vice versa)
 - Verifies provided class IDs to extract/target exist in the provided mapping
 - Converts YOLO normalized coordinates to pixel bounding boxes
 - Crops and saves bounding boxes into per-class folders
 - Prints a summary of all crops extracted per class
 - Save important info in temporary txt file if argument sent

"""

import os
import cv2
import argparse
from tqdm import tqdm

def yolo_to_bbox(x_center, y_center, width, height, img_width, img_height):
    """Convert YOLO format (normalized) to absolute pixel coordinates"""
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    width_abs = width * img_width
    height_abs = height * img_height

    x1 = int(x_center_abs - width_abs / 2)
    y1 = int(y_center_abs - height_abs / 2)
    x2 = int(x_center_abs + width_abs / 2)
    y2 = int(y_center_abs + height_abs / 2)

    return x1, y1, x2, y2


def _detect_splits(imgs_dir, label_dir):
    """Detect train/val/test split subdirectories.
    Returns list of (split_name, img_split_dir, label_split_dir).
    If no splits found, returns [('', imgs_dir, label_dir)]."""
    splits = []
    for split in ['train', 'val', 'test']:
        si = os.path.join(imgs_dir, split)
        sl = os.path.join(label_dir, split)
        if os.path.isdir(si) and os.path.isdir(sl):
            splits.append((split, si, sl))
    if splits:
        return splits
    return [('', imgs_dir, label_dir)]


def main():
    p = argparse.ArgumentParser(description="Labelled Frames -> sampled cropped images by class")
    p.add_argument("--imgs_label_path", help="Path containing both images AND labels (use this OR --imgs_path + --label_path)")
    p.add_argument("--imgs_path", help="Path to images folder (use with --label_path for separate folders)")
    p.add_argument("--label_path", help="Path to labels folder (use with --imgs_path for separate folders)")
    p.add_argument("--classes", nargs="+", required=True, help="Class IDs required")
    p.add_argument("--class_ids_to_names", nargs="+", required=True, help="Pairs of class_id class_name (e.g. 0 class0 1 class1 2 class2)")
    p.add_argument("--output_dir", default="cropped_imgs_by_class", help="Output Directory to store cropped images")
    p.add_argument("--output_txt_file", help="Temporary TXT file to store details for PDF generation at the end")
    args = p.parse_args()

    # Resolve imgs_dir and label_dir
    if args.imgs_label_path:
        imgs_dir = args.imgs_label_path
        label_dir = args.imgs_label_path
    elif args.imgs_path and args.label_path:
        imgs_dir = args.imgs_path
        label_dir = args.label_path
    else:
        raise ValueError("Provide either --imgs_label_path OR both --imgs_path and --label_path")

    for path, name in [(imgs_dir, "imgs"), (label_dir, "label")]:
        if not os.path.exists(path):
            raise OSError(f"{name.upper()} PATH DOESN'T EXIST: {path}")

    # Detect train/val/test splits
    split_dirs = _detect_splits(imgs_dir, label_dir)
    has_splits = split_dirs[0][0] != ''
    if has_splits:
        print(f"\n📂  Detected dataset splits: {', '.join(s[0] for s in split_dirs)}")
        print(f"    Processing all splits into combined output\n")

    ## CHECK FOR CLASSES
    class_map = dict(zip(args.class_ids_to_names[::2], args.class_ids_to_names[1::2]))
    classes_to_target = set(args.classes)

    missing_in_map = [cls for cls in classes_to_target if cls not in class_map]
    if missing_in_map:
        raise RuntimeError(
            f"Class ID(s) {missing_in_map} are not present in class_ids_to_names mapping. "
            f"Available keys: {list(class_map.keys())}"
        )

    ## CREATE OUTPUT DIRECTORIES
    output_base = args.output_dir
    os.makedirs(output_base, exist_ok=True)

    # Precompute output dir per class_id (fix #3: avoid repeated os.path.join in inner loop)
    class_output_dirs = {}
    for class_id in classes_to_target:
        class_name = class_map[class_id]
        class_dir = os.path.join(output_base, class_name)
        os.makedirs(class_dir, exist_ok=True)
        class_output_dirs[class_id] = class_dir

    # Accumulators across all splits
    total_img_count = 0
    total_txt_count = 0
    all_missing_txts = []
    unexpected_classes = {}
    class_annotation_counts = {}
    empty_annotation_files = []
    invalid_annotation_files = []
    error_reading_files = []
    empty_crop_files = []
    degenerate_crop_files = []  # non-empty but min(W,H) < 3px
    all_missing_imgs = []       # txt files whose image was not found on disk
    crop_counts = {cls: 0 for cls in classes_to_target}

    # =========================================================================
    # PROCESS EACH SPLIT
    # =========================================================================
    for split_name, split_imgs_dir, split_label_dir in split_dirs:
        split_label = f" [{split_name}]" if split_name else ""

        print("\n------------------------------------------------------------")
        print(f"|                     CHECKING DATASET{split_label:>22} |")
        print("------------------------------------------------------------")
        img_files = [f for f in os.listdir(split_imgs_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
        txt_files = [f for f in os.listdir(split_label_dir) if f.endswith(".txt")]
        img_files_set = set(img_files)
        txt_files_set = set(txt_files)

        total_img_count += len(img_files)
        total_txt_count += len([t for t in txt_files if t != 'classes.txt' and not t.endswith('_image_list.txt')])

        missing_txts = []
        for img in img_files:
            base_name = os.path.splitext(img)[0]
            txt_exp_file = base_name + ".txt"
            if txt_exp_file not in txt_files_set:
                missing_txts.append(txt_exp_file)

        missing_imgs = []
        for txt in txt_files:
            if txt == "classes.txt" or txt.endswith("_image_list.txt"):
                print(f"✅  IGNORE NON-ANNOTATION FILE: {txt}")
                continue

            base_name = os.path.splitext(txt)[0]
            found = False
            for ext in [".png", ".jpg", ".jpeg"]:
                img_exp_file = base_name + ext
                if img_exp_file in img_files_set:
                    found = True
                    break
            if not found:
                missing_imgs.append(txt)

        if missing_imgs:
            print(f"⚠️  {len(missing_imgs)} TXT FILE(S) HAVE NO MATCHING IMAGE{split_label}:")
            for m in missing_imgs:
                print(f"    - {m}")
            all_missing_imgs.extend(f"{split_name}/{m}" if split_name else m for m in missing_imgs)
        elif missing_txts:
            print(f"⚠️  NUMBER OF BACKGROUND PNG FILES WITH NO ANNOTATION/TXT FILE: {len(missing_txts)}\n")
        else:
            print(f"✅  ALL PNG FILES HAVE THEIR CORRESPONDING TXT FILES\n")

        all_missing_txts.extend(f"{split_name}/{t}" if split_name else t for t in missing_txts)

        # =================================================================
        # VALIDATE ANNOTATIONS + PROCESS CROPS (single pass, fix #1)
        # Each .txt is read once. cv2.imread is skipped if no target-class
        # annotations are present in the file.
        # =================================================================
        print("\n------------------------------------------------------------")
        print(f"|          VALIDATING & PROCESSING ANNOTATIONS{split_label:>14} |")
        print("------------------------------------------------------------")

        # Filename prefix to avoid collisions across splits
        fname_prefix = f"{split_name}_" if split_name else ""

        for txt_file in tqdm(txt_files, desc=f"Processing annotations{split_label}", unit="file"):
            if txt_file == "classes.txt" or txt_file.endswith("_image_list.txt"):
                continue

            txt_path = os.path.join(split_label_dir, txt_file)
            entry = f"{split_name}/{txt_file}" if split_name else txt_file

            with open(txt_path, 'r') as f:
                lines = f.readlines()

            if len(lines) == 0:
                empty_annotation_files.append(entry)
                continue

            # --- Validate all lines and collect target-class crops to perform ---
            pending_crops = []  # (idx, class_id, x_center, y_center, w, h)
            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    print(f"⚠️  Invalid Annotation File: {txt_file}")
                    if entry not in invalid_annotation_files:
                        invalid_annotation_files.append(entry)
                    continue

                class_id = parts[0]
                class_annotation_counts[class_id] = class_annotation_counts.get(class_id, 0) + 1

                if class_id not in class_map:
                    if class_id not in unexpected_classes:
                        unexpected_classes[class_id] = []
                    if entry not in unexpected_classes[class_id]:
                        unexpected_classes[class_id].append(entry)
                    continue

                if class_id not in classes_to_target:
                    continue

                try:
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width    = float(parts[3])
                    height   = float(parts[4])
                except ValueError:
                    print(f"⚠️  Invalid annotation in {txt_file}: {line.strip()}")
                    if entry not in invalid_annotation_files:
                        invalid_annotation_files.append(entry)
                    continue

                pending_crops.append((idx, class_id, x_center, y_center, width, height))

            # Skip imread entirely if this file has no target-class annotations
            if not pending_crops:
                continue

            # Find corresponding image
            base_name = os.path.splitext(txt_file)[0]
            img_file = None
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = base_name + ext
                if candidate in img_files_set:
                    img_file = candidate
                    break

            if not img_file:
                continue

            img = cv2.imread(os.path.join(split_imgs_dir, img_file))
            if img is None:
                print(f"\n⚠️  Could not read image: {img_file}")
                error_reading_files.append(f"{split_name}/{img_file}" if split_name else img_file)
                continue

            img_height, img_width = img.shape[:2]
            img_base = os.path.splitext(img_file)[0]

            for idx, class_id, x_center, y_center, width, height in pending_crops:
                x1, y1, x2, y2 = yolo_to_bbox(x_center, y_center, width, height, img_width, img_height)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_width, x2)
                y2 = min(img_height, y2)

                cropped = img[y1:y2, x1:x2]
                if cropped.size == 0:
                    print(f"\n⚠️  Empty crop from {img_file} for class {class_id}")
                    empty_crop_files.append(f"{img_file} (class {class_id})")
                    continue

                h_crop, w_crop = cropped.shape[:2]
                if min(w_crop, h_crop) < 3:
                    print(f"\n⚠️  Degenerate crop ({w_crop}x{h_crop}) from {img_file} for class {class_id}")
                    degenerate_crop_files.append(f"{img_file} | class {class_id} | {w_crop}x{h_crop}px")
                    continue

                output_path = os.path.join(
                    class_output_dirs[class_id],
                    f"{fname_prefix}{img_base}_crop_{idx}.png"
                )
                cv2.imwrite(output_path, cropped)
                crop_counts[class_id] += 1

        # Report unexpected classes
        if unexpected_classes:
            print(f"\n⚠️  UNEXPECTED CLASS IDs FOUND (not in class_ids_to_names) — skipped, will appear in PDF:")
            print(f"    Expected class IDs: 0 to {len(class_map) - 1}")
            print(f"    -----------------------------------------------")
            for cls_id, files in unexpected_classes.items():
                count = class_annotation_counts.get(cls_id, 0)
                print(f"    Class ID '{cls_id}': {count} annotations in {len(files)} file(s)")
                for f in files[:5]:
                    print(f"        - {f}")
                if len(files) > 5:
                    print(f"        ... and {len(files) - 5} more")
        else:
            print(f"✅  All annotations use valid class IDs")

        # Report class distribution
        print(f"\n📊  CLASS DISTRIBUTION (all annotations in dataset so far):")
        expected_ids = set(class_map.keys())
        total_annotations = sum(class_annotation_counts.values())

        for cls_id in sorted(class_annotation_counts.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
            count = class_annotation_counts[cls_id]
            percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
            name = class_map.get(cls_id, "UNKNOWN")
            marker = "⚠️ " if cls_id not in expected_ids else "   "
            print(f"{marker} Class {cls_id} ({name}): {count} annotations ({percentage:.1f}%)")

        # Class imbalance warning
        if class_annotation_counts:
            counts = [class_annotation_counts.get(c, 0) for c in expected_ids]
            counts = [c for c in counts if c > 0]
            if counts:
                max_count = max(counts)
                min_count = min(counts)
                if min_count > 0 and max_count / min_count > 10:
                    print(f"\n⚠️  CLASS IMBALANCE WARNING: Ratio {max_count}:{min_count} = {max_count/min_count:.1f}x")
                    print(f"    Consider balancing your dataset for better YOLO training.")

        if empty_annotation_files:
            print(f"\n⚠️  {len(empty_annotation_files)} annotation file(s) are empty (no objects labelled)")

        print("------------------------------------------------------------\n")

    # =========================================================================
    # COMBINED SUMMARY
    # =========================================================================
    total_txt_processed = sum(1 for s, si, sl in split_dirs
                              for f in os.listdir(sl)
                              if f.endswith(".txt") and f != "classes.txt" and not f.endswith("_image_list.txt"))

    print("\n------------------------------------------------------------")
    print("|                        SUMMARY                           |")
    print("------------------------------------------------------------")
    if has_splits:
        print(f"Splits processed: {', '.join(s[0] for s in split_dirs)}")
    print(f"Total image files: {total_img_count}")
    print(f"Total annotation files processed: {total_txt_processed}")
    print(f"Output directory: {output_base}")
    print("\nSuccessful Crops extracted per class:")
    for class_id in sorted(crop_counts.keys()):
        class_name = class_map[class_id]
        print(f"  Class {class_id} ({class_name}): {crop_counts[class_id]} crops")
    print("------------------------------------------------------------\n")

    # =========================================================================
    # WRITE TEMPORARY FILE FOR PDF GENERATION (if requested)
    # =========================================================================
    if args.output_txt_file:
        print(f"📝 Writing statistics to temporary file: {args.output_txt_file}")

        with open(args.output_txt_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("SCRIPT 1: ANNOTATION BBOX CROPPING - STATISTICS\n")
            f.write("="*70 + "\n\n")

            # Dataset overview
            f.write("DATASET OVERVIEW:\n")
            f.write("-"*70 + "\n")
            if has_splits:
                f.write(f"Splits: {', '.join(s[0] for s in split_dirs)}\n")
            f.write(f"Total image files found: {total_img_count}\n")
            f.write(f"Total annotation files found: {total_txt_count}\n")

            # Missing annotations (background images)
            f.write("\nPNG FILES WITH NO TXT ANNOTATIONS (Background Images) =>\n")
            f.write("-"*70 + "\n")
            f.write(f"Count: {len(all_missing_txts)}\n")
            if all_missing_txts:
                f.write("Files:\n")
                for txt in all_missing_txts[:10]:
                    f.write(f"  - {txt}\n")
                if len(all_missing_txts) > 10:
                    f.write(f"  ... and {len(all_missing_txts) - 10} more\n")
            f.write("\n")

            # Empty annotation files
            f.write("EMPTY ANNOTATION FILES (No Objects Labelled):\n")
            f.write("-"*70 + "\n")
            f.write(f"Count: {len(empty_annotation_files)}\n")
            if empty_annotation_files:
                f.write("Files:\n")
                for file in empty_annotation_files[:10]:
                    f.write(f"  - {file}\n")
                if len(empty_annotation_files) > 10:
                    f.write(f"  ... and {len(empty_annotation_files) - 10} more\n")
            f.write("\n")

            # Invalid annotation files
            f.write("INVALID ANNOTATION FILES (Malformed Data):\n")
            f.write("-"*70 + "\n")
            f.write(f"Count: {len(invalid_annotation_files)}\n")
            if invalid_annotation_files:
                f.write("Files:\n")
                for file in invalid_annotation_files:
                    f.write(f"  - {file}\n")
            f.write("\n")

            # Errors while reading files
            f.write("ERRORS WHILE READING FILES:\n")
            f.write("-"*70 + "\n")
            f.write(f"Count: {len(error_reading_files)}\n")
            if error_reading_files:
                f.write("Files:\n")
                for file in error_reading_files:
                    f.write(f"  - {file}\n")
            f.write("\n")

            # Empty crop files
            f.write("EMPTY CROP FILES (Zero-size Crops):\n")
            f.write("-"*70 + "\n")
            f.write(f"Count: {len(empty_crop_files)}\n")
            if empty_crop_files:
                f.write("Files:\n")
                for file in empty_crop_files:
                    f.write(f"  - {file}\n")
            f.write("\n")

            # Degenerate crop files
            f.write("DEGENERATE CROP FILES (min dimension < 3px):\n")
            f.write("-"*70 + "\n")
            f.write(f"Count: {len(degenerate_crop_files)}\n")
            if degenerate_crop_files:
                f.write("Files:\n")
                for file in degenerate_crop_files:
                    f.write(f"  - {file}\n")
            f.write("\n")

            # TXT files missing their image on disk
            f.write("TXT FILES WITH NO MATCHING IMAGE ON DISK:\n")
            f.write("-"*70 + "\n")
            f.write(f"Count: {len(all_missing_imgs)}\n")
            if all_missing_imgs:
                f.write("Files:\n")
                for file in all_missing_imgs:
                    f.write(f"  - {file}\n")
            f.write("\n")

            # Unexpected / out-of-range class IDs
            f.write("UNEXPECTED CLASS IDs IN ANNOTATIONS (out of range — labelling ignored):\n")
            f.write("-"*70 + "\n")
            total_unexpected = sum(len(v) for v in unexpected_classes.values())
            f.write(f"Count: {total_unexpected} file(s) affected\n")
            if unexpected_classes:
                f.write("Details:\n")
                for cls_id in sorted(unexpected_classes.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
                    files = unexpected_classes[cls_id]
                    count = class_annotation_counts.get(cls_id, 0)
                    f.write(f"  Class ID '{cls_id}' ({count} annotations, {len(files)} file(s)):\n")
                    for file in files:
                        f.write(f"    - {file}\n")
            f.write("\n")

            # Class distribution & successful crops
            f.write("CLASS DISTRIBUTION & SUCCESSFUL CROPS:\n")
            f.write("-"*70 + "\n")
            total_annotations = sum(class_annotation_counts.values())
            total_crops = sum(crop_counts.values())
            f.write(f"Total annotations in dataset: {total_annotations}\n")
            f.write(f"Total crops successfully extracted: {total_crops}\n\n")

            for cls_id in sorted(class_annotation_counts.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
                count = class_annotation_counts[cls_id]
                percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
                name = class_map.get(cls_id, "UNKNOWN")
                crops = crop_counts.get(cls_id, 0)
                f.write(f"  Class {cls_id} ({name}): {count} annotations → {crops} crops ({percentage:.1f}%)\n")
            f.write("\n")

            # Class imbalance
            f.write("CLASS IMBALANCE ANALYSIS:\n")
            f.write("-"*70 + "\n")
            if class_annotation_counts:
                expected_ids = set(class_map.keys())
                counts = [class_annotation_counts.get(c, 0) for c in expected_ids]
                counts = [c for c in counts if c > 0]
                if counts and len(counts) > 1:
                    max_count = max(counts)
                    min_count = min(counts)
                    ratio = max_count / min_count if min_count > 0 else float('inf')
                    f.write(f"Max class count: {max_count}\n")
                    f.write(f"Min class count: {min_count}\n")
                    f.write(f"Imbalance ratio: {ratio:.1f}x\n")
                    if ratio > 10:
                        f.write("⚠️  WARNING: Severe class imbalance detected (>10x)\n")
                    elif ratio > 5:
                        f.write("⚠️  WARNING: Moderate class imbalance detected (>5x)\n")
                    else:
                        f.write("✅ Class distribution is relatively balanced\n")
                else:
                    f.write("N/A (insufficient data)\n")
            else:
                f.write("N/A (no annotations found)\n")
            f.write("\n")

            f.write("="*70 + "\n")
            f.write("END OF SCRIPT 1 STATISTICS\n")
            f.write("="*70 + "\n")

        print(f"✅ Statistics written successfully to {args.output_txt_file}\n")

if __name__ == "__main__":
    main()
