"""
Annotation TXT Files → Cropped Bounding Box Images by Class

This script processes a folder containing YOLO-format labelled images (.png/.jpg and .txt pairs), and extracts cropped object images from bounding boxes for specified class IDs.

Usage:
    python ann_txt_files_crop_bbox.py \
        --imgs_label_path "path/to/LabelledData" \
        --classes 0 1 2 3 4 \
        --class_ids_to_names 0 board 1 screw 2 screw_holder 3 tape 4 case \
        --output_dir cropped_imgs_by_class

This script:
 - Validates dataset consistency (ensures each image has a label file and vice versa)
 - Verifies provided class IDs to extract/target exist in the provided mapping
 - Converts YOLO normalized coordinates to pixel bounding boxes
 - Crops and saves bounding boxes into per-class folders
 - Prints a summary of all crops extracted per class

python '.\1. ann_txt_files_crop_bbox.py' --imgs_label_path "C:/VkRetro/YoloDetectExtractFrames/EDA_intra_class_variation_scripts/CarrierInspectionLabelledData" --classes 0 1 2 3 4 --class_ids_to_names 0 board 1 screw 2 screw_holder 3 tape 4 case
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

def main():
    p = argparse.ArgumentParser(description="Labelled Frames -> sampled cropped images by class")
    p.add_argument("--imgs_label_path", required=True, help="Path to Images+Labels Path")
    p.add_argument("--classes", nargs="+", required=True, help="Class IDs required")
    p.add_argument("--class_ids_to_names", nargs="+", required=True, help="Pairs of class_id class_name (e.g. 0 board 1 screw 2 screw_holder)")
    p.add_argument("--output_dir", default="cropped_imgs_by_class", help="Output Directory to store cropped images")
    args = p.parse_args()

    dataset_path = args.imgs_label_path
    if not os.path.exists(dataset_path):
        raise OSError(f"DATASET PATH DOESN'T EXIST here: {dataset_path}")
    
    print("\n------------------------------------------------------------")
    print("|                     CHECKING DATASET                     |")
    print("------------------------------------------------------------")
    img_files = [f for f in os.listdir(dataset_path) if f.endswith((".png", ".jpg", ".jpeg"))]
    txt_files = [f for f in os.listdir(dataset_path) if f.endswith(".txt")]
    missing_txts = []
    for img in img_files:
        base_name = os.path.splitext(img)[0]
        txt_exp_file = base_name + ".txt"
        if txt_exp_file not in txt_files:
            missing_txts.append(txt_exp_file)

    missing_imgs = []
    for txt in txt_files:
        if txt == "classes.txt":
            print("✅  IGNORE CLASSES.TXT!")
            continue

        base_name = os.path.splitext(txt)[0]
        found = False
        for ext in [".png", ".jpg", ".jpeg"]:
            img_exp_file = base_name + ext
            if img_exp_file in img_files:
                found = True
                break
        if not found:
            missing_imgs.append(txt)

    if missing_imgs:
        raise RuntimeError(f"❌  HALT PROCESS! THE TXT FILES MISSING IMG FILES: {missing_imgs}")
    elif missing_txts:
        print(f"⚠️  NUMBER OF BACKGROUND PNG FILES WITH NO ANNOTATION/TXT FILE: {len(missing_txts)}\n")
    else:
        print(f"✅  ALL PNG FILES HAVE THEIR CORRESPONDING TXT FILES\n")

    ## CHECK FOR CLASSES
    class_map = dict(zip(args.class_ids_to_names[::2], args.class_ids_to_names[1::2]))

    # =========================================================================
    # DATASET VALIDATION: Scan for unexpected class IDs and class distribution
    # =========================================================================
    print("\n------------------------------------------------------------")
    print("|                 VALIDATING ANNOTATIONS                   |")
    print("------------------------------------------------------------")

    unexpected_classes = {}  # {class_id: [list of files]}
    class_annotation_counts = {}  # {class_id: count} for ALL classes found
    empty_annotation_files = []

    for txt_file in txt_files:
        if txt_file == "classes.txt":
            continue
        txt_path = os.path.join(dataset_path, txt_file)
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        if len(lines) == 0:
            empty_annotation_files.append(txt_file)
            continue

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                print(f"⚠️  Invalid Annotation File: {txt_file}")
                continue
            class_id = parts[0]

            # Count all annotations
            class_annotation_counts[class_id] = class_annotation_counts.get(class_id, 0) + 1

            # Check if class is unexpected (not in our mapping)
            if class_id not in class_map:
                if class_id not in unexpected_classes:
                    unexpected_classes[class_id] = []
                if txt_file not in unexpected_classes[class_id]:
                    unexpected_classes[class_id].append(txt_file)

    # Report unexpected classes
    if unexpected_classes:
        print(f"\n❌  UNEXPECTED CLASS IDs FOUND (not in class_ids_to_names):")
        print(f"    Expected class IDs: 0 to {len(class_map) - 1}")
        print(f"    -----------------------------------------------")
        for cls_id, files in unexpected_classes.items():
            count = class_annotation_counts.get(cls_id, 0)
            print(f"    Class ID '{cls_id}': {count} annotations in {len(files)} file(s)")
            # Show all files where invalid annotation
            for f in files:
                print(f"        - {f}")
        raise RuntimeError(f"\n    ⚠️  FIX THESE ANNOTATIONS BEFORE YOLO TRAINING!")
    else:
        print(f"✅  All annotations use valid class IDs")

    # Report class distribution
    print(f"\n📊  CLASS DISTRIBUTION (all annotations in dataset):")
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
        counts = [c for c in counts if c > 0]  # Only count classes that exist
        if counts:
            max_count = max(counts)
            min_count = min(counts)
            if min_count > 0 and max_count / min_count > 10:
                print(f"\n⚠️  CLASS IMBALANCE WARNING: Ratio {max_count}:{min_count} = {max_count/min_count:.1f}x")
                print(f"    Consider balancing your dataset for better YOLO training.")

    # Empty files warning
    if empty_annotation_files:
        print(f"\n⚠️  {len(empty_annotation_files)} annotation file(s) are empty (no objects labelled)")

    print("------------------------------------------------------------\n")
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
    
    for class_id in classes_to_target:
        class_name = class_map[class_id]
        class_dir = os.path.join(output_base, f"{class_name}")
        os.makedirs(class_dir, exist_ok=True)
    
    print("\n------------------------------------------------------------")
    print("|                  PROCESSING ANNOTATIONS                  |")
    print("------------------------------------------------------------")
    
    ## WALK THROUGH WHOLE DATASET
    crop_counts = {cls: 0 for cls in classes_to_target}

    for txt_file in tqdm(txt_files, desc="Processing annotations", unit="file"):
        if txt_file == "classes.txt":
            continue

        txt_path = os.path.join(dataset_path, txt_file)

        # Find corresponding image
        base_name = os.path.splitext(txt_file)[0]
        img_file = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = base_name + ext
            if candidate in img_files:
                img_file = candidate
                break
        
        if not img_file:
            continue
        
        img_path = os.path.join(dataset_path, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"⚠️  Could not read image: {img_file}")
            continue
        
        img_height, img_width = img.shape[:2]
        
        # Parse annotations
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = parts[0]
            if class_id not in classes_to_target:
                continue
            
            try:
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError:
                print(f"⚠️  Invalid annotation in {txt_file}: {line.strip()}")
                continue
            
            # Convert to pixel coordinates
            x1, y1, x2, y2 = yolo_to_bbox(x_center, y_center, width, height, img_width, img_height)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)
            
            # Crop the image
            cropped = img[y1:y2, x1:x2]
            
            if cropped.size == 0:
                print(f"⚠️  Empty crop from {img_file} for class {class_id}")
                continue
            
            # Save cropped image
            class_name = class_map[class_id]
            output_dir = os.path.join(output_base, f"{class_name}")
            base_name = os.path.splitext(img_file)[0]
            output_filename = f"{base_name}_crop_{idx}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            cv2.imwrite(output_path, cropped)
            crop_counts[class_id] += 1

    print("\n------------------------------------------------------------")
    print("|                        SUMMARY                           |")
    print("------------------------------------------------------------")
    print(f"Total files processed: {len(txt_files)}")
    print(f"Output directory: {output_base}")
    print("\nSuccessful Crops extracted per class:")
    for class_id in sorted(crop_counts.keys()):
        class_name = class_map[class_id]
        print(f"  Class {class_id} ({class_name}): {crop_counts[class_id]} crops")
    print("------------------------------------------------------------\n")

if __name__ == "__main__":
    main()