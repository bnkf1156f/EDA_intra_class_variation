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
        txt_exp_file = img[:-3] + "txt"
        if txt_exp_file not in txt_files:
            missing_txts.append(txt_exp_file)

    missing_imgs = []
    for txt in txt_files:
        found = False
        for ext in ["png", "jpg", "jpeg"]:
            img_exp_file = txt[:-3] + ext
            if img_exp_file in img_files:
                found = True
                break
        if not found:
            missing_imgs.append(txt)

    if missing_imgs:
        raise RuntimeError(f"❌  HALT PROCESS! THE TXT FILES MISSING IMG FILES: {missing_imgs}")
    elif missing_txts:
        print(f"⚠️  NUMBER OF BACKGROUND PNG FILES WITH NO TXT: {len(missing_txts)}\n")
    else:
        print(f"✅  ALL PNG FILES HAVE THEIR CORRESPONDING TXT FILES\n")

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
    
    for class_id in classes_to_target:
        class_name = class_map[class_id]
        class_dir = os.path.join(output_base, f"{class_name}")
        os.makedirs(class_dir, exist_ok=True)
    
    print("\n------------------------------------------------------------")
    print("|                  PROCESSING ANNOTATIONS                  |")
    print("------------------------------------------------------------")
    
    ## WALK THROUGH WHOLE DATASET
    crop_counts = {cls: 0 for cls in classes_to_target}
    processed_files = 0
    
    for txt_file in txt_files:
        txt_path = os.path.join(dataset_path, txt_file)
        
        # Find corresponding image
        img_file = None
        for ext in ["png", "jpg", "jpeg"]:
            candidate = txt_file[:-3] + ext
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
        
        processed_files += 1
        if processed_files % 100 == 0:
            print(f"Processed {processed_files}/{len(txt_files)} files...")
    
    print("\n------------------------------------------------------------")
    print("|                        SUMMARY                           |")
    print("------------------------------------------------------------")
    print(f"Total files processed: {processed_files}")
    print(f"Output directory: {output_base}")
    print("\nCrops extracted per class:")
    for class_id in classes_to_target:
        class_name = class_map[class_id]
        print(f"  Class {class_id} ({class_name}): {crop_counts[class_id]} crops")
    print("------------------------------------------------------------\n")

if __name__ == "__main__":
    main()