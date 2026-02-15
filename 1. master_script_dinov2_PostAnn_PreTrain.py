"""
Master script to run entire intra and inter class variation EDA pipeline.
Includes memory management and cooling breaks for laptop GPUs.

(with PDF) 34 classes, 5789 annotations: 15-16 mins (2 mins by pdf)
"""

import subprocess
import sys
import gc
import time
import psutil
import os
import torch

def clear_memory():
    """Aggressive memory cleanup."""
    print("\n🧹 Clearing memory...")
    
    # Python garbage collection
    gc.collect()
    
    # Clear PyTorch cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Get GPU memory stats
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"   GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    # Get system memory stats
    mem = psutil.virtual_memory()
    print(f"   RAM Usage: {mem.percent}% ({mem.used/1e9:.1f}GB / {mem.total/1e9:.1f}GB)")
    print("   ✅ Memory cleared\n")

def cooling_break(seconds):
    """Pause execution to let GPU cool down."""
    print(f"\n❄️  Cooling break: {seconds} seconds...")
    print("   This prevents thermal throttling on your laptop GPU")
    
    for remaining in range(seconds, 0, -5):
        print(f"   ⏳ {remaining} seconds remaining...", end='\r')
        time.sleep(5)
    
    print("\n   ✅ Cooling break complete\n")

def check_system_resources():
    """Check if system has enough resources to continue."""
    mem = psutil.virtual_memory()
    
    if mem.percent > 90:
        print("⚠️  WARNING: RAM usage is very high (>90%)")
        print("   Consider closing other applications")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        usage_percent = (allocated / total) * 100
        
        if usage_percent > 85:
            print("⚠️  WARNING: GPU memory usage is very high (>85%)")
            print("   The script will attempt to clear cache")
            clear_memory()

def run_step(script, args, cool_down_after=True):
    """Run a pipeline step with memory management."""
    print(f"\n{'='*60}")
    print(f"Running: {script}")
    print(f"{'='*60}\n")
    
    # Check resources before starting
    check_system_resources()
    
    # Run the script
    result = subprocess.run([sys.executable, script] + args)
    
    if result.returncode != 0:
        print(f"❌ Error in {script}")
        clear_memory()  # Clean up even on failure
        sys.exit(1)
    
    print(f"\n✅ {script} completed successfully")
    
    # Post-execution cleanup
    if cool_down_after:
        clear_memory()
        cooling_break(10)  # 10 second break between steps

def main():
    ## ---------------------------------##
    ##  GLOBAL VARIABLES PRE-TRAINING   ##
    ## ---------------------------------##
    imgs_label_path = r"path/to/LabelledData"  # Replace with your annotated images directory
    class_names = [
        "class0",
        "class1",
        "class2",
    ]  # Replace with your class names

    cropped_bbox_dir = "cropped_imgs_by_class"

    batch_size = 64
    save_suffix = "embeddings_dinov2.npy"

    epsilon = 0.15  # Only imp when auto-tune is NOT selected during clustering
    min_pts = 3
    output_cluster_dir = "clustering_results_txt_files"
    max_cluster_samples = 20

    temp_file = "temp_ann_file.txt"
    pdf_generate = True
    pdf_name = "PDF_REPORT"
    

    # GPU specs print
    print("="*60)
    print("INTRA-CLASS VARIATION EDA PIPELINE")
    print("="*60)
    
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {gpu_props.name}")
        print(f"  VRAM: {gpu_props.total_memory/1e9:.1f}GB")
    else:
        print("  GPU: Not available or PyTorch not installed")
    
    print("="*60)

    # Classes to ids and etc
    class_ids = [str(i) for i in range(len(class_names))]
    class_ids_to_names = []
    for i, name in enumerate(class_names):
        class_ids_to_names.extend([str(i), name])


    # Step 1: Crop YOLO bboxes -- Only send temp txt file if requested for PDF Generation
    print("\n" + "="*60)
    total_steps = 4 if pdf_generate else 3
    print(f"STEP 1/{total_steps}: EXTRACTING BOUNDING BOXES FROM LABELLED DATA")
    print("⚠️  This step is time-consuming")
    print("="*60)
    if pdf_generate:
        run_step("postannotation_scripts/1. ann_txt_files_crop_bbox.py", [
            "--imgs_label_path", imgs_label_path,
            "--classes"] + class_ids + [  # Unpack list
            "--class_ids_to_names"] + class_ids_to_names + [  # Unpack list
            "--output_dir", cropped_bbox_dir,
            "--output_txt_file", temp_file
        ])
    else:
        run_step("postannotation_scripts/1. ann_txt_files_crop_bbox.py", [
            "--imgs_label_path", imgs_label_path,
            "--classes"] + class_ids + [  # Unpack list
            "--class_ids_to_names"] + class_ids_to_names + [  # Unpack list
            "--output_dir", cropped_bbox_dir
        ])


    # Step 2: Embed the cropped YOLO bbox
    print("\n" + "="*60)
    print(f"STEP 2/{total_steps}: GENERATING DINOV2 EMBEDDINGS")
    print("="*60)
    print("⚠️  This step is GPU-intensive")
    
    run_step("postannotation_scripts/2. save_dinov2_embeddings_per_class.py", [
        "--root", cropped_bbox_dir,
        "--batch", str(batch_size),
        "--save_suffix", save_suffix
    ], cool_down_after=True)


    # Step 3: Cluster using DBSCAN to visualize
    print("\n" + "="*60)
    print(f"STEP 3/{total_steps}: CLUSTERING ANALYSIS")
    print("="*60)
    run_step("postannotation_scripts/3. clustering_of_classes_embeddings.py", [
        "--root", cropped_bbox_dir,
        "--eps", str(epsilon),
        "--min_samples", str(min_pts),
        "--output_dir", output_cluster_dir,
        "--max_samples", str(max_cluster_samples),
        "--save_suffix", save_suffix,
        "--auto_tune",
        "--save_montage",
        "--cross_class"
    ], cool_down_after=False)  # No cooling needed after last step

    # Step 4 (Optional): Generate PDF Report
    if pdf_generate:
        print("\n" + "="*60)
        print("STEP 4/4: GENERATING PDF REPORT")
        print("="*60)
        run_step("postannotation_scripts/4. generate_pdf.py", [
            "--temp_txt_file", temp_file,
            "--clustering_dir", output_cluster_dir,
            "--pdf_name", pdf_name
        ], cool_down_after=False)

        # Ask user whether to delete temp file
        print("\n" + "="*60)
        print(f"📄 PDF Report generated: {pdf_name}.pdf")
        print(f"📁 Temporary annotation file: {temp_file}")
        print("="*60)
        response = input("\n🗑️  Delete temporary annotation file? (y/n): ")
        if response.lower() == 'y':
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"   ✅ Deleted: {temp_file}")
                else:
                    print(f"   ⚠️  File not found: {temp_file}")
            except Exception as e:
                print(f"   ❌ Error deleting file: {e}")
        else:
            print(f"   📌 Keeping: {temp_file}")

    # Final cleanup
    clear_memory()

    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE!")
    print("="*60)
    print("\nOutput Locations:")
    print(f"  Cropped Images: {cropped_bbox_dir}/")
    print(f"  Clustering Results: {output_cluster_dir}/")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        clear_memory()
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        clear_memory()
        sys.exit(1)