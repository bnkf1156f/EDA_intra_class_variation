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
import questionary

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

def _prompt_required(message):
    """Prompt until user provides a non-empty value."""
    while True:
        val = questionary.text(message).ask()
        if val and val.strip():
            return val.strip()
        print("   ⚠️  This field is required, please enter a value.")


def _prompt_default(message, default):
    """Prompt with a default — pressing Enter keeps the default."""
    val = questionary.text(f"{message} (default: {default})", default=str(default)).ask()
    return val.strip() if val and val.strip() else str(default)


def _load_classes_txt(path):
    """Read class names from a classes.txt file (one name per line, skip blanks)."""
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def main():
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

    ## -----------------------------------------------##
    ##   REQUIRED INPUTS (prompted, no defaults)       ##
    ## -----------------------------------------------##
    print("\n📁  STEP 0: CONFIGURE PIPELINE INPUTS")
    print("="*60)

    imgs_path   = _prompt_required("Path to images folder")
    label_path  = _prompt_required("Path to labels folder (can be same as images folder)")
    classes_txt = _prompt_required("Path to classes.txt file")

    if not os.path.isdir(imgs_path):
        raise OSError(f"Images path does not exist: {imgs_path}")
    if not os.path.isdir(label_path):
        raise OSError(f"Labels path does not exist: {label_path}")
    if not os.path.isfile(classes_txt):
        raise OSError(f"classes.txt not found: {classes_txt}")

    class_names = _load_classes_txt(classes_txt)
    print(f"\n✅  Loaded {len(class_names)} classes from {classes_txt}")

    ## -----------------------------------------------##
    ##   OPTIONAL PARAMETERS (Enter = keep default)   ##
    ## -----------------------------------------------##
    print("\n⚙️   PARAMETERS  (press Enter to keep default)\n")

    # Embedding Handling
    cropped_bbox_dir    = _prompt_default("Output folder for cropped images", "cropped_imgs_by_class")
    batch_size          = int(_prompt_default("DINOv2 batch size (reduce if GPU OOM)", 64))
    save_suffix         = _prompt_default("Embeddings filename", "embeddings_dinov2.npy")
    use_embedding_cache = _prompt_default("Use embedding cache? (True/False)", True).lower() == "true"

    # Cluster Handling
    epsilon                      = float(_prompt_default("DBSCAN eps fallback (used if auto-tune disabled)", 0.15))
    min_pts                      = int(_prompt_default("Min points per cluster", 3))
    auto_tune_percentile         = int(_prompt_default("Auto-tune k-NN percentile (90=tight, 95=balanced, 98=loose)", 95))
    umap_min_dist                = float(_prompt_default("UMAP min_dist (lower=tighter packing)", 0.05))
    output_cluster_dir           = _prompt_default("Output folder for clustering results", "clustering_results_txt_files")
    max_cluster_samples          = int(_prompt_default("Max sample images saved per cluster", 20))

    # Uniform class handling
    uniform_class_eps_threshold      = float(_prompt_default("Uniform class eps threshold", 0.1))
    uniform_class_downsample_target  = int(_prompt_default("Uniform class downsample target", 4000))
    uniform_class_min_samples        = int(_prompt_default("Uniform class min size to trigger downsampling", 12000))

    # PDF handling
    temp_file    = "temp_ann_file.txt"
    pdf_generate = _prompt_default("Generate PDF report? (True/False)", True).lower() == "true"
    pdf_name     = _prompt_default("Output PDF filename (no extension)", "PDF_REPORT")

    print("\n" + "="*60)

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
            "--imgs_path", imgs_path,
            "--label_path", label_path,
            "--classes"] + class_ids + [
            "--class_ids_to_names"] + class_ids_to_names + [
            "--output_dir", cropped_bbox_dir,
            "--output_txt_file", temp_file
        ])
    else:
        run_step("postannotation_scripts/1. ann_txt_files_crop_bbox.py", [
            "--imgs_path", imgs_path,
            "--label_path", label_path,
            "--classes"] + class_ids + [
            "--class_ids_to_names"] + class_ids_to_names + [
            "--output_dir", cropped_bbox_dir
        ])


    # Step 2: Embed the cropped YOLO bbox
    print("\n" + "="*60)
    print(f"STEP 2/{total_steps}: GENERATING DINOV2 EMBEDDINGS")
    print("="*60)
    print("⚠️  This step is GPU-intensive")
    
    embedding_args = [
        "--root", cropped_bbox_dir,
        "--batch", str(batch_size),
        "--save_suffix", save_suffix
    ]
    if use_embedding_cache:
        embedding_args.append("--use_cache")

    run_step("postannotation_scripts/2. save_dinov2_embeddings_per_class.py", embedding_args, cool_down_after=True)


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
        "--auto_tune_percentile", str(auto_tune_percentile),
        "--umap_min_dist", str(umap_min_dist),
        "--save_montage",
        "--cross_class",
        "--uniform_eps_threshold", str(uniform_class_eps_threshold),
        "--uniform_downsample_target", str(uniform_class_downsample_target),
        "--uniform_min_samples", str(uniform_class_min_samples)
    ], cool_down_after=False)  # No cooling needed after last step

    # Step 4 (Optional): Generate PDF Report
    if pdf_generate:
        print("\n" + "="*60)
        print("STEP 4/4: GENERATING PDF REPORT")
        print("="*60)
        run_step("postannotation_scripts/4. generate_pdf.py", [
            "--temp_txt_file", temp_file,
            "--clustering_dir", output_cluster_dir,
            "--pdf_name", os.path.join(output_cluster_dir, pdf_name)
        ], cool_down_after=False)

        # Ask user whether to delete temp file
        print("\n" + "="*60)
        print(f"📄 PDF Report generated: {os.path.join(output_cluster_dir, pdf_name)}.pdf")
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
    print(f"  Cropped Images:     {cropped_bbox_dir}/")
    print(f"  Clustering Results: {output_cluster_dir}/")
    if pdf_generate:
        print(f"  PDF Report:         {os.path.join(output_cluster_dir, pdf_name)}.pdf")
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