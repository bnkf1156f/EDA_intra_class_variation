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

def ask_or_exit(prompt_result):
    """Exit cleanly if user cancels a questionary prompt (Ctrl+C)."""
    if prompt_result is None:
        print("\n   Cancelled by user. Exiting.")
        sys.exit(0)
    return prompt_result


def _ask(prompt_fn):
    """Wrap any questionary prompt with cancel + RuntimeError handling."""
    try:
        return ask_or_exit(prompt_fn.ask())
    except (KeyboardInterrupt, EOFError, RuntimeError):
        print("\n   Cancelled by user. Exiting.")
        sys.exit(0)


def _prompt_path(message, must_exist_dir=False, must_exist_file=False):
    """Prompt for a path with autocomplete and validation."""
    def _validate(p):
        p = p.strip().strip('"').strip("'")
        if not p:
            return "This field is required."
        if must_exist_dir and not os.path.isdir(p):
            return f"Directory does not exist: {p}"
        if must_exist_file and not os.path.isfile(p):
            return f"File does not exist: {p}"
        return True
    val = _ask(questionary.path(message, validate=_validate))
    return val.strip().strip('"').strip("'")


def _prompt_text(message, default=None):
    """Prompt for text with an optional default."""
    if default is not None:
        val = _ask(questionary.text(message, default=str(default)))
    else:
        val = _ask(questionary.text(message))
    val = val.strip().strip('"').strip("'")
    return val if val else (str(default) if default is not None else "")


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

    imgs_path   = _prompt_path("Path to images folder:", must_exist_dir=True)
    label_path  = _prompt_path("Path to labels folder (can be same as images):", must_exist_dir=True)

    # Detect train/val/test splits
    _splits = [s for s in ['train', 'val']
               if os.path.isdir(os.path.join(imgs_path, s)) and os.path.isdir(os.path.join(label_path, s))]
    if _splits:
        print(f"\n📂  Detected dataset splits: {', '.join(_splits)}")
        print(f"    Images and labels from all splits will be processed together")

    classes_txt = _prompt_path("Path to classes.txt file:", must_exist_file=True)

    class_names = _load_classes_txt(classes_txt)
    print(f"\n✅  Loaded {len(class_names)} classes from {classes_txt}")

    ## -----------------------------------------------##
    ##   RESULTS DIRECTORY (fixed, not changeable)      ##
    ## -----------------------------------------------##
    RESULTS_DIR = "postann_pretrain_results"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"\n📂  All results will be saved under: {os.path.abspath(RESULTS_DIR)}/")

    # Derive smart defaults from the images folder name
    _folder_name = os.path.basename(os.path.normpath(imgs_path))

    ## -----------------------------------------------##
    ##   OUTPUT NAMES — always asked, with defaults    ##
    ## -----------------------------------------------##
    _default_cropped   = f"cropped_imgs_by_class_{_folder_name}"
    _default_clust_dir = f"clustering_results_{_folder_name}"
    _default_pdf_name  = f"PDF_REPORT_{_folder_name}"

    cropped_bbox_dir   = os.path.join(RESULTS_DIR, _prompt_text("Subfolder for cropped images", _default_cropped))
    output_cluster_dir = os.path.join(RESULTS_DIR, _prompt_text("Subfolder for clustering results", _default_clust_dir))

    ## -----------------------------------------------##
    ##   OPTIONAL PARAMETERS                           ##
    ## -----------------------------------------------##

    change_defaults = _ask(questionary.confirm(
        "Change default parameters?", default=False))

    # Embedding Handling
    if change_defaults:
        batch_size          = int(_prompt_text("DINOv2 batch size (reduce if GPU OOM)", 64))
        save_suffix         = _prompt_text("Embeddings filename", "embeddings_dinov2.npy")
        use_embedding_cache = _ask(questionary.confirm("Use embedding cache?", default=True))
    else:
        batch_size          = 64
        save_suffix         = "embeddings_dinov2.npy"
        use_embedding_cache = True

    # Clustering flags
    auto_tune   = _ask(questionary.confirm("Enable auto-tune eps?", default=True))
    cross_class = _ask(questionary.confirm("Enable cross-class analysis?", default=True))

    # Cluster Handling — conditionally prompt eps OR percentile
    if auto_tune:
        auto_tune_percentile = int(_prompt_text("Auto-tune k-NN percentile (90=tight, 95=balanced, 98=loose)", 90))
        epsilon = 0.15  # unused fallback
    else:
        epsilon = float(_prompt_text("DBSCAN epsilon (0.10=strict, 0.15=balanced, 0.20-0.30=lenient)", 0.15))
        auto_tune_percentile = 95  # unused fallback

    if change_defaults:
        min_pts             = int(_prompt_text("Min points per cluster", 3))
        umap_min_dist       = float(_prompt_text("UMAP min_dist (lower=tighter packing)", 0.05))
        max_cluster_samples = int(_prompt_text("Max sample images saved per cluster", 20))

        # Uniform class handling
        uniform_class_eps_threshold     = float(_prompt_text("Uniform class eps threshold", 0.1))
        uniform_class_downsample_target = int(_prompt_text("Uniform class downsample target", 4000))
        uniform_class_min_samples       = int(_prompt_text("Uniform class min size to trigger downsampling", 12000))
    else:
        min_pts             = 3
        umap_min_dist       = 0.05
        max_cluster_samples = 20
        uniform_class_eps_threshold     = 0.1
        uniform_class_downsample_target = 4000
        uniform_class_min_samples       = 12000

    # PDF handling
    temp_file    = os.path.join(RESULTS_DIR, "temp_ann_file.txt")
    pdf_generate = _ask(questionary.confirm("Generate PDF report?", default=True))
    pdf_name     = _prompt_text("Output PDF filename (no extension)", _default_pdf_name) if pdf_generate else _default_pdf_name

    ## -----------------------------------------------##
    ##   PRE-FLIGHT: CHECK OUTPUT DIRS ARE EMPTY      ##
    ## -----------------------------------------------##
    dirs_to_check = [
        (cropped_bbox_dir,   "Cropped images folder"),
        (output_cluster_dir, "Clustering results folder"),
    ]

    print("\n" + "="*60)
    for dir_path, dir_label in dirs_to_check:
        if os.path.isdir(dir_path):
            print(f"\n⚠️  {dir_label} already exists and is NOT empty:")
            print(f"   {os.path.abspath(dir_path)}")
            overwrite = _ask(questionary.confirm(
                f"   Overwrite existing contents? (No = exit)", default=True))
            if not overwrite:
                print("\n   Exiting. Please choose a different output folder or clear the existing one.")
                sys.exit(0)
            print(f"   ✅ Will overwrite: {dir_path}")

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
    cluster_args = [
        "--root", cropped_bbox_dir,
        "--eps", str(epsilon),
        "--min_samples", str(min_pts),
        "--output_dir", output_cluster_dir,
        "--max_samples", str(max_cluster_samples),
        "--save_suffix", save_suffix,
        "--auto_tune_percentile", str(auto_tune_percentile),
        "--umap_min_dist", str(umap_min_dist),
        "--save_montage",
        "--uniform_eps_threshold", str(uniform_class_eps_threshold),
        "--uniform_downsample_target", str(uniform_class_downsample_target),
        "--uniform_min_samples", str(uniform_class_min_samples)
    ]
    if auto_tune:
        cluster_args.append("--auto_tune")
    if cross_class:
        cluster_args.append("--cross_class")

    run_step("postannotation_scripts/3. clustering_of_classes_embeddings.py",
             cluster_args, cool_down_after=False)  # No cooling needed after last step

    # Step 4 (Optional): Generate PDF Report
    if pdf_generate:
        print("\n" + "="*60)
        print("STEP 4/4: GENERATING PDF REPORT")
        print("="*60)
        pdf_args = [
            "--temp_txt_file", temp_file,
            "--clustering_dir", output_cluster_dir,
            "--pdf_name", os.path.join(output_cluster_dir, pdf_name),
            "--imgs_path", imgs_path,
            "--label_path", label_path,
            "--classes_txt", classes_txt,
            "--auto_tune_percentile", str(auto_tune_percentile),
            "--epsilon", str(epsilon),
        ]
        if auto_tune:
            pdf_args.append("--auto_tune")
        if cross_class:
            pdf_args.append("--cross_class")
        run_step("postannotation_scripts/4. generate_pdf.py", pdf_args, cool_down_after=False)

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

    # Ask user whether to delete cropped images folder
    if os.path.isdir(cropped_bbox_dir):
        cropped_size_gb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, files in os.walk(cropped_bbox_dir)
            for f in files
        ) / 1e9
        print("\n" + "="*60)
        print(f"📁 Cropped images folder: {cropped_bbox_dir}/")
        print(f"   Size on disk: {cropped_size_gb:.2f} GB")
        print("⚠️  WARNING: Deleting this folder is PERMANENT — embeddings (.npy) are kept,")
        print("   but you will NOT be able to view the original cropped images again.")
        response = input("\n🗑️  Delete cropped images folder to free disk space? (y/n): ")
        if response.lower() == 'y':
            import shutil
            try:
                shutil.rmtree(cropped_bbox_dir)
                print(f"   ✅ Deleted: {cropped_bbox_dir}/")
            except Exception as e:
                print(f"   ❌ Error deleting folder: {e}")
        else:
            print(f"   📌 Keeping: {cropped_bbox_dir}/")

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