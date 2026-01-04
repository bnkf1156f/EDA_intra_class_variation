"""
PDF Report Generator for Intra-Class Variation Analysis

This script generates a comprehensive PDF report combining:
- Dataset quality statistics from Script 1 (annotation cropping)
- Clustering analysis visualizations from Script 3 (DBSCAN clustering)

Usage:
    python "4. generate_pdf.py" \
        --temp_txt_file temp_ann_file.txt \
        --clustering_dir clustering_results \
        --pdf_name "REPORT"

Output:
    - Page 1: Dataset quality report with charts
    - Pages 2-N: Per-class clustering montages with captions
"""

import os
import argparse
from pathlib import Path
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def parse_temp_txt_file(txt_path):
    """Parse Script 1 temporary txt file and extract all statistics."""

    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    stats = {
        'raw_content': content,
        'class_data': [],  # List of {class_id, name, annotations, crops, percentage}
        'total_annotations': 0,
        'total_crops': 0,
        'imbalance_ratio': None,
        'imbalance_warning': None
    }

    # Parse class distribution section
    in_class_section = False
    for line in content.split('\n'):
        if 'CLASS DISTRIBUTION & SUCCESSFUL CROPS:' in line:
            in_class_section = True
            continue

        if in_class_section:
            if line.startswith('Total annotations in dataset:'):
                stats['total_annotations'] = int(line.split(':')[1].strip())
            elif line.startswith('Total crops successfully extracted:'):
                stats['total_crops'] = int(line.split(':')[1].strip())
            elif line.strip().startswith('Class '):
                # Parse: "  Class 0 (person): 1234 annotations → 1230 crops (18.2%)"
                parts = line.strip().split(':')
                if len(parts) >= 2:
                    class_info = parts[0].replace('Class ', '').strip()
                    class_id = class_info.split('(')[0].strip()
                    class_name = class_info.split('(')[1].replace(')', '').strip()

                    detail = parts[1].strip()
                    annotations = int(detail.split('annotations')[0].strip())
                    crops_part = detail.split('→')[1].strip()
                    crops = int(crops_part.split('crops')[0].strip())
                    percentage = float(crops_part.split('(')[1].replace('%)', '').strip())

                    stats['class_data'].append({
                        'class_id': class_id,
                        'name': class_name,
                        'annotations': annotations,
                        'crops': crops,
                        'percentage': percentage
                    })
            elif 'CLASS IMBALANCE ANALYSIS:' in line:
                in_class_section = False

        # Parse imbalance info
        if line.startswith('Imbalance ratio:'):
            ratio_str = line.split(':')[1].strip().replace('x', '')
            stats['imbalance_ratio'] = float(ratio_str)
        elif '⚠️  WARNING:' in line and 'imbalance' in line.lower():
            stats['imbalance_warning'] = line.strip()
        elif '✅ Class distribution is relatively balanced' in line:
            stats['imbalance_warning'] = line.strip()

    return stats


def create_annotation_histogram(stats, output_path):
    """Generate bar chart for annotation counts per class."""
    if not stats['class_data']:
        return None

    class_names = [c['name'] for c in stats['class_data']]
    annotations = [c['annotations'] for c in stats['class_data']]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(class_names)), annotations, color='steelblue', edgecolor='black', linewidth=0.7)

    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Annotation Count', fontsize=12, fontweight='bold')
    ax.set_title('Annotation Distribution per Class', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on top of bars
    for i, (bar, val) in enumerate(zip(bars, annotations)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}',
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def create_imbalance_visualization(stats, output_path):
    """Generate imbalance ratio visualization."""
    if not stats['imbalance_ratio']:
        return None

    ratio = stats['imbalance_ratio']

    fig, ax = plt.subplots(figsize=(8, 4))

    # Color based on severity
    if ratio > 10:
        color = 'darkred'
        severity = 'Severe Imbalance'
    elif ratio > 5:
        color = 'darkorange'
        severity = 'Moderate Imbalance'
    else:
        color = 'green'
        severity = 'Balanced'

    ax.barh([0], [ratio], color=color, height=0.5, edgecolor='black', linewidth=2)
    ax.set_xlim(0, max(ratio * 1.2, 15))
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Imbalance Ratio (Max/Min)', fontsize=12, fontweight='bold')
    ax.set_title(f'Class Imbalance: {severity}', fontsize=14, fontweight='bold')
    ax.set_yticks([])
    ax.text(ratio + 0.5, 0, f'{ratio:.1f}x', va='center', fontsize=14, fontweight='bold')

    # Add threshold lines
    ax.axvline(5, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='Moderate (5x)')
    ax.axvline(10, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Severe (10x)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def load_cluster_statistics(clustering_dir):
    """Load cluster statistics CSV from Script 3 output."""
    csv_path = Path(clustering_dir) / "cluster_statistics.csv"

    if not csv_path.exists():
        print(f"⚠️  WARNING: cluster_statistics.csv not found at {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    return df


def find_montage_image(clustering_dir, class_name):
    """Find the montage PNG for a given class name."""
    clustering_path = Path(clustering_dir)

    # Try exact match for montage (Script 3 with --save_montage flag)
    montage_path = clustering_path / f"{class_name}_montage.png"
    if montage_path.exists():
        return str(montage_path)

    # Try case-insensitive search
    for file in clustering_path.glob("*_montage.png"):
        if file.stem.lower().replace('_montage', '') == class_name.lower():
            return str(file)

    return None


def create_placeholder_image(class_name, output_path):
    """Create a placeholder image when montage is missing."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, f'⚠️  Montage Not Available\n\nClass: {class_name}\n\n'
            f'Possible reasons:\n• Insufficient samples for clustering\n• Clustering failed\n• File missing',
            ha='center', va='center', fontsize=14, color='red',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    return output_path


def build_pdf(temp_txt_file, clustering_dir, pdf_name, pdf_quality):
    """Build the complete PDF report."""
    print("\n" + "="*70)
    print("PDF REPORT GENERATION")
    print("="*70)

    # Validate inputs
    if not os.path.isfile(temp_txt_file):
        raise FileNotFoundError(f"❌ Temp txt file not found or is not a file: {temp_txt_file}")

    if not os.path.isdir(clustering_dir):
        raise NotADirectoryError(f"❌ Clustering directory not found or is not a directory: {clustering_dir}")

    # Parse Script 1 statistics
    print("\n📊 Parsing Script 1 statistics...")
    stats = parse_temp_txt_file(temp_txt_file)
    print(f"   ✅ Found {len(stats['class_data'])} classes")
    print(f"   ✅ Total annotations: {stats['total_annotations']}")

    # Load Script 3 cluster statistics
    print("\n📊 Loading Script 3 cluster statistics...")
    cluster_df = load_cluster_statistics(clustering_dir)
    if cluster_df is not None:
        print(f"   ✅ Loaded statistics for {len(cluster_df)} classes")
    else:
        print("   ⚠️  No cluster statistics found - will skip cluster metrics in captions")

    # Create temporary directory for charts
    temp_dir = Path("temp_pdf_charts")
    temp_dir.mkdir(exist_ok=True)

    # Generate charts for Page 1
    print("\n📈 Generating charts...")
    hist_path = temp_dir / "annotation_histogram.png"
    imbalance_path = temp_dir / "imbalance_chart.png"

    create_annotation_histogram(stats, hist_path)
    print(f"   ✅ Annotation histogram created")

    if stats['imbalance_ratio']:
        create_imbalance_visualization(stats, imbalance_path)
        print(f"   ✅ Imbalance chart created")

    # Create PDF
    pdf_filename = f"{pdf_name}.pdf" if not pdf_name.endswith('.pdf') else pdf_name
    print(f"\n📄 Creating PDF: {pdf_filename}")

    doc = SimpleDocTemplate(pdf_filename, pagesize=letter,
                           topMargin=0.5*inch, bottomMargin=0.5*inch,
                           leftMargin=0.5*inch, rightMargin=0.5*inch)

    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=12,
        alignment=TA_CENTER
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=10,
        spaceBefore=10
    )

    normal_style = styles['Normal']
    normal_style.fontSize = 10
    normal_style.leading = 14

    # =========================================================================
    # PAGE 1: Dataset Quality Report
    # =========================================================================
    story.append(Paragraph("INTRA-CLASS VARIATION ANALYSIS", title_style))
    story.append(Paragraph("Dataset Quality Report", heading_style))
    story.append(Spacer(1, 0.2*inch))

    # Add raw statistics as formatted text
    content_lines = stats['raw_content'].split('\n')
    for line in content_lines:
        if line.strip():
            # Skip chart sections (we'll add them as images)
            if 'CLASS DISTRIBUTION & SUCCESSFUL CROPS:' in line:
                break

            # Format section headers
            if line.startswith('='):
                continue
            elif line.strip().endswith(':') and not line.startswith(' '):
                story.append(Paragraph(f"<b>{line.strip()}</b>", heading_style))
            elif line.startswith('-'):
                continue
            else:
                story.append(Paragraph(line.replace('⚠️', '').replace('✅', ''), normal_style))
                story.append(Spacer(1, 0.05*inch))

    story.append(Spacer(1, 0.3*inch))

    # Add charts
    story.append(Paragraph("Class Distribution Visualization", heading_style))
    if hist_path.exists():
        img = Image(str(hist_path), width=7*inch, height=3.5*inch)
        story.append(img)
        story.append(Spacer(1, 0.2*inch))

    if stats['imbalance_ratio'] and imbalance_path.exists():
        story.append(Paragraph("Class Imbalance Analysis", heading_style))
        img = Image(str(imbalance_path), width=6*inch, height=3*inch)
        story.append(img)

    story.append(PageBreak())

    # =========================================================================
    # PAGES 2-N: Per-Class Clustering Montages
    # =========================================================================
    print("\n🖼️  Adding class montages...")

    missing_montages = []

    for class_info in stats['class_data']:
        class_name = class_info['name']

        # Find montage image
        montage_path = find_montage_image(clustering_dir, class_name)

        if montage_path is None:
            print(f"   ⚠️  WARNING: Montage missing for class '{class_name}' - creating placeholder")
            missing_montages.append(class_name)
            placeholder_path = temp_dir / f"{class_name}_placeholder.png"
            create_placeholder_image(class_name, placeholder_path)
            montage_path = str(placeholder_path)
        else:
            print(f"   ✅ Found montage for class '{class_name}'")

        # Get cluster statistics for this class
        cluster_info = None
        if cluster_df is not None:
            cluster_row = cluster_df[cluster_df['class'] == class_name]
            if not cluster_row.empty:
                cluster_info = cluster_row.iloc[0]

        # Add montage title
        story.append(Paragraph(f"Class: {class_name}", title_style))
        story.append(Spacer(1, 0.1*inch))

        # Add montage image (scale to fit page width, split by cluster rows if too tall)
        try:
            # Calculate image dimensions
            from PIL import Image as PILImage

            # Disable decompression bomb protection for large montages
            PILImage.MAX_IMAGE_PIXELS = None

            img_pil = PILImage.open(montage_path)
            img_width, img_height = img_pil.size

            # Scale to fit page width (7 inches max)
            max_width = 7 * inch
            max_page_height = 8.5 * inch  # Max height per page

            width_scale = max_width / img_width
            scaled_height = img_height * width_scale

            # If image fits on one page, add it directly (compress first)
            if scaled_height <= max_page_height:
                # Compress to JPEG for smaller file size
                compressed_path = temp_dir / f"{class_name}_compressed.jpg"
                img_pil.convert('RGB').save(compressed_path, 'JPEG', quality=pdf_quality, optimize=True)
                img = Image(str(compressed_path), width=max_width, height=scaled_height)
                story.append(img)
            else:
                # Split image at cluster row boundaries
                # Get cluster info to calculate number of rows
                if cluster_info is not None:
                    num_cluster_rows = int(cluster_info['n_clusters']) + (1 if cluster_info['n_outliers'] > 0 else 0)
                else:
                    # Fallback: estimate based on typical cluster height
                    num_cluster_rows = max(1, int(img_height / 600))  # Assume ~600px per cluster row

                # Calculate height per cluster row
                row_height_px = img_height / num_cluster_rows
                row_height_scaled = row_height_px * width_scale

                # Calculate how many rows fit per page
                rows_per_page = max(1, int(max_page_height / row_height_scaled))

                # Split into chunks by whole rows
                num_chunks = (num_cluster_rows + rows_per_page - 1) // rows_per_page

                for chunk_idx in range(num_chunks):
                    # Calculate which rows to include
                    start_row = chunk_idx * rows_per_page
                    end_row = min((chunk_idx + 1) * rows_per_page, num_cluster_rows)

                    # Calculate pixel coordinates (crop at row boundaries)
                    top = int(start_row * row_height_px)
                    bottom = int(end_row * row_height_px)

                    # Crop the chunk
                    chunk = img_pil.crop((0, top, img_width, bottom))

                    # Save chunk temporarily as compressed JPEG (much smaller)
                    chunk_path = temp_dir / f"{class_name}_chunk_{chunk_idx}.jpg"
                    chunk.convert('RGB').save(chunk_path, 'JPEG', quality=pdf_quality, optimize=True)

                    # Calculate chunk height after scaling
                    chunk_height = (bottom - top) * width_scale

                    # Add chunk to PDF
                    if chunk_idx > 0:
                        story.append(Spacer(1, 0.05*inch))
                        story.append(Paragraph(f"<i>(Continuation - Clusters {start_row} to {end_row-1})</i>",
                                             ParagraphStyle('Continuation', parent=normal_style,
                                                          fontSize=9, alignment=TA_CENTER,
                                                          textColor=colors.grey)))
                        story.append(Spacer(1, 0.05*inch))

                    img = Image(str(chunk_path), width=max_width, height=chunk_height)
                    story.append(img)

                    # Add page break between chunks (except last one)
                    if chunk_idx < num_chunks - 1:
                        story.append(PageBreak())

        except Exception as e:
            import traceback
            print(f"   ⚠️  Error loading image for {class_name}: {e}")
            print(f"   ⚠️  Path attempted: {montage_path}")
            print(f"   ⚠️  Full error:\n{traceback.format_exc()}")
            story.append(Paragraph(f"⚠️ Error loading montage image", normal_style))

        story.append(Spacer(1, 0.1*inch))

        # Build caption
        if cluster_info is not None:
            caption = (f"<b>Samples:</b> {cluster_info['n_samples']} | "
                      f"<b>Clusters:</b> {cluster_info['n_clusters']} | "
                      f"<b>Outliers:</b> {cluster_info['n_outliers']} ({cluster_info['outlier_rate']}) | "
                      f"<b>Epsilon:</b> {cluster_info['eps_used']:.4f}")
        else:
            caption = f"<b>Samples:</b> {class_info['crops']} | <b>Clustering statistics unavailable</b>"

        caption_style = ParagraphStyle('Caption', parent=normal_style, fontSize=11, alignment=TA_CENTER)
        story.append(Paragraph(caption, caption_style))

        story.append(PageBreak())

    # Build PDF
    print("\n📦 Building PDF document...")
    doc.build(story)

    # Cleanup temp files
    print("🧹 Cleaning up temporary files...")
    import shutil
    shutil.rmtree(temp_dir)

    print("\n" + "="*70)
    print("✅ PDF GENERATION COMPLETE")
    print("="*70)
    print(f"📄 Output: {pdf_filename}")
    print(f"📊 Total pages: ~{len(stats['class_data']) + 1}")

    if missing_montages:
        print(f"\n⚠️  WARNING: {len(missing_montages)} class(es) had missing montages:")
        for cls in missing_montages:
            print(f"   - {cls}")

    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate PDF report from clustering analysis")
    parser.add_argument("--temp_txt_file", required=True, help="Path to Script 1 temp txt file")
    parser.add_argument("--clustering_dir", required=True, help="Path to Script 3 clustering output directory")
    parser.add_argument("--pdf_name", required=True, help="Output PDF filename (without .pdf extension)")

    args = parser.parse_args()

    pdf_quality = 75

    build_pdf(args.temp_txt_file, args.clustering_dir, args.pdf_name, pdf_quality)


if __name__ == "__main__":
    main()
