"""
Test script for V-JEPA 2.1 ViT-B/16 image embeddings.

Config:
    https://github.com/facebookresearch/vjepa2/tree/main/configs/train_2_1/vitb16
Checkpoint download (V-JEPA 2.1 Pretrain - ViT-B/16):
    https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt

# V-JEPA 2.1 via torch.hub (alternative, needs full repo):
# encoder, predictor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_base_384')

# Basic test (dummy image only)
python "test_v_jepa_2.1.py"

# Test with real images
python "test_v_jepa_2.1.py" img1.jpg img2.jpg img3.jpg

# Test with a folder
python "test_v_jepa_2.1.py" path/to/frames/
"""

import sys
import time
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path

# ─── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = Path(__file__).parent / "models" / "vjepa2_1_vitb_dist_vitG_384.pt"
IMG_SIZE = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet normalization (same as their eval uses)
NORMALIZE = transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)

TRANSFORM = transforms.Compose([
    transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    NORMALIZE,
])


# ─── Build ViT-B/16 from scratch (no need to clone vjepa2 repo) ──────────────
def build_vit_base_384():
    """
    Manually construct the VisionTransformer matching vjepa2_1_vit_base_384 config.
    Uses torch.hub to pull only the model definition from the repo.
    """
    # Pull VisionTransformer class from the repo via torch.hub
    vit_src = torch.hub.load(
        'facebookresearch/vjepa2',
        'vjepa2_1_vit_base_384',
        pretrained=False,
    )
    # vit_src returns (encoder, predictor) - we only need encoder
    encoder = vit_src[0]

    # Load our local checkpoint
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    # V-JEPA 2.1 ViT-B uses 'ema_encoder' key
    state_dict = ckpt["ema_encoder"]

    # Clean up keys (remove 'module.' or 'backbone.' prefixes if present)
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("backbone.", "")
        cleaned[k] = v

    encoder.load_state_dict(cleaned, strict=True)
    encoder.eval()
    encoder.to(DEVICE)

    print(f"  Model: ViT-B/16 | embed_dim={encoder.embed_dim} | params={sum(p.numel() for p in encoder.parameters()) / 1e6:.1f}M")
    print(f"  Device: {DEVICE}")

    return encoder


# ─── Extract embeddings ───────────────────────────────────────────────────────
@torch.no_grad()
def extract_embedding(encoder, img_path):
    """Extract a single 768d embedding from an image via mean-pooling patch tokens."""
    img = Image.open(img_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)  # [1, 3, 384, 384]

    # V-JEPA 2.1 image mode: 5D with T=1 triggers patch_embed_img + img_mod_embed
    # (img_temporal_dim_size=1 in model config → check_temporal_dim matches T=1)
    tensor = tensor.unsqueeze(2)  # [1, 3, 1, 384, 384]

    patch_tokens = encoder(tensor)  # [1, num_patches, 768]

    # Mean-pool patch tokens → single 768d vector
    embedding = patch_tokens.mean(dim=1)  # [1, 768]

    # L2 normalize for cosine similarity
    embedding = nn.functional.normalize(embedding, dim=-1)

    return embedding.cpu().numpy().squeeze()  # (768,)


@torch.no_grad()
def extract_embeddings_batch(encoder, img_paths, batch_size=16):
    """Extract embeddings for multiple images in batches."""
    all_embeddings = []

    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i + batch_size]
        tensors = []
        valid_indices = []

        for j, p in enumerate(batch_paths):
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(TRANSFORM(img))
                valid_indices.append(j)
            except Exception as e:
                print(f"  Skipping {p}: {e}")

        if not tensors:
            continue

        batch = torch.stack(tensors).to(DEVICE)  # [B, 3, 384, 384]
        # V-JEPA 2.1 image mode: T=1 triggers native image path
        batch = batch.unsqueeze(2)  # [B, 3, 1, 384, 384]
        patch_tokens = encoder(batch)  # [B, num_patches, 768]
        embeddings = patch_tokens.mean(dim=1)  # [B, 768]
        embeddings = nn.functional.normalize(embeddings, dim=-1)
        all_embeddings.append(embeddings.cpu().numpy())

        # Free GPU memory
        del batch, patch_tokens, embeddings
        torch.cuda.empty_cache()

    if all_embeddings:
        return np.concatenate(all_embeddings, axis=0)
    return np.array([])


# ─── Similarity test ──────────────────────────────────────────────────────────
def test_similarity(encoder, img_paths):
    """Compute pairwise cosine similarity between images."""
    if len(img_paths) < 2:
        print("Need at least 2 images for similarity test.")
        return

    print(f"\nExtracting embeddings for {len(img_paths)} images...")
    embeddings = extract_embeddings_batch(encoder, img_paths)

    print(f"  Embeddings shape: {embeddings.shape}")

    # Pairwise cosine similarity (embeddings already L2-normalized)
    sim_matrix = embeddings @ embeddings.T

    print("\nPairwise cosine similarity:")
    names = [Path(p).name for p in img_paths]

    # Header
    max_name_len = max(len(n) for n in names)
    header = " " * (max_name_len + 2)
    for n in names:
        header += f"{n[:10]:>12}"
    print(header)

    # Rows
    for i, name in enumerate(names):
        row = f"{name:<{max_name_len}}  "
        for j in range(len(names)):
            row += f"{sim_matrix[i, j]:12.4f}"
        print(row)


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("V-JEPA 2.1 ViT-B/16 - Image Embedding Test")
    print("=" * 60)

    # Check checkpoint exists
    if not CHECKPOINT_PATH.exists():
        print(f"\nCheckpoint not found at: {CHECKPOINT_PATH}")
        print(f"Download from: https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt")
        sys.exit(1)

    # Build model
    print("\n[1] Loading V-JEPA 2.1 encoder...")
    t0 = time.time()
    encoder = build_vit_base_384()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Test with a single dummy image first
    print("\n[2] Testing with dummy image (random noise)...")
    # V-JEPA 2.1 image mode: [B, C, T=1, H, W] uses patch_embed_img (tubelet=1)
    dummy = torch.randn(1, 3, 1, IMG_SIZE, IMG_SIZE).to(DEVICE)
    t0 = time.time()
    with torch.no_grad():
        output = encoder(dummy)
    print(f"  Output shape: {output.shape}")
    print(f"  spatial_patches = (384/16)^2 = {(IMG_SIZE // 16) ** 2}, temporal_patches = T/tubelet = 1/1 = 1")
    print(f"  Inference time: {time.time() - t0:.3f}s")

    # Mean pool
    pooled = output.mean(dim=1)
    print(f"  Pooled embedding shape: {pooled.shape}")  # expect [1, 768]

    # VRAM usage
    if torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"  Peak VRAM: {vram_mb:.0f} MB")

    # If user provides image paths, test real images
    if len(sys.argv) > 1:
        img_paths = sys.argv[1:]
        # Expand glob patterns
        expanded = []
        for p in img_paths:
            path = Path(p)
            if path.is_dir():
                expanded.extend(sorted(path.glob("*.png")) + sorted(path.glob("*.jpg")) + sorted(path.glob("*.jpeg")))
            elif path.exists():
                expanded.append(path)

        if expanded:
            print(f"\n[3] Testing with {len(expanded)} real images...")
            test_similarity(encoder, expanded)
        else:
            print(f"\nNo valid images found in: {img_paths}")
    else:
        print("\n[3] To test with real images:")
        print(f'    python "{Path(__file__).name}" path/to/image1.jpg path/to/image2.jpg')
        print(f'    python "{Path(__file__).name}" path/to/image_folder/')

    print("\nDone!")
