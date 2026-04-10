"""
Embedding model comparison: V-JEPA 2.1 vs DINOv2 vs SigLIP vs DinoV3
Runs all three on the same images and prints pairwise cosine similarity + stats.

Checkpoint download (V-JEPA 2.1 Pretrain - ViT-B/16):
    https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt

Usage:
    python "test_v_jepa_2.1.py" path/to/folder/
    python "test_v_jepa_2.1.py" img1.jpg img2.jpg img3.jpg
"""

import sys
import time
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────
VJEPA_CHECKPOINT = Path(__file__).parent.parent / "models" / "vjepa2_1_vitb_dist_vitG_384.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Transforms ───────────────────────────────────────────────────────────────
TRANSFORM_VJEPA = transforms.Compose([
    transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

TRANSFORM_DINO = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# SigLIP uses its own processor (loaded later)


# ─── V-JEPA 2.1 ───────────────────────────────────────────────────────────────
def load_vjepa():
    if not VJEPA_CHECKPOINT.exists():
        print(f"  [SKIP] Checkpoint not found: {VJEPA_CHECKPOINT}")
        print(f"         Download from: https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt")
        return None

    encoder, _ = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_base_384', pretrained=False)
    ckpt = torch.load(VJEPA_CHECKPOINT, map_location="cpu", weights_only=False)
    state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in ckpt["ema_encoder"].items()}
    encoder.load_state_dict(state_dict, strict=True)
    encoder.eval().to(DEVICE)
    print(f"  embed_dim={encoder.embed_dim} | params={sum(p.numel() for p in encoder.parameters()) / 1e6:.1f}M | device={DEVICE}")
    return encoder


@torch.no_grad()
def embed_vjepa(encoder, img_paths, batch_size=8):
    all_embs = []
    for i in range(0, len(img_paths), batch_size):
        batch = []
        for p in img_paths[i:i + batch_size]:
            try:
                batch.append(TRANSFORM_VJEPA(Image.open(p).convert("RGB")))
            except Exception as e:
                print(f"  Skipping {p}: {e}")
        if not batch:
            continue
        t = torch.stack(batch).unsqueeze(2).to(DEVICE)  # [B, 3, 1, 384, 384]
        emb = encoder(t).mean(dim=1)                     # [B, 768]
        emb = nn.functional.normalize(emb, dim=-1)
        all_embs.append(emb.cpu().numpy())
        del t, emb
        torch.cuda.empty_cache()
    return np.concatenate(all_embs, axis=0) if all_embs else np.array([])


# ─── DINOv2 ───────────────────────────────────────────────────────────────────
def load_dinov2():
    from transformers import AutoModel
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    model.eval().to(DEVICE)
    print(f"  embed_dim=768 | params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M | device={DEVICE}")
    return model


@torch.no_grad()
def embed_dinov2(model, img_paths, batch_size=16):
    all_embs = []
    for i in range(0, len(img_paths), batch_size):
        batch = []
        for p in img_paths[i:i + batch_size]:
            try:
                batch.append(TRANSFORM_DINO(Image.open(p).convert("RGB")))
            except Exception as e:
                print(f"  Skipping {p}: {e}")
        if not batch:
            continue
        t = torch.stack(batch).to(DEVICE)
        out = model(pixel_values=t)
        emb = out.last_hidden_state[:, 0]  # CLS token
        emb = nn.functional.normalize(emb, dim=-1)
        all_embs.append(emb.cpu().numpy())
        del t, out, emb
        torch.cuda.empty_cache()
    return np.concatenate(all_embs, axis=0) if all_embs else np.array([])


# ─── DINOv3 ───────────────────────────────────────────────────────────────────
def load_dinov3():
    from transformers import DINOv3ViTImageProcessorFast, AutoModel
    processor = DINOv3ViTImageProcessorFast.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
    model = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
    model.eval().to(DEVICE)
    print(f"  embed_dim={model.config.hidden_size} | params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M | device={DEVICE}")
    return model, processor


@torch.no_grad()
def embed_dinov3(model, processor, img_paths, batch_size=16):
    all_embs = []
    for i in range(0, len(img_paths), batch_size):
        imgs = []
        for p in img_paths[i:i + batch_size]:
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"  Skipping {p}: {e}")
        if not imgs:
            continue
        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        out = model(**inputs)
        emb = out.last_hidden_state[:, 0]  # CLS token (index 0, before register tokens)
        emb = nn.functional.normalize(emb, dim=-1)
        all_embs.append(emb.cpu().numpy())
        del inputs, out, emb
        torch.cuda.empty_cache()
    return np.concatenate(all_embs, axis=0) if all_embs else np.array([])


# ─── SigLIP ───────────────────────────────────────────────────────────────────
def load_siglip():
    from transformers import AutoProcessor, AutoModel
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
    model.eval().to(DEVICE)
    print(f"  embed_dim=768 | params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M | device={DEVICE}")
    return model, processor


@torch.no_grad()
def embed_siglip(model, processor, img_paths, batch_size=16):
    all_embs = []
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"  Skipping {p}: {e}")
        if not imgs:
            continue
        inputs = processor(images=imgs, return_tensors="pt", padding=True).to(DEVICE)
        out = model.vision_model(**inputs)
        emb = out.pooler_output  # [B, 768]
        emb = nn.functional.normalize(emb, dim=-1)
        all_embs.append(emb.cpu().numpy())
        del inputs, out, emb
        torch.cuda.empty_cache()
    return np.concatenate(all_embs, axis=0) if all_embs else np.array([])


# ─── Reporting ────────────────────────────────────────────────────────────────
def print_similarity_stats(name, embeddings, img_paths):
    sim = embeddings @ embeddings.T
    n = len(sim)
    # Off-diagonal values only
    mask = ~np.eye(n, dtype=bool)
    off_diag = sim[mask]

    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    print(f"  min={off_diag.min():.4f}  max={off_diag.max():.4f}  "
          f"mean={off_diag.mean():.4f}  std={off_diag.std():.4f}  "
          f"range={off_diag.max()-off_diag.min():.4f}")

    # Group by video/source prefix (first 10 chars of filename)
    names = [Path(p).name for p in img_paths]
    prefixes = [n[:10] for n in names]
    unique = sorted(set(prefixes))

    if len(unique) > 1:
        within, cross = [], []
        for i in range(n):
            for j in range(i + 1, n):
                s = sim[i, j]
                if prefixes[i] == prefixes[j]:
                    within.append(s)
                else:
                    cross.append(s)
        if within:
            print(f"  within-video:  mean={np.mean(within):.4f}  min={np.min(within):.4f}  max={np.max(within):.4f}")
        if cross:
            print(f"  cross-video:   mean={np.mean(cross):.4f}  min={np.min(cross):.4f}  max={np.max(cross):.4f}")
        if within and cross:
            sep = np.mean(within) - np.mean(cross)
            print(f"  separation (within - cross mean): {sep:+.4f}  {'✓ good' if sep > 0.05 else '✗ poor'}")

    # Full matrix (truncated names)
    col_w = 10
    header = " " * 52 + "  " + "  ".join(f"{n[:col_w]:>{col_w}}" for n in names)
    print(f"\n{header}")
    for i, name in enumerate(names):
        row = f"  {name:<50}  " + "  ".join(f"{sim[i,j]:>{col_w}.4f}" for j in range(n))
        print(row)


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Embedding Model Comparison: V-JEPA 2.1 vs DINOv2 vs DINOv3 vs SigLIP")
    print("=" * 60)

    # Collect image paths
    img_paths = []
    if len(sys.argv) > 1:
        for p in sys.argv[1:]:
            path = Path(p)
            if path.is_dir():
                img_paths.extend(sorted(path.glob("*.png")) + sorted(path.glob("*.jpg")) + sorted(path.glob("*.jpeg")))
            elif path.exists():
                img_paths.append(path)

    if not img_paths:
        print("\nUsage: python test_v_jepa_2.1.py path/to/folder/")
        print("       python test_v_jepa_2.1.py img1.jpg img2.jpg ...")
        sys.exit(1)

    print(f"\nImages: {len(img_paths)}")
    for p in img_paths:
        print(f"  {p.name}")

    results = {}

    # ── V-JEPA 2.1 ──
    print("\n\n[1/4] V-JEPA 2.1 ViT-B/16 (384px)")
    print("─" * 40)
    t0 = time.time()
    vjepa = load_vjepa()
    if vjepa is not None:
        emb = embed_vjepa(vjepa, img_paths)
        print(f"  Embeddings: {emb.shape} | time={time.time()-t0:.1f}s")
        results["V-JEPA 2.1 ViT-B/16"] = emb
        del vjepa
        torch.cuda.empty_cache()

    # ── DINOv2 ──
    print("\n[2/4] DINOv2 ViT-B/14 (224px)")
    print("─" * 40)
    t0 = time.time()
    dino = load_dinov2()
    emb = embed_dinov2(dino, img_paths)
    print(f"  Embeddings: {emb.shape} | time={time.time()-t0:.1f}s")
    results["DINOv2 ViT-B/14"] = emb
    del dino
    torch.cuda.empty_cache()

    # ── DINOv3 ──
    print("\n[3/4] DINOv3 ViT-B/16 (224px)")
    print("─" * 40)
    t0 = time.time()
    dinov3_model, dinov3_proc = load_dinov3()
    emb = embed_dinov3(dinov3_model, dinov3_proc, img_paths)
    print(f"  Embeddings: {emb.shape} | time={time.time()-t0:.1f}s")
    results["DINOv3 ViT-B/16"] = emb
    del dinov3_model, dinov3_proc
    torch.cuda.empty_cache()

    # ── SigLIP ──
    print("\n[4/4] SigLIP ViT-B/16 (224px)")
    print("─" * 40)
    t0 = time.time()
    siglip_model, siglip_proc = load_siglip()
    emb = embed_siglip(siglip_model, siglip_proc, img_paths)
    print(f"  Embeddings: {emb.shape} | time={time.time()-t0:.1f}s")
    results["SigLIP ViT-B/16"] = emb
    del siglip_model, siglip_proc
    torch.cuda.empty_cache()

    # ── Results ──
    print("\n\n" + "=" * 60)
    print("SIMILARITY RESULTS")
    print("=" * 60)
    for model_name, emb in results.items():
        print_similarity_stats(model_name, emb, img_paths)

    print("\n\nDone!")
