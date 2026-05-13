"""
Minimal Qwen3-VL test: embeddings + captions, sequential to avoid OOM.
Run from project root:  python utils/test_qwen.py
Point FRAMES_DIR at a real folder of frames before running.
"""

import gc
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoProcessor, AutoModel, Qwen3VLForConditionalGeneration

# ── CONFIG ────────────────────────────────────────────────────────────────────

FRAMES_DIR = Path("/home/retrocausal-train/Documents/yolov5/data/Dematic_ITDMS_Split/test")  # ← point at your frames folder
MAX_FRAMES = 20
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# Auto-select models based on available VRAM
# ≥16GB (training server): 8B embedding + 8B-Instruct caption
# <16GB (8GB laptop):       2B embedding + 3B-Instruct caption (sequential, fits in 8GB)
_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
if _vram_gb >= 16:
    EMBED_MODEL   = "Qwen/Qwen3-VL-Embedding-8B"
    CAPTION_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
else:
    EMBED_MODEL   = "Qwen/Qwen3-VL-Embedding-2B"
    CAPTION_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

print(f"VRAM detected: {_vram_gb:.1f}GB → embed={EMBED_MODEL.split('/')[-1]}, caption={CAPTION_MODEL.split('/')[-1]}")

CAPTION_INSTRUCTION = (
    "Overhead CCTV of industrial assembly workstation. "
    "Answer in order, stop when answered:\n"
    "1. Is worker PICKING something? → say what and from where "
    "(e.g. 'picking screw from bin', 'grabbing tool from cart', 'taking component from shelf')\n"
    "2. Is worker at MAIN ASSEMBLY unit doing something? → say action + tool if any "
    "(e.g. 'torquing bolt with torque gun', 'aligning component by hand', "
    "'tightening with wrench', 'applying adhesive', 'inspecting joint')\n"
    "3. Is worker CARRYING or MOVING something large? → say what "
    "(e.g. 'carrying rack to fixture', 'sliding unit along track')\n"
    "4. Otherwise → 'worker idle at workstation'\n"
    "One sentence, max 12 words, no hedging words like 'likely' or 'possibly'."
)


# ── LOAD FRAMES ───────────────────────────────────────────────────────────────

frame_paths = sorted(FRAMES_DIR.glob("*.jpg"))[:MAX_FRAMES]
frame_paths += sorted(FRAMES_DIR.glob("*.png"))[:MAX_FRAMES]
frame_paths = sorted(set(frame_paths))[:MAX_FRAMES]

if not frame_paths:
    raise FileNotFoundError(f"No frames found in {FRAMES_DIR.resolve()}")

print(f"Found {len(frame_paths)} frames in {FRAMES_DIR}")

# ── PHASE 1: EMBEDDINGS ───────────────────────────────────────────────────────

print(f"\n[Phase 1] Loading embedding model: {EMBED_MODEL}")
embed_processor = AutoProcessor.from_pretrained(EMBED_MODEL)
embed_model = AutoModel.from_pretrained(
    EMBED_MODEL,
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
).to(DEVICE).eval()
print(f"  Model on device: {next(embed_model.parameters()).device}")

embeddings = []

with torch.no_grad():
    for p in frame_paths:
        img = Image.open(p).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": ""},
        ]}]
        text = embed_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = embed_processor(
            text=[text], images=[img], return_tensors="pt", padding=True
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        outputs = embed_model(**inputs)
        hidden = outputs.last_hidden_state  # (1, seq_len, D)

        # EOS pooling: last real token via attention_mask
        attn_mask = inputs.get("attention_mask")
        if attn_mask is not None:
            last_idx = int(attn_mask[0].sum().item()) - 1
        else:
            last_idx = hidden.shape[1] - 1

        emb = hidden[0, last_idx, :].cpu().float().numpy()
        embeddings.append(emb)

        del inputs, outputs, hidden
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

embeddings = np.array(embeddings, dtype=np.float32)

# L2 normalize
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / np.where(norms == 0, 1.0, norms)

print(f"  Embeddings shape: {embeddings.shape}")

# Free embed model before loading caption model
del embed_model, embed_processor
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()
print("  Embed model freed.")

# ── PHASE 2: CAPTIONS ─────────────────────────────────────────────────────────

print(f"\n[Phase 2] Loading caption model: {CAPTION_MODEL}")
caption_processor = AutoProcessor.from_pretrained(CAPTION_MODEL)
caption_model = Qwen3VLForConditionalGeneration.from_pretrained(
    CAPTION_MODEL,
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    low_cpu_mem_usage=True,
).to(DEVICE).eval()
print(f"  Model on device: {next(caption_model.parameters()).device}")

captions = []

with torch.no_grad():
    for p in frame_paths:
        img = Image.open(p).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": CAPTION_INSTRUCTION},
                ],
            }
        ]
        text = caption_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = caption_processor(text=[text], images=[img], return_tensors="pt").to(DEVICE)
        inputs.pop("token_type_ids", None)

        generated_ids = caption_model.generate(**inputs, max_new_tokens=64, do_sample=False)
        trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
        caption = caption_processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        captions.append(caption)

        del inputs, generated_ids
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

del caption_model, caption_processor
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()
print("  Caption model freed.")

# ── RESULTS ───────────────────────────────────────────────────────────────────

print("\n=== Captions ===")
for p, cap in zip(frame_paths, captions):
    print(f"  {p.name}: {cap}")

sim_matrix = cosine_similarity(embeddings)

print("\n=== Cosine Similarity Matrix ===")
print(np.round(sim_matrix, 3))

THRESHOLD_SAME    = 0.85
THRESHOLD_SIMILAR = 0.65

print("\n=== Pair Report ===")
for i in range(len(frame_paths)):
    for j in range(i + 1, len(frame_paths)):
        sim = sim_matrix[i][j]
        if sim > THRESHOLD_SAME:
            tag = "SAME"
        elif sim > THRESHOLD_SIMILAR:
            tag = "SIMILAR"
        else:
            tag = "DIFFERENT"
        print(f"  {frame_paths[i].name} vs {frame_paths[j].name}: {sim:.3f}  [{tag}]")

avg_sim = sim_matrix[np.triu_indices(len(frame_paths), k=1)].mean()
print(f"\nAverage pairwise similarity: {avg_sim:.3f}")
if avg_sim > 0.85:
    print("Overall: Single continuous activity")
elif avg_sim > 0.65:
    print("Overall: Related workflow steps")
else:
    print("Overall: Multiple different activities")
