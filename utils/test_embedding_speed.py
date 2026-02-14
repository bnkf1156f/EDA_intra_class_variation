from transformers import AutoModel, AutoImageProcessor, CLIPModel, CLIPProcessor, SiglipVisionModel
import torch
import time
import numpy as np
from PIL import Image

models = [
    ('facebook/dinov2-base', 'Current baseline', 'auto'),
    ('openai/clip-vit-base-patch32', 'Fastest option', 'clip'),
    ('google/siglip-base-patch16-224', 'Balanced: speed + quality', 'siglip'),
    ('google/siglip-base-patch16-384', 'Better quality, slower', 'siglip'),
]

print("="*80)
print("🔍 COMPLETE MODEL COMPARISON - Dimensions + Speed + VRAM")
print("Target: 10K frames in 15-20 mins on 8GB GPU")
print("="*80)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🖥️  Device: {device}")

if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Test batches
batch_sizes = [16, 32, 64]
num_test_batches = 5

results_summary = []

for model_name, description, model_type in models:
    try:
        print(f'\n{"="*80}')
        print(f'📦 {model_name}')
        print(f'   ({description})')
        print(f'{"="*80}')

        # Load model with type-specific handling
        print("   ⏳ Loading...", end=" ", flush=True)
        start = time.time()

        if model_type == 'clip':
            from transformers import CLIPModel, CLIPProcessor
            processor = CLIPProcessor.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(model_name).to(device)
        elif model_type == 'siglip':
            from transformers import AutoProcessor, SiglipVisionModel
            processor = AutoProcessor.from_pretrained(model_name)
            model = SiglipVisionModel.from_pretrained(model_name).to(device)
        else:  # auto (DINOv2)
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device)

        model.eval()
        load_time = time.time() - start
        print(f"✓ ({load_time:.1f}s)")

        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f'   📊 Parameters: {total_params/1e6:.0f}M')

        # Get embedding dimension (create proper PIL-compatible dummy data)
        dummy_img = Image.new('RGB', (224, 224), color=(128, 128, 128))

        if model_type == 'siglip':
            # SigLIP vision model only needs pixel_values
            inputs = processor(images=dummy_img, return_tensors='pt')
            inputs = {'pixel_values': inputs['pixel_values'].to(device)}
        else:
            inputs = processor(images=dummy_img, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        embedding_dim = None
        if model_type == 'clip':
            # CLIP uses image_embeds from vision model
            embedding_dim = outputs.image_embeds.shape[-1]
            print(f'   📐 Embedding: {embedding_dim}d (image_embeds)')
        elif model_type == 'siglip':
            # SigLIP uses pooler_output
            embedding_dim = outputs.pooler_output.shape[-1]
            print(f'   📐 Embedding: {embedding_dim}d (pooler_output)')
        elif hasattr(outputs, 'last_hidden_state'):
            # DINOv2 uses last_hidden_state
            embedding_dim = outputs.last_hidden_state.shape[-1]
            print(f'   📐 Embedding: {embedding_dim}d (last_hidden_state)')
        elif hasattr(outputs, 'pooler_output'):
            embedding_dim = outputs.pooler_output.shape[-1]
            print(f'   📐 Embedding: {embedding_dim}d (pooler_output)')

        if embedding_dim:
            if embedding_dim == 768:
                print(f'       ✅ MATCHES DINOv2 → Drop-in replacement!')
            else:
                print(f'       ⚠️  Different from DINOv2 (768d) → Code changes needed')

        # Speed benchmark
        print(f'\n   ⚡ Speed Test:')
        print(f'   {"Batch":<8} {"FPS":<12} {"10K Time":<12} {"VRAM":<10} {"Status"}')
        print(f'   {"-"*8} {"-"*12} {"-"*12} {"-"*10} {"-"*15}')

        best_batch = None
        best_fps = 0
        best_time_10k = 999

        for batch_size in batch_sizes:
            try:
                # Create batch of PIL images
                dummy_batch = [Image.new('RGB', (224, 224), color=(128, 128, 128)) for _ in range(batch_size)]

                if model_type == 'siglip':
                    inputs_batch = processor(images=dummy_batch, return_tensors='pt')
                    inputs_batch = {'pixel_values': inputs_batch['pixel_values'].to(device)}
                else:
                    inputs_batch = processor(images=dummy_batch, return_tensors='pt').to(device)

                # Warmup
                with torch.no_grad():
                    _ = model(**inputs_batch)

                if device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()

                # Benchmark
                times = []
                for _ in range(num_test_batches):
                    start = time.time()
                    with torch.no_grad():
                        _ = model(**inputs_batch)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    times.append(time.time() - start)

                avg_time = np.mean(times)
                fps = batch_size / avg_time
                time_10k = 10000 / fps / 60  # minutes

                if device == "cuda":
                    peak_vram = torch.cuda.max_memory_allocated() / 1e9
                    vram_str = f"{peak_vram:.1f}GB"
                else:
                    vram_str = "N/A"

                # Status indicator
                if time_10k <= 15:
                    status = "✅ FAST"
                elif time_10k <= 20:
                    status = "⚠️  OK"
                else:
                    status = "❌ SLOW"

                print(f'   {batch_size:<8} {fps:>6.0f} fps{"":>3} {time_10k:>5.1f} min{"":>3} {vram_str:<10} {status}')

                if time_10k < best_time_10k:
                    best_fps = fps
                    best_batch = batch_size
                    best_time_10k = time_10k

                if device == "cuda":
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f'   {batch_size:<8} {"OOM":<12} {"N/A":<12} {"N/A":<10} ❌ OUT OF MEMORY')
                    if device == "cuda":
                        torch.cuda.empty_cache()
                else:
                    raise

        if best_batch:
            meets_target = "✅" if best_time_10k <= 20 else "❌"
            print(f'\n   🏆 BEST: batch={best_batch} → {best_time_10k:.1f} min for 10K frames {meets_target}')

            results_summary.append({
                'model': model_name.split('/')[-1],
                'emb_dim': embedding_dim,
                'time_10k': best_time_10k,
                'batch': best_batch,
                'meets_target': best_time_10k <= 20
            })

        del model, processor
        if device == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        print(f'   ❌ Error: {e}')
        if device == "cuda":
            torch.cuda.empty_cache()

# Final summary
print("\n" + "="*80)
print("📋 FINAL SUMMARY - BEST OPTIONS FOR YOUR USE CASE")
print("="*80)

if results_summary:
    print(f"\n{'Model':<35} {'Emb':<8} {'Time':<10} {'Target'}")
    print(f"{'-'*35} {'-'*8} {'-'*10} {'-'*10}")
    for r in results_summary:
        target_icon = "✅" if r['meets_target'] else "❌"
        emb_match = "768d ✅" if r['emb_dim'] == 768 else f"{r['emb_dim']}d ⚠️"
        print(f"{r['model']:<35} {emb_match:<8} {r['time_10k']:>5.1f} min  {target_icon}")

if results_summary:
    print("\n💡 RECOMMENDATION:")
    fast_and_compatible = [r for r in results_summary if r['meets_target'] and r['emb_dim'] == 768]
    if fast_and_compatible:
        best = min(fast_and_compatible, key=lambda x: x['time_10k'])
        print(f"   ✅ BEST: {best['model']} → {best['time_10k']:.1f} min (768d = drop-in replacement)")
    else:
        print("   ⚠️  No models meet both speed (<20 min) AND dimension (768d) requirements")
        fast_only = [r for r in results_summary if r['meets_target']]
        if fast_only:
            print("   📌 Fast but require code changes for embedding dimension:")
            for r in fast_only:
                print(f"      • {r['model']}: {r['time_10k']:.1f} min ({r['emb_dim']}d)")

print("="*80)
