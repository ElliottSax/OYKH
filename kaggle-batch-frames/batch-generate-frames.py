"""
Batch Frame Generator for OYKH Videos
Generate 100-500 frames for FREE on Kaggle

Instead of: $0.05 per frame on Replicate = $25 for 500 frames
On Kaggle: $0 for 500 frames
"""

import torch
from diffusers import FluxPipeline
import json
from pathlib import Path

print("=" * 60)
print("OYKH Batch Frame Generator")
print("Generating 100s of frames for FREE!")
print("=" * 60)

# Load FLUX with trained LoRA
print("\nLoading FLUX + OYKHCHAR LoRA...")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)
pipe.load_lora_weights("/kaggle/input/oykhchar-lora/oykhchar-lora-final")
pipe = pipe.to("cuda")
print("Model loaded!")

# Common scenarios for viral videos
scenarios = {
    "celebration": [
        "OYKHCHAR standing with arms raised in celebration",
        "OYKHCHAR jumping with joy",
        "OYKHCHAR doing a victory pose",
        "OYKHCHAR clapping hands happily"
    ],
    "thinking": [
        "OYKHCHAR sitting and thinking deeply",
        "OYKHCHAR with hand on chin contemplating",
        "OYKHCHAR looking confused with question mark above head",
        "OYKHCHAR having an aha moment"
    ],
    "active": [
        "OYKHCHAR running forward energetically",
        "OYKHCHAR walking confidently",
        "OYKHCHAR doing exercise",
        "OYKHCHAR dancing"
    ],
    "working": [
        "OYKHCHAR typing on laptop",
        "OYKHCHAR reading a book",
        "OYKHCHAR taking notes",
        "OYKHCHAR presenting with pointer"
    ],
    "emotions": [
        "OYKHCHAR looking happy and smiling",
        "OYKHCHAR looking sad",
        "OYKHCHAR looking surprised",
        "OYKHCHAR looking determined"
    ]
}

# Generate library
print("\nGenerating frame library...")
print("This will take 2-3 hours but is 100% FREE\n")

output_dir = Path("/kaggle/working/frame_library")
output_dir.mkdir(exist_ok=True)

total_frames = 0
manifest = {}

for category, prompts in scenarios.items():
    print(f"\n📁 Category: {category}")
    category_dir = output_dir / category
    category_dir.mkdir(exist_ok=True)

    manifest[category] = []

    for i, prompt in enumerate(prompts):
        print(f"  Generating: {prompt}")

        # Generate 5 variations of each prompt
        for variation in range(5):
            image = pipe(
                prompt,
                num_inference_steps=28,
                guidance_scale=3.5,
                height=1024,
                width=1024
            ).images[0]

            filename = f"{category}_{i+1}_var{variation+1}.png"
            filepath = category_dir / filename
            image.save(filepath)

            manifest[category].append({
                "prompt": prompt,
                "filename": filename,
                "variation": variation + 1
            })

            total_frames += 1
            print(f"    ✅ {filename} (Total: {total_frames})")

# Save manifest
manifest_path = output_dir / "manifest.json"
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)

print("\n" + "=" * 60)
print(f"✅ COMPLETE! Generated {total_frames} frames")
print(f"📁 Location: {output_dir}")
print(f"📋 Manifest: {manifest_path}")
print("=" * 60)
print("\nCost on Replicate: $" + str(total_frames * 0.05))
print("Cost on Kaggle: $0")
print(f"Savings: ${total_frames * 0.05}")
print("\nDownload all frames and use in video production!")
