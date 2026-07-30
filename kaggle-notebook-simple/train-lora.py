"""
OYKHCHAR LoRA Training Script for Kaggle
Trains a custom character LoRA on FLUX.1-dev
"""

import torch
import os
from pathlib import Path
import zipfile
from diffusers import FluxPipeline, AutoencoderKL
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torch.optim import AdamW
from tqdm import tqdm

print("=" * 60)
print("OYKHCHAR LoRA Training on FLUX.1-dev")
print("=" * 60)

# Step 1: Find and extract training images
print("\n[1/7] Finding training images...")
input_base = Path('/kaggle/input/oykhchar-training')

# List all files to debug
print("Available files:")
if input_base.exists():
    for item in input_base.rglob('*'):
        print(f"  {item}")

# Try to find images.zip
zip_path = None
for possible_path in input_base.rglob('*.zip'):
    if 'images' in possible_path.name.lower():
        zip_path = possible_path
        break

if not zip_path:
    print("ERROR: images.zip not found!")
    print("Looking for .jpg files directly...")
    image_files = list(input_base.rglob('*.jpg'))
    if image_files:
        print(f"Found {len(image_files)} .jpg files directly")
        extract_path = input_base
    else:
        raise FileNotFoundError("No training images found!")
else:
    print(f"Found zip: {zip_path}")
    extract_path = Path('/kaggle/working/training_images')
    extract_path.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    image_files = list(extract_path.rglob('*.jpg'))
    print(f"Extracted {len(image_files)} training images")

# Step 2: Check GPU
print("\n[2/7] Checking GPU...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Step 3: Load FLUX.1-dev
print("\n[3/7] Loading FLUX.1-dev model...")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")
print("Model loaded successfully")

# Step 4: Configure LoRA
print("\n[4/7] Configuring LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    lora_dropout=0.1,
)

pipe.transformer = get_peft_model(pipe.transformer, lora_config)
print("LoRA configured")
pipe.transformer.print_trainable_parameters()

# Step 5: Prepare dataset
print("\n[5/7] Preparing dataset...")

class LoRADataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = list(Path(image_dir).rglob('*.jpg'))
        print(f"Found {len(self.image_paths)} images in dataset")

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.captions = []
        for img_path in self.image_paths:
            caption_path = img_path.with_suffix('.txt')
            if caption_path.exists():
                with open(caption_path, 'r') as f:
                    self.captions.append(f.read().strip())
            else:
                self.captions.append("OYKHCHAR character")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        return {'image': image, 'caption': self.captions[idx]}

dataset = LoRADataset(extract_path)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
print(f"Dataset ready: {len(dataset)} images")

# Step 6: Train
print("\n[6/7] Training LoRA...")
optimizer = AdamW(pipe.transformer.parameters(), lr=1e-4)
num_epochs = 50

pipe.transformer.train()
global_step = 0

for epoch in range(num_epochs):
    epoch_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch in progress_bar:
        images = batch['image'].to("cuda")
        captions = batch['caption']

        with torch.cuda.amp.autocast():
            text_embeddings = pipe.encode_prompt(
                captions,
                device="cuda",
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
            )

            latents = pipe.vae.encode(images).latent_dist.sample()
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device="cuda")

            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            model_pred = pipe.transformer(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings[0]
            ).sample

            loss = torch.nn.functional.mse_loss(model_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        global_step += 1
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

    if (epoch + 1) % 10 == 0:
        checkpoint_path = f"/kaggle/working/checkpoint_epoch_{epoch+1}"
        pipe.transformer.save_pretrained(checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

print("\nTraining complete!")

# Step 7: Save and test
print("\n[7/7] Saving final LoRA...")
output_dir = "/kaggle/working/oykhchar-lora-final"
pipe.transformer.save_pretrained(output_dir)
print(f"LoRA saved: {output_dir}")

print("\nTesting trained LoRA...")
pipe.transformer.eval()

test_prompts = [
    "OYKHCHAR character standing with arms raised",
    "OYKHCHAR character sitting and thinking",
    "OYKHCHAR character running forward"
]

for i, prompt in enumerate(test_prompts):
    print(f"Generating test {i+1}: {prompt}")
    image = pipe(
        prompt,
        num_inference_steps=28,
        guidance_scale=3.5,
        height=1024,
        width=1024
    ).images[0]

    image.save(f"/kaggle/working/test_{i+1}.png")

print("\n" + "=" * 60)
print("ALL DONE!")
print("Download LoRA from: /kaggle/working/oykhchar-lora-final/")
print("=" * 60)
