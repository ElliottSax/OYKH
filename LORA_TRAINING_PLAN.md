# LoRA Training Plan for Perfect Character Consistency

## What is LoRA?
**Low-Rank Adaptation** - Fine-tunes FLUX model to "memorize" your exact character design.
- Train on 15-30 images of your character
- Model learns your stick figure style
- **Result:** 90-95%+ consistency forever

---

## Step 1: Collect Training Images (15-30 images)

### Option A: Curate from existing videos
1. Extract frames from your best videos
2. Keep only shots where character looks perfect
3. Crop/clean as needed

### Option B: Generate fresh training set
1. Create 30 variations with Replicate (best quality)
2. Manually review and keep only perfect ones
3. Ensure variety: different poses, different props

### Training Image Requirements:
- ✅ Same character design across all images
- ✅ Variety of poses (pointing, thinking, holding, etc.)
- ✅ Variety of props (coffee, brain, lightbulb, etc.)
- ✅ Same background style (blue-purple gradient)
- ✅ High quality (1024x576 minimum)
- ❌ No inconsistent shots
- ❌ No bad generations

---

## Step 2: Prepare Training Data

### Directory Structure:
```
C:/projects/oykh-temp/lora-training/
├── images/
│   ├── shot_001.png  (character pointing)
│   ├── shot_002.png  (character thinking)
│   ├── shot_003.png  (character holding coffee)
│   └── ... (15-30 total)
└── captions/
    ├── shot_001.txt  (caption for each image)
    ├── shot_002.txt
    └── ...
```

### Caption Format:
Each `.txt` file should describe the image:
```
white stick figure character with round head and dot eyes, pointing gesture, blue-purple gradient background, educational illustration style
```

Keep captions consistent - they teach the LoRA what to look for.

---

## Step 3: Train LoRA

### Option A: Replicate (Easiest, $5-10)
```javascript
// Use Replicate's FLUX LoRA training
// https://replicate.com/ostris/flux-dev-lora-trainer

const training = await replicate.trainings.create(
  "ostris",
  "flux-dev-lora-trainer",
  "e440909d3512c31646ee2e0c7d6f6f4923224863a6a10c494606e79fb5844497",
  {
    destination: "your-username/oykh-character-lora",
    input: {
      input_images: "https://your-zip-file-url.zip", // ZIP of training images
      steps: 1000,
      lora_rank: 16,
      optimizer: "adamw8bit",
      batch_size: 1,
      resolution: "512,768,1024",
      autocaption: true, // Or provide manual captions
      trigger_word: "OYKHCHAR", // Special word to activate your character
    },
  }
);
```

### Option B: HuggingFace/Diffusers (Free, requires GPU)
Use Google Colab or HuggingFace Spaces with free GPU:
```python
# Install dependencies
!pip install diffusers transformers accelerate peft

# Training script (simplified)
from diffusers import FluxPipeline, DiffusionPipeline
from peft import LoraConfig

# Load base model
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["to_q", "to_v", "to_k", "to_out.0"],
)

# Train (30min - 2hrs depending on GPU)
# ... training loop ...

# Save LoRA weights
pipe.save_lora_weights("oykh-character-lora")
```

---

## Step 4: Use Trained LoRA

### With Replicate:
```javascript
const output = await replicate.run(
  "your-username/oykh-character-lora", // Your trained model
  {
    input: {
      prompt: "OYKHCHAR pointing at viewer, coffee mug, blue-purple gradient",
      // OYKHCHAR triggers your trained character
    }
  }
);
```

### With HuggingFace:
```javascript
// Load FLUX with your LoRA
const output = await hf.textToImage({
  model: 'black-forest-labs/FLUX.1-dev',
  inputs: "OYKHCHAR thinking pose, brain icon, Kurzgesagt style",
  parameters: {
    lora: "your-username/oykh-character-lora",
    lora_scale: 0.8, // How strongly to apply LoRA (0.6-1.0)
  }
});
```

---

## Expected Results

**Before LoRA:** 60-70% character consistency
**After LoRA:** 90-95% character consistency

**Training Time:** 30min - 2hrs
**Training Cost:** $5-10 (Replicate) or FREE (Colab)
**Long-term Value:** Perfect consistency forever, reusable for all future videos

---

## Next Steps

1. ✅ Set up Replicate API (done)
2. 🔄 Test Replicate FLUX-dev (immediate consistency boost)
3. 📸 Generate/curate 20-30 perfect character images
4. 🎓 Train LoRA using Replicate's trainer
5. 🚀 Use LoRA for all future video generation

Once you have the Replicate API token, we'll:
1. Test FLUX-dev first (should be 15-20% better than current)
2. If good enough, proceed with LoRA training for perfection
