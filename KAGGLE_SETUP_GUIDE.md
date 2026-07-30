# Kaggle LoRA Training Setup Guide

## ✅ What Was Fixed

**Previous Issues:**
- ❌ No Hugging Face authentication (FLUX.1-dev requires login)
- ❌ Memory inefficient (would crash on free tier)
- ❌ Manual training loop had bugs
- ❌ No proper checkpointing

**New Fixes:**
- ✅ Uses Kaggle Secrets for HF authentication
- ✅ Memory-optimized with quantization & gradient checkpointing
- ✅ Uses proven ai-toolkit training framework
- ✅ Automatic checkpointing every 250 steps
- ✅ Generates sample images during training
- ✅ Better error messages

---

## 🔑 Step 1: Get Hugging Face Token

1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: `Kaggle LoRA Training`
4. Type: **Read** (not write)
5. Click "Generate token"
6. **Copy the token** (you'll need it in next step)

---

## 🔐 Step 2: Add Token to Kaggle Secrets

1. Go to: https://www.kaggle.com/settings
2. Click "Add-ons" tab in the left sidebar
3. Scroll to "Secrets" section
4. Click "Add a new secret"
5. **Label:** `HF_TOKEN`
6. **Value:** Paste your Hugging Face token from Step 1
7. Click "Add secret"

✅ Your token is now securely stored!

---

## 🚀 Step 3: Start Training

The notebook has been pushed to Kaggle. To access it:

### Option A: Via Kaggle Web Interface

1. Go to: https://www.kaggle.com/code
2. Find "OYKHCHAR LoRA Training V3 - Fixed"
3. Click to open
4. Click "Run All" or run cells one by one
5. Monitor progress (45-60 minutes)

### Option B: Via Command Line

```bash
# View the notebook
kaggle kernels status elliottsax/oykhchar-lora-training-v3

# Pull and run locally (if you want to edit first)
kaggle kernels pull elliottsax/oykhchar-lora-training-v3
```

---

## 📊 What to Expect

**Training Timeline:**

| Time | Status | What's Happening |
|------|--------|------------------|
| 0-5 min | Setup | Installing ai-toolkit, downloading FLUX base model |
| 5-10 min | Prep | Loading dataset, creating config, initializing training |
| 10-55 min | Training | 1000 steps with checkpoints every 250 steps |
| 55-60 min | Finalize | Packaging LoRA, generating final test images |

**Training will:**
- Save checkpoints every 250 steps
- Generate sample images at steps 250, 500, 750, 1000
- Show loss metrics
- Create final `.safetensors` file

---

## 📥 Step 4: Download Your LoRA

After training completes:

1. Click "Output" tab in Kaggle notebook
2. Download `oykhchar_lora_final` folder
3. You'll get:
   - `oykhchar_lora.safetensors` (the trained LoRA)
   - Sample test images
   - Checkpoints

---

## 🧪 Step 5: Test Your LoRA

Use the trained LoRA with FLUX:

```python
from diffusers import FluxPipeline
import torch

# Load base model
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)

# Load your LoRA
pipe.load_lora_weights("./oykhchar_lora_final")

# Generate with trigger word
image = pipe(
    "OYKHCHAR character pointing forward with coffee mug",
    num_inference_steps=28,
    guidance_scale=3.5,
    width=1024,
    height=1024
).images[0]

image.save("test.png")
```

**Important:** Always include `OYKHCHAR` in your prompts to activate the LoRA!

---

## 🔧 Troubleshooting

### Error: "HF_TOKEN not found"
**Fix:** Complete Step 2 above to add token to Kaggle Secrets

### Error: "Out of memory"
**Fix:**
- Make sure GPU is enabled (Runtime → Change runtime type → GPU T4)
- Restart kernel and clear outputs
- The config uses quantization - should fit in 15GB

### Error: "FLUX model not found"
**Fix:** Make sure your HF token has read access and you've accepted FLUX license:
- Go to: https://huggingface.co/black-forest-labs/FLUX.1-dev
- Click "Agree and access repository"

### Training seems stuck
**Check:**
- Look for progress bars in output
- Download is ~20GB (first time only, takes 5-10 min)
- Training shows step count increasing
- If truly stuck >20 min, restart kernel

### Results not consistent enough
**Try:**
- Increase training steps to 1500-2000
- Increase LoRA rank to 32
- Train for more epochs
- Add more training images

---

## 💰 Cost

**Kaggle Free Tier:**
- 30 GPU hours per week
- This training uses ~1 hour
- Completely FREE! 🎉

---

## 📝 Notes

- Training runs in background - you can close browser
- Kaggle will save outputs automatically
- You can restart/resume if it crashes
- First run downloads FLUX (~20GB) - subsequent runs are faster
- Your HF token stays secret - never visible in notebook output

---

## ✅ Ready to Start?

1. ✅ HF token created
2. ✅ Token added to Kaggle secrets
3. ✅ Notebook pushed to Kaggle
4. ▶️ Ready to click "Run All"!

Good luck! 🚀
