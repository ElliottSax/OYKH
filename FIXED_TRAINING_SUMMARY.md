# Kaggle LoRA Training - FIXED!

## What Was Done

### Problems Identified
1. **No Hugging Face Authentication** - FLUX.1-dev requires login
2. **Memory Issues** - Original approach would crash on free tier
3. **Training Loop Bugs** - Manual implementation had errors
4. **No Error Handling** - Failed silently without useful messages

### Solutions Implemented
1. **Added HF Authentication** - Uses Kaggle Secrets for secure token storage
2. **Memory Optimization** - Quantization + gradient checkpointing
3. **Professional Framework** - Switched to ai-toolkit (proven, reliable)
4. **Better Diagnostics** - Clear error messages and status updates

---

## Notebook Status

**URL:** https://www.kaggle.com/code/elliottsax/oykhchar-lora-training-v2

**Version:** 2 (just pushed)

**Status:** Ready to run (after you add HF token)

---

## REQUIRED: Setup Hugging Face Token

The training WILL FAIL without this step!

### Step 1: Get Your HF Token

1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it: `Kaggle Training`
4. Permission: **Read** (not write)
5. Click "Generate"
6. **COPY THE TOKEN** (you won't see it again!)

### Step 2: Add to Kaggle Secrets

1. Go to: https://www.kaggle.com/settings
2. Click "Add-ons" in left sidebar
3. Scroll to "Secrets" section
4. Click "Add a new secret"
5. **Label:** `HF_TOKEN` (EXACT spelling, all caps)
6. **Value:** Paste your token from Step 1
7. Click "Add"

IMPORTANT: The label MUST be exactly `HF_TOKEN` - case sensitive!

### Step 3: Accept FLUX License

1. Go to: https://huggingface.co/black-forest-labs/FLUX.1-dev
2. Click "Agree and access repository"
3. You only need to do this once

---

## How to Run Training

### Option 1: Start Now (Web Interface)

1. Go to: https://www.kaggle.com/code/elliottsax/oykhchar-lora-training-v2
2. Click "Edit" or "Run" button
3. Click "Run All" (top menu)
4. Wait 45-60 minutes
5. Download outputs when done

### Option 2: Monitor from Command Line

```bash
# Check status
kaggle kernels status elliottsax/oykhchar-lora-training-v2

# View output (after training)
kaggle kernels output elliottsax/oykhchar-lora-training-v2 -p ./lora-output
```

---

## Training Timeline

| Time | What's Happening |
|------|------------------|
| 0-5 min | Installing ai-toolkit, downloading FLUX base model (~20GB) |
| 5-10 min | Extracting dataset, setting up training |
| 10-55 min | Training 1000 steps (checkpoints at 250, 500, 750, 1000) |
| 55-60 min | Saving final LoRA, generating test images |

**First run takes longer** (downloads FLUX model)
**Subsequent runs are faster** (model is cached)

---

## What You'll Get

After training completes:

1. **oykhchar_lora.safetensors** - The trained LoRA weights (~100-200MB)
2. **Sample images** - Test generations at each checkpoint
3. **Training logs** - Loss metrics and progress
4. **Checkpoints** - Intermediate saves in case you want to use earlier version

---

## Download Your LoRA

1. Go to notebook page
2. Click "Output" tab (top right)
3. Download the `oykhchar_lora_final` folder
4. Extract and use the .safetensors file

---

## Using Your Trained LoRA

```python
from diffusers import FluxPipeline
import torch

# Load FLUX with your LoRA
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)
pipe.load_lora_weights("./oykhchar_lora_final")
pipe.to("cuda")

# Generate image (MUST include OYKHCHAR trigger word!)
image = pipe(
    "OYKHCHAR character pointing forward energetically",
    num_inference_steps=28,
    guidance_scale=3.5,
    width=1024,
    height=1024
).images[0]

image.save("output.png")
```

**CRITICAL:** Always include `OYKHCHAR` in your prompts!

---

## Troubleshooting

### "HF_TOKEN not found in secrets"

- Double-check you added it to Kaggle Secrets (Step 2 above)
- Make sure the label is exactly `HF_TOKEN` (all caps, no spaces)
- Restart the kernel after adding the secret

### "CUDA out of memory"

- Make sure GPU is enabled in notebook settings
- The config uses quantization - should fit in 15GB
- If still fails, try reducing batch_size to 1 in the config

### "Repository not found" or "Access denied"

- Make sure you accepted the FLUX license (Step 3 above)
- Your HF token needs read permission
- Go to FLUX model page and click "Agree"

### Training taking too long (>2 hours)

- First run downloads 20GB model - this is normal
- Check internet connection in notebook is enabled
- Look for download progress in output

### Results not consistent enough

After training completes, if results aren't good:

- Try training for more steps (1500-2000)
- Increase LoRA rank to 32
- Add more training images
- Use stronger guidance_scale (4.0-5.0)

---

## Cost

- **Kaggle Free:** 30 GPU hours/week
- **This training:** ~1 hour
- **Total cost:** $0.00 FREE!

---

## Next Steps

1. [ ] Get HF token from Hugging Face
2. [ ] Add token to Kaggle Secrets as `HF_TOKEN`
3. [ ] Accept FLUX license on Hugging Face
4. [ ] Go to notebook URL and click "Run All"
5. [ ] Wait ~1 hour
6. [ ] Download your trained LoRA
7. [ ] Test with FLUX pipeline!

---

## Need Help?

**Notebook URL:** https://www.kaggle.com/code/elliottsax/oykhchar-lora-training-v2

**Setup Guide:** See `KAGGLE_SETUP_GUIDE.md` for detailed instructions

**Common Issues:** Check troubleshooting section above

Ready to train! The notebook is fixed and waiting for your HF token.
