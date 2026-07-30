# Deploy LoRA on Banana.dev

## Overview

Banana.dev is a serverless GPU platform (cheaper than Replicate for high volume).

**Pros:**
- Pay per second (not per request)
- Better pricing at scale
- Full control over model configuration

**Cons:**
- More complex setup
- Need to manage Docker containers
- Longer cold starts

---

## Setup Steps

### 1. Download Your LoRA Weights

From Replicate training page:
- Download `oykhchar-v1-lora.safetensors` file
- This is your trained LoRA (typically 20-100 MB)

---

### 2. Create Banana Deployment

**Option A: Use Existing FLUX Template**

Banana has FLUX templates - you add your LoRA to it:

```python
# app.py (Banana deployment)
import torch
from diffusers import FluxPipeline
from safetensors.torch import load_file

# Load base FLUX model
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16
)

# Load your LoRA weights
lora_weights = load_file("oykhchar-v1-lora.safetensors")
pipe.load_lora_weights(lora_weights)
pipe.fuse_lora()  # Merge for faster inference

def inference(model_inputs):
    prompt = model_inputs.get('prompt', '')

    # Add trigger word
    if not prompt.startswith('OYKHCHAR'):
        prompt = f"OYKHCHAR {prompt}"

    image = pipe(
        prompt=prompt,
        width=1024,
        height=576,
        num_inference_steps=28,
        guidance_scale=3.5,
    ).images[0]

    return {"image": image}
```

---

### 3. Create requirements.txt

```
torch>=2.0.0
diffusers>=0.25.0
transformers>=4.30.0
accelerate>=0.20.0
safetensors>=0.3.0
```

---

### 4. Create Dockerfile

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
COPY oykhchar-v1-lora.safetensors .

# Download base model during build (faster cold starts)
RUN python -c "from diffusers import FluxPipeline; FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-dev')"

CMD ["python", "app.py"]
```

---

### 5. Deploy to Banana

```bash
# Install Banana CLI
npm install -g banana-cli

# Login
banana login

# Deploy
banana deploy
```

---

## Cost Comparison

### Replicate:
- $0.05-0.10 per image (28 steps)
- Easy to use, instant
- Good for low-medium volume

### Banana:
- ~$0.01-0.03 per image (same quality)
- 50-70% cheaper at scale
- Better for high volume (100+ images/day)

**Breakeven:** ~50 images/day

---

## Alternative: ComfyUI + Your LoRA

Even cheaper - run locally!

1. Install ComfyUI
2. Download FLUX.1-dev model
3. Add your LoRA to ComfyUI/models/loras/
4. Use in workflow

**Cost:** $0 (just electricity)
**Speed:** Depends on your GPU

---

## Recommendation

**Start with Replicate:**
- Training: Keep LoRA on Replicate
- Production videos: Use Replicate (easy)

**Scale to Banana when:**
- Generating 50+ videos per day
- Want 50%+ cost savings
- Have dev resources for setup

**Or use ComfyUI for:**
- Local generation
- Offline work
- Maximum control

---

## Integration with OYKH Server

### For Banana:

```javascript
// server-simple.js with Banana
import axios from 'axios';

const BANANA_API_KEY = process.env.BANANA_API_KEY;
const BANANA_MODEL_KEY = process.env.BANANA_MODEL_KEY;

async function generateImageWithBanana(prompt) {
  const response = await axios.post(
    'https://api.banana.dev/start/v4/',
    {
      apiKey: BANANA_API_KEY,
      modelKey: BANANA_MODEL_KEY,
      modelInputs: {
        prompt: `OYKHCHAR ${prompt}`,
        width: 1024,
        height: 576,
        num_inference_steps: 28,
        guidance_scale: 3.5,
      }
    }
  );

  return response.data.modelOutputs[0].image;
}
```

---

## Next Steps

1. **Test with Replicate first** (easiest)
2. **Measure usage** (how many videos/day?)
3. **Calculate costs** (Replicate vs Banana)
4. **Switch if volume justifies it**

Your LoRA works everywhere - you're not locked in! 🎯
