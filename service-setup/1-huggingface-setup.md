# Hugging Face Setup Guide

## Step 1: Create Account
1. Go to https://huggingface.co/join
2. Sign up (free)
3. Verify email

## Step 2: Get API Token
1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: "OYKH Production"
4. Type: "Write" (to upload models)
5. Copy token

## Step 3: Login CLI
```bash
huggingface-cli login
# Paste token when prompted
```

## Step 4: Test Access
```bash
huggingface-cli whoami
```

## Step 5: Upload Trained LoRA (After Kaggle training completes)
```bash
cd C:/projects/oykh-temp/lora-output

# Create model card
cat > README.md << EOF
---
license: other
tags:
- flux
- lora
- character
---

# OYKHCHAR LoRA

Custom character LoRA trained on FLUX.1-dev for consistent viral video generation.

**Trigger word:** OYKHCHAR

**Training:**
- 26 images
- 50 epochs
- Trained on Kaggle (FREE!)

**Usage:**
\`\`\`python
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
pipe.load_lora_weights("your-username/oykhchar-lora")

image = pipe("OYKHCHAR character celebrating", num_inference_steps=28).images[0]
\`\`\`
EOF

# Upload
huggingface-cli upload your-username/oykhchar-lora . --repo-type model
```

## Step 6: Use FREE Inference
```javascript
// server-simple.js integration

async function generateWithHuggingFace(prompt) {
  const response = await fetch(
    'https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell',
    {
      headers: {
        Authorization: `Bearer ${process.env.HF_TOKEN}`,
        'Content-Type': 'application/json'
      },
      method: 'POST',
      body: JSON.stringify({
        inputs: prompt,
        parameters: {
          num_inference_steps: 4, // schnell is fast!
          guidance_scale: 0
        }
      })
    }
  );

  const blob = await response.blob();
  return blob;
}

// Cost: $0 (FREE tier: 1000 requests/day)
// vs Replicate: $0.05 per image
// Savings: $50/day if generating 1000 images!
```

## Models to Try (All FREE)

### Image Generation
- `black-forest-labs/FLUX.1-schnell` - Fastest (4 steps)
- `stabilityai/sdxl-turbo` - 1-step generation!
- `stabilityai/stable-diffusion-xl-base-1.0` - High quality

### Text (for RepurposeAI)
- `facebook/bart-large-cnn` - Summarization
- `google/flan-t5-xxl` - Text transformation
- `meta-llama/Meta-Llama-3-8B-Instruct` - Chat/generation

### Computer Vision (for quality scoring)
- `openai/clip-vit-large-patch14` - Image understanding
- `facebook/detr-resnet-50` - Object detection

## Pro Tier ($9/month) - Worth It?

**Free Tier:**
- 1000 requests/day per model
- Shared GPU (slower)
- Public models only

**Pro Tier:**
- Unlimited requests
- Faster inference
- Private model hosting
- Priority support

**Recommendation:** Start free, upgrade if you hit limits

## Cost Comparison

**Generating 10,000 images/month:**
- Hugging Face Free: $0 (within limits)
- Hugging Face Pro: $9
- Replicate: $500
- **Savings: $491/month**
