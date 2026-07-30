# LoRA Training Instructions - OYKH Character

## ✅ Training Data Ready!

**Location:** `C:\projects\oykh-temp\lora-training\oykh-training-data.tar.gz` (19MB)
**Images:** 29 high-quality character variations
**Cost:** ~$5-10 for training
**Time:** ~30-60 minutes

---

## 🚀 Start Training (Web Interface - EASIEST)

### Step 1: Go to Replicate LoRA Trainer
https://replicate.com/ostris/flux-dev-lora-trainer/train

### Step 2: Upload Training Data
1. Click "Choose file" under **input_images**
2. Select: `C:\projects\oykh-temp\lora-training\oykh-training-data.tar.gz`
3. Wait for upload to complete (~30 seconds)

### Step 3: Configure Training Parameters

**Required Settings:**
- **trigger_word:** `OYKHCHAR`
- **steps:** `1000`
- **lora_rank:** `16`

**Optional (Recommended):**
- **autocaption:** `false` (we have manual captions)
- **optimizer:** `adamw8bit`
- **learning_rate:** `0.0004`
- **resolution:** `512,768,1024`
- **batch_size:** `1`

### Step 4: Start Training
1. Click "Create training"
2. Bookmark the training page URL
3. Wait ~30-60 minutes

### Step 5: Monitor Progress
- Training status updates automatically
- Check logs for progress
- You'll get an email when complete

---

## 🎯 After Training Completes

### Step 1: Get Your Model ID
Training will output a model like:
```
username/oykh-character-v1
```

### Step 2: Update server-simple.js

Replace the Replicate model call with your LoRA:

```javascript
const output = await replicate.run(
  "YOUR-USERNAME/oykh-character-v1",  // Your trained LoRA
  {
    input: {
      prompt: "OYKHCHAR pointing at viewer, coffee mug, blue-purple gradient",
      // Always include OYKHCHAR to trigger your character
      aspect_ratio: "16:9",
      num_inference_steps: 28,
      guidance_scale: 3.5,
      output_format: "png",
      output_quality: 90,
    }
  }
);
```

### Step 3: Update Prompts

Add `OYKHCHAR` at the beginning of every prompt:

**Before:**
```
White stick figure with thick black vector outline, round head, dot eyes, pointing gesture
```

**After:**
```
OYKHCHAR pointing gesture, coffee mug, blue-purple gradient
```

The LoRA already knows what OYKHCHAR looks like, so prompts can be simpler!

---

## 🎬 Expected Results

**Before LoRA:** 60-70% character consistency
**After LoRA:** 90-95% character consistency

**Props:** Will also match training style
**Background:** Consistent gradient
**Quality:** Professional, sprite-level consistency

---

## 💡 Tips

1. **Always use "OYKHCHAR"** in prompts to activate your trained character
2. **Shorter prompts work better** - LoRA knows the character details
3. **First few generations** may vary slightly while model "warms up"
4. **Can retrain** if results aren't perfect (adjust training steps)
5. **LoRA is reusable** - use for all future videos

---

## 🔧 Troubleshooting

**If training fails:**
- Check file upload completed
- Verify trigger_word is set
- Try reducing steps to 500

**If results aren't consistent enough:**
- Increase steps to 1500-2000
- Increase lora_rank to 32
- Review training images for inconsistencies

**If character looks wrong:**
- Check OYKHCHAR is in prompt
- Verify using correct model ID
- Try increasing guidance_scale

---

## 📊 Cost Breakdown

- **Training:** ~$5-10 (one-time)
- **Generation after training:** ~$0.05-0.10 per image
- **Total video (24 shots):** ~$1.20-2.40

**Long-term value:** Perfect consistency forever!

---

## ⏭️ Next Steps After Training

1. ✅ Get trained model ID from Replicate
2. 🔧 Update server-simple.js with model ID
3. 📝 Update prompt template to include OYKHCHAR
4. 🎬 Generate test video
5. 🎉 Enjoy 90%+ character consistency!
