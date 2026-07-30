# Video Consistency Analysis

## Test Videos Comparison

### Video 1: Coffee & Brain (First Test - Loose Prompts)

- **File:** `Coffee___Your_Brain__The_UNTOLD_Science_1771627258713.mp4`
- **Shots:** 22
- **Prompt Style:** Basic character description without strict specifications

### Video 2: Procrastination (Strict Prompts)

- **File:** `Unlock_the_REAL_Reason_You_Procrastinate__1771628099624.mp4`
- **Shots:** 24
- **Prompt Style:** Strict STYLE_LOCK with exact specifications

### Video 3: Procrastination v2 (Attempted img2img)

- **File:** `Unlock_the_REAL_Reason_You_Procrastinate__1771628582914.mp4`
- **Shots:** 24
- **Prompt Style:** Strict prompts + attempted img2img (fell back to text-to-image due to API limitations)

---

## Consistency Improvements with Strict Prompts

### What Changed:

**Before (Loose Prompts):**

```
"White stick figure with thick black vector outline teaching concept"
```

**After (Strict Prompts):**

```
Character is pure white stick figure with perfect geometric circle head (190px diameter),
two black dot eyes (12px, 45px apart), black 10px outline on head only.
Simple stick body with black lines (10px width) for torso, arms, legs.
Background is smooth linear gradient from deep purple (#4A148C) to #7B1FA2.
NO textures, NO shadows, NO complexity.
```

### Expected Improvements:

1. **Character Proportions** ✓
   - Head should be consistent size (190px)
   - Eye spacing should be uniform (45px apart)
   - Line weights should match (10px)

2. **Color Consistency** ✓
   - Background always #4A148C → #7B1FA2 gradient
   - Props use exact hex codes (#E91E63 pink, #FFEB3B yellow)

3. **Style Consistency** ✓
   - "Kurzgesagt style" reference enforced
   - Negative prompts prevent unwanted variations
   - Vector illustration style locked in

4. **Prop Consistency** ✓
   - Coffee mug: Always white with 3 steam lines
   - Brain: Always pink with 5 segments
   - Clock: Always white face with simple hands

---

## Img2img Workflow Status

### What Happened:

The img2img approach fell back to text-to-image because:

- SDXL refiner model not available on free HuggingFace tier
- Error: "No Inference Provider available for model stabilityai/stable-diffusion-xl-refiner-1.0"

### Solutions:

**Option 1: Use Different Model for img2img**

- Switch to `stabilityai/stable-diffusion-2-1` (available on free tier)
- Or use FLUX with controlnet approach
- Or use Replicate API (paid but better img2img support)

**Option 2: Stick with Strict Prompts Only**

- Current strict prompts already provide significant improvement
- No img2img needed if consistency is acceptable with strict prompts alone
- Simpler implementation, faster generation

**Option 3: Use Stability AI Official API**

- Use official Stability AI API key
- Better img2img support
- Costs ~$0.02-0.04 per image

---

## Current Status

**Working:** Strict prompt system with exact specifications ✓
**Blocked:** Img2img workflow (needs model change or paid API) ⚠️
**Fallback:** All shots generated with text-to-image using strict prompts ✓

**Recommendation:**

1. Review Video 2 (procrastination with strict prompts)
2. If consistency is acceptable → stick with strict prompts
3. If more consistency needed → implement Option 1 or 3 above

---

## How to Review Videos:

Open both videos and check:

- Do characters look the same across shots?
- Do repeated props (coffee mug) look identical?
- Is the background gradient consistent?
- Any unwanted style variations (textures, shadows, etc.)?

Report findings and we can further refine the prompts or switch to img2img with a working model.
