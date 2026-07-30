# Advanced Consistency Techniques

## Current Status
✅ Coffee-style prompts = BEST results so far
🎯 Goal: Push consistency even further

---

## Technique 1: Consistent Seed (Easiest)

**What:** Use the same random seed for all image generations
**Why:** Forces AI to use same "starting point" for creativity
**Expected:** More consistent character proportions and style

### Implementation:
```javascript
const CONSISTENT_SEED = 42; // Same seed for all shots

const imageBlob = await hf.textToImage({
  model: 'black-forest-labs/FLUX.1-schnell',
  inputs: shot.imagenPrompt,
  parameters: {
    width: 1024,
    height: 576,
    num_inference_steps: 4,
    seed: CONSISTENT_SEED, // ← Key change
  }
});
```

**Pros:** Simple, free, might work immediately
**Cons:** May reduce variety in poses/props

---

## Technique 2: Increased Inference Steps

**What:** Increase from 4 steps to 6-8 steps
**Why:** More refinement = more accurate prompt following
**Expected:** Better adherence to character specs

### Implementation:
```javascript
num_inference_steps: 6, // Was 4, now 6 (or try 8)
```

**Pros:** Better quality, still fast enough
**Cons:** Slightly slower (~50% more time)

---

## Technique 3: Ultra-Minimal Prompts

**What:** Make prompts even SHORTER (25-30 words)
**Why:** Less information = less room for variation
**Expected:** AI focuses on core elements only

### Example:
**Current (40-50 words):**
```
White stick figure with thick black vector outline, perfectly round head,
two simple black dot eyes, pointing at viewer with inviting gesture,
coffee mug with steam lines. Blue-purple gradient background with
sophisticated lighting. Educational vector illustration, Kurzgesagt style.
```

**Ultra-Minimal (25-30 words):**
```
White stick figure, round head, dot eyes, pointing at viewer,
coffee mug. Blue-purple gradient. Kurzgesagt educational style.
```

---

## Technique 4: Character Reference Embedding

**What:** Add strong character consistency phrase at START of every prompt
**Why:** AI prioritizes beginning of prompts more
**Expected:** Character becomes "locked in" early in generation

### Implementation:
```javascript
const CHARACTER_ANCHOR = "EXACT same white stick figure character: ";
const enhancedPrompt = CHARACTER_ANCHOR + shot.imagenPrompt;
```

**Example:**
```
EXACT same white stick figure character: White stick figure with
thick black vector outline, perfectly round head...
```

---

## Technique 5: Two-Pass Generation

**What:** Generate character base first, then add props/poses
**Why:** Separates character consistency from scene elements
**Expected:** Character locked, only props vary

### Implementation:
```javascript
// Pass 1: Generate base character
const basePrompt = "White stick figure, round head, dot eyes,
                    blue-purple gradient, Kurzgesagt style";
const base = await generateImage(basePrompt, seed: 42);

// Pass 2: Add pose/props as variation
const finalPrompt = basePrompt + ", " + shot.action + ", " + shot.props;
const final = await generateImage(finalPrompt, seed: 42 + shot.number);
```

**Pros:** Maximum character consistency
**Cons:** 2x generation time, more complex

---

## Technique 6: Model Temperature/Guidance

**What:** Adjust guidance_scale if available
**Why:** Higher guidance = stricter prompt following
**Expected:** Less creative variation from prompts

### Implementation:
```javascript
parameters: {
  width: 1024,
  height: 576,
  num_inference_steps: 6,
  guidance_scale: 7.5, // Higher = stricter (if supported by FLUX)
}
```

---

## Technique 7: Prompt Structure Lock

**What:** NEVER vary the structure, only swap key words
**Why:** AI learns pattern and follows it precisely
**Expected:** Very consistent base, controlled variations

### Template:
```
[CHARACTER_BASE] + [ACTION] + [PROP] + [BACKGROUND] + [STYLE]
```

**Every prompt EXACTLY:**
```
White stick figure, round head, dot eyes,
[ACTION: pointing at viewer | thinking | explaining],
[PROP: coffee mug | brain icon | clock | none],
blue-purple gradient,
Kurzgesagt style.
```

---

## Recommended Test Order

1. **Start with Technique 1 (Seed)** - Easiest, might solve it immediately
2. **Add Technique 2 (More steps)** - Small effort, likely improvement
3. **Try Technique 3 (Shorter)** - Test if simpler is better
4. **Combine 1+2+3** - Best of all three
5. **If still not enough, try Technique 4 (Anchor)** - More aggressive

---

## Test Plan

### Quick Test (10 minutes):
1. Add consistent seed to current system
2. Increase inference steps to 6
3. Generate 1 test video
4. Compare to current best

### Medium Test (30 minutes):
1. Implement all 3 quick techniques
2. Create ultra-minimal prompt version
3. Generate 2 test videos (current vs ultra-minimal)
4. Compare all results

### Full Test (1 hour):
1. Implement all techniques
2. Generate 5 videos with different combinations
3. Scientific comparison of consistency
4. Document winning combination

---

## Expected Results

**Seed alone:** +10-20% consistency
**Seed + More steps:** +20-30% consistency
**Seed + Steps + Shorter:** +30-40% consistency
**All techniques:** +50%+ consistency (approaching sprite-level)

At some point, diminishing returns kick in and only sprite system will give 100%.
