# Image Consistency Solutions for OYKH

You're absolutely right - AI-generated images lack the ruthless consistency needed for educational content. Here are the solutions, ranked by effectiveness:

---

## ⭐ RECOMMENDED: Template-Based SVG System

### Why This Is Best:

- **100% consistency** - Same character every time, pixel-perfect
- **Fast** - No AI generation = instant renders
- **Free** - No API costs per shot
- **Scalable** - Easy to add new poses/props
- **Professional** - Kurzgesagt/TED-Ed quality control

### How It Works:

**Step 1: Create Core Assets (One-time setup)**

```
assets/
  characters/
    pose-pointing.svg
    pose-explaining.svg
    pose-thinking.svg
    pose-holding.svg
    pose-excited.svg
    ... (10-15 total)

  props/
    coffee-mug.svg
    brain-icon.svg
    clock.svg
    lightbulb.svg
    book.svg
    ... (20-30 total)

  backgrounds/
    gradient-blue-purple.svg
    gradient-orange-yellow.svg
    ... (5-10 variations)
```

**Step 2: Modified Pipeline**

```javascript
// Instead of AI image generation:
async function generateShot(shot) {
  // 1. Select character pose
  const character = loadSVG(`assets/characters/${shot.pose}.svg`);

  // 2. Add props
  const props = shot.props.map((p) => loadSVG(`assets/props/${p}.svg`));

  // 3. Background gradient
  const bg = createGradient(shot.bgColors || ['#4A148C', '#7B1FA2']);

  // 4. Composite with Sharp
  return await compositeImage({
    background: bg,
    layers: [character, ...props],
    width: 1920,
    height: 1080,
  });
}
```

**Step 3: Enhanced Script Generation**

```javascript
// Script now includes pose/prop references:
{
  "shotNumber": 1,
  "duration": 5,
  "characterPose": "pointing",  // ← SVG filename
  "props": ["coffee-mug"],      // ← SVG filenames
  "propPositions": {
    "coffee-mug": { x: 1400, y: 300, scale: 1.2 }
  },
  "backgroundGradient": ["#4A148C", "#7B1FA2"],
  "narration": "Ever wonder what coffee does to your brain?"
}
```

---

## Option 2: Strict Prompt Engineering (Quick Fix)

### Enhanced Prompt Template:

```javascript
const STYLE_LOCK = `
STRICT STYLE REQUIREMENTS (DO NOT DEVIATE):
- Character: EXACTLY like Kurzgesagt YouTube style
- Head: Perfect geometric circle, diameter 180px, white fill #FFFFFF
- Eyes: Two black dots, 12px diameter, 45px apart horizontally
- Body: Straight vertical line, 10px black stroke
- Arms/Legs: Simple lines, 10px black stroke, no curves
- Outline: EXACTLY 10px black stroke on ALL elements
- Props: Same vector style, 10px black outline, flat single colors
- Background: Linear gradient ONLY, no textures
- NO: Shadows on character, textures, 3D effects, realistic details

EXACT COLOR CODES:
- Character fill: #FFFFFF
- All outlines: #000000
- Background: Linear gradient #4A148C to #7B1FA2

REFERENCE STYLE: Kurzgesagt, TED-Ed, CGP Grey educational videos
`;

// Prepend to every prompt
const fullPrompt = `${STYLE_LOCK}\n\n${shot.imagenPrompt}`;
```

**Pros:** Can implement in 30 minutes
**Cons:** Still relies on AI, ~70-80% consistency

---

## Option 3: Hybrid Approach (AI + Templates)

Use templates for character, AI for backgrounds/atmosphere:

```javascript
async function generateHybridShot(shot) {
  // 1. Template character (100% consistent)
  const character = loadSVG(`assets/characters/${shot.pose}.svg`);

  // 2. AI-generated background atmosphere (variety)
  const bgAtmosphere = await generateAI({
    prompt: `Abstract ${shot.mood} background, blue-purple gradient,
             minimalist, suitable for educational content`,
    model: 'flux-schnell',
  });

  // 3. Composite
  return composite(bgAtmosphere, character, shot.props);
}
```

**Pros:** Character consistency + creative backgrounds
**Cons:** More complex pipeline

---

## Option 4: ControlNet Reference Image

Use first shot as reference for all subsequent shots:

```javascript
// Generate master reference once
const masterReference = await generateAI({
  prompt: STRICT_CHARACTER_PROMPT,
  model: 'sdxl',
});

// All subsequent shots use it as reference
for (const shot of shots) {
  const image = await generateWithControlNet({
    prompt: shot.prompt,
    referenceImage: masterReference,
    controlnetType: 'canny', // or 'openpose'
    strength: 0.8,
  });
}
```

**Pros:** Good consistency with variety
**Cons:** Requires ControlNet API access (Replicate, Stability AI)

---

## My Recommendation

### Immediate (Today):

**Implement Template-Based System:**

1. Create 10 basic character pose SVGs (2 hours)
2. Create 15 common prop SVGs (1 hour)
3. Update pipeline to use templates instead of AI (2 hours)
4. Test with coffee video

This gives you Kurzgesagt-level consistency immediately.

### Future Enhancements:

1. Add simple animations (SVG transforms)
2. Create pose generator (programmatically generate variations)
3. Add AI for special complex scenes only

---

## Quick SVG Character Example

```svg
<svg viewBox="0 0 1920 1080" xmlns="http://www.w3.org/2000/svg">
  <!-- Background Gradient -->
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#4A148C"/>
      <stop offset="100%" stop-color="#7B1FA2"/>
    </linearGradient>
  </defs>
  <rect width="1920" height="1080" fill="url(#bg)"/>

  <!-- Character (centered) -->
  <g transform="translate(960, 540)">
    <!-- Head -->
    <circle cx="0" cy="-120" r="90" fill="white" stroke="black" stroke-width="10"/>
    <!-- Eyes -->
    <circle cx="-25" cy="-120" r="10" fill="black"/>
    <circle cx="25" cy="-120" r="10" fill="black"/>
    <!-- Body -->
    <line x1="0" y1="-30" x2="0" y2="120" stroke="black" stroke-width="10"/>
    <!-- Arms - pointing right -->
    <line x1="0" y1="20" x2="180" y2="-40" stroke="black" stroke-width="10"/>
    <line x1="0" y1="20" x2="-60" y2="60" stroke="black" stroke-width="10"/>
    <!-- Legs -->
    <line x1="0" y1="120" x2="-60" y2="240" stroke="black" stroke-width="10"/>
    <line x1="0" y1="120" x2="60" y2="240" stroke="black" stroke-width="10"/>
  </g>
</svg>
```

Want me to build the template-based system? I can have it running in a few hours.
