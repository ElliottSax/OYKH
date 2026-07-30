/**
 * Generate ULTRA-CONSISTENT Training Images for LoRA
 * Strategy: Minimal prompt variation, maximum consistency
 */

import Replicate from 'replicate';
import fs from 'fs/promises';
import path from 'path';
import 'dotenv/config';

const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });

// Create training data directory
const trainingDir = 'C:/projects/oykh-temp/lora-training-v2';
const imagesDir = path.join(trainingDir, 'images');
const captionsDir = path.join(trainingDir, 'captions');

await fs.mkdir(imagesDir, { recursive: true });
await fs.mkdir(captionsDir, { recursive: true });

// HYPER-CONSISTENT BASE - Use exact same structure every time
const CHARACTER_BASE = "White stick figure character, perfectly round head, two simple black dot eyes, thick black outline";
const BACKGROUND = "blue-purple gradient background";
const STYLE = "minimalist educational illustration, Kurzgesagt style";

// Minimal variations - only change the action, keep everything else identical
const actions = [
  // Core poses (repeat with slight variations for consistency check)
  "pointing forward",
  "pointing at viewer",
  "pointing upward",
  "arms spread wide",
  "one arm raised",

  // Thinking poses
  "hand on chin thinking",
  "hand touching temple",
  "arms crossed",
  "looking upward pondering",
  "hand raised to head",

  // With coffee mug prop
  "holding coffee mug",
  "holding coffee mug up",
  "holding coffee mug with steam",
  "coffee mug in hand",
  "drinking coffee",

  // With brain icon prop
  "brain icon floating above",
  "brain icon next to head",
  "pointing at brain icon",
  "brain icon overhead",

  // With lightbulb prop
  "lightbulb above head",
  "lightbulb appearing",
  "pointing at lightbulb",
  "holding lightbulb",

  // With clock prop
  "holding clock",
  "clock icon floating",
  "pointing at clock",

  // Simple gestures
  "hands on hips",
  "standing straight",
  "walking forward",
  "sitting down",
  "leaning forward",

  // Emotions (minimal variation)
  "excited jumping",
  "celebrating arms up",
  "relaxed standing",
  "shrugging shoulders",

  // Additional consistency tests (same pose, regenerate)
  "pointing forward confidently",
  "pointing at viewer inviting",
  "thinking pose contemplative",
  "holding coffee mug thoughtfully",
  "brain icon thought bubble",
  "lightbulb idea moment",
  "standing presentation pose",
  "explaining with gesture",
  "teaching pose pointing",
  "curious head tilt",
  "focused concentration",
  "friendly welcoming stance",
  "professional standing",
  "animated explaining",
  "calm neutral pose",
];

console.log(`🎨 Generating ${actions.length} ultra-consistent training images...`);
console.log(`📁 Output: ${imagesDir}`);
console.log('');
console.log('Strategy:');
console.log('  - Strict prompt template');
console.log('  - Maximum quality settings');
console.log('  - Generate 50 → Curate best 25');
console.log('');

let completed = 0;
const results = [];

// Generate in batches
const batchSize = 3;
for (let i = 0; i < actions.length; i += batchSize) {
  const batch = actions.slice(i, i + batchSize);

  await Promise.all(
    batch.map(async (action, batchIndex) => {
      const imageNum = i + batchIndex + 1;
      const actionSlug = action.replace(/\s+/g, '_').replace(/[^a-z0-9_]/gi, '');
      const imageName = `train_${String(imageNum).padStart(3, '0')}_${actionSlug}`;
      const imagePath = path.join(imagesDir, `${imageName}.png`);
      const captionPath = path.join(captionsDir, `${imageName}.txt`);

      try {
        // ULTRA-CONSISTENT PROMPT STRUCTURE
        const prompt = `${CHARACTER_BASE}, ${action}, ${BACKGROUND}, ${STYLE}`;

        console.log(`   [${imageNum}/${actions.length}] ${action}...`);

        const output = await replicate.run(
          "black-forest-labs/flux-dev",
          {
            input: {
              prompt: prompt,
              aspect_ratio: "16:9",
              num_inference_steps: 40, // HIGHER quality (vs 28)
              guidance_scale: 5.0, // STRICTER adherence (vs 3.5)
              output_format: "png",
              output_quality: 100, // MAX quality
            }
          }
        );

        // Download image
        const response = await fetch(output[0]);
        const arrayBuffer = await response.arrayBuffer();
        await fs.writeFile(imagePath, Buffer.from(arrayBuffer));

        // Create caption - simple and consistent
        const caption = `OYKHCHAR ${action}`;
        await fs.writeFile(captionPath, caption);

        completed++;
        results.push({
          number: imageNum,
          action: action,
          path: imagePath,
          prompt: prompt
        });

        console.log(`   ✓ [${completed}/${actions.length}] Saved`);

      } catch (error) {
        console.error(`   ✗ Failed ${action}:`, error.message);
        results.push({
          number: imageNum,
          action: action,
          error: error.message
        });
      }
    })
  );

  // Small delay between batches
  if (i + batchSize < actions.length) {
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
}

console.log('');
console.log(`✅ Generated ${completed}/${actions.length} images`);
console.log('');
console.log('📊 Next Steps:');
console.log('');
console.log('1. Review images in: ' + imagesDir);
console.log('2. Delete any inconsistent images (different head size, wrong style, etc.)');
console.log('3. Keep 20-25 most consistent images');
console.log('4. Run: node package-curated-training.js');
console.log('');
console.log('Look for consistency in:');
console.log('  ✓ Round head size (should be similar across all)');
console.log('  ✓ Eye position (same distance apart)');
console.log('  ✓ Line thickness (same outline width)');
console.log('  ✓ Background gradient (same blue-purple)');
console.log('  ✓ Overall character proportions');
console.log('');

// Save generation report
const report = {
  total: actions.length,
  succeeded: completed,
  failed: actions.length - completed,
  images: results.filter(r => !r.error),
  errors: results.filter(r => r.error)
};

await fs.writeFile(
  path.join(trainingDir, 'generation-report.json'),
  JSON.stringify(report, null, 2)
);

console.log('📋 Generation report saved: generation-report.json');
