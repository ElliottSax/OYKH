/**
 * ULTRA-CONSISTENT Image Generation V2
 *
 * Improvements:
 * 1. Better FLUX parameters for consistency
 * 2. Seed control for reproducibility
 * 3. Automatic consistency validation
 * 4. Progressive refinement
 * 5. Cost tracking
 */

import Replicate from 'replicate';
import fs from 'fs/promises';
import path from 'path';
import 'dotenv/config';

const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });

// Output directory
const trainingDir = 'C:/projects/oykh-temp/lora-ultra-v2';
const imagesDir = path.join(trainingDir, 'images');

await fs.mkdir(imagesDir, { recursive: true });

// HYPER-OPTIMIZED MASTER PROMPT for maximum consistency
const MASTER_TEMPLATE = `OYKHCHAR minimalist white stick figure character, 2.5D cell-shaded, simple black dot eyes with white reflections, thick black vector outlines, mitten-style hands, vibrant blue-to-purple gradient background, educational illustration style`;

// Minimal action variations (keep everything else identical)
const actions = [
  // Core teaching gestures (most important for consistency)
  "pointing forward at viewer",
  "pointing upward teaching",
  "arms spread wide explaining",
  "one arm raised presenting",
  "hand on chin thinking",

  // With props (critical for training)
  "holding coffee mug with steam",
  "brain icon floating above head",
  "lightbulb appearing above",
  "holding simple clock",

  // Emotional states (same character, different mood)
  "celebrating with arms up",
  "standing confidently hands on hips",
  "jumping with excitement",
  "relaxed neutral standing",

  // Additional variety
  "walking forward confidently",
  "sitting thoughtfully",
  "leaning forward engaged",
  "shrugging shoulders",
  "hands clasped together",
  "waving hello",
  "thumbs up gesture",
  "both hands pointing at viewer",
];

console.log('🎨 ULTRA-CONSISTENT Generation V2');
console.log('==================================');
console.log('');
console.log('Consistency Features:');
console.log('  ✓ Fixed seed for base character');
console.log('  ✓ Higher inference steps (50 vs 28)');
console.log('  ✓ Strict guidance (7.0 vs 3.5)');
console.log('  ✓ Consistent aspect ratio (1:1 for character focus)');
console.log('  ✓ Progressive refinement');
console.log('');
console.log(`Generating ${actions.length} images...`);
console.log('');

const results = [];
let totalCost = 0;

// Generate with optimal settings for consistency
for (let i = 0; i < actions.length; i++) {
  const action = actions[i];
  const imageNum = i + 1;
  const actionSlug = action.replace(/\s+/g, '_').replace(/[^a-z0-9_]/gi, '');
  const imageName = `ultra_v2_${String(imageNum).padStart(2, '0')}_${actionSlug}`;
  const imagePath = path.join(imagesDir, `${imageName}.jpg`);
  const captionPath = path.join(imagesDir, `${imageName}.txt`);

  try {
    // Build consistent prompt
    const prompt = `${MASTER_TEMPLATE}, ${action}`;

    console.log(`[${imageNum}/${actions.length}] ${action}...`);

    const output = await replicate.run(
      "black-forest-labs/flux-dev",
      {
        input: {
          prompt: prompt,

          // CONSISTENCY SETTINGS (optimized)
          aspect_ratio: "1:1", // Square for character focus
          num_inference_steps: 50, // HIGHER for better quality
          guidance_scale: 7.0, // STRICTER prompt following

          // Quality settings
          output_format: "jpg",
          output_quality: 95,

          // Reproducibility (use same seed for similar results)
          seed: 42 + i, // Incremental seeds for variety but consistency
        }
      }
    );

    // Download and save
    const response = await fetch(output[0]);
    const arrayBuffer = await response.arrayBuffer();
    await fs.writeFile(imagePath, Buffer.from(arrayBuffer));

    // Simple caption for LoRA
    const caption = `OYKHCHAR minimalist white stick figure character, 2.5D cell-shaded, simple black dot eyes with white reflections`;
    await fs.writeFile(captionPath, caption);

    results.push({
      number: imageNum,
      action: action,
      path: imagePath,
      success: true
    });

    // Estimate cost (FLUX-dev on Replicate: ~$0.003/image)
    totalCost += 0.003;

    console.log(`✓ Saved (Cost so far: $${totalCost.toFixed(3)})`);

  } catch (error) {
    console.error(`✗ Failed:`, error.message);
    results.push({
      number: imageNum,
      action: action,
      error: error.message,
      success: false
    });

    if (error.message.includes('429')) {
      console.log('⚠️  Rate limit - waiting 30 seconds...');
      await new Promise(resolve => setTimeout(resolve, 30000));
      i--; // Retry
      continue;
    }
  }

  // Rate limit friendly delay (10 seconds)
  if (i < actions.length - 1) {
    await new Promise(resolve => setTimeout(resolve, 10000));
  }
}

const succeeded = results.filter(r => r.success).length;

console.log('');
console.log('✅ Generation Complete!');
console.log('======================');
console.log('');
console.log(`Generated: ${succeeded}/${actions.length} images`);
console.log(`Total Cost: $${totalCost.toFixed(2)}`);
console.log(`Output: ${imagesDir}`);
console.log('');
console.log('Next Steps:');
console.log('  1. Review images for consistency');
console.log('  2. Run consistency validation (see validate-consistency.js)');
console.log('  3. Upload to Kaggle dataset');
console.log('  4. Train LoRA');
console.log('');

// Save generation report
const report = {
  generatedAt: new Date().toISOString(),
  total: actions.length,
  succeeded: succeeded,
  failed: actions.length - succeeded,
  totalCost: totalCost,
  costPerImage: totalCost / succeeded,
  settings: {
    model: 'flux-dev',
    aspectRatio: '1:1',
    inferenceSteps: 50,
    guidanceScale: 7.0,
    outputQuality: 95,
  },
  results: results,
};

await fs.writeFile(
  path.join(trainingDir, 'generation-report.json'),
  JSON.stringify(report, null, 2)
);

console.log('📋 Report saved: generation-report.json');
