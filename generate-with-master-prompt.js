/**
 * Generate Training Images with MASTER PROMPT
 * Uses the specific 2.5D minimalist style template
 */

import Replicate from 'replicate';
import fs from 'fs/promises';
import path from 'path';
import 'dotenv/config';

const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });

// NEW DIRECTORY for master prompt images
const trainingDir = 'C:/projects/oykh-temp/lora-master';
const imagesDir = path.join(trainingDir, 'images');
const captionsDir = path.join(trainingDir, 'captions');

await fs.mkdir(imagesDir, { recursive: true });
await fs.mkdir(captionsDir, { recursive: true });

// CORE MASTER PROMPT - The foundation
const MASTER_PROMPT = `A minimalist white stick figure character with thick black vector outlines (3px stroke). The head is a perfect circle with two simple black oval eyes (no eyebrows or mouth unless specified). The body features thick, rounded limbs with no lines separating the hands from the arms or the feet from the legs. Hands are rendered as simple mitten-style blobs. The style is a 2.5D flat vector illustration with subtle cell-shading and soft reflections. Background is a vibrant blue-to-purple gradient.`;

// Actions to combine with master prompt
const actions = [
  // Core gestures
  "Character is pointing at the viewer with one arm extended forward",
  "Character is pointing upward with one arm raised",
  "Character has both arms spread wide in an explaining gesture",
  "Character has one arm raised in a teaching gesture",
  "Character has hand on chin in a thinking pose",
  "Character has hand touching temple in contemplative pose",
  "Character has arms crossed in pondering pose",
  "Character is standing with hands on hips confidently",

  // With coffee props
  "Character is holding a simple coffee mug with steam rising",
  "Character is holding a coffee mug up near head",
  "Character is drinking from a coffee mug",
  "Character has coffee mug in one hand looking thoughtful",

  // With brain/lightbulb props
  "Character with a simple brain icon floating above head",
  "Character is pointing at a brain icon beside them",
  "Character with a lightbulb appearing above head",
  "Character is holding a lightbulb excitedly",

  // With clock props
  "Character is holding a simple clock",
  "Character is pointing at a clock urgently",

  // Emotions/states
  "Character is jumping with excitement",
  "Character is standing relaxed and calm",
  "Character has both arms raised in celebration",
  "Character is in a focused concentration pose",
  "Character is shrugging shoulders uncertainly",

  // Basic poses
  "Character is standing straight in neutral pose",
  "Character is walking forward",
];

console.log(`🎨 Generating ${actions.length} images with MASTER PROMPT`);
console.log(`📁 Output: ${imagesDir}`);
console.log('');
console.log('Master Prompt Template:');
console.log('  ✓ 3px vector outlines');
console.log('  ✓ No lines separating hands/arms');
console.log('  ✓ Mitten-style hands');
console.log('  ✓ 2.5D flat vector with cell-shading');
console.log('  ✓ Blue-purple gradient background');
console.log('');
console.log('⏱️  Rate limit aware: 10 sec delays');
console.log(`⏳ ETA: ~${Math.ceil(actions.length / 6)} minutes`);
console.log('');

let completed = 0;

// Generate ONE AT A TIME with 10 second delays
for (let i = 0; i < actions.length; i++) {
  const action = actions[i];
  const imageNum = i + 1;
  const actionSlug = action.substring(14, 40).replace(/\s+/g, '_').replace(/[^a-z0-9_]/gi, '');
  const imageName = `master_${String(imageNum).padStart(2, '0')}_${actionSlug}`;
  const imagePath = path.join(imagesDir, `${imageName}.png`);
  const captionPath = path.join(captionsDir, `${imageName}.txt`);

  try {
    // Combine master prompt with specific action
    const fullPrompt = `${MASTER_PROMPT} ${action}`;

    console.log(`[${imageNum}/${actions.length}] ${action.substring(14, 50)}...`);

    const output = await replicate.run(
      "black-forest-labs/flux-dev",
      {
        input: {
          prompt: fullPrompt,
          aspect_ratio: "16:9",
          num_inference_steps: 40,
          guidance_scale: 5.0,
          output_format: "png",
          output_quality: 100,
        }
      }
    );

    const response = await fetch(output[0]);
    const arrayBuffer = await response.arrayBuffer();
    await fs.writeFile(imagePath, Buffer.from(arrayBuffer));

    // Caption for LoRA - simple trigger word
    const caption = `OYKHCHAR ${action}`;
    await fs.writeFile(captionPath, caption);

    completed++;
    console.log(`✓ [${completed}/${actions.length}] Saved`);

  } catch (error) {
    if (error.message.includes('429')) {
      console.log(`⚠️  Rate limit - waiting 15 seconds...`);
      await new Promise(resolve => setTimeout(resolve, 15000));
      i--; // Retry
      continue;
    }
    console.error(`✗ Failed:`, error.message);
  }

  // Wait 10 seconds between requests
  if (i < actions.length - 1) {
    await new Promise(resolve => setTimeout(resolve, 10000));
  }
}

console.log('');
console.log(`✅ Generated ${completed}/${actions.length} images with master prompt`);
console.log('');

// Auto-package
try {
  const { exec } = await import('child_process');
  const { promisify } = await import('util');
  const execAsync = promisify(exec);

  const archiveName = `oykh-master-${completed}imgs.tar.gz`;
  await execAsync(`cd "${trainingDir}" && tar -czf "${archiveName}" images/ captions/`);

  const stats = await fs.stat(path.join(trainingDir, archiveName));
  const sizeMB = (stats.size / 1024 / 1024).toFixed(1);

  console.log(`✅ Archive: ${trainingDir}/${archiveName}`);
  console.log(`📊 Size: ${sizeMB} MB`);
  console.log('');
  console.log('🎓 Ready for LoRA Training!');
  console.log('Upload to: https://replicate.com/ostris/flux-dev-lora-trainer/train');
  console.log('');
  console.log('Settings:');
  console.log('  trigger_word: OYKHCHAR');
  console.log('  steps: 1000-1500 (more for this detailed style)');
  console.log('  lora_rank: 16');
  console.log('  learning_rate: 0.0004');
  console.log('');
} catch (err) {
  console.log('Package with: cd lora-master && tar -czf training.tar.gz images/ captions/');
}
