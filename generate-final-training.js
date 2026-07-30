/**
 * Generate Final Curated Training Set - 25 Images
 * New clean directory, ultra-consistent prompts
 */

import Replicate from 'replicate';
import fs from 'fs/promises';
import path from 'path';
import 'dotenv/config';

const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });

// NEW CLEAN DIRECTORY
const trainingDir = 'C:/projects/oykh-temp/lora-final';
const imagesDir = path.join(trainingDir, 'images');
const captionsDir = path.join(trainingDir, 'captions');

await fs.mkdir(imagesDir, { recursive: true });
await fs.mkdir(captionsDir, { recursive: true });

// ULTRA-CONSISTENT BASE
const BASE = "White stick figure character, perfectly round head, two simple black dot eyes, thick black outline";
const BG = "blue-purple gradient background";
const STYLE = "minimalist educational illustration, Kurzgesagt style";

// 25 CAREFULLY SELECTED POSES - Most important for videos
const poses = [
  // Core gestures (8)
  "pointing at viewer",
  "pointing upward",
  "arms spread wide explaining",
  "one arm raised teaching",
  "hand on chin thinking",
  "hand touching temple contemplating",
  "arms crossed pondering",
  "hands on hips confident",

  // With props - Coffee (4)
  "holding coffee mug",
  "holding coffee mug with steam",
  "drinking from coffee mug",
  "coffee mug in hand thoughtful",

  // With props - Brain/Lightbulb (4)
  "brain icon floating above head",
  "pointing at brain icon",
  "lightbulb above head",
  "holding lightbulb excited",

  // With props - Clock (2)
  "holding clock",
  "pointing at clock urgent",

  // Emotions/States (5)
  "excited jumping celebration",
  "relaxed standing calm",
  "shrugging shoulders uncertain",
  "celebrating arms raised",
  "focused concentration pose",

  // Basic poses (2)
  "standing straight neutral",
  "walking forward",
];

console.log(`🎨 Generating ${poses.length} final training images`);
console.log(`📁 Clean directory: ${imagesDir}`);
console.log('');
console.log('Settings:');
console.log('  - Ultra-consistent prompts');
console.log('  - 40 inference steps (maximum quality)');
console.log('  - Guidance 5.0 (strict adherence)');
console.log('  - 100% output quality');
console.log('');

let completed = 0;

// Generate in batches of 3
const batchSize = 3;
for (let i = 0; i < poses.length; i += batchSize) {
  const batch = poses.slice(i, i + batchSize);

  await Promise.all(
    batch.map(async (pose, batchIndex) => {
      const imageNum = i + batchIndex + 1;
      const poseSlug = pose.replace(/\s+/g, '_').replace(/[^a-z0-9_]/gi, '');
      const imageName = `img_${String(imageNum).padStart(2, '0')}_${poseSlug}`;
      const imagePath = path.join(imagesDir, `${imageName}.png`);
      const captionPath = path.join(captionsDir, `${imageName}.txt`);

      try {
        const prompt = `${BASE}, ${pose}, ${BG}, ${STYLE}`;

        console.log(`   [${imageNum}/${poses.length}] ${pose}...`);

        const output = await replicate.run(
          "black-forest-labs/flux-dev",
          {
            input: {
              prompt: prompt,
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

        const caption = `OYKHCHAR ${pose}`;
        await fs.writeFile(captionPath, caption);

        completed++;
        console.log(`   ✓ [${completed}/${poses.length}] Saved`);

      } catch (error) {
        console.error(`   ✗ Failed:`, error.message);
      }
    })
  );

  if (i + batchSize < poses.length) {
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
}

console.log('');
console.log(`✅ Complete: ${completed}/${poses.length} images`);
console.log('');
console.log('📦 Auto-packaging...');

// Auto-package the training data
try {
  const { exec } = await import('child_process');
  const { promisify } = await import('util');
  const execAsync = promisify(exec);

  const archiveName = `oykh-lora-training-${completed}imgs.tar.gz`;
  const archivePath = path.join(trainingDir, archiveName);

  await execAsync(`cd "${trainingDir}" && tar -czf "${archiveName}" images/ captions/`);

  const stats = await fs.stat(archivePath);
  const sizeMB = (stats.size / 1024 / 1024).toFixed(1);

  console.log(`✅ Archive created: ${archiveName}`);
  console.log(`📊 Size: ${sizeMB} MB`);
  console.log(`📁 Path: ${archivePath}`);
  console.log('');
  console.log('🎓 Ready for LoRA Training!');
  console.log('');
  console.log('Next Steps:');
  console.log('1. Go to: https://replicate.com/ostris/flux-dev-lora-trainer/train');
  console.log('2. Upload:', archivePath);
  console.log('3. Settings:');
  console.log('   - trigger_word: OYKHCHAR');
  console.log('   - steps: 1000');
  console.log('   - lora_rank: 16');
  console.log('4. Start training!');
  console.log('');

} catch (err) {
  console.log('Package manually with:');
  console.log(`cd "${trainingDir}" && tar -czf oykh-training.tar.gz images/ captions/`);
}
