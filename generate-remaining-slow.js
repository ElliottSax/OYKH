/**
 * Generate Remaining Training Images - SLOW MODE
 * Respects rate limit: 6 requests/minute
 */

import Replicate from 'replicate';
import fs from 'fs/promises';
import path from 'path';
import 'dotenv/config';

const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });

const trainingDir = 'C:/projects/oykh-temp/lora-final';
const imagesDir = path.join(trainingDir, 'images');
const captionsDir = path.join(trainingDir, 'captions');

const BASE = "White stick figure character, perfectly round head, two simple black dot eyes, thick black outline";
const BG = "blue-purple gradient background";
const STYLE = "minimalist educational illustration, Kurzgesagt style";

// Check what we already have
const existing = await fs.readdir(imagesDir);
const existingCount = existing.filter(f => f.endsWith('.png')).length;

console.log(`📊 Current progress: ${existingCount} images`);
console.log(`🎯 Target: 25 images`);
console.log(`📝 Need: ${25 - existingCount} more images`);
console.log('');

// Remaining poses to generate
const remainingPoses = [
  "hand on chin thinking",
  "hand touching temple contemplating",
  "arms crossed pondering",
  "hands on hips confident",
  "holding coffee mug with steam",
  "drinking from coffee mug",
  "coffee mug in hand thoughtful",
  "brain icon floating above head",
  "pointing at brain icon",
  "holding lightbulb excited",
  "holding clock",
  "pointing at clock urgent",
  "excited jumping celebration",
  "relaxed standing calm",
  "celebrating arms raised",
  "focused concentration pose",
  "standing straight neutral",
  "walking forward",
];

console.log('⏱️  SLOW MODE: 10 seconds between requests');
console.log(`⏳ Estimated time: ~${Math.ceil(remainingPoses.length / 6)} minutes`);
console.log('');

let completed = existingCount;
let startNum = existingCount + 1;

// Generate ONE AT A TIME with 10 second delays
for (let i = 0; i < remainingPoses.length; i++) {
  const pose = remainingPoses[i];
  const imageNum = startNum + i;
  const poseSlug = pose.replace(/\s+/g, '_').replace(/[^a-z0-9_]/gi, '');
  const imageName = `img_${String(imageNum).padStart(2, '0')}_${poseSlug}`;
  const imagePath = path.join(imagesDir, `${imageName}.png`);
  const captionPath = path.join(captionsDir, `${imageName}.txt`);

  try {
    const prompt = `${BASE}, ${pose}, ${BG}, ${STYLE}`;

    console.log(`[${imageNum}/25] ${pose}...`);

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
    console.log(`✓ [${completed}/25] Saved - ${remainingPoses.length - i - 1} remaining`);

  } catch (error) {
    if (error.message.includes('429')) {
      console.log(`⚠️  Rate limit hit - waiting 15 seconds...`);
      await new Promise(resolve => setTimeout(resolve, 15000));
      i--; // Retry this one
      continue;
    }
    console.error(`✗ Failed:`, error.message);
  }

  // Wait 10 seconds between requests (6 per minute = safe)
  if (i < remainingPoses.length - 1) {
    console.log(`⏳ Waiting 10 seconds...`);
    await new Promise(resolve => setTimeout(resolve, 10000));
  }
}

console.log('');
console.log(`✅ Generation complete: ${completed}/25 images`);
console.log('');
console.log('Packaging...');

// Auto-package
try {
  const { exec } = await import('child_process');
  const { promisify } = await import('util');
  const execAsync = promisify(exec);

  const archiveName = `oykh-lora-training-${completed}imgs.tar.gz`;
  await execAsync(`cd "${trainingDir}" && tar -czf "${archiveName}" images/ captions/`);

  console.log(`✅ Archive: ${trainingDir}/${archiveName}`);
} catch (err) {
  console.log('Package with: cd lora-final && tar -czf training.tar.gz images/ captions/');
}
