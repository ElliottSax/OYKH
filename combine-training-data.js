/**
 * Combine Generated Images + Curated Video Frames
 * Creates final training archive
 */

import fs from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

const finalDir = 'C:/projects/oykh-temp/lora-final';
const generatedDir = path.join(finalDir, 'images');
const videoFramesDir = path.join(finalDir, 'images-from-video');
const combinedDir = path.join(finalDir, 'training-combined');
const captionsDir = path.join(combinedDir, 'captions');

console.log('🔄 Combining Training Data');
console.log('==========================');
console.log('');

// Create combined directory
await fs.mkdir(combinedDir, { recursive: true });
await fs.mkdir(path.join(combinedDir, 'images'), { recursive: true });
await fs.mkdir(captionsDir, { recursive: true });

// Step 1: Copy generated images
console.log('📦 Step 1: Copying generated images...');
const generated = await fs.readdir(generatedDir);
const generatedPngs = generated.filter(f => f.endsWith('.png'));

for (const file of generatedPngs) {
  const src = path.join(generatedDir, file);
  const dest = path.join(combinedDir, 'images', file);
  await fs.copyFile(src, dest);

  // Copy/create caption
  const baseName = path.basename(file, '.png');
  const captionSrc = path.join(finalDir, 'captions', `${baseName}.txt`);
  const captionDest = path.join(captionsDir, `${baseName}.txt`);

  if (existsSync(captionSrc)) {
    await fs.copyFile(captionSrc, captionDest);
  } else {
    await fs.writeFile(captionDest, 'OYKHCHAR educational stick figure character');
  }
}

console.log(`✓ Copied ${generatedPngs.length} generated images`);
console.log('');

// Step 2: Check for curated video frames
console.log('📦 Step 2: Checking for curated video frames...');

if (existsSync(videoFramesDir)) {
  const videoFrames = await fs.readdir(videoFramesDir);
  const framePngs = videoFrames.filter(f => f.endsWith('.png'));

  console.log(`Found ${framePngs.length} video frames`);
  console.log('');

  if (framePngs.length > 50) {
    console.log('⚠️  You have more than 50 frames!');
    console.log('');
    console.log('Please curate first:');
    console.log(`1. Open: ${videoFramesDir}`);
    console.log('2. Delete inconsistent/bad frames');
    console.log('3. Keep only 15-20 BEST frames');
    console.log('4. Run this script again');
    console.log('');
    process.exit(0);
  }

  // Copy curated video frames
  let frameNum = generatedPngs.length + 1;
  for (const file of framePngs) {
    const newName = `img_${String(frameNum).padStart(2, '0')}_video_frame.png`;
    const src = path.join(videoFramesDir, file);
    const dest = path.join(combinedDir, 'images', newName);

    await fs.copyFile(src, dest);

    // Create caption
    const captionPath = path.join(captionsDir, `img_${String(frameNum).padStart(2, '0')}_video_frame.txt`);
    await fs.writeFile(captionPath, 'OYKHCHAR educational stick figure character');

    frameNum++;
  }

  console.log(`✓ Copied ${framePngs.length} curated video frames`);
} else {
  console.log('⚠️  No video frames found - using generated images only');
}

console.log('');

// Count total
const allImages = await fs.readdir(path.join(combinedDir, 'images'));
const totalImages = allImages.filter(f => f.endsWith('.png')).length;

console.log(`📊 Total Training Images: ${totalImages}`);
console.log('');

if (totalImages < 15) {
  console.log('⚠️  Warning: Less than 15 images - LoRA quality may be poor');
  console.log('   Recommendation: Add more video frames or generate more images');
  console.log('');
}

if (totalImages > 40) {
  console.log('💡 Tip: More than 40 images - consider keeping only the best 30');
  console.log('');
}

// Create archive
console.log('📦 Creating training archive...');
const archiveName = `oykh-lora-final-${totalImages}imgs.tar.gz`;
const archivePath = path.join(combinedDir, archiveName);

try {
  await execAsync(`cd "${combinedDir}" && tar -czf "${archiveName}" images/ captions/`);

  const stats = await fs.stat(archivePath);
  const sizeMB = (stats.size / 1024 / 1024).toFixed(1);

  console.log(`✅ Training archive created!`);
  console.log('');
  console.log(`📁 File: ${archivePath}`);
  console.log(`📊 Size: ${sizeMB} MB`);
  console.log(`🖼️  Images: ${totalImages}`);
  console.log('');
  console.log('🎓 Ready for LoRA Training!');
  console.log('');
  console.log('Upload to: https://replicate.com/ostris/flux-dev-lora-trainer/train');
  console.log('');
  console.log('Settings:');
  console.log('  trigger_word: OYKHCHAR');
  console.log('  steps: 1000');
  console.log('  lora_rank: 16');
  console.log('  learning_rate: 0.0004');
  console.log('');

} catch (error) {
  console.error('Error creating archive:', error.message);
  console.log('');
  console.log('Create manually:');
  console.log(`cd "${combinedDir}" && tar -czf ${archiveName} images/ captions/`);
}
