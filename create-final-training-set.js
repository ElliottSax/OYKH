/**
 * Create Final Premium Training Set
 * Combines: 6 master + best from previous + curated video frames
 */

import fs from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

const outputDir = 'C:/projects/oykh-temp/lora-final-premium';
const imagesDir = path.join(outputDir, 'images');
const captionsDir = path.join(outputDir, 'captions');

await fs.mkdir(imagesDir, { recursive: true });
await fs.mkdir(captionsDir, { recursive: true });

console.log('🎨 Creating Premium Training Set');
console.log('=================================');
console.log('');

let imageCount = 0;

// Step 1: Copy ALL 6 master prompt images (these are gold!)
console.log('Step 1: Copying 6 MASTER PROMPT images (highest quality)...');
const masterDir = 'C:/projects/oykh-temp/lora-master/images';
const masterFiles = await fs.readdir(masterDir);

for (const file of masterFiles.filter(f => f.endsWith('.png'))) {
  imageCount++;
  const newName = `premium_${String(imageCount).padStart(2, '0')}_master.png`;
  await fs.copyFile(
    path.join(masterDir, file),
    path.join(imagesDir, newName)
  );
  await fs.writeFile(
    path.join(captionsDir, newName.replace('.png', '.txt')),
    'OYKHCHAR minimalist 2.5D vector character'
  );
}
console.log(`✓ Copied ${imageCount} master images`);
console.log('');

// Step 2: Copy BEST 12 from previous batch (variety)
console.log('Step 2: Selecting best 12 from previous 23 images...');
const prevDir = 'C:/projects/oykh-temp/lora-final/images';
const prevFiles = await fs.readdir(prevDir);
const prevPngs = prevFiles.filter(f => f.endsWith('.png'));

// Select diverse poses (every other image for variety)
const selectedPrev = prevPngs.filter((_, i) => i % 2 === 0).slice(0, 12);

for (const file of selectedPrev) {
  imageCount++;
  const newName = `premium_${String(imageCount).padStart(2, '0')}_prev.png`;
  await fs.copyFile(
    path.join(prevDir, file),
    path.join(imagesDir, newName)
  );
  await fs.writeFile(
    path.join(captionsDir, newName.replace('.png', '.txt')),
    'OYKHCHAR stick figure character'
  );
}
console.log(`✓ Selected ${selectedPrev.length} previous images`);
console.log('');

// Step 3: Add curated video frames (if available)
console.log('Step 3: Checking for curated video frames...');
const videoDir = 'C:/projects/oykh-temp/lora-final/images-from-video';

if (existsSync(videoDir)) {
  const videoFiles = await fs.readdir(videoDir);
  const videoPngs = videoFiles.filter(f => f.endsWith('.png'));

  if (videoPngs.length <= 20) {
    // Already curated - add all
    for (const file of videoPngs) {
      imageCount++;
      const newName = `premium_${String(imageCount).padStart(2, '0')}_video.png`;
      await fs.copyFile(
        path.join(videoDir, file),
        path.join(imagesDir, newName)
      );
      await fs.writeFile(
        path.join(captionsDir, newName.replace('.png', '.txt')),
        'OYKHCHAR character from video'
      );
    }
    console.log(`✓ Added ${videoPngs.length} curated video frames`);
  } else {
    console.log(`⚠️  Found ${videoPngs.length} video frames - please curate first`);
    console.log('   Keep only 10-15 best frames, then run this script again');
  }
} else {
  console.log('ℹ️  No video frames - using generated images only');
}

console.log('');
console.log('=================================');
console.log(`📊 Total Training Images: ${imageCount}`);
console.log('=================================');
console.log('');

if (imageCount < 15) {
  console.log('⚠️  Warning: Less than 15 images');
  console.log('   Add 5-10 curated video frames for better results');
  console.log('');
} else if (imageCount >= 20 && imageCount <= 35) {
  console.log('✅ Perfect range for LoRA training!');
  console.log('');
}

// Create archive
console.log('📦 Creating training archive...');
const archiveName = `oykh-premium-${imageCount}imgs.tar.gz`;

try {
  await execAsync(`cd "${outputDir}" && tar -czf "${archiveName}" images/ captions/`);

  const stats = await fs.stat(path.join(outputDir, archiveName));
  const sizeMB = (stats.size / 1024 / 1024).toFixed(1);

  console.log('');
  console.log('🎉 TRAINING ARCHIVE READY!');
  console.log('==========================');
  console.log('');
  console.log(`📁 File: ${path.join(outputDir, archiveName)}`);
  console.log(`📊 Size: ${sizeMB} MB`);
  console.log(`🖼️  Images: ${imageCount}`);
  console.log('');
  console.log('🎓 Upload to: https://replicate.com/ostris/flux-dev-lora-trainer/train');
  console.log('');
  console.log('⚙️  Recommended Settings:');
  console.log('   trigger_word: OYKHCHAR');
  console.log('   steps: 1000-1500');
  console.log('   lora_rank: 16');
  console.log('   learning_rate: 0.0004');
  console.log('   optimizer: adamw8bit');
  console.log('');
  console.log('💰 Cost: ~$5-10 for training');
  console.log('⏱️  Time: ~30-60 minutes');
  console.log('');

} catch (error) {
  console.error('Error:', error.message);
}
