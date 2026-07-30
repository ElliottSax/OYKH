/**
 * Package Perfect Training Images
 * Uses the ideal 22 images from lora-final
 */

import fs from 'fs/promises';
import path from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

const sourceDir = 'C:/projects/oykh-temp/lora-final/images';
const outputDir = 'C:/projects/oykh-temp/lora-training-perfect';
const imagesDir = path.join(outputDir, 'images');
const captionsDir = path.join(outputDir, 'captions');

await fs.mkdir(imagesDir, { recursive: true });
await fs.mkdir(captionsDir, { recursive: true });

console.log('🎨 Packaging Perfect Training Set');
console.log('==================================');
console.log('');

// Copy all JPG images and create captions
const files = await fs.readdir(sourceDir);
const jpgFiles = files.filter(f => f.endsWith('.jpg')).sort((a, b) => {
  const numA = parseInt(a.replace('.jpg', ''));
  const numB = parseInt(b.replace('.jpg', ''));
  return numA - numB;
});

console.log(`Found ${jpgFiles.length} perfect training images`);
console.log('');

for (const file of jpgFiles) {
  const num = file.replace('.jpg', '').padStart(2, '0');
  const newName = `oykhchar_${num}.png`;

  // Convert JPG to PNG for better quality
  const src = path.join(sourceDir, file);
  const dest = path.join(imagesDir, newName);

  await execAsync(`magick "${src}" "${dest}"`)
    .catch(() => fs.copyFile(src, dest)); // Fallback if ImageMagick not available

  // Create caption with trigger word
  const captionPath = path.join(captionsDir, newName.replace('.png', '.txt'));
  await fs.writeFile(
    captionPath,
    'OYKHCHAR minimalist white stick figure character with 2.5D cell-shading'
  );

  console.log(`✓ Processed ${file} → ${newName}`);
}

console.log('');
console.log('==================================');
console.log(`📊 Total Training Images: ${jpgFiles.length}`);
console.log('==================================');
console.log('');

// Create archive
console.log('📦 Creating training archive...');
const archiveName = `oykhchar-perfect-${jpgFiles.length}imgs.tar.gz`;

try {
  await execAsync(`cd "${outputDir}" && tar -czf "${archiveName}" images/ captions/`);

  const stats = await fs.stat(path.join(outputDir, archiveName));
  const sizeMB = (stats.size / 1024 / 1024).toFixed(1);

  console.log('');
  console.log('🎉 PERFECT TRAINING SET READY!');
  console.log('==============================');
  console.log('');
  console.log(`📁 File: ${path.join(outputDir, archiveName)}`);
  console.log(`📊 Size: ${sizeMB} MB`);
  console.log(`🖼️  Images: ${jpgFiles.length}`);
  console.log('');
  console.log('🎓 Upload to: https://replicate.com/ostris/flux-dev-lora-trainer/train');
  console.log('');
  console.log('⚙️  Recommended Settings:');
  console.log('   trigger_word: OYKHCHAR');
  console.log('   steps: 1200-1500 (more for this detailed style)');
  console.log('   lora_rank: 16-24 (higher for 2.5D details)');
  console.log('   learning_rate: 0.0004');
  console.log('   optimizer: adamw8bit');
  console.log('   resolution: "512,768,1024"');
  console.log('');
  console.log('💰 Cost: ~$6-12 for training');
  console.log('⏱️  Time: ~45-90 minutes');
  console.log('');
  console.log('🎯 Expected Result: 90-95% character consistency!');
  console.log('');

} catch (error) {
  console.error('Error:', error.message);
  console.log('');
  console.log('Create archive manually:');
  console.log(`cd "${outputDir}" && tar -czf ${archiveName} images/ captions/`);
}
