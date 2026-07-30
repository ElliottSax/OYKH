/**
 * Package Curated Training Data for LoRA
 * Run AFTER manually reviewing and deleting inconsistent images
 */

import fs from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

const trainingDir = 'C:/projects/oykh-temp/lora-training-v2';
const imagesDir = path.join(trainingDir, 'images');
const captionsDir = path.join(trainingDir, 'captions');

console.log('📦 Packaging Curated Training Data');
console.log('=====================================');
console.log('');

// Count remaining images
const imageFiles = (await fs.readdir(imagesDir)).filter(f => f.endsWith('.png'));
const captionFiles = (await fs.readdir(captionsDir)).filter(f => f.endsWith('.txt'));

console.log(`📸 Images found: ${imageFiles.length}`);
console.log(`📝 Captions found: ${captionFiles.length}`);
console.log('');

if (imageFiles.length < 15) {
  console.log('⚠️  Warning: Less than 15 images may result in poor LoRA quality');
  console.log('   Recommended: 20-30 images for best results');
  console.log('');
}

if (imageFiles.length > 35) {
  console.log('💡 Tip: More than 35 images may slow training');
  console.log('   Consider keeping only the 25 most consistent');
  console.log('');
}

// Clean up orphaned captions (captions without matching images)
let orphanedCaptions = 0;
for (const captionFile of captionFiles) {
  const imageName = captionFile.replace('.txt', '.png');
  if (!imageFiles.includes(imageName)) {
    await fs.unlink(path.join(captionsDir, captionFile));
    orphanedCaptions++;
  }
}

if (orphanedCaptions > 0) {
  console.log(`🧹 Cleaned up ${orphanedCaptions} orphaned captions`);
  console.log('');
}

// Create archive
console.log('📦 Creating training archive...');
const archiveName = `oykh-training-curated-${imageFiles.length}imgs.tar.gz`;
const archivePath = path.join(trainingDir, archiveName);

try {
  await execAsync(`cd "${trainingDir}" && tar -czf "${archiveName}" images/ captions/`);
  const stats = await fs.stat(archivePath);
  const sizeMB = (stats.size / 1024 / 1024).toFixed(1);

  console.log(`✅ Archive created: ${archiveName}`);
  console.log(`📊 Size: ${sizeMB} MB`);
  console.log(`📁 Location: ${archivePath}`);
  console.log('');
  console.log('🎓 Ready for LoRA Training!');
  console.log('');
  console.log('Next Steps:');
  console.log('1. Go to: https://replicate.com/ostris/flux-dev-lora-trainer/train');
  console.log('2. Upload file:', archivePath);
  console.log('3. Settings:');
  console.log('   - trigger_word: OYKHCHAR');
  console.log('   - steps: 1000');
  console.log('   - lora_rank: 16');
  console.log('4. Start training (~30-60 min, ~$5-10)');
  console.log('');

  // Save training manifest
  const manifest = {
    images: imageFiles.length,
    archive: archiveName,
    archivePath: archivePath,
    sizeBytes: stats.size,
    sizeMB: sizeMB,
    created: new Date().toISOString(),
    settings: {
      trigger_word: 'OYKHCHAR',
      recommended_steps: 1000,
      recommended_lora_rank: 16,
      recommended_learning_rate: 0.0004
    }
  };

  await fs.writeFile(
    path.join(trainingDir, 'training-manifest.json'),
    JSON.stringify(manifest, null, 2)
  );

  console.log('📋 Training manifest saved');

} catch (error) {
  console.error('Error creating archive:', error.message);
  console.log('');
  console.log('Manual packaging:');
  console.log(`cd "${trainingDir}"`);
  console.log(`tar -czf ${archiveName} images/ captions/`);
}
