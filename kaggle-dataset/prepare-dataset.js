/**
 * Prepare Kaggle Dataset for LoRA Training
 * Copies images and creates captions
 */

import fs from 'fs';
import path from 'path';

const sourceDir = 'C:/projects/oykh-temp/lora-final/images';
const destDir = 'C:/projects/oykh-temp/kaggle-dataset/images';

// Create images directory
if (!fs.existsSync(destDir)) {
  fs.mkdirSync(destDir, { recursive: true });
}

// Copy all .jpg files (except sheet.jpg)
const files = fs.readdirSync(sourceDir).filter(f => f.endsWith('.jpg') && f !== 'sheet.jpg');

console.log(`📦 Preparing ${files.length} training images...`);

files.forEach((file, index) => {
  const sourcePath = path.join(sourceDir, file);
  const newName = `oykhchar_${String(index + 1).padStart(2, '0')}.jpg`;
  const destPath = path.join(destDir, newName);

  // Copy image
  fs.copyFileSync(sourcePath, destPath);

  // Create caption file
  const captionPath = destPath.replace('.jpg', '.txt');
  const caption = 'OYKHCHAR, minimalist white stick figure character with round head, simple black dot eyes with white reflections, mitten-style hands, 2.5D cell-shaded illustration';
  fs.writeFileSync(captionPath, caption);

  console.log(`✅ ${newName} + caption`);
});

console.log('');
console.log('✅ Dataset ready!');
console.log(`📁 Location: ${destDir}`);
console.log(`📊 Total images: ${files.length}`);
console.log('');
console.log('Next: Upload to Kaggle');
console.log('  kaggle datasets create -p C:/projects/oykh-temp/kaggle-dataset');
