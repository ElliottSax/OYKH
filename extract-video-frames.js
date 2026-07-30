/**
 * Extract Training Images from Best Video
 * Extracts 1 frame per second from consistent video
 */

import { exec } from 'child_process';
import { promisify } from 'util';
import fs from 'fs/promises';
import path from 'path';

const execAsync = promisify(exec);

// Your best video (most consistent)
const videoPath = 'C:/projects/oykh-temp/output/Unlock_the_REAL_Reason_You_Procrastinate__1771630418342.mp4';

// Output to lora-final directory to combine with generated images
const outputDir = 'C:/projects/oykh-temp/lora-final/images-from-video';
await fs.mkdir(outputDir, { recursive: true });

console.log('🎬 Extracting frames from best video...');
console.log(`📹 Video: ${videoPath}`);
console.log(`📁 Output: ${outputDir}`);
console.log('');

// Extract 1 frame per second (video is ~2 minutes = ~120 frames)
console.log('Extracting frames (1 per second)...');
try {
  await execAsync(
    `ffmpeg -i "${videoPath}" -vf "fps=1" "${outputDir}/frame_%03d.png" -y`,
    { maxBuffer: 50 * 1024 * 1024 }
  );

  const frames = await fs.readdir(outputDir);
  const pngFrames = frames.filter(f => f.endsWith('.png'));

  console.log(`✅ Extracted ${pngFrames.length} frames`);
  console.log('');
  console.log('📊 Next steps:');
  console.log('1. Review frames in:', outputDir);
  console.log('2. Delete frames with:');
  console.log('   - Inconsistent character (different head size, etc.)');
  console.log('   - Blurry or low quality');
  console.log('   - Text overlays');
  console.log('   - Scene transitions');
  console.log('3. Keep 15-20 best frames with consistent character');
  console.log('4. Copy good frames to: C:/projects/oykh-temp/lora-final/images/');
  console.log('   Rename as: img_30_extracted_01.png, img_31_extracted_02.png, etc.');
  console.log('');

  // Create captions for extracted frames
  const captionsDir = 'C:/projects/oykh-temp/lora-final/captions-from-video';
  await fs.mkdir(captionsDir, { recursive: true });

  for (const frame of pngFrames) {
    const baseName = path.basename(frame, '.png');
    const captionPath = path.join(captionsDir, `${baseName}.txt`);
    await fs.writeFile(captionPath, 'OYKHCHAR educational stick figure character');
  }

  console.log(`📝 Created ${pngFrames.length} caption files`);
  console.log('');

} catch (error) {
  console.error('Error extracting frames:', error.message);
  console.log('');
  console.log('Manual extraction:');
  console.log(`ffmpeg -i "${videoPath}" -vf "fps=1" "${outputDir}/frame_%03d.png"`);
}
