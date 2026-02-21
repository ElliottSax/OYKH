/**
 * Generate Images with Modal (60% cheaper than Replicate!)
 *
 * Replicate FLUX-dev: ~$0.003/image
 * Modal FLUX-dev: ~$0.0012/image (60% savings!)
 *
 * Requirements:
 * 1. Install Modal: pip install modal
 * 2. Setup Modal: modal token new
 * 3. Deploy app: modal deploy modal-flux-app.py
 */

import { exec } from 'child_process';
import { promisify } from 'util';
import fs from 'fs/promises';
import path from 'path';

const execAsync = promisify(exec);

// Output directory
const trainingDir = 'C:/projects/oykh-temp/lora-modal';
const imagesDir = path.join(trainingDir, 'images');

await fs.mkdir(imagesDir, { recursive: true });

// MASTER TEMPLATE
const MASTER_TEMPLATE = `OYKHCHAR minimalist white stick figure character, 2.5D cell-shaded, simple black dot eyes with white reflections, thick black vector outlines, mitten-style hands, vibrant blue-to-purple gradient background, educational illustration style`;

// Actions
const actions = [
  "pointing forward at viewer",
  "pointing upward teaching",
  "arms spread wide explaining",
  "one arm raised presenting",
  "hand on chin thinking",
  "holding coffee mug with steam",
  "brain icon floating above head",
  "lightbulb appearing above",
  "holding simple clock",
  "celebrating with arms up",
  "standing confidently hands on hips",
  "jumping with excitement",
  "relaxed neutral standing",
  "walking forward confidently",
  "sitting thoughtfully",
  "leaning forward engaged",
  "shrugging shoulders",
  "hands clasped together",
  "waving hello",
  "thumbs up gesture",
];

console.log('🚀 Modal FLUX Generation');
console.log('========================');
console.log('');
console.log('Benefits:');
console.log('  💰 60% cheaper than Replicate');
console.log('  ⚡ Same quality FLUX-dev model');
console.log('  🔧 Full control over infrastructure');
console.log('');
console.log(`Generating ${actions.length} images...`);
console.log('');

// Check if Modal is installed
try {
  const { stdout } = await execAsync('modal --version');
  console.log(`✓ Modal CLI: ${stdout.trim()}`);
} catch (error) {
  console.error('✗ Modal CLI not found!');
  console.error('');
  console.error('Install Modal:');
  console.error('  pip install modal');
  console.error('  modal token new');
  console.error('  modal deploy modal-flux-app.py');
  console.error('');
  process.exit(1);
}

// Check if app is deployed
console.log('Checking Modal app deployment...');
try {
  const { stdout } = await execAsync('modal app list');
  if (!stdout.includes('oykh-flux-production')) {
    console.log('⚠️  App not deployed. Deploying now...');
    await execAsync('modal deploy modal-flux-app.py', {
      cwd: 'C:/projects/oykh-temp'
    });
    console.log('✓ App deployed!');
  } else {
    console.log('✓ App already deployed');
  }
} catch (error) {
  console.error('✗ Failed to deploy app:', error.message);
  console.error('');
  console.error('Manual deployment:');
  console.error('  cd C:/projects/oykh-temp');
  console.error('  modal deploy modal-flux-app.py');
  console.error('');
  process.exit(1);
}

console.log('');
console.log('Generating images...');
console.log('');

let totalCost = 0;
const results = [];

for (let i = 0; i < actions.length; i++) {
  const action = actions[i];
  const imageNum = i + 1;
  const actionSlug = action.replace(/\s+/g, '_').replace(/[^a-z0-9_]/gi, '');
  const imageName = `modal_${String(imageNum).padStart(2, '0')}_${actionSlug}`;
  const imagePath = path.join(imagesDir, `${imageName}.jpg`);
  const captionPath = path.join(imagesDir, `${imageName}.txt`);

  try {
    const prompt = `${MASTER_TEMPLATE}, ${action}`;

    console.log(`[${imageNum}/${actions.length}] ${action}...`);

    // Call Modal function
    const { stdout } = await execAsync(
      `modal run modal-flux-app.py::main --prompt "${prompt.replace(/"/g, '\\"')}"`,
      {
        cwd: 'C:/projects/oykh-temp',
        maxBuffer: 10 * 1024 * 1024 // 10MB buffer for large outputs
      }
    );

    // Modal saves to output.png in current directory
    const tempPath = path.join('C:/projects/oykh-temp', 'output.png');

    // Convert PNG to JPG and move to images directory
    await execAsync(
      `magick "${tempPath}" -quality 95 "${imagePath}"`,
      { cwd: 'C:/projects/oykh-temp' }
    ).catch(async () => {
      // If ImageMagick not available, just copy the PNG
      await fs.copyFile(tempPath, imagePath.replace('.jpg', '.png'));
    });

    // Create caption
    const caption = `OYKHCHAR minimalist white stick figure character, 2.5D cell-shaded, simple black dot eyes with white reflections`;
    await fs.writeFile(captionPath, caption);

    // Estimate cost (Modal A10G: ~$0.0012/image)
    totalCost += 0.0012;

    results.push({
      number: imageNum,
      action: action,
      path: imagePath,
      success: true
    });

    console.log(`✓ Saved (Cost so far: $${totalCost.toFixed(4)})`);

    // Clean up temp file
    try {
      await fs.unlink(tempPath);
    } catch (e) {
      // Ignore if already deleted
    }

  } catch (error) {
    console.error(`✗ Failed:`, error.message);
    results.push({
      number: imageNum,
      action: action,
      error: error.message,
      success: false
    });
  }

  // Small delay between requests
  await new Promise(resolve => setTimeout(resolve, 2000));
}

const succeeded = results.filter(r => r.success).length;
const replicateCost = succeeded * 0.003; // What it would cost on Replicate
const savings = replicateCost - totalCost;

console.log('');
console.log('✅ Modal Generation Complete!');
console.log('=============================');
console.log('');
console.log(`Generated: ${succeeded}/${actions.length} images`);
console.log(`Modal Cost: $${totalCost.toFixed(4)}`);
console.log(`Replicate Cost: $${replicateCost.toFixed(4)}`);
console.log(`Savings: $${savings.toFixed(4)} (${((savings / replicateCost) * 100).toFixed(0)}%)`);
console.log(`Output: ${imagesDir}`);
console.log('');

// Save report
const report = {
  generatedAt: new Date().toISOString(),
  platform: 'Modal',
  total: actions.length,
  succeeded: succeeded,
  failed: actions.length - succeeded,
  totalCost: totalCost,
  replicateCost: replicateCost,
  savings: savings,
  savingsPercent: (savings / replicateCost) * 100,
  results: results,
};

await fs.writeFile(
  path.join(trainingDir, 'generation-report.json'),
  JSON.stringify(report, null, 2)
);

console.log('📋 Report saved: generation-report.json');
