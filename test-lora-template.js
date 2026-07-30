/**
 * Test Your Trained LoRA
 * Replace YOUR-MODEL-URL with your actual Replicate model
 */

import Replicate from 'replicate';
import fs from 'fs/promises';
import 'dotenv/config';

const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });

// ⚠️ REPLACE THIS with your actual model URL from Replicate
const YOUR_LORA_MODEL = "your-username/oykhchar-v1";  // ← UPDATE THIS!

const testPrompts = [
  "OYKHCHAR pointing at viewer with inviting gesture",
  "OYKHCHAR holding coffee mug with steam rising",
  "OYKHCHAR with brain icon floating above head",
  "OYKHCHAR arms spread wide in explaining gesture",
  "OYKHCHAR hand on chin in thinking pose",
  "OYKHCHAR celebrating with arms raised",
];

console.log('🧪 Testing Trained LoRA');
console.log('======================');
console.log('');
console.log('Model:', YOUR_LORA_MODEL);
console.log('Tests:', testPrompts.length);
console.log('');

if (YOUR_LORA_MODEL.includes('your-username')) {
  console.log('⚠️  ERROR: Please update YOUR_LORA_MODEL with your actual model URL!');
  console.log('');
  console.log('After training completes, Replicate will give you a URL like:');
  console.log('  username/oykhchar-v1');
  console.log('');
  console.log('Replace "your-username/oykhchar-v1" with that URL in this file.');
  process.exit(1);
}

const outputDir = 'C:/projects/oykh-temp/lora-tests';
await fs.mkdir(outputDir, { recursive: true });

for (let i = 0; i < testPrompts.length; i++) {
  const prompt = testPrompts[i];
  console.log(`Test ${i + 1}/${testPrompts.length}: ${prompt.substring(0, 50)}...`);

  try {
    const output = await replicate.run(YOUR_LORA_MODEL, {
      input: {
        prompt: prompt,
        aspect_ratio: "16:9",
        num_inference_steps: 28,
        guidance_scale: 3.5,
        output_format: "png",
        output_quality: 90,
      }
    });

    const response = await fetch(output[0]);
    const buffer = Buffer.from(await response.arrayBuffer());
    const filename = `test_${String(i + 1).padStart(2, '0')}.png`;
    await fs.writeFile(`${outputDir}/${filename}`, buffer);

    console.log(`✓ Saved ${filename}`);

  } catch (error) {
    console.error(`✗ Failed:`, error.message);
  }

  // Small delay between requests
  if (i < testPrompts.length - 1) {
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
}

console.log('');
console.log('✅ Test complete!');
console.log('');
console.log('📁 Check results in:', outputDir);
console.log('');
console.log('👀 What to look for:');
console.log('  - Same character across ALL images');
console.log('  - Identical head size and proportions');
console.log('  - Consistent eye position and reflections');
console.log('  - Same 2.5D cell-shading style');
console.log('  - Props match the character style');
console.log('');
console.log('If consistency is 90%+, you\'re ready for production! 🎉');
