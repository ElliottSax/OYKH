/**
 * Train LoRA for OYKH Character Consistency
 * Uses Replicate's FLUX LoRA Trainer
 */

import Replicate from 'replicate';
import fs from 'fs';
import 'dotenv/config';

const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });

console.log('🎓 Starting OYKH Character LoRA Training');
console.log('================================================');
console.log('');

// Training configuration
const trainingConfig = {
  destination: "oykh-character-v1", // Model name
  input: {
    // Training data - we'll need to upload the tar.gz file
    input_images: "https://upload-to-replicate-url", // Will be replaced

    // Training parameters
    steps: 1000, // Number of training steps (1000 = ~30min, 2000 = ~1hr)
    lora_rank: 16, // LoRA rank (16 is good balance of quality/size)
    optimizer: "adamw8bit", // Memory-efficient optimizer
    batch_size: 1, // Batch size (1 for stability)
    learning_rate: 0.0004, // Learning rate

    // Image settings
    resolution: "512,768,1024", // Train on multiple resolutions

    // Caption settings
    autocaption: false, // We have manual captions
    caption_dropout_rate: 0.05, // Slight dropout for generalization

    // Trigger word
    trigger_word: "OYKHCHAR", // Use this in prompts to activate character

    // Seed for reproducibility
    seed: 424242,
  }
};

console.log('Training Configuration:');
console.log('  - Steps:', trainingConfig.input.steps);
console.log('  - LoRA Rank:', trainingConfig.input.lora_rank);
console.log('  - Trigger Word:', trainingConfig.input.trigger_word);
console.log('  - Resolutions:', trainingConfig.input.resolution);
console.log('');

console.log('📦 Step 1: Upload training data to Replicate');
console.log('================================================');
console.log('');
console.log('To upload training data:');
console.log('');
console.log('1. Go to https://replicate.com/');
console.log('2. Navigate to your account');
console.log('3. Upload the training archive:');
console.log('   File: C:/projects/oykh-temp/lora-training/oykh-training-data.tar.gz');
console.log('');
console.log('OR use Replicate file upload API (recommended):');
console.log('');

// Method 1: Using Replicate's file upload
try {
  console.log('Attempting to upload training data via Replicate API...');

  const file = fs.readFileSync('C:/projects/oykh-temp/lora-training/oykh-training-data.tar.gz');

  // Note: Replicate's Node SDK doesn't have direct file upload yet
  // We'll need to use their HTTP API or upload manually
  console.log('');
  console.log('⚠️  Manual upload required:');
  console.log('');
  console.log('Option A - Use Replicate web interface:');
  console.log('1. Go to https://replicate.com/account');
  console.log('2. Create a new model or use file hosting');
  console.log('3. Upload: oykh-training-data.tar.gz (19MB)');
  console.log('4. Get the public URL');
  console.log('');
  console.log('Option B - Use temporary file hosting:');
  console.log('1. Upload to: https://file.io or https://tmpfiles.org');
  console.log('2. Upload: C:/projects/oykh-temp/lora-training/oykh-training-data.tar.gz');
  console.log('3. Copy the download URL');
  console.log('');
  console.log('Option C - Use GitHub/Hugging Face:');
  console.log('1. Create a public repository');
  console.log('2. Upload the tar.gz file');
  console.log('3. Get raw file URL');
  console.log('');
  console.log('Once you have the URL, run:');
  console.log('  node train-lora.js <your-url>');
  console.log('');

  // Check if URL was provided as argument
  const uploadUrl = process.argv[2];

  if (!uploadUrl) {
    console.log('⏸️  Waiting for training data URL...');
    console.log('');
    console.log('After uploading, run:');
    console.log('  node train-lora.js https://your-file-url/oykh-training-data.tar.gz');
    process.exit(0);
  }

  console.log('✅ Using training data URL:', uploadUrl);
  console.log('');

  // Update config with actual URL
  trainingConfig.input.input_images = uploadUrl;

  console.log('🎓 Step 2: Start LoRA Training');
  console.log('================================================');
  console.log('');
  console.log('Training FLUX-dev LoRA with your character...');
  console.log('This will take approximately 30-60 minutes.');
  console.log('Cost: ~$5-10');
  console.log('');

  const training = await replicate.trainings.create(
    "ostris",
    "flux-dev-lora-trainer",
    "e440909d3512c31646ee2e0c7d6f6f4923224863a6a10c494606e79fb5844497",
    trainingConfig
  );

  console.log('✅ Training started!');
  console.log('');
  console.log('Training ID:', training.id);
  console.log('Status:', training.status);
  console.log('');
  console.log('Monitor progress at:');
  console.log(`https://replicate.com/p/${training.id}`);
  console.log('');
  console.log('Checking training status...');
  console.log('');

  // Poll for training completion
  let currentTraining = training;
  while (currentTraining.status !== 'succeeded' && currentTraining.status !== 'failed') {
    await new Promise(resolve => setTimeout(resolve, 30000)); // Check every 30 seconds
    currentTraining = await replicate.trainings.get(training.id);

    console.log(`[${new Date().toLocaleTimeString()}] Status: ${currentTraining.status}`);

    if (currentTraining.logs) {
      const lines = currentTraining.logs.split('\n');
      const lastLine = lines[lines.length - 2] || lines[lines.length - 1];
      if (lastLine) {
        console.log(`  Latest: ${lastLine.substring(0, 100)}`);
      }
    }
  }

  console.log('');
  if (currentTraining.status === 'succeeded') {
    console.log('🎉 LoRA Training Complete!');
    console.log('');
    console.log('Your trained model:');
    console.log(`  ${currentTraining.output.model}`);
    console.log('');
    console.log('To use in video generation:');
    console.log('  1. Update server-simple.js');
    console.log('  2. Use model:', currentTraining.output.model);
    console.log('  3. Include "OYKHCHAR" in prompts');
    console.log('');
    console.log('Example prompt:');
    console.log('  "OYKHCHAR pointing at viewer, coffee mug, blue-purple gradient"');
    console.log('');
  } else {
    console.log('❌ Training failed');
    console.log('Error:', currentTraining.error);
  }

} catch (error) {
  console.error('Error:', error.message);
  console.log('');
  console.log('If you encounter issues, you can manually start training at:');
  console.log('https://replicate.com/ostris/flux-dev-lora-trainer/train');
}
