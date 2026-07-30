/**
 * Generate Training Images for LoRA
 * Creates 30 high-quality character images with variety
 */

import Replicate from 'replicate';
import fs from 'fs/promises';
import path from 'path';
import 'dotenv/config';

const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });

// Create training data directory
const trainingDir = 'C:/projects/oykh-temp/lora-training';
const imagesDir = path.join(trainingDir, 'images');
const captionsDir = path.join(trainingDir, 'captions');

await fs.mkdir(imagesDir, { recursive: true });
await fs.mkdir(captionsDir, { recursive: true });

// Training prompts - variety of poses and props
const trainingPrompts = [
  // Pointing/Gesturing (5 variations)
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, pointing at viewer with inviting gesture. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "pointing" },
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, pointing upward with excited gesture. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "pointing_up" },
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, pointing to the side with explaining gesture. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "pointing_side" },
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, both arms spread wide in welcoming gesture. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "arms_spread" },
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, one arm raised in teaching gesture. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "teaching" },

  // Thinking/Contemplating (5 variations)
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, hand touching chin in thoughtful pose. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "thinking_chin" },
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, hand on temple in contemplative pose. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "thinking_temple" },
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, arms crossed in pondering pose. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "arms_crossed" },
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, looking upward in wonder. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "looking_up" },
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, hand raised to head in curious gesture. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "curious" },

  // With Props - Coffee (3 variations)
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, holding coffee mug with steam lines rising. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "coffee_holding" },
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, drinking from coffee mug. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "coffee_drinking" },
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, offering coffee mug forward. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "coffee_offering" },

  // With Props - Brain/Lightbulb (4 variations)
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, brain icon floating above head. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "brain_icon" },
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, lightbulb appearing above head. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "lightbulb" },
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, pointing at brain icon beside. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "pointing_brain" },
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, holding lightbulb with excited expression. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "holding_lightbulb" },

  // With Props - Clock/Time (3 variations)
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, holding simple clock. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "clock_holding" },
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, pointing at clock urgently. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "clock_urgent" },
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, clock icon floating nearby. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "clock_floating" },

  // Emotional States (5 variations)
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, jumping with joy and excitement. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "excited_jumping" },
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, slumped shoulders in frustrated pose. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "frustrated" },
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, relaxed standing pose. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "relaxed" },
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, celebrating with arms raised. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "celebrating" },
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, shrugging gesture with uncertainty. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "shrugging" },

  // Basic Poses (5 variations)
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, standing straight in neutral pose. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "neutral" },
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, walking forward. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "walking" },
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, sitting position. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "sitting" },
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, leaning forward slightly. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "leaning" },
  { prompt: "White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, standing with hands on hips. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.", pose: "hands_on_hips" },
];

console.log(`🎨 Generating ${trainingPrompts.length} training images with Replicate FLUX-dev...`);
console.log(`📁 Output: ${imagesDir}`);
console.log('');

let completed = 0;

// Generate images in batches of 3 to avoid rate limits
const batchSize = 3;
for (let i = 0; i < trainingPrompts.length; i += batchSize) {
  const batch = trainingPrompts.slice(i, i + batchSize);

  await Promise.all(
    batch.map(async ({ prompt, pose }, batchIndex) => {
      const imageNum = i + batchIndex + 1;
      const imageName = `training_${String(imageNum).padStart(3, '0')}_${pose}`;
      const imagePath = path.join(imagesDir, `${imageName}.png`);
      const captionPath = path.join(captionsDir, `${imageName}.txt`);

      try {
        console.log(`   [${imageNum}/${trainingPrompts.length}] Generating: ${pose}...`);

        const output = await replicate.run(
          "black-forest-labs/flux-dev",
          {
            input: {
              prompt: prompt,
              aspect_ratio: "16:9",
              num_inference_steps: 28,
              guidance_scale: 3.5,
              output_format: "png",
              output_quality: 100, // Max quality for training
            }
          }
        );

        // Download image
        const response = await fetch(output[0]);
        const arrayBuffer = await response.arrayBuffer();
        await fs.writeFile(imagePath, Buffer.from(arrayBuffer));

        // Create caption file (simplified for LoRA training)
        const caption = "OYKHCHAR white stick figure character with round head and dot eyes, " +
                       pose.replace(/_/g, ' ') + ", blue-purple gradient background, " +
                       "educational illustration style";
        await fs.writeFile(captionPath, caption);

        completed++;
        console.log(`   ✓ [${completed}/${trainingPrompts.length}] Saved: ${imageName}.png`);

      } catch (error) {
        console.error(`   ✗ Failed to generate ${pose}:`, error.message);
      }
    })
  );

  // Small delay between batches
  if (i + batchSize < trainingPrompts.length) {
    console.log(`   Waiting 2 seconds before next batch...`);
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
}

console.log('');
console.log(`✅ Training dataset complete: ${completed}/${trainingPrompts.length} images`);
console.log(`📁 Images: ${imagesDir}`);
console.log(`📝 Captions: ${captionsDir}`);
console.log('');
console.log('Next steps:');
console.log('1. Review images and remove any inconsistent/bad generations');
console.log('2. Create ZIP file of training images');
console.log('3. Upload to Replicate for LoRA training');
