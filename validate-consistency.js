/**
 * Automatic Consistency Validation
 *
 * Uses Together.ai Vision API to check if images match the OYKHCHAR style
 * 97% cheaper than OpenAI GPT-4V!
 */

import Together from 'together-ai';
import fs from 'fs/promises';
import path from 'path';
import 'dotenv/config';

const together = new Together({ apiKey: process.env.TOGETHER_API_KEY });

// Directory to validate
const imagesDir = process.argv[2] || 'C:/projects/oykh-temp/lora-ultra-v2/images';

console.log('🔍 Consistency Validation');
console.log('========================');
console.log('');
console.log(`Checking images in: ${imagesDir}`);
console.log('');

// OYKHCHAR style requirements
const STYLE_CHECKLIST = `
Check if this image matches the OYKHCHAR character style:

REQUIRED ELEMENTS:
✓ White stick figure with round head
✓ Simple black dot eyes (two dots, no other facial features)
✓ Thick black outline around character
✓ Smooth, rounded limbs (no sharp angles)
✓ Mitten-style hands (no fingers)
✓ Blue-to-purple gradient background
✓ Minimalist/educational illustration style
✓ 2.5D or cell-shaded look

CONSISTENCY ISSUES TO FLAG:
✗ Wrong head shape (not round)
✗ Extra facial features (mouth, eyebrows, nose)
✗ Thin or inconsistent outlines
✗ Detailed hands with fingers
✗ Wrong background color
✗ Too much detail or realistic rendering
✗ Multiple characters
✗ Wrong proportions (head too small/large)

Rate the image on consistency (1-10) and list any issues.
Return JSON: { "score": 8, "consistent": true, "issues": ["list", "of", "issues"], "notes": "brief description" }
`;

// Get all images
const files = await fs.readdir(imagesDir);
const imageFiles = files.filter(f => f.endsWith('.jpg') || f.endsWith('.png'));

console.log(`Found ${imageFiles.length} images to validate`);
console.log('');

const validationResults = [];
let totalCost = 0;

for (let i = 0; i < imageFiles.length; i++) {
  const filename = imageFiles[i];
  const filepath = path.join(imagesDir, filename);

  console.log(`[${i + 1}/${imageFiles.length}] Validating ${filename}...`);

  try {
    // Read image as base64
    const imageBuffer = await fs.readFile(filepath);
    const base64Image = imageBuffer.toString('base64');
    const mimeType = filename.endsWith('.png') ? 'image/png' : 'image/jpeg';

    // Use Together.ai vision model (Llama 3.2 90B Vision is much cheaper than GPT-4V)
    const response = await together.chat.completions.create({
      model: "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
      messages: [
        {
          role: "user",
          content: [
            {
              type: "image_url",
              image_url: {
                url: `data:${mimeType};base64,${base64Image}`
              }
            },
            {
              type: "text",
              text: STYLE_CHECKLIST
            }
          ]
        }
      ],
      max_tokens: 500,
      temperature: 0.1, // Low temperature for consistent scoring
    });

    const result = response.choices[0]?.message?.content;

    // Try to parse JSON response
    let validation = { score: 0, consistent: false, issues: [], notes: "Failed to parse" };
    try {
      const jsonMatch = result.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        validation = JSON.parse(jsonMatch[0]);
      }
    } catch (e) {
      validation.notes = result.substring(0, 100);
    }

    validationResults.push({
      filename,
      ...validation
    });

    // Cost tracking (Llama Vision is ~$0.0002/image vs GPT-4V ~$0.01/image = 98% savings!)
    const tokens = response.usage?.total_tokens || 500;
    const cost = (tokens / 1000) * 0.0008; // $0.80 per 1M tokens
    totalCost += cost;

    const emoji = validation.consistent ? '✓' : '✗';
    const scoreStr = validation.score ? `Score: ${validation.score}/10` : '';
    console.log(`  ${emoji} ${scoreStr} - ${validation.notes || 'OK'}`);

    if (validation.issues && validation.issues.length > 0) {
      validation.issues.forEach(issue => {
        console.log(`    ⚠️  ${issue}`);
      });
    }

  } catch (error) {
    console.error(`  ✗ Validation failed: ${error.message}`);
    validationResults.push({
      filename,
      score: 0,
      consistent: false,
      issues: ['Validation error: ' + error.message],
      notes: 'Failed to validate'
    });
  }

  // Small delay to avoid rate limits
  if (i < imageFiles.length - 1) {
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
}

// Calculate summary statistics
const consistent = validationResults.filter(r => r.consistent).length;
const avgScore = validationResults.reduce((sum, r) => sum + (r.score || 0), 0) / validationResults.length;

console.log('');
console.log('📊 Validation Summary');
console.log('====================');
console.log('');
console.log(`Total Images: ${imageFiles.length}`);
console.log(`Consistent: ${consistent} (${((consistent / imageFiles.length) * 100).toFixed(1)}%)`);
console.log(`Avg Score: ${avgScore.toFixed(1)}/10`);
console.log(`Total Cost: $${totalCost.toFixed(4)} (vs GPT-4V: $${(imageFiles.length * 0.01).toFixed(2)}) = ${(((1 - totalCost / (imageFiles.length * 0.01)) * 100).toFixed(0))}% savings!`);
console.log('');

// Save detailed report
const report = {
  validatedAt: new Date().toISOString(),
  totalImages: imageFiles.length,
  consistentImages: consistent,
  consistencyRate: (consistent / imageFiles.length) * 100,
  averageScore: avgScore,
  totalCost: totalCost,
  results: validationResults,
  recommendations: []
};

// Add recommendations
const lowScoreImages = validationResults.filter(r => r.score < 7);
if (lowScoreImages.length > 0) {
  report.recommendations.push({
    action: 'remove_low_score',
    count: lowScoreImages.length,
    files: lowScoreImages.map(r => r.filename)
  });
}

const inconsistentImages = validationResults.filter(r => !r.consistent);
if (inconsistentImages.length > 0) {
  report.recommendations.push({
    action: 'review_inconsistent',
    count: inconsistentImages.length,
    files: inconsistentImages.map(r => r.filename)
  });
}

const reportPath = path.join(path.dirname(imagesDir), 'validation-report.json');
await fs.writeFile(reportPath, JSON.stringify(report, null, 2));

console.log('Low Score Images (< 7/10):');
if (lowScoreImages.length === 0) {
  console.log('  ✓ None! All images scored well.');
} else {
  lowScoreImages.forEach(img => {
    console.log(`  - ${img.filename} (${img.score}/10): ${img.notes}`);
  });
}

console.log('');
console.log(`📋 Detailed report saved: ${reportPath}`);
console.log('');
console.log('Recommendations:');
if (report.recommendations.length === 0) {
  console.log('  ✓ All images are consistent! Ready for LoRA training.');
} else {
  report.recommendations.forEach(rec => {
    console.log(`  - ${rec.action}: ${rec.count} images`);
  });
}
