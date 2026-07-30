import sharp from 'sharp';
import fs from 'fs/promises';

console.log('Testing sharp image generation...');

const svgText = `
  <svg width="1920" height="1080">
    <rect width="1920" height="1080" fill="#1e1b4b"/>
    <text x="50%" y="50%" font-size="120" fill="white" text-anchor="middle" dy=".3em" font-family="Arial">
      Test Shot 1
    </text>
  </svg>
`;

try {
  await sharp(Buffer.from(svgText)).png().toFile('test-output.png');

  const stats = await fs.stat('test-output.png');
  console.log('✅ Success! Generated test-output.png:', stats.size, 'bytes');
} catch (error) {
  console.error('❌ Error:', error);
}
