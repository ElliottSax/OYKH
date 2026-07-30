/**
 * OYKH Video Generation Backend Server
 *
 * Handles:
 * - Script generation (Gemini)
 * - Image generation (Imagen 3 / Mock)
 * - Audio generation (Google TTS / Mock)
 * - Video assembly (FFmpeg)
 */

import express from 'express';
import cors from 'cors';
import {
  generateViralVideoScript,
  generateAllImages,
  generateNarration,
  assembleVideo,
} from './services';
import type { ProductionVibe, ViralVideoScript } from './types';

const app = express();
const PORT = process.env.PORT || 3101;

app.use(cors());
app.use(express.json({ limit: '50mb' }));

// Serve static files from output directory
app.use('/output', express.static('output'));

/**
 * POST /api/generate-video
 * Generate complete video from topic
 */
app.post('/api/generate-video', async (req, res) => {
  try {
    const { topic, hook, vibe } = req.body as {
      topic: string;
      hook?: string;
      vibe: ProductionVibe;
    };

    if (!topic) {
      return res.status(400).json({ error: 'Topic is required' });
    }

    console.log('🎬 Starting video generation for:', topic);

    // Step 1: Generate script (5%)
    res.write(JSON.stringify({ step: 'script', progress: 5 }) + '\n');
    const script = await generateViralVideoScript(topic, hook, vibe);
    console.log('✅ Script generated');

    // Step 2: Generate images (5% → 50%)
    res.write(JSON.stringify({ step: 'images', progress: 10 }) + '\n');
    const allShots = script.chapters.flatMap((c) => c.shots);
    const imageUrls = await generateAllImages(allShots, vibe, (current, total) => {
      const progress = 10 + (current / total) * 40; // 10% → 50%
      res.write(JSON.stringify({ step: 'images', progress, current, total }) + '\n');
    });
    console.log('✅ Images generated');

    // Step 3: Generate audio (50% → 70%)
    res.write(JSON.stringify({ step: 'audio', progress: 50 }) + '\n');
    const narrationText = script.chapters.map((c) => c.narration).join('\n\n');
    const audioData = await generateNarration(narrationText, 'en-US-Journey-D', true); // Mock for now
    console.log('✅ Audio generated');
    res.write(JSON.stringify({ step: 'audio', progress: 70 }) + '\n');

    // Step 4: Assemble video (70% → 100%)
    res.write(JSON.stringify({ step: 'assembly', progress: 75 }) + '\n');
    const videoPath = await assembleVideo(script, imageUrls, audioData, (percent) => {
      const progress = 75 + percent * 0.25; // 75% → 100%
      res.write(JSON.stringify({ step: 'assembly', progress }) + '\n');
    });
    console.log('✅ Video assembled');

    // Send final result
    const videoUrl = `/output/${videoPath.split(/[\\/]/).pop()}`;
    res.write(
      JSON.stringify({
        step: 'complete',
        progress: 100,
        videoUrl,
        script,
      }) + '\n'
    );

    res.end();
  } catch (error) {
    console.error('Video generation error:', error);
    res.status(500).json({
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * POST /api/generate-script
 * Generate script only (for preview)
 */
app.post('/api/generate-script', async (req, res) => {
  try {
    const { topic, hook, vibe } = req.body;

    if (!topic) {
      return res.status(400).json({ error: 'Topic is required' });
    }

    const script = await generateViralVideoScript(topic, hook, vibe);
    res.json({ script });
  } catch (error) {
    console.error('Script generation error:', error);
    res.status(500).json({
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * GET /api/health
 * Health check
 */
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.listen(PORT, () => {
  console.log(`🚀 OYKH Backend Server running on http://localhost:${PORT}`);
  console.log(`📁 Output directory: ${process.cwd()}/output`);
});
