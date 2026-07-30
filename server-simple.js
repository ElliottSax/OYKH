/**
 * OYKH Simple Video Generation Server
 * Generates complete videos with mock images/audio for now
 */

import express from 'express';
import cors from 'cors';
import { exec } from 'child_process';
import { promisify } from 'util';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { HfInference } from '@huggingface/inference';
import Replicate from 'replicate';
import googleTTS from 'google-tts-api';
import 'dotenv/config';

const execAsync = promisify(exec);
const __dirname = path.dirname(fileURLToPath(import.meta.url));

const app = express();
const PORT = 3100;

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use('/output', express.static(path.join(__dirname, 'output')));

/**
 * POST /api/generate-video-simple
 * Quick video generation with placeholders
 */
app.post('/api/generate-video-simple', async (req, res) => {
  try {
    const { script } = req.body;

    if (!script || !script.chapters) {
      return res.status(400).json({ error: 'Valid script required' });
    }

    console.log('🎬 Generating video for:', script.metadata.title);

    // Set headers for streaming
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Transfer-Encoding', 'chunked');

    const sendProgress = (data) => {
      res.write(JSON.stringify(data) + '\n');
    };

    // Create temp and output directories
    sendProgress({ step: 'Initializing', progress: 2 });
    const tempDir = path.join(__dirname, 'temp', `video-${Date.now()}`);
    const outputDir = path.join(__dirname, 'output');
    await fs.mkdir(tempDir, { recursive: true });
    await fs.mkdir(outputDir, { recursive: true });

    // Get all shots
    const allShots = script.chapters.flatMap((c) => c.shots);
    console.log(`📸 Total shots: ${allShots.length}`);

    // AI Image Generation - Replicate (better quality) or HuggingFace (free tier)
    const USE_REPLICATE = true; // Switch to Replicate for better consistency

    console.log(`🎨 Generating AI images with ${USE_REPLICATE ? 'Replicate (FLUX-dev)' : 'HuggingFace (FLUX-schnell)'}...`);
    sendProgress({ step: 'Generating Images', progress: 5, current: 0, total: allShots.length });

    let hf, replicate;

    if (USE_REPLICATE) {
      const replicateToken = process.env.REPLICATE_API_TOKEN;
      if (!replicateToken) {
        throw new Error('REPLICATE_API_TOKEN not found in environment. Get one at https://replicate.com/account/api-tokens');
      }
      replicate = new Replicate({ auth: replicateToken });
    } else {
      const hfToken = process.env.HUGGINGFACE_TOKEN;
      if (!hfToken) {
        throw new Error('HUGGINGFACE_TOKEN not found in environment');
      }
      hf = new HfInference(hfToken);
    }

    // STRICT STYLE LOCK - Master visual specification for 100% consistency
    const STYLE_LOCK = `CRITICAL VISUAL REQUIREMENTS - DO NOT DEVIATE:

REFERENCE STYLE: Kurzgesagt YouTube educational videos, TED-Ed animations, CGP Grey style

CHARACTER SPECIFICATIONS (EXACT):
- Head: Perfect geometric circle, pure white fill (#FFFFFF), diameter 180-200px
- Eyes: Two perfect black circles (#000000), 12px diameter each, positioned 45px apart horizontally
- Body: Single straight vertical line from head to hips, 10px width, pure black (#000000)
- Arms: Two straight lines extending from shoulders, 10px width, pure black (#000000)
- Legs: Two straight lines extending from hips, 10px width, pure black (#000000)
- Outline: EXACTLY 10px black (#000000) stroke on head only, NO outline on stick body/limbs
- Fill: Pure white (#FFFFFF) on head only, limbs are just black lines
- NO: Gradients on character, shadows on character, textures, 3D effects, rounded joints, fingers, toes

PROPS SPECIFICATIONS (EXACT):
- Style: Same minimalist vector style as character
- Outline: EXACTLY 10px black (#000000) stroke on all props
- Fill: Single flat color per prop, NO gradients within props
- Examples: Coffee mug (white fill, simple handle), Brain (pink #E91E63, 5 simple curves), Clock (white face, simple hands)
- Size: Props should be 20-30% of character size
- NO: Realistic details, textures, complex shapes, multiple colors per prop

BACKGROUND SPECIFICATIONS (EXACT):
- Type: Smooth linear gradient ONLY, top to bottom
- Colors: Deep blue-purple (#4A148C) at top transitioning to medium purple (#7B1FA2) at bottom
- Lighting: Subtle radial glow behind character (optional, very subtle)
- NO: Complex backgrounds, patterns, textures, objects in background, scenery

COMPOSITION RULES (EXACT):
- Character: Always centered horizontally, positioned at 40-60% vertical
- Props: Positioned around character, never overlapping character
- Spacing: Minimum 100px padding from all edges
- Focus: Character is primary element, 50-60% of frame height

COLOR PALETTE (USE ONLY THESE):
- Character: White (#FFFFFF) and Black (#000000) only
- Background: Purple range (#4A148C to #7B1FA2)
- Props: Limited palette - Pink (#E91E63), Yellow (#FFEB3B), Cyan (#00BCD4), Orange (#FF9800)

FORBIDDEN ELEMENTS:
- NO photorealistic elements
- NO complex shading or lighting on character
- NO textures (paper, grain, noise)
- NO gradients on character or props (background only)
- NO curved or organic shapes (use geometric shapes only)
- NO facial features beyond two dot eyes
- NO hands or feet details
- NO background objects or scenery
- NO text or labels in the image

ART STYLE KEYWORDS: flat design, vector illustration, educational graphics, ultra-minimalist, geometric shapes, Kurzgesagt style, high contrast, clean lines, infographic style`;

    // Standardized pose library for consistency
    const POSE_LIBRARY = {
      pointing: 'right arm extended forward pointing at viewer, left arm at side',
      explaining: 'both arms spread wide at 45-degree angles, palms facing forward',
      thinking: 'right hand touching side of head, left arm at side',
      holding: 'both arms forward holding object at chest height',
      excited: 'both arms raised above head in celebration',
      presenting: 'one arm extended to side presenting, other arm at side',
      questioning: 'one arm raised with palm up in questioning gesture',
      teaching: 'one arm pointing, other arm gesturing',
    };

    // Standardized prop descriptions for consistency
    const PROP_LIBRARY = {
      'coffee-mug':
        'Simple coffee mug with handle on right side, white fill, black 10px outline, 3 curved steam lines above',
      brain:
        'Simplified brain icon with 5 curved segments, pink fill (#E91E63), black 10px outline',
      clock:
        'Circular clock face, white fill, black outline, only 12-3-6-9 markers visible, simple black hands',
      lightbulb:
        'Classic lightbulb shape, yellow fill (#FFEB3B), black 10px outline, 3 straight light rays',
      book: 'Rectangular book standing upright, white pages, simple cover, black 10px outline',
      heart: 'Geometric heart shape, red fill (#F44336), black 10px outline',
      star: 'Five-pointed star, yellow fill (#FFEB3B), black 10px outline',
      checkmark: 'Simple checkmark symbol, green (#4CAF50), black 10px outline, bold and geometric',
    };

    // OPTIMIZED GENERATION: Simple natural prompts work best (based on Coffee video success)
    // ULTRA-MINIMAL TEST: Less information = less variation (25-30 words vs 40-50)
    const INFERENCE_STEPS = 4; // Sweet spot for FLUX.1-schnell

    console.log('🎨 Using ultra-minimal prompts (25-30 words) for consistency...');

    const batchSize = 3;
    for (let i = 0; i < allShots.length; i += batchSize) {
      const batch = allShots.slice(i, i + batchSize);

      sendProgress({
        step: 'Generating Images',
        progress: 5 + Math.floor((i / allShots.length) * 60),
        current: i,
        total: allShots.length,
      });

      await Promise.all(
        batch.map(async (shot) => {
          const imagePath = path.join(
            tempDir,
            `shot-${String(shot.shotNumber).padStart(4, '0')}.png`
          );

          try {
            console.log(
              `   Generating shot ${shot.shotNumber}: ${shot.imagenPrompt.substring(0, 50)}...`
            );

            let imageBuffer;

            if (USE_REPLICATE) {
              // Replicate: FLUX-dev for better quality and consistency
              const output = await replicate.run(
                "black-forest-labs/flux-dev",
                {
                  input: {
                    prompt: shot.imagenPrompt,
                    aspect_ratio: "16:9",
                    num_inference_steps: 28, // FLUX-dev sweet spot
                    guidance_scale: 3.5, // Higher = stricter prompt following
                    output_format: "png",
                    output_quality: 90,
                  }
                }
              );

              // Replicate returns a URL, fetch the image
              const response = await fetch(output[0]);
              imageBuffer = Buffer.from(await response.arrayBuffer());
            } else {
              // HuggingFace: FLUX-schnell (free tier)
              const imageBlob = await hf.textToImage({
                model: 'black-forest-labs/FLUX.1-schnell',
                inputs: shot.imagenPrompt,
                parameters: {
                  width: 1024,
                  height: 576,
                  num_inference_steps: INFERENCE_STEPS,
                },
              });
              imageBuffer = Buffer.from(await imageBlob.arrayBuffer());
            }

            await fs.writeFile(imagePath, imageBuffer);
          } catch (err) {
            console.warn(
              `   Failed to generate shot ${shot.shotNumber}, using fallback:`,
              err.message
            );
            // Fallback: simple SVG placeholder
            const placeholderSvg = `
            <svg width="1920" height="1080">
              <rect width="1920" height="1080" fill="#1e1b4b"/>
              <text x="50%" y="50%" font-size="80" fill="white" text-anchor="middle" font-family="Arial">Shot ${shot.shotNumber}</text>
            </svg>
          `;
            const sharp = (await import('sharp')).default;
            await sharp(Buffer.from(placeholderSvg)).png().toFile(imagePath);
          }
        })
      );

      console.log(`   Generated ${i + batchSize + 1}/${allShots.length} shots (img2img)`);

      const currentCount = Math.min(i + batchSize + 1, allShots.length);
      const imageProgress = 5 + (currentCount / allShots.length) * 45; // 5% -> 50%
      sendProgress({
        step: 'Generating Images',
        progress: Math.round(imageProgress),
        current: currentCount,
        total: allShots.length,
      });

      // Small delay between batches to respect rate limits
      if (i + batchSize < allShots.length) {
        await new Promise((resolve) => setTimeout(resolve, 2000));
      }
    }

    console.log(`✓ All ${allShots.length} shots generated with optimized prompts`);

    // Generate voiceover audio for each chapter
    console.log('🎤 Generating voiceover...');
    sendProgress({ step: 'Generating Audio', progress: 50 });
    const audioFiles = [];

    for (let i = 0; i < script.chapters.length; i++) {
      const chapter = script.chapters[i];

      try {
        // Split narration into chunks < 200 chars
        const narration = chapter.narration;
        const chunks = [];
        const sentences = narration.match(/[^.!?]+[.!?]+/g) || [narration];

        let currentChunk = '';
        for (const sentence of sentences) {
          if ((currentChunk + sentence).length < 190) {
            currentChunk += sentence;
          } else {
            if (currentChunk) chunks.push(currentChunk.trim());
            currentChunk = sentence;
          }
        }
        if (currentChunk) chunks.push(currentChunk.trim());

        // Generate audio for each chunk
        for (let j = 0; j < chunks.length; j++) {
          const audioPath = path.join(tempDir, `audio-${i}-${j}.mp3`);

          const audioUrl = googleTTS.getAudioUrl(chunks[j], {
            lang: 'en',
            slow: false,
            host: 'https://translate.google.com',
          });

          const response = await fetch(audioUrl);
          const buffer = await response.arrayBuffer();
          await fs.writeFile(audioPath, Buffer.from(buffer));
          audioFiles.push(audioPath);
        }

        const audioProgress = 50 + ((i + 1) / script.chapters.length) * 20; // 50% -> 70%
        sendProgress({
          step: 'Generating Audio',
          progress: Math.round(audioProgress),
          current: i + 1,
          total: script.chapters.length,
        });

        console.log(
          `   Generated audio ${i + 1}/${script.chapters.length} (${chunks.length} chunks)`
        );
      } catch (err) {
        console.warn(`   Failed to generate audio for chapter ${i + 1}:`, err.message);
      }
    }

    // Concatenate audio files (only if we have audio)
    let mergedAudioPath = null;
    if (audioFiles.length > 0) {
      console.log('🎵 Merging audio...');
      sendProgress({ step: 'Merging Audio', progress: 70 });
      const audioListPath = path.join(tempDir, 'audio-list.txt');
      const audioInputs = audioFiles.map((f) => `file '${f.replace(/\\/g, '/')}'`).join('\n');
      await fs.writeFile(audioListPath, audioInputs);

      mergedAudioPath = path.join(tempDir, 'narration.mp3');
      await execAsync(
        `ffmpeg -y -f concat -safe 0 -i "${audioListPath}" -c copy "${mergedAudioPath}"`,
        {
          maxBuffer: 50 * 1024 * 1024,
        }
      );
    } else {
      console.log('⚠️  No audio generated, creating silent video');
    }

    // Create video from images using FFmpeg
    console.log('🎞️  Assembling video with voiceover...');
    sendProgress({ step: 'Assembling Video', progress: 75 });
    const outputPath = path.join(
      outputDir,
      `${script.metadata.title.replace(/[^a-z0-9]/gi, '_')}_${Date.now()}.mp4`
    );

    // Create input list for FFmpeg
    const inputListPath = path.join(tempDir, 'inputs.txt');
    const inputs = allShots
      .map((shot, idx) => {
        const imagePath = path.join(
          tempDir,
          `shot-${String(shot.shotNumber).padStart(4, '0')}.png`
        );
        return `file '${imagePath.replace(/\\/g, '/')}'\nduration ${shot.duration}`;
      })
      .join('\n');

    // Add last file again (FFmpeg requirement)
    if (allShots.length > 0) {
      const lastShot = allShots[allShots.length - 1];
      const lastPath = path.join(
        tempDir,
        `shot-${String(lastShot.shotNumber).padStart(4, '0')}.png`
      );
      await fs.writeFile(inputListPath, inputs + `\nfile '${lastPath.replace(/\\/g, '/')}'`);
    }

    // Run FFmpeg (with or without audio)
    let ffmpegCommand;
    if (mergedAudioPath) {
      ffmpegCommand = `ffmpeg -y -f concat -safe 0 -i "${inputListPath}" -i "${mergedAudioPath}" -vf "scale=1920:1080,fps=30" -c:v libx264 -c:a aac -preset fast -crf 23 -pix_fmt yuv420p -shortest "${outputPath}"`;
      console.log('Running FFmpeg with voiceover...');
    } else {
      ffmpegCommand = `ffmpeg -y -f concat -safe 0 -i "${inputListPath}" -vf "scale=1920:1080,fps=30" -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p "${outputPath}"`;
      console.log('Running FFmpeg (silent video)...');
    }

    await execAsync(ffmpegCommand, { maxBuffer: 50 * 1024 * 1024 });
    sendProgress({ step: 'Finalizing', progress: 95 });

    // Cleanup temp files
    await fs.rm(tempDir, { recursive: true, force: true });

    const videoUrl = `/output/${path.basename(outputPath)}`;
    console.log(`✅ Video created: ${videoUrl}`);

    sendProgress({
      success: true,
      step: 'Complete',
      progress: 100,
      videoUrl: `http://localhost:3100${videoUrl}`,
      videoPath: outputPath,
      shots: allShots.length,
      duration: allShots.reduce((sum, shot) => sum + shot.duration, 0),
    });

    res.end();
  } catch (error) {
    console.error('Video generation error:', error);
    res.write(
      JSON.stringify({
        error: error.message || 'Unknown error',
      }) + '\n'
    );
    res.end();
  }
});

/**
 * POST /api/generate-script
 * Generate video script using Gemini API
 */
app.post('/api/generate-script', async (req, res) => {
  try {
    const { topic, hook, vibe } = req.body;

    if (!topic) {
      return res.status(400).json({ error: 'Topic required' });
    }

    console.log('🎬 Generating script for:', topic);

    const apiKey = process.env.GEMINI_API_KEY || process.env.VITE_GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error('GEMINI_API_KEY not found in environment variables');
    }

    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });

    const prompt = `You are an expert viral YouTube video scriptwriter.

TOPIC: "${topic}"
HOOK: "${hook || `The Secret of ${topic}`}"
VIBE: ${vibe || 'minimal'}

CRITICAL: ALL shots must feature the SAME CHARACTER for visual consistency.

CHARACTER SPECIFICATIONS (EXACT - NEVER DEVIATE):
- Head: Perfect geometric circle, pure white (#FFFFFF), black 10px outline
- Eyes: Two black dots, 12px diameter, 45px apart
- Body: Simple stick figure, pure black lines, 10px width
- NO gradients, shadows, textures, or 3D effects on character
- Reference style: EXACTLY like Kurzgesagt YouTube educational videos

STANDARDIZED POSES (use these descriptions):
- "pointing": right arm extended pointing forward, left arm at side
- "explaining": both arms spread wide at 45 degrees
- "thinking": right hand touching side of head
- "holding": both arms forward holding object
- "excited": both arms raised above head
- "presenting": one arm extended to side, other at side
- "questioning": one arm raised with palm up

STANDARDIZED PROPS (use these exact descriptions):
- "coffee-mug": Simple mug, white fill, black 10px outline, 3 steam lines
- "brain": Pink (#E91E63) with 5 curved segments, black 10px outline
- "clock": White circle face, black outline, simple hands
- "lightbulb": Yellow (#FFEB3B), black outline, 3 light rays
- "book": White rectangular book, black outline
- "heart": Red (#F44336), geometric heart, black outline
- "checkmark": Green (#4CAF50), simple check, black outline

BACKGROUND (EXACT):
- ALWAYS: Smooth gradient from deep purple (#4A148C) at top to medium purple (#7B1FA2) at bottom
- NO patterns, textures, objects, or scenery in background
- Optional: Subtle glow behind character

VISUAL CONSISTENCY RULES:
- Same character design in EVERY shot (only pose changes)
- Props use same minimalist vector style
- Background gradient consistent across all shots
- Shot pacing: 4-7 seconds per shot for retention

Create a SHORT 2-minute (120 second) video script with 20-30 shots total.

CRITICAL SHOT PACING:
- Each shot MUST be 4-7 seconds (never less than 4, never more than 7)
- Rapid scene changes designed to combat viewer drop-off at 30-second mark
- This fast-cut approach keeps attention high and prevents early scroll-away

Return as JSON with this structure:
{
  "metadata": {
    "topic": "${topic}",
    "hook": "${hook || `The Secret of ${topic}`}",
    "title": "Catchy 60-char video title",
    "vibe": "${vibe || 'minimal'}",
    "targetDuration": 120,
    "thumbnailConcept": {
      "mainElement": "Main visual",
      "emotion": "shocked/curious",
      "text": "3-5 WORDS",
      "colorScheme": "high contrast colors"
    }
  },
  "chapters": [
    {
      "chapterNumber": 1,
      "title": "Chapter title",
      "timestamp": "0:00",
      "duration": 30,
      "purpose": "Hook viewers",
      "narration": "Natural conversational narration",
      "keyMessage": "Main point",
      "emotionalTone": "curious",
      "shots": [
        {
          "shotNumber": 1,
          "duration": 5,
          "pose": "pointing",
          "props": ["coffee-mug"],
          "characterEmotion": "excited, engaging",
          "cameraAngle": "medium shot",
          "imagenPrompt": "Character in 'pointing' pose. Coffee mug prop in upper right. Character is pure white stick figure with perfect circle head, two black dot eyes, black 10px outline on head. Simple stick body and limbs in pure black. Mug is white with black 10px outline and 3 steam lines. Background is smooth gradient from deep purple (#4A148C) at top to medium purple (#7B1FA2) at bottom. Kurzgesagt educational video style. Clean vector illustration. NO textures, NO shadows on character, NO complex details.",
          "animation": "fade in",
          "transition": "cut"
        }
      ]
    }
  ],
  "openLoops": [],
  "retentionBombs": [],
  "totalShots": 35,
  "estimatedCost": 17.50
}

CRITICAL IMAGENPROMPT INSTRUCTIONS - REFINED MASTER PROMPT:

Use this EXACT template structure for EVERY imagenPrompt. This achieves perfect 2.5D minimalist consistency:

REFINED MASTER PROMPT (use as base for ALL images):
"A minimalist white stick figure character [ACTION/POSE], centered on a simple blue-purple gradient background. The character features a perfectly round head, thick black vector outlines (3px stroke), and simple black dot eyes with subtle white reflections. No lines separating hands from arms or feet from legs. Hands are rendered as simple ovals with a separate small thumb (mitten-style), and feet are simple minimalist blobs. Flat vector illustration, cell-shaded with 2.5D depth shading, clean lines, high-contrast, high-quality digital art."

KEY COMPONENTS EXPLAINED:
- "No lines separating hands from arms": CRITICAL for seamless limb look
- "Simple ovals with separate small thumb": Creates mitten-style without individual fingers
- "Feet as simple minimalist blobs": No toes, no ankle definition
- "2.5D depth shading": Cell-shading with edge lighting for polished look
- "Subtle white reflections" in eyes: Adds life without complexity

HOW TO CREATE imagenPrompt:
1. Replace [ACTION/POSE] with specific character action
2. If props exist, mention them naturally in the action
3. That's it - master prompt handles all style details

EXAMPLE EXCELLENT imagenPrompts:

1. "A minimalist white stick figure character pointing at the viewer with one arm extended forward, centered on a simple blue-purple gradient background. The character features a perfectly round head, thick black vector outlines (3px stroke), and simple black dot eyes with subtle white reflections. No lines separating hands from arms or feet from legs. Hands are rendered as simple ovals with a separate small thumb (mitten-style), and feet are simple minimalist blobs. Flat vector illustration, cell-shaded with 2.5D depth shading, clean lines, high-contrast, high-quality digital art."

2. "A minimalist white stick figure character holding a simple coffee mug with steam rising, centered on a simple blue-purple gradient background. The character features a perfectly round head, thick black vector outlines (3px stroke), and simple black dot eyes with subtle white reflections. No lines separating hands from arms or feet from legs. Hands are rendered as simple ovals with a separate small thumb (mitten-style), and feet are simple minimalist blobs. Flat vector illustration, cell-shaded with 2.5D depth shading, clean lines, high-contrast, high-quality digital art."

3. "A minimalist white stick figure character with both arms spread wide in explaining gesture, brain icon floating above, centered on a simple blue-purple gradient background. The character features a perfectly round head, thick black vector outlines (3px stroke), and simple black dot eyes with subtle white reflections. No lines separating hands from arms or feet from legs. Hands are rendered as simple ovals with a separate small thumb (mitten-style), and feet are simple minimalist blobs. Flat vector illustration, cell-shaded with 2.5D depth shading, clean lines, high-contrast, high-quality digital art."

CRITICAL RULES:
- ALWAYS use the complete REFINED MASTER PROMPT
- Only vary the [ACTION/POSE] section and props
- Keep actions natural: "pointing at viewer", "holding coffee mug", "arms spread wide"
- Props are simple: "coffee mug", "brain icon", "lightbulb", "clock"
- Total length: 70-90 words
- This prompt structure guarantees 2.5D consistency

RETURN ONLY VALID JSON. No markdown code blocks.`;

    console.log('📡 Calling Gemini API...');
    const result = await Promise.race([
      model.generateContent(prompt),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error('Request timed out after 120 seconds')), 120000)
      ),
    ]);

    let text = result.response.text();
    console.log('✅ Response received:', text.length, 'chars');

    // Strip markdown code blocks if present
    text = text.replace(/^```json\s*/i, '').replace(/\s*```$/s, '');

    const script = JSON.parse(text);

    // Add status to all shots
    script.chapters.forEach((chapter) => {
      chapter.shots.forEach((shot) => {
        shot.status = 'pending';
      });
    });

    console.log('✅ Script generated:', script.totalShots, 'shots');
    res.json(script);
  } catch (error) {
    console.error('Script generation error:', error);
    res.status(500).json({
      error: error.message || 'Script generation failed',
    });
  }
});

/**
 * GET /api/health
 */
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', ffmpeg: 'available' });
});

app.listen(PORT, () => {
  console.log(`🚀 OYKH Backend running on http://localhost:${PORT}`);
  console.log(`📹 Video output: ${path.join(__dirname, 'output')}`);
});
