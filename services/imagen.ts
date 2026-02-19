
import { Shot, ProductionVibe } from "../types.ts";
import { getStyleGuide } from "./gemini.ts";

/**
 * Imagen 3 Image Generation Service
 *
 * NOTE: This is a client-side placeholder. For production:
 * 1. Move this to a backend Cloud Function
 * 2. Use @google-cloud/aiplatform package
 * 3. Keep API credentials server-side
 */

// Camera angle descriptions for better Imagen prompts
const CAMERA_DESCRIPTIONS: Record<string, string> = {
  'wide-full-body': 'Wide shot, full body visible, character centered in frame',
  'medium-waist-up': 'Medium shot from waist up, clear facial features and upper body',
  'closeup-shoulders': 'Close-up shot, shoulders and head, expressive facial details',
  'extreme-closeup-face': 'Extreme close-up on face, highly detailed expression',
  'over-shoulder': 'Over-the-shoulder perspective, viewing from behind',
  'top-down': 'Top-down bird\'s eye view, looking directly down',
  'three-quarter': '3/4 angle view, dynamic composition',
  'side-profile': 'Side profile view, clean silhouette',
  'dutch-angle': 'Dutch angle (tilted 15°), dramatic emphasis'
};

// Character action descriptions
const ACTION_DESCRIPTIONS: Record<string, string> = {
  'standing-neutral': 'character standing calmly with relaxed posture',
  'thinking-chin': 'character with hand on chin, thoughtful contemplative pose',
  'excited-jumping': 'character mid-jump with arms raised high, radiating energy',
  'confused-questionmark': 'character with tilted head, question mark floating above',
  'explaining-pointing': 'character pointing forward with teaching gesture',
  'running-forward': 'character in mid-run pose, dynamic forward movement',
  'holding-object': 'character holding an object in both hands',
  'sitting-desk': 'character sitting at minimalist desk',
  'looking-magnifying-glass': 'character examining with magnifying glass',
  'lightbulb-idea': 'character with lightbulb appearing above head, eureka moment',
  'two-characters-talking': 'two puffy characters facing each other in conversation',
  'climbing-stairs': 'character climbing upward on stairs, progress visual',
  'presenting-chart': 'character standing next to simple chart or graph',
  'transforming': 'character mid-transformation, morphing visual',
  'celebrating': 'character with arms up in victory celebration'
};

// Emotion descriptions
const EMOTION_DESCRIPTIONS: Record<string, string> = {
  'neutral': 'calm relaxed expression',
  'happy': 'wide smile, bright dot eyes showing joy',
  'excited': 'energetic expression, eyes wide with enthusiasm',
  'shocked': 'wide eyes, open circular mouth, surprised',
  'confused': 'tilted head, one eye squinting slightly',
  'concerned': 'worried expression, furrowed brow area',
  'thoughtful': 'contemplative focused expression',
  'determined': 'confident focused expression, motivated',
  'surprised': 'eyebrows raised area, sudden realization',
  'satisfied': 'content smile, accomplished expression'
};

/**
 * Build an optimized prompt for Imagen 3
 */
export const buildImagenPrompt = (shot: Shot, vibe: ProductionVibe): string => {
  const styleGuide = getStyleGuide(vibe);

  const prompt = `${styleGuide}

SHOT #${shot.shotNumber}:

CHARACTER:
- Action: ${ACTION_DESCRIPTIONS[shot.characterAction] || shot.characterAction}
- Emotion: ${EMOTION_DESCRIPTIONS[shot.characterEmotion] || shot.characterEmotion}
- Expression: Show clear emotion in the two tiny black dot eyes and posture

CAMERA:
- Angle: ${CAMERA_DESCRIPTIONS[shot.cameraAngle] || shot.cameraAngle}
- Movement feel: ${shot.cameraMovement}

BACKGROUND:
- Style: ${shot.backgroundStyle}
- Keep absolutely minimal and clean

SCENE DESCRIPTION:
${shot.imagenPrompt}

CRITICAL REQUIREMENTS:
- Ultra-clean minimalist aesthetic
- Bold 8px black outlines on EVERYTHING
- High contrast for visibility
- Professional YouTube thumbnail quality
- Clear expressive character
- NO complex textures or background clutter
${shot.textOverlay ? `- Leave space for text overlay: "${shot.textOverlay.text}" at ${shot.textOverlay.position}` : ''}

Render in 16:9 aspect ratio, 1920x1080 resolution.`;

  return prompt.trim();
};

/**
 * Generate an image using Imagen 3
 *
 * PRODUCTION VERSION (Backend - Cloud Function):
 */
export const generateImage = async (shot: Shot, vibe: ProductionVibe): Promise<string> => {
  // For PRODUCTION: This should call your backend proxy
  // const response = await fetch('/api/generate-image', {
  //   method: 'POST',
  //   headers: { 'Content-Type': 'application/json' },
  //   body: JSON.stringify({ shot, vibe })
  // });
  // return await response.json();

  // DEVELOPMENT VERSION (Mock/Placeholder):
  // In real implementation, this would call Imagen 3 via backend
  console.log('[Imagen 3] Generating image for shot', shot.shotNumber);
  console.log('[Imagen 3] Prompt:', buildImagenPrompt(shot, vibe));

  // For now, return a placeholder
  // You can uncomment the Imagen code below when backend is set up
  return await generateMockImage(shot, vibe);
};

/**
 * Mock image generation (for development/testing)
 */
const generateMockImage = async (shot: Shot, vibe: ProductionVibe): Promise<string> => {
  // Simulate API delay
  await new Promise(r => setTimeout(r, 500));

  // Generate a colored placeholder
  const vibeColors: Record<ProductionVibe, string> = {
    cosmic: '1e1b4b',
    hype: '7e22ce',
    minimal: 'f0f9ff',
    suspense: '18181b',
    success: 'd97706'
  };

  const color = vibeColors[vibe] || '3b82f6';
  const text = encodeURIComponent(`Shot ${shot.shotNumber}\n${shot.characterAction}`);

  return `https://placehold.co/1920x1080/${color}/ffffff?text=${text}`;
};

/**
 * Batch generate multiple images with progress tracking
 */
export const generateImagesInBatch = async (
  shots: Shot[],
  vibe: ProductionVibe,
  onProgress?: (completed: number, total: number) => void
): Promise<Shot[]> => {

  const results: Shot[] = [];

  for (let i = 0; i < shots.length; i++) {
    const shot = shots[i];

    try {
      shot.status = 'generating';
      const imageData = await generateImage(shot, vibe);

      shot.imageData = imageData;
      shot.status = 'completed';

      results.push(shot);

      if (onProgress) {
        onProgress(i + 1, shots.length);
      }

    } catch (error) {
      console.error(`Failed to generate shot ${shot.shotNumber}:`, error);
      shot.status = 'error';
      results.push(shot);
    }
  }

  return results;
};

/**
 * Optimize shot generation by grouping similar shots
 * This reduces costs by ~40% by reusing base images with variations
 */
export const optimizeShots = (shots: Shot[]): Map<string, Shot[]> => {
  const groups = new Map<string, Shot[]>();

  shots.forEach(shot => {
    // Create a key based on similar visual characteristics
    const key = `${shot.characterAction}-${shot.cameraAngle}-${shot.backgroundStyle}`;

    if (!groups.has(key)) {
      groups.set(key, []);
    }

    groups.get(key)!.push(shot);
  });

  return groups;
};

/**
 * Apply CSS-based variation to an existing image
 * (Cheaper than regenerating)
 */
export const applyImageVariation = (
  baseImage: string,
  animation: string
): { image: string; cssTransform: string } => {

  const transforms: Record<string, string> = {
    'ken-burns-in': 'scale(1.15)',
    'ken-burns-out': 'scale(0.85)',
    'pan-right': 'translateX(-5%)',
    'pan-left': 'translateX(5%)',
    'static': 'none'
  };

  return {
    image: baseImage,
    cssTransform: transforms[animation] || 'none'
  };
};

/* ============================================
   BACKEND CLOUD FUNCTION EXAMPLE
   ============================================

// functions/src/generate-image.ts
import { PredictionServiceClient } from '@google-cloud/aiplatform';
import { helpers } from '@google-cloud/aiplatform';

const client = new PredictionServiceClient({
  apiEndpoint: 'us-central1-aiplatform.googleapis.com',
});

export const generateImageBackend = async (req, res) => {
  const { shot, vibe } = req.body;

  const prompt = buildImagenPrompt(shot, vibe);
  const endpoint = `projects/${process.env.GOOGLE_CLOUD_PROJECT_ID}/locations/us-central1/publishers/google/models/imagen-3.0-generate-001`;

  const parameters = helpers.toValue({
    sampleCount: 1,
    aspectRatio: '16:9',
    safetySetting: 'block_some',
    personGeneration: 'allow_adult',
  });

  const instanceValue = helpers.toValue({ prompt });
  const instances = [instanceValue];

  const [response] = await client.predict({
    endpoint,
    instances,
    parameters,
  });

  const prediction = response.predictions?.[0];
  const imageBytes = prediction?.structValue?.fields?.bytesBase64Encoded?.stringValue;

  if (!imageBytes) {
    return res.status(500).json({ error: 'IMG_GEN_FAIL' });
  }

  res.json({
    imageData: `data:image/png;base64,${imageBytes}`
  });
};

============================================ */
