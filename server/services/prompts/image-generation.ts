import { getStyleGuide } from './script-generation';
import type { ProductionVibe } from '../../types';

export const IMAGE_GENERATION_PROMPT = (shot: any, vibe: ProductionVibe) => `
${getStyleGuide(vibe)}

SCENE: ${shot.imagenPrompt}
`;

export const PROMPT_REFINEMENT_PROMPT = (failedPrompt: string, error: string) => `
The following Imagen 3 prompt failed with this error: "${error}".

Original prompt: "${failedPrompt}"

Please rewrite the prompt to avoid this error while keeping the same visual concept. Return ONLY the fixed prompt, with no additional text or formatting.
`;

// Camera angle descriptions for better Imagen prompts
export const CAMERA_DESCRIPTIONS: Record<string, string> = {
  'wide-full-body': 'Wide shot, full body visible, character centered in frame',
  'medium-waist-up': 'Medium shot from waist up, clear facial features and upper body',
  'closeup-shoulders': 'Close-up shot, shoulders and head, expressive facial details',
  'extreme-closeup-face': 'Extreme close-up on face, highly detailed expression',
  'over-shoulder': 'Over-the-shoulder perspective, viewing from behind',
  'top-down': "Top-down bird's eye view, looking directly down",
  'three-quarter': '3/4 angle view, dynamic composition',
  'side-profile': 'Side profile view, clean silhouette',
  'dutch-angle': 'Dutch angle (tilted 15°), dramatic emphasis',
};

// Character action descriptions
export const ACTION_DESCRIPTIONS: Record<string, string> = {
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
  transforming: 'character mid-transformation, morphing visual',
  celebrating: 'character with arms up in victory celebration',
};

// Emotion descriptions
export const EMOTION_DESCRIPTIONS: Record<string, string> = {
  neutral: 'calm relaxed expression',
  happy: 'wide smile, bright dot eyes showing joy',
  excited: 'energetic expression, eyes wide with enthusiasm',
  shocked: 'wide eyes, open circular mouth, surprised',
  confused: 'tilted head, one eye squinting slightly',
  concerned: 'worried expression, furrowed brow area',
  thoughtful: 'contemplative focused expression',
  determined: 'confident focused expression, motivated',
  surprised: 'eyebrows raised area, sudden realization',
  satisfied: 'content smile, accomplished expression',
};
