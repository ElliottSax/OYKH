import { GoogleGenerativeAI, Schema } from '@google/generative-ai';
import { ViralVideoScript, ProductionVibe } from '../types.ts';
import {
  SCRIPT_GENERATION_PROMPT,
  SCRIPT_SCHEMA,
  REFINEMENT_PROMPT,
  SUGGESTIONS_PROMPT,
  HOOKS_PROMPT,
} from './prompts';

const ai = new GoogleGenerativeAI(process.env.API_KEY);

/**
 * Generate a complete viral 5-minute video script optimized for retention
 */
export const generateViralVideoScript = async (
  topic: string,
  hook: string,
  vibe: ProductionVibe
): Promise<ViralVideoScript> => {
  const model = ai.getGenerativeModel({ model: 'gemini-1.5-flash' });
  const result = await model.generateContent({
    contents: [{ role: 'user', parts: [{ text: SCRIPT_GENERATION_PROMPT(topic, hook, vibe) }] }],
    generationConfig: {
      responseMimeType: 'application/json',
      responseSchema: SCRIPT_SCHEMA as Schema,
    },
  });

  const parsed = JSON.parse(result.response.candidates[0].content.parts[0].text || '{}');

  // Add status to all shots
  parsed.chapters.forEach((chapter: any) => {
    chapter.shots.forEach((shot: any) => {
      shot.status = 'pending';
    });
  });

  return parsed as ViralVideoScript;
};

/**
 * Refine an existing script based on user feedback
 */
export const refineScript = async (
  script: ViralVideoScript,
  feedback: string
): Promise<ViralVideoScript> => {
  const model = ai.getGenerativeModel({ model: 'gemini-1.5-flash' });
  const result = await model.generateContent({
    contents: [{ role: 'user', parts: [{ text: REFINEMENT_PROMPT(script, feedback) }] }],
    generationConfig: {
      responseMimeType: 'application/json',
      responseSchema: SCRIPT_SCHEMA as Schema, // Re-use the same schema
    },
  });

  return JSON.parse(
    result.response.candidates[0].content.parts[0].text || '{}'
  ) as ViralVideoScript;
};

/**
 * Fetch viral topic suggestions
 */
export const fetchSuggestions = async (): Promise<string[]> => {
  const model = ai.getGenerativeModel({ model: 'gemini-1.5-flash' });
  const result = await model.generateContent({
    contents: [{ role: 'user', parts: [{ text: SUGGESTIONS_PROMPT }] }],
    generationConfig: {
      responseMimeType: 'application/json',
      responseSchema: { type: 'array', items: { type: 'string' } } as Schema,
    },
  });

  return JSON.parse(result.response.candidates[0].content.parts[0].text || '[]');
};

/**
 * Fetch viral hooks for a specific topic
 */
export const fetchHooks = async (
  topic: string
): Promise<{ hooks: string[]; vibe: ProductionVibe }> => {
  const model = ai.getGenerativeModel({ model: 'gemini-1.5-flash' });
  const result = await model.generateContent({
    contents: [{ role: 'user', parts: [{ text: HOOKS_PROMPT(topic) }] }],
    generationConfig: {
      responseMimeType: 'application/json',
      responseSchema: {
        type: 'object',
        properties: {
          hooks: { type: 'array', items: { type: 'string' } },
          vibe: { type: 'string' },
        },
        required: ['hooks', 'vibe'],
      } as Schema,
    },
  });

  return JSON.parse(
    result.response.candidates[0].content.parts[0].text || '{"hooks": [], "vibe": "minimal"}'
  );
};
