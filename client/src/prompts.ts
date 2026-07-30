import { SchemaType } from '@google/generative-ai';
import { ProductionVibe } from './types';

// Vibe configurations
const VIBE_CONFIGS: Record<ProductionVibe, { gradient: string; music: string; color: string }> = {
  cosmic: {
    gradient: 'Deep Space (Indigo #1e1b4b to Black #000000)',
    music: 'https://cdn.pixabay.com/audio/2022/02/10/audio_097486411d.mp3',
    color: '#6366f1',
  },
  hype: {
    gradient: 'Electric Neon (Purple #7e22ce to Cyan #0891b2)',
    music: 'https://cdn.pixabay.com/audio/2021/08/04/audio_10860570b5.mp3',
    color: '#a855f7',
  },
  minimal: {
    gradient: 'Clean Studio (Light Blue #f0f9ff to White #ffffff)',
    music: 'https://cdn.pixabay.com/audio/2022/05/27/audio_18087374a6.mp3',
    color: '#3b82f6',
  },
  suspense: {
    gradient: 'Noir Shadow (Dark Gray #18181b to Black #000000)',
    music: 'https://cdn.pixabay.com/audio/2022/03/10/audio_c8c8a14e1f.mp3',
    color: '#4b5563',
  },
  success: {
    gradient: 'Royal Gold (Amber #d97706 to Orange #ea580c)',
    music: 'https://cdn.pixabay.com/audio/2022/01/18/audio_d0c6ff11bd.mp3',
    color: '#f59e0b',
  },
};

export const getStyleGuide = (vibe: ProductionVibe): string => `
STYLE: Ultra-Minimalist 3D Digital Art / Cel-shaded.
CHARACTER: A 'puffy' volumetric white character. Think 'Baymax' but as a stick figure. Smooth, rounded 3D limbs. Large spherical head with two tiny black dot eyes.
TEXTURE: Soft matte plastic / clay. Subsurface scattering enabled.
SHADING: High-contrast ambient occlusion. Soft drop shadows underneath the character to ground them in 3D space.
OUTLINE: Clean, bold 8px black ink stroke around everything.
ENVIRONMENT: Vertical background gradient using ${VIBE_CONFIGS[vibe].gradient}. Absolutely no complex textures or background objects.
`;

export const SCRIPT_GENERATION_PROMPT = (
  topic: string,
  hook: string,
  vibe: ProductionVibe
) => `You are an expert viral YouTube video scriptwriter.

TOPIC: "${topic}"
HOOK: "${hook}"
VIBE: ${vibe}

Create a SHORT 2-minute (120 second) video script with 30-40 shots total:

RETENTION STRUCTURE:
- Cold Open (0-3s): Immediate shock/curiosity. "Wait... WHAT?!" moment
- Setup (3-30s): Establish the problem/question
- Body with Retention Bombs (30-270s): Place hooks every 30 seconds
  * 30s: "And it gets worse..."
  * 60s: "But wait, there's a twist..."
  * 90s: "Here's what scientists discovered..."
  * 120s: "This changes everything..."
  * 180s: "The implications are insane..."
  * 240s: "And here's the crazy part..."
- Resolution (270-290s): Answer the main question
- CTA + Cliffhanger (290-300s): "But there's ONE exception..." (tease next video)

SHOT REQUIREMENTS:
- 30-40 total shots (average 3 seconds each)
- Every shot needs a PURPOSE
- Mix of character reactions, visual metaphors
- 50% of shots should have text overlays

VISUAL STYLE (for each shot):
- Puffy 3D stick figure character (Baymax-style)
- Different emotions: neutral, excited, shocked, confused, thoughtful, etc.
- Camera angles: wide, medium, close-up, top-down, etc.
- Actions: thinking, pointing, jumping, holding objects, etc.
- Simple gradient backgrounds only
- Bold 8px black outlines

OPEN LOOPS (create curiosity):
- Pose 3-5 questions in first 60 seconds
- Resolve gradually throughout video
- Example: "What if I told you your brain is lying to you?" (resolve at 2:15)

VIDEO TITLE (use curiosity formula):
- [Number/Statement] + [Unexpected Element] + [Curiosity Gap]
- Under 60 characters
- Examples:
  * "Your Brain Deletes This Every Night (And It's Saving Your Life)"
  * "Scientists Found The One Memory Your Brain Can't Delete"
  * "Why 90% Of Your Memories Vanish (On Purpose)"

THUMBNAIL CONCEPT:
- Main visual: Character with strong emotion (shocked/curious)
- Text: 3-5 words max, ALL CAPS
- High contrast colors
- Example: "90% DELETED" with shocked character

Return comprehensive JSON.`;

export const SCRIPT_SCHEMA = {
  type: SchemaType.OBJECT,
  properties: {
    metadata: {
      type: SchemaType.OBJECT,
      properties: {
        topic: { type: SchemaType.STRING },
        hook: { type: SchemaType.STRING },
        title: {
          type: SchemaType.STRING,
          description: '60 chars max, curiosity-driven',
        },
        vibe: { type: SchemaType.STRING },
        targetDuration: { type: SchemaType.NUMBER },
        thumbnailConcept: {
          type: SchemaType.OBJECT,
          properties: {
            mainElement: { type: SchemaType.STRING },
            emotion: { type: SchemaType.STRING },
            text: { type: SchemaType.STRING },
            colorScheme: { type: SchemaType.STRING },
          },
          required: ['mainElement', 'emotion', 'text', 'colorScheme'],
        },
      },
      required: ['topic', 'hook', 'title', 'vibe', 'targetDuration', 'thumbnailConcept'],
    },

    chapters: {
      type: SchemaType.ARRAY,
      items: {
        type: SchemaType.OBJECT,
        properties: {
          chapterNumber: { type: SchemaType.NUMBER },
          title: { type: SchemaType.STRING },
          timestamp: { type: SchemaType.STRING },
          duration: { type: SchemaType.NUMBER },
          purpose: { type: SchemaType.STRING },
          narration: {
            type: SchemaType.STRING,
            description: 'Natural, conversational narration text',
          },
          keyMessage: { type: SchemaType.STRING },
          emotionalTone: { type: SchemaType.STRING },

          shots: {
            type: SchemaType.ARRAY,
            items: {
              type: SchemaType.OBJECT,
              properties: {
                shotNumber: { type: SchemaType.NUMBER },
                duration: { type: SchemaType.NUMBER },

                // Visual elements
                characterAction: { type: SchemaType.STRING },
                characterEmotion: { type: SchemaType.STRING },
                cameraAngle: { type: SchemaType.STRING },
                cameraMovement: { type: SchemaType.STRING },
                backgroundStyle: { type: SchemaType.STRING },

                // AI generation prompt
                imagenPrompt: {
                  type: SchemaType.STRING,
                  description: 'Detailed Imagen 3 prompt for this specific shot',
                },

                // Text overlay (optional)
                textOverlay: {
                  type: SchemaType.OBJECT,
                  properties: {
                    text: { type: SchemaType.STRING },
                    position: { type: SchemaType.STRING },
                    style: { type: SchemaType.STRING },
                    animation: { type: SchemaType.STRING },
                    timing: {
                      type: SchemaType.OBJECT,
                      properties: {
                        delay: { type: SchemaType.NUMBER },
                        duration: { type: SchemaType.NUMBER },
                        fadeOut: { type: SchemaType.NUMBER },
                      },
                    },
                    highlightWords: {
                      type: SchemaType.ARRAY,
                      items: { type: SchemaType.STRING },
                    },
                    fontSize: { type: SchemaType.STRING },
                  },
                },

                // Animation and transitions
                animation: { type: SchemaType.STRING },
                transition: { type: SchemaType.STRING },
              },
              required: [
                'shotNumber',
                'duration',
                'characterAction',
                'characterEmotion',
                'cameraAngle',
                'cameraMovement',
                'backgroundStyle',
                'imagenPrompt',
                'animation',
                'transition',
              ],
            },
          },
        },
        required: [
          'chapterNumber',
          'title',
          'timestamp',
          'duration',
          'purpose',
          'narration',
          'keyMessage',
          'emotionalTone',
          'shots',
        ],
      },
    },

    openLoops: {
      type: SchemaType.ARRAY,
      items: {
        type: SchemaType.OBJECT,
        properties: {
          question: { type: SchemaType.STRING },
          posedAt: { type: SchemaType.NUMBER },
          resolvedAt: { type: SchemaType.NUMBER },
          intensity: { type: SchemaType.STRING },
        },
        required: ['question', 'posedAt', 'resolvedAt', 'intensity'],
      },
    },

    retentionBombs: {
      type: SchemaType.ARRAY,
      items: {
        type: SchemaType.OBJECT,
        properties: {
          timestamp: { type: SchemaType.NUMBER },
          type: { type: SchemaType.STRING },
          content: { type: SchemaType.STRING },
          shotNumbers: {
            type: SchemaType.ARRAY,
            items: { type: SchemaType.NUMBER },
          },
        },
        required: ['timestamp', 'type', 'content', 'shotNumbers'],
      },
    },

    totalShots: {
      type: SchemaType.NUMBER,
      description: 'Total number of shots in the video',
    },

    estimatedCost: {
      type: SchemaType.NUMBER,
      description: 'Estimated cost in USD to generate all images and audio',
    },
  },
  required: ['metadata', 'chapters', 'openLoops', 'retentionBombs', 'totalShots', 'estimatedCost'],
};
