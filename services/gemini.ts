
import { GoogleGenAI, Type } from "@google/genai";
import { ViralVideoScript, ProductionVibe, Shot } from "../types.ts";

// Vibe configurations
const VIBE_CONFIGS: Record<ProductionVibe, { gradient: string, music: string, color: string }> = {
  cosmic: { gradient: "Deep Space (Indigo #1e1b4b to Black #000000)", music: "https://cdn.pixabay.com/audio/2022/02/10/audio_097486411d.mp3", color: "#6366f1" },
  hype: { gradient: "Electric Neon (Purple #7e22ce to Cyan #0891b2)", music: "https://cdn.pixabay.com/audio/2021/08/04/audio_10860570b5.mp3", color: "#a855f7" },
  minimal: { gradient: "Clean Studio (Light Blue #f0f9ff to White #ffffff)", music: "https://cdn.pixabay.com/audio/2022/05/27/audio_18087374a6.mp3", color: "#3b82f6" },
  suspense: { gradient: "Noir Shadow (Dark Gray #18181b to Black #000000)", music: "https://cdn.pixabay.com/audio/2022/03/10/audio_c8c8a14e1f.mp3", color: "#4b5563" },
  success: { gradient: "Royal Gold (Amber #d97706 to Orange #ea580c)", music: "https://cdn.pixabay.com/audio/2022/01/18/audio_d0c6ff11bd.mp3", color: "#f59e0b" }
};

export const getStyleGuide = (vibe: ProductionVibe): string => `
STYLE: Ultra-Minimalist 3D Digital Art / Cel-shaded.
CHARACTER: A 'puffy' volumetric white character. Think 'Baymax' but as a stick figure. Smooth, rounded 3D limbs. Large spherical head with two tiny black dot eyes.
TEXTURE: Soft matte plastic / clay. Subsurface scattering enabled.
SHADING: High-contrast ambient occlusion. Soft drop shadows underneath the character to ground them in 3D space.
OUTLINE: Clean, bold 8px black ink stroke around everything.
ENVIRONMENT: Vertical background gradient using ${VIBE_CONFIGS[vibe].gradient}. Absolutely no complex textures or background objects.
`;

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

/**
 * Generate a complete viral 5-minute video script optimized for retention
 */
export const generateViralVideoScript = async (
  topic: string,
  hook: string,
  vibe: ProductionVibe
): Promise<ViralVideoScript> => {

  const response = await ai.models.generateContent({
    model: 'gemini-1.5-flash', // CORRECTED: Using real model
    contents: `You are an expert viral YouTube video scriptwriter specializing in 5-minute educational explainers in the style of Kurzgesagt and CGP Grey.

TOPIC: "${topic}"
HOOK: "${hook}"
VIBE: ${vibe}

Create a complete 5-minute (300 second) video script following viral best practices:

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
- 150-180 total shots (average 1.7-2 seconds each)
- Every shot needs a PURPOSE (advance story, reveal info, create emotion)
- Mix of character reactions, visual metaphors, stats, comparisons
- 90% of shots should have text overlays (viewers watch on mute)

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

Return comprehensive JSON.`,

    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          metadata: {
            type: Type.OBJECT,
            properties: {
              topic: { type: Type.STRING },
              hook: { type: Type.STRING },
              title: {
                type: Type.STRING,
                description: "Viral video title, under 60 chars"
              },
              vibe: { type: Type.STRING },
              targetDuration: { type: Type.NUMBER },
              thumbnailConcept: {
                type: Type.OBJECT,
                properties: {
                  mainElement: { type: Type.STRING },
                  emotion: { type: Type.STRING },
                  text: { type: Type.STRING },
                  colorScheme: { type: Type.STRING }
                },
                required: ["mainElement", "emotion", "text", "colorScheme"]
              }
            },
            required: ["topic", "hook", "title", "vibe", "targetDuration", "thumbnailConcept"]
          },

          chapters: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                chapterNumber: { type: Type.NUMBER },
                title: { type: Type.STRING },
                timestamp: { type: Type.STRING },
                duration: { type: Type.NUMBER },
                purpose: { type: Type.STRING },
                narration: {
                  type: Type.STRING,
                  description: "Full narration text for this chapter"
                },
                keyMessage: { type: Type.STRING },
                emotionalTone: { type: Type.STRING },

                shots: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      shotNumber: { type: Type.NUMBER },
                      duration: { type: Type.NUMBER },

                      // Visual elements
                      characterAction: { type: Type.STRING },
                      characterEmotion: { type: Type.STRING },
                      cameraAngle: { type: Type.STRING },
                      cameraMovement: { type: Type.STRING },
                      backgroundStyle: { type: Type.STRING },

                      // Imagen 3 prompt
                      imagenPrompt: {
                        type: Type.STRING,
                        description: "Detailed visual description for Imagen 3 generation"
                      },

                      // Text overlay
                      textOverlay: {
                        type: Type.OBJECT,
                        properties: {
                          text: { type: Type.STRING },
                          position: { type: Type.STRING },
                          style: { type: Type.STRING },
                          animation: { type: Type.STRING },
                          timing: {
                            type: Type.OBJECT,
                            properties: {
                              delay: { type: Type.NUMBER },
                              duration: { type: Type.NUMBER },
                              fadeOut: { type: Type.NUMBER }
                            },
                            required: ["delay", "duration", "fadeOut"]
                          },
                          highlightWords: {
                            type: Type.ARRAY,
                            items: { type: Type.STRING }
                          },
                          fontSize: { type: Type.STRING }
                        },
                        required: ["text", "position", "style", "animation", "timing"]
                      },

                      // Animation
                      animation: { type: Type.STRING },
                      transition: { type: Type.STRING }
                    },
                    required: [
                      "shotNumber",
                      "duration",
                      "characterAction",
                      "characterEmotion",
                      "cameraAngle",
                      "cameraMovement",
                      "backgroundStyle",
                      "imagenPrompt",
                      "animation",
                      "transition"
                    ]
                  }
                }
              },
              required: [
                "chapterNumber",
                "title",
                "timestamp",
                "duration",
                "purpose",
                "narration",
                "keyMessage",
                "emotionalTone",
                "shots"
              ]
            }
          },

          openLoops: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                question: { type: Type.STRING },
                posedAt: { type: Type.NUMBER },
                resolvedAt: { type: Type.NUMBER },
                intensity: { type: Type.STRING }
              },
              required: ["question", "posedAt", "resolvedAt", "intensity"]
            }
          },

          retentionBombs: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                timestamp: { type: Type.NUMBER },
                type: { type: Type.STRING },
                content: { type: Type.STRING },
                shotNumbers: {
                  type: Type.ARRAY,
                  items: { type: Type.NUMBER }
                }
              },
              required: ["timestamp", "type", "content", "shotNumbers"]
            }
          },

          totalShots: {
            type: Type.NUMBER,
            description: "Total number of shots in the video"
          },

          estimatedCost: {
            type: Type.NUMBER,
            description: "Estimated cost in USD to generate this video"
          }
        },
        required: ["metadata", "chapters", "openLoops", "retentionBombs", "totalShots", "estimatedCost"]
      }
    }
  });

  const parsed = JSON.parse(response.text || '{}');

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

  const response = await ai.models.generateContent({
    model: 'gemini-1.5-flash',
    contents: `You are refining a viral video script based on user feedback.

CURRENT SCRIPT:
${JSON.stringify(script, null, 2)}

USER FEEDBACK:
"${feedback}"

Refine the script while maintaining its structure and viral elements. Return the improved version as JSON using the same schema.`,

    config: {
      responseMimeType: "application/json",
      responseSchema: {
        // Same schema as generateViralVideoScript
        type: Type.OBJECT,
        properties: {
          metadata: { type: Type.OBJECT },
          chapters: { type: Type.ARRAY },
          openLoops: { type: Type.ARRAY },
          retentionBombs: { type: Type.ARRAY },
          totalShots: { type: Type.NUMBER },
          estimatedCost: { type: Type.NUMBER }
        }
      }
    }
  });

  return JSON.parse(response.text || '{}') as ViralVideoScript;
};

/**
 * Fetch viral topic suggestions
 */
export const fetchSuggestions = async (): Promise<string[]> => {
  const response = await ai.models.generateContent({
    model: 'gemini-1.5-flash',
    contents: `List 6 viral educational explainer topics about psychology, science, or technology that would get millions of views. Topics that make people say "Wait, what?!"

Examples:
- "Your Brain Deletes 90% of Your Memories Every Night"
- "The One Color Humans Can't Actually See"
- "Why Time Speeds Up As You Age (It's Not What You Think)"

Return a JSON array of 6 topics.`,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.ARRAY,
        items: { type: Type.STRING }
      }
    }
  });

  return JSON.parse(response.text || '[]');
};

/**
 * Fetch viral hooks for a specific topic
 */
export const fetchHooks = async (topic: string): Promise<{ hooks: string[], vibe: ProductionVibe }> => {
  const response = await ai.models.generateContent({
    model: 'gemini-1.5-flash',
    contents: `Analyze: "${topic}".

1. Suggest 3 viral "hook" titles that would make people click (short, clickable, curiosity-driven)
2. Choose the best Production Vibe for this topic: cosmic, hype, minimal, suspense, or success

Return JSON.`,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          hooks: {
            type: Type.ARRAY,
            items: { type: Type.STRING }
          },
          vibe: { type: Type.STRING }
        },
        required: ["hooks", "vibe"]
      }
    }
  });

  return JSON.parse(response.text || '{"hooks": [], "vibe": "minimal"}');
};

// Export vibe configs and style guide for use in other services
export { VIBE_CONFIGS };
export const MUSIC_FOR_VIBE = (vibe: ProductionVibe) => VIBE_CONFIGS[vibe]?.music || VIBE_CONFIGS.minimal.music;
