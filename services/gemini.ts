
import { GoogleGenAI, Type, Modality } from "@google/genai";
import { Scene, ProductionVibe } from "../types.ts";

const VIBE_CONFIGS: Record<ProductionVibe, { gradient: string, music: string, color: string }> = {
  cosmic: { gradient: "Deep Space (Indigo #1e1b4b to Black #000000)", music: "https://cdn.pixabay.com/audio/2022/02/10/audio_097486411d.mp3", color: "#6366f1" },
  hype: { gradient: "Electric Neon (Purple #7e22ce to Cyan #0891b2)", music: "https://cdn.pixabay.com/audio/2021/08/04/audio_10860570b5.mp3", color: "#a855f7" },
  minimal: { gradient: "Clean Studio (Light Blue #f0f9ff to White #ffffff)", music: "https://cdn.pixabay.com/audio/2022/05/27/audio_18087374a6.mp3", color: "#3b82f6" },
  suspense: { gradient: "Noir Shadow (Dark Gray #18181b to Black #000000)", music: "https://cdn.pixabay.com/audio/2022/03/10/audio_c8c8a14e1f.mp3", color: "#4b5563" },
  success: { gradient: "Royal Gold (Amber #d97706 to Orange #ea580c)", music: "https://cdn.pixabay.com/audio/2022/01/18/audio_d0c6ff11bd.mp3", color: "#f59e0b" }
};

const getStyleGuide = (vibe: ProductionVibe) => `
STYLE: Ultra-Minimalist 3D Digital Art / Cel-shaded.
CHARACTER: A 'puffy' volumetric white character. Think 'Baymax' but as a stick figure. Smooth, rounded 3D limbs. Large spherical head with two tiny black dot eyes. 
TEXTURE: Soft matte plastic / clay. Subsurface scattering enabled.
SHADING: High-contrast ambient occlusion. Soft drop shadows underneath the character to ground them in 3D space.
OUTLINE: Clean, bold 8px black ink stroke around everything.
ENVIRONMENT: Vertical background gradient using ${VIBE_CONFIGS[vibe].gradient}. Absolutely no complex textures or background objects.
`;

const MOCK_AUDIO_SILENT = "data:audio/wav;base64,UklGRjIAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YRAAAAAAAAAAAAAAAAAAAAAAAAAA";

function pcmToWav(pcmBase64: string): string {
  try {
    const binaryString = atob(pcmBase64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    const wavHeader = new ArrayBuffer(44);
    const view = new DataView(wavHeader);
    
    // RIFF identifier
    view.setUint32(0, 0x52494646, false); 
    // File length
    view.setUint32(4, 36 + len, true);    
    // RIFF type
    view.setUint32(8, 0x57415645, false); 
    // Format chunk identifier
    view.setUint32(12, 0x666d7420, false); 
    // Format chunk length
    view.setUint32(16, 16, true);
    // Sample format (PCM)
    view.setUint16(20, 1, true);           
    // Channels
    view.setUint16(22, 1, true);           
    // Sample rate
    view.setUint32(24, 24000, true);       
    // Byte rate
    view.setUint32(28, 24000 * 2, true);   
    // Block align
    view.setUint16(32, 2, true);           
    // Bits per sample
    view.setUint16(34, 16, true);          
    // Data chunk identifier
    view.setUint32(36, 0x64617461, false); 
    // Data chunk length
    view.setUint32(40, len, true);         
    
    const blob = new Blob([wavHeader, bytes], { type: 'audio/wav' });
    return URL.createObjectURL(blob);
  } catch (e) {
    console.error("PCM2WAV conversion error:", e);
    return MOCK_AUDIO_SILENT;
  }
}

export const fetchHooks = async (topic: string): Promise<{ hooks: string[], vibe: ProductionVibe }> => {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: `Analyze: "${topic}". 
    1. Suggest 3 viral "Hook" titles (short, clickable).
    2. Choose the best Production Vibe: cosmic, hype, minimal, suspense, success.
    Return JSON.`,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          hooks: { type: Type.ARRAY, items: { type: Type.STRING } },
          vibe: { type: Type.STRING }
        },
        required: ["hooks", "vibe"]
      }
    }
  });
  return JSON.parse(response.text || "{}");
};

export const fetchSuggestions = async (): Promise<string[]> => {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: "List 6 viral explainer topics about psychology or science. JSON array.",
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.ARRAY,
        items: { type: Type.STRING }
      }
    }
  });
  return JSON.parse(response.text || "[]");
};

export const generateScript = async (topic: string, hook: string, vibe: ProductionVibe, isMock: boolean = false): Promise<Scene[]> => {
  if (isMock) {
    await new Promise(r => setTimeout(r, 600));
    const mockTitles = ["GENESIS", "PROBLEM", "TWIST", "SOLUTION", "FUTURE", "FINAL"];
    return mockTitles.map((t, i) => ({
      title: `Scene ${i+1}`,
      script: `Welcome to this exploration of ${topic}. This is point number ${i+1}.`,
      visualDescription: `Puffy character demonstrating ${t} in a ${vibe} background.`,
      screenText: t,
      status: 'idle'
    }));
  }

  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: `Script for "${topic}" with hook "${hook}". Vibe: ${vibe}. 6 scenes, 12 words each. 3.5D puffy stick figure. JSON.`,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            title: { type: Type.STRING },
            script: { type: Type.STRING },
            visualDescription: { type: Type.STRING },
            screenText: { type: Type.STRING },
          },
          required: ["title", "script", "visualDescription", "screenText"],
        }
      }
    }
  });
  return JSON.parse(response.text || "[]").map((s: any) => ({ ...s, status: 'idle' }));
};

export const generateSceneImage = async (scene: Scene, vibe: ProductionVibe, isMock: boolean = false): Promise<string> => {
  if (isMock) {
    await new Promise(r => setTimeout(r, 400));
    const hex = VIBE_CONFIGS[vibe]?.color.replace('#','') || '3b82f6';
    return `https://placehold.co/1920x1080/${hex}/ffffff?text=${encodeURIComponent(scene.screenText)}`;
  }

  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  const prompt = `${getStyleGuide(vibe)} SCENE: ${scene.visualDescription}. Extremely clean, minimalist masterpiece.`;
  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash-image',
    contents: { parts: [{ text: prompt }] },
    config: { imageConfig: { aspectRatio: "16:9" } }
  });
  for (const part of response.candidates?.[0].content.parts || []) {
    if (part.inlineData) return `data:image/png;base64,${part.inlineData.data}`;
  }
  throw new Error("IMG_GEN_FAIL");
};

export const generateSceneAudio = async (scene: Scene, isMock: boolean = false): Promise<string> => {
  if (isMock) return MOCK_AUDIO_SILENT;
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash-preview-tts",
    contents: [{ parts: [{ text: `Speak with high intensity: ${scene.script}` }] }],
    config: {
      responseModalities: [Modality.AUDIO],
      speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Puck' } } },
    },
  });
  const data = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
  if (!data) throw new Error("TTS_GEN_FAIL");
  return pcmToWav(data);
};

export const MUSIC_FOR_VIBE = (vibe: ProductionVibe) => VIBE_CONFIGS[vibe]?.music || VIBE_CONFIGS.minimal.music;
