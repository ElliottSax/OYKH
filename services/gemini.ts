
import { GoogleGenAI, Type, Modality } from "@google/genai";
import { Scene } from "../types";

const VISUAL_STYLE_PROMPT = `
Style Guide: Ultra-clean professional vector-style educational illustration.
- Character: A minimalist white stick figure.
- Anatomy: A perfectly round, clearly defined white head with exactly two small black dot eyes. The body and limbs are slightly thicker than typical stick figures for better balance and visibility.
- Outlines: Thick, heavy, uniform black vector outlines for every single element.
- Shading: Heavy, stylized black or dark gray shadows placed beneath and around the character and objects to create depth and contrast.
- Reflections: Pronounced light reflections consisting of small white curved lines or circles placed strictly in the upper right quadrant of all major shapes (head, body parts, and objects).
- Background: A modern, vibrant mixed-color gradient background.
- Pose: The character must always be in a dynamic, expressive pose that conveys action or instruction.
- Objects: Any tools, items, or background elements must share the exact same thick black outlines, heavy shadows, and upper-right white reflection treatment.
- Composition: 16:9 aspect ratio, clean, professional, and balanced for high-quality video frames.
`;

export const generateScript = async (topic: string): Promise<Scene[]> => {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: `Write a high-tier, 60-scene instructional "How-To" script for a 10-minute video titled: "${topic}". 
    The video is for the premium "Once You Know How" channel. The tone must be snappy, professional, and authoritative.
    
    Instructional Structure:
    1. Scenes 1-5: The Hook. Display the complex outcome. It looks impossible.
    2. Scenes 6-15: Setup. The tools and the secret "How-To" mindset.
    3. Scenes 16-45: Step-by-Step execution. Rapid-fire, clear instructional phases.
    4. Scenes 46-55: Pro-tips. The subtle nuances that separate amateurs from pros.
    5. Scenes 56-60: The "Aha!" moment and Final Mastery.
    
    Catchphrase Integration:
    - Use the phrase "once you know how" at least 8 times throughout the production.
    - The VERY LAST line of Scene 60 must be: "once you know how".
    
    Script Length: Each scene's narration should be approximately 15-20 words to fit a 10-second window.
    
    Visual Style: ${VISUAL_STYLE_PROMPT}.
    
    Return a JSON array of 60 scene objects.`,
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
          },
          required: ["title", "script", "visualDescription"],
        }
      }
    }
  });

  try {
    const scenes = JSON.parse(response.text || "[]");
    return scenes.map((s: any) => ({ ...s, status: 'idle' }));
  } catch (e) {
    console.error("Failed to parse long-form script", e);
    throw new Error("Could not generate the full 60-scene production. Try a simpler topic.");
  }
};

export const generateSceneImage = async (scene: Scene): Promise<string> => {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  const prompt = `${VISUAL_STYLE_PROMPT} 
  Visual Scenario: ${scene.visualDescription}. 
  Focus on: dynamic stick figure pose, round head, dot eyes, thick lines, heavy shadows, and white circles/lines in the upper right quadrant of shapes. Objects must match style.`;
  
  const response = await ai.models.generateContent({
    model: 'gemini-3-pro-image-preview',
    contents: { parts: [{ text: prompt }] },
    config: {
      imageConfig: { aspectRatio: "16:9", imageSize: "1K" }
    }
  });

  const part = response.candidates?.[0]?.content?.parts.find(p => p.inlineData);
  if (part?.inlineData?.data) {
    return part.inlineData.data;
  }
  throw new Error("Image generation failed");
};

export const generateSceneVideo = async (scene: Scene, imageData: string): Promise<string> => {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  
  let operation = await ai.models.generateVideos({
    model: 'veo-3.1-fast-generate-preview',
    prompt: `Professional educational animation: ${scene.visualDescription}. High-end vector movement. Crisp lines, maintain stylized heavy shadows and upper-right reflections. Duration: 10 seconds. Smooth instructional motion.`,
    image: {
      imageBytes: imageData,
      mimeType: 'image/png',
    },
    config: {
      numberOfVideos: 1,
      resolution: '720p',
      aspectRatio: '16:9'
    }
  });

  while (!operation.done) {
    await new Promise(resolve => setTimeout(resolve, 8000));
    operation = await ai.operations.getVideosOperation({ operation: operation });
  }

  const downloadLink = operation.response?.generatedVideos?.[0]?.video?.uri;
  if (!downloadLink) throw new Error("Video generation failed");

  const videoResponse = await fetch(`${downloadLink}&key=${process.env.API_KEY}`);
  const blob = await videoResponse.blob();
  return URL.createObjectURL(blob);
};

export const generateSceneAudio = async (scene: Scene): Promise<string> => {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash-preview-tts",
    contents: [{ parts: [{ text: `Narrate with a crisp, snappy, professional instructor voice. Pace for exactly 10 seconds. Script: ${scene.script}` }] }],
    config: {
      responseModalities: [Modality.AUDIO],
      speechConfig: {
        voiceConfig: {
          prebuiltVoiceConfig: { voiceName: 'Puck' },
        },
      },
    },
  });

  const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
  if (!base64Audio) throw new Error("Audio generation failed");

  const audioBytes = decode(base64Audio);
  const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
  const audioBuffer = await decodeAudioData(audioBytes, audioContext, 24000, 1);
  
  return audioBufferToWavUrl(audioBuffer);
};

function decode(base64: string) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

function audioBufferToWavUrl(buffer: AudioBuffer): string {
  const numOfChan = buffer.numberOfChannels,
    length = buffer.length * numOfChan * 2 + 44,
    bufferArr = new ArrayBuffer(length),
    view = new DataView(bufferArr),
    channels = [],
    sampleRate = buffer.sampleRate;
  let offset = 0, pos = 0;

  function setUint16(data: number) { view.setUint16(pos, data, true); pos += 2; }
  function setUint32(data: number) { view.setUint32(pos, data, true); pos += 4; }

  setUint32(0x46464952);                         
  setUint32(length - 8);                         
  setUint32(0x45564157);                         
  setUint32(0x20746d66);                         
  setUint32(16);                                 
  setUint16(1);                                  
  setUint16(numOfChan);
  setUint32(sampleRate);
  setUint32(sampleRate * 2 * numOfChan);         
  setUint16(numOfChan * 2);                      
  setUint16(16);                                 
  setUint32(0x61746164);                         
  setUint32(length - pos - 4);                   

  for (let i = 0; i < buffer.numberOfChannels; i++) channels.push(buffer.getChannelData(i));

  while (pos < length) {
    for (let i = 0; i < numOfChan; i++) {
      let sample = Math.max(-1, Math.min(1, channels[i][offset]));
      sample = (sample < 0 ? sample * 0x8000 : sample * 0x7FFF);
      view.setInt16(pos, sample, true);
      pos += 2;
    }
    offset++;
  }

  const blob = new Blob([bufferArr], { type: 'audio/wav' });
  return URL.createObjectURL(blob);
}
