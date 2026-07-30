import { GoogleVoice, ViralChapter } from '../types.ts';

/**
 * Google Cloud Text-to-Speech Service
 *
 * NOTE: This is a client-side placeholder. For production:
 * 1. Move this to a backend Cloud Function
 * 2. Use @google-cloud/text-to-speech package
 * 3. Keep API credentials server-side
 */

// Mock silent audio for development
const MOCK_AUDIO_SILENT =
  'data:audio/wav;base64,UklGRjIAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YRAAAAAAAAAAAAAAAAAAAAAAAAAA';

/**
 * Generate narration audio using Google Cloud TTS
 *
 * Supports both text strings and chapter arrays
 * PRODUCTION VERSION (Backend - Cloud Function):
 */
export async function generateNarration(
  textOrChapters: string | ViralChapter[],
  voiceName: GoogleVoice = 'en-US-Journey-D',
  isMock: boolean = true
): Promise<string> {
  // Handle chapter array input (for job manager compatibility)
  if (Array.isArray(textOrChapters)) {
    const chapters = textOrChapters;
    const fullText = chapters.map((c) => c.narration).join('\n\n');

    console.log(`[Google TTS] Generating audio for ${chapters.length} chapters`);
    console.log(`[Google TTS] Total text length: ${fullText.length} characters`);

    return await generateNarrationFromText(fullText, voiceName, isMock);
  }

  // Handle text string input
  return await generateNarrationFromText(textOrChapters, voiceName, isMock);
}

/**
 * Internal: Generate audio from text string
 */
async function generateNarrationFromText(
  text: string,
  voiceName: GoogleVoice,
  isMock: boolean
): Promise<string> {
  if (isMock) {
    console.log('[Google TTS Mock] Simulating audio generation...');
    await new Promise((r) => setTimeout(r, 800));
    return MOCK_AUDIO_SILENT;
  }

  // For PRODUCTION: This should call your backend proxy
  // const response = await fetch('/api/generate-audio', {
  //   method: 'POST',
  //   headers: { 'Content-Type': 'application/json' },
  //   body: JSON.stringify({ text, voiceName })
  // });
  // const { audioUrl } = await response.json();
  // return audioUrl;

  // DEVELOPMENT VERSION (Mock):
  console.log('[Google TTS] Generating audio');
  console.log('[Google TTS] Voice:', voiceName);
  console.log('[Google TTS] Text length:', text.length, 'characters');

  // Simulate API delay
  await new Promise((r) => setTimeout(r, 800));

  return MOCK_AUDIO_SILENT;
}

/**
 * Generate audio for all chapters in a video
 */
export const generateChapterAudio = async (
  chapters: ViralChapter[],
  voiceName: GoogleVoice = 'en-US-Journey-D',
  isMock: boolean = false,
  onProgress?: (completed: number, total: number) => void
): Promise<{ audioMap: Map<number, string>; failedCount: number }> => {
  const audioMap = new Map<number, string>();
  let failedCount = 0;

  for (let i = 0; i < chapters.length; i++) {
    const chapter = chapters[i];

    try {
      const audioUrl = await generateNarration(chapter.narration, voiceName, isMock);
      audioMap.set(chapter.chapterNumber, audioUrl);

      if (onProgress) {
        onProgress(i + 1, chapters.length);
      }
    } catch (error) {
      console.error(`Failed to generate audio for chapter ${chapter.chapterNumber}:`, error);
      // Use silent audio as fallback
      audioMap.set(chapter.chapterNumber, MOCK_AUDIO_SILENT);
      failedCount++;
    }
  }

  return { audioMap, failedCount };
};

/**
 * Get voice display name for UI
 */
export const getVoiceDisplayName = (voiceName: GoogleVoice): string => {
  const names: Record<GoogleVoice, string> = {
    'en-US-Journey-D': 'Journey D (Energetic Male)',
    'en-US-Journey-F': 'Journey F (Warm Female)',
    'en-US-Journey-O': 'Journey O (Authoritative)',
    'en-US-Studio-M': 'Studio M (Documentary Male)',
    'en-US-Studio-O': 'Studio O (Professional Female)',
    'en-US-Neural2-D': 'Neural2 D (Friendly Male)',
    'en-US-Neural2-F': 'Neural2 F (Clear Female)',
  };

  return names[voiceName] || voiceName;
};

/**
 * Estimate audio duration (for planning)
 * Rough estimate: ~150 words per minute, ~5 characters per word
 */
export const estimateAudioDuration = (text: string): number => {
  const characters = text.length;
  const words = characters / 5; // Average word length
  const minutes = words / 150; // Words per minute
  return minutes * 60; // Convert to seconds
};

/* ============================================
   BACKEND CLOUD FUNCTION EXAMPLE
   ============================================

// functions/src/generate-audio.ts
import { TextToSpeechClient } from '@google-cloud/text-to-speech';

const ttsClient = new TextToSpeechClient();

export const generateAudioBackend = async (req, res) => {
  const { text, voiceName } = req.body;

  const [response] = await ttsClient.synthesizeSpeech({
    input: { text },
    voice: {
      languageCode: 'en-US',
      name: voiceName || 'en-US-Journey-D',
    },
    audioConfig: {
      audioEncoding: 'MP3',
      pitch: 0,
      speakingRate: 1.1, // Slightly faster for viral content
      volumeGainDb: 0,
      effectsProfileId: ['headphone-class-device'], // Optimized for mobile
    },
  });

  if (!response.audioContent) {
    return res.status(500).json({ error: 'TTS_GEN_FAIL' });
  }

  // Convert to base64 for transport
  const audioBase64 = response.audioContent.toString('base64');

  // Or upload to Cloud Storage and return URL
  // const bucket = storage.bucket(process.env.STORAGE_BUCKET);
  // const file = bucket.file(`audio/${Date.now()}.mp3`);
  // await file.save(response.audioContent);
  // const [url] = await file.getSignedUrl({ action: 'read', expires: Date.now() + 3600000 });

  res.json({
    audioData: `data:audio/mp3;base64,${audioBase64}`
    // or: audioUrl: url
  });
};

============================================ */
