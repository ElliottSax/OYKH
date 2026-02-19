# OYKH - Once You Know How AI Video Generator

**AI-powered viral explainer video generation system** following Google AI Studio best practices.

Generates 5-minute educational YouTube videos in the style of Kurzgesagt and CGP Grey using:
- **Gemini 1.5 Flash** for viral script generation
- **Imagen 3** for minimalist character animation
- **Google Cloud TTS** for professional narration

---

## 🚀 Quick Start (Development Mode)

### Prerequisites
- Node.js 18+
- Google Cloud account (for production)

### Installation

```bash
# Install dependencies
npm install

# Create environment file
cp .env.example .env.local

# Add your API key
# GEMINI_API_KEY=your_key_here

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

---

## 📁 Project Structure

```
oykh-temp/
├── services/
│   ├── gemini.ts          # Script generation (Gemini 1.5 Flash)
│   ├── imagen.ts          # Image generation (Imagen 3)
│   └── tts.ts             # Audio generation (Google TTS)
├── types.ts               # TypeScript definitions
├── App.tsx                # Main application
├── index.tsx              # Entry point
├── index.html             # HTML template
└── vite.config.ts         # Build configuration
```

---

## 🎯 Current Implementation Status

### ✅ Working (Development Mode)
- [x] Viral script generation with Gemini 1.5 Flash
- [x] Structured output with retention optimization
- [x] Shot-by-shot planning
- [x] Mock image placeholders
- [x] Mock audio generation
- [x] Step-based UI workflow

### 🚧 In Progress
- [ ] Imagen 3 integration (needs backend)
- [ ] Google TTS integration (needs backend)
- [ ] Video assembly (Shotstack API)
- [ ] Script review UI
- [ ] Batch generation mode

### 📋 Planned
- [ ] Multi-platform export (YouTube, Shorts, TikTok)
- [ ] Shot optimization (cost reduction)
- [ ] Analytics dashboard
- [ ] Thumbnail generator

---

## 🔧 Production Setup

### Phase 1: Backend Proxy (Security)

**Why**: Keep API keys server-side, prevent abuse

**Setup Cloud Functions**:

```bash
# Install Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# Initialize project
gcloud init

# Enable required APIs
gcloud services enable \
  aiplatform.googleapis.com \
  texttospeech.googleapis.com \
  storage.googleapis.com \
  cloudfunctions.googleapis.com

# Deploy backend functions
cd functions
npm install
gcloud functions deploy generate-video \
  --runtime nodejs20 \
  --trigger-http \
  --allow-unauthenticated
```

**Backend Code** (`functions/src/index.ts`):

```typescript
import { onRequest } from 'firebase-functions/v2/https';
import { GoogleGenAI } from '@google/genai';
import { PredictionServiceClient } from '@google-cloud/aiplatform';
import { TextToSpeechClient } from '@google-cloud/text-to-speech';

// Script generation
export const generateScript = onRequest(async (req, res) => {
  const { topic, hook, vibe } = req.body;

  const gemini = new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY
  });

  const script = await gemini.models.generateContent({
    model: 'gemini-1.5-flash',
    contents: /* viral script prompt */
  });

  res.json(script);
});

// Image generation
export const generateImage = onRequest(async (req, res) => {
  const { shot, vibe } = req.body;

  const client = new PredictionServiceClient({
    apiEndpoint: 'us-central1-aiplatform.googleapis.com'
  });

  const [response] = await client.predict({
    endpoint: `projects/${process.env.PROJECT_ID}/locations/us-central1/publishers/google/models/imagen-3.0-generate-001`,
    instances: [{ prompt: buildPrompt(shot, vibe) }],
    parameters: {
      sampleCount: 1,
      aspectRatio: '16:9'
    }
  });

  res.json({ imageData: response.predictions[0].bytesBase64Encoded });
});

// Audio generation
export const generateAudio = onRequest(async (req, res) => {
  const { text, voiceName } = req.body;

  const ttsClient = new TextToSpeechClient();

  const [response] = await ttsClient.synthesizeSpeech({
    input: { text },
    voice: { languageCode: 'en-US', name: voiceName },
    audioConfig: {
      audioEncoding: 'MP3',
      speakingRate: 1.1
    }
  });

  res.json({ audioData: response.audioContent.toString('base64') });
});
```

**Update Frontend** (`services/gemini.ts`, `services/imagen.ts`, `services/tts.ts`):

```typescript
// Instead of calling Google APIs directly:
const response = await ai.models.generateContent({ /* ... */ });

// Call YOUR backend:
const response = await fetch('/api/generate-script', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ topic, hook, vibe })
});
```

### Phase 2: Video Assembly (Shotstack API)

**Why**: Professional video rendering with transitions, audio mixing

```bash
npm install shotstack-sdk
```

```typescript
// services/video-assembly.ts
import Shotstack from 'shotstack-sdk';

const client = new Shotstack.Client(process.env.SHOTSTACK_API_KEY);

export const assembleVideo = async (shots, audioUrl, musicUrl) => {
  const edit = {
    timeline: {
      soundtrack: {
        src: audioUrl,
        effect: 'fadeIn'
      },
      background: musicUrl,
      tracks: [
        {
          clips: shots.map((shot, i) => ({
            asset: {
              type: 'image',
              src: shot.imageData
            },
            start: shot.startTime,
            length: shot.duration,
            transition: {
              in: shot.transition,
              out: 'fade'
            },
            effect: shot.animation // ken-burns, etc.
          }))
        },
        {
          clips: shots
            .filter(shot => shot.textOverlay)
            .map(shot => ({
              asset: {
                type: 'html',
                html: renderTextOverlay(shot.textOverlay)
              },
              start: shot.startTime,
              length: shot.duration
            }))
        }
      ]
    },
    output: {
      format: 'mp4',
      resolution: '1080'
    }
  };

  const response = await client.render.postRender(edit);
  return response.data.response.url; // Final video URL
};
```

### Phase 3: Cost Optimization

**Shot Reuse Strategy** (40% savings):

```typescript
// services/shot-optimizer.ts
export const optimizeShots = (shots: Shot[]) => {
  const groups = new Map<string, Shot[]>();

  shots.forEach(shot => {
    const key = `${shot.characterAction}-${shot.cameraAngle}`;

    if (!groups.has(key)) {
      groups.set(key, []);
    }

    groups.get(key)!.push(shot);
  });

  // Generate only unique shots
  const uniqueShots = Array.from(groups.entries())
    .map(([key, shots]) => shots[0]);

  // Reuse with CSS variations
  const optimizedShots = shots.map(shot => {
    const groupKey = `${shot.characterAction}-${shot.cameraAngle}`;
    const baseShot = groups.get(groupKey)![0];

    return {
      ...shot,
      imageData: baseShot.imageData,
      cssTransform: getAnimationTransform(shot.animation)
    };
  });

  return optimizedShots;
};
```

---

## 💰 Cost Analysis

### Per 5-Minute Video (150 shots):

**Development (Mock Mode)**:
- Script generation: $0.002
- Total: **$0.002**

**Production (All Google)**:
- Gemini 1.5 Flash (script): $0.002
- Imagen 3 (150 images @ $0.02): $3.00
- Google TTS (narration): $0.08
- Shotstack (video assembly): $0.05
- Total: **$3.13 per video**

**Optimized (40% shot reuse)**:
- Gemini 1.5 Flash (script): $0.002
- Imagen 3 (90 unique @ $0.02): $1.80
- Google TTS (narration): $0.08
- Shotstack (video assembly): $0.05
- Total: **$1.93 per video**

**Monthly Estimates**:
- 10 videos/month: $19.30
- 50 videos/month: $96.50
- 100 videos/month: $193.00

---

## 🎨 Visual Style Guide

The system generates minimalist 3D character animations:

- **Character**: Puffy volumetric white stick figure (Baymax-style)
- **Texture**: Soft matte plastic/clay with subsurface scattering
- **Outlines**: Bold 8px black strokes
- **Backgrounds**: Simple gradients only (no clutter)
- **Vibes**: cosmic, hype, minimal, suspense, success

---

## 📊 Retention Optimization

Videos are structured for maximum viewer retention:

- **Cold Open** (0-3s): Shock value hook
- **Retention Bombs**: Every 30 seconds
- **Open Loops**: Questions posed early, resolved later
- **Pattern Interrupts**: Visual change every 1.5-2.5 seconds
- **Text Overlays**: 90% of shots (for muted viewing)

---

## 🔑 Environment Variables

Create `.env.local`:

```env
# Development
GEMINI_API_KEY=your_gemini_key

# Production (Backend)
GOOGLE_CLOUD_PROJECT_ID=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=./key.json
SHOTSTACK_API_KEY=your_shotstack_key
STORAGE_BUCKET=your-storage-bucket
```

---

## 📚 Documentation

- [Google AI Studio Guide](https://ai.google.dev/gemini-api/docs)
- [Imagen 3 Documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/image/overview)
- [Google Cloud TTS](https://cloud.google.com/text-to-speech/docs)
- [Shotstack API](https://shotstack.io/docs/guide/)

---

## 🐛 Troubleshooting

**"Model not found" error**:
- Ensure you're using `gemini-1.5-flash` (NOT `gemini-3-flash-preview`)

**Images not generating**:
- Development mode uses placeholders by default
- For real images, set up Imagen 3 backend (see Production Setup)

**No audio**:
- Development mode uses silent audio
- For real narration, set up Google TTS backend

**API rate limits**:
- Add delays between batch requests
- Implement exponential backoff
- Use shot optimization to reduce API calls

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📞 Support

For issues and questions:
- GitHub Issues: [Report a bug](https://github.com/your-repo/issues)
- Documentation: See `/docs` folder

---

Built with ❤️ following Google AI Studio best practices
