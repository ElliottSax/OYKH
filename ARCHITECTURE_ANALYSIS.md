# OYKH Architecture Analysis - Topic2Manim Integration

**Date**: February 19, 2026
**Decision**: Keep OYKH's core architecture, adopt Topic2Manim's orchestration patterns

---

## Executive Summary

After analyzing 9 competitor repositories, **Topic2Manim** has the best architectural match for OYKH's needs. However, **OYKH should NOT be replaced** - instead, we should integrate Topic2Manim's proven orchestration patterns while keeping OYKH's unique advantages.

### Why Topic2Manim is the Best Match

1. ✅ **Educational focus** - Built specifically for educational explainer videos
2. ✅ **Production-ready** - Working end-to-end pipeline with error recovery
3. ✅ **Clean 4-agent pipeline** - Script → TTS → Animation → Assembly
4. ✅ **Parallel processing** - TTS fragments generated concurrently
5. ✅ **Self-healing code** - REPL loop automatically fixes compilation errors
6. ✅ **LLM fallback system** - Claude (priority) → OpenAI (fallback)
7. ✅ **Background job system** - Non-blocking video generation with progress tracking
8. ✅ **Video assembly** - FFmpeg concatenation with audio merging

### Why OYKH Remains Superior

1. ✅ **Full Google AI stack** - Gemini + Imagen 3 + Google TTS (better integration)
2. ✅ **Viral retention optimization** - Cold open, retention bombs, open loops (no competitor has this)
3. ✅ **Character-based animation** - Minimalist 3D stick figures (not Manim math animations)
4. ✅ **5-minute format** - Long-form educational content (not 60s shorts)
5. ✅ **Retention science** - Education-specific engagement patterns
6. ✅ **Shot-level granularity** - 150-180 shots vs Topic2Manim's 5-10 scenes

---

## Detailed Architecture Comparison

### Topic2Manim Architecture

```
Pipeline (60-second videos, 5-10 scenes):
┌─────────────────────────────────────────────────────────────┐
│ Agent 1: Script Generation (Claude/GPT)                    │
│   - Generates 5-10 scene descriptions                      │
│   - JSON output with text + animation description          │
├─────────────────────────────────────────────────────────────┤
│ Agent 2: TTS Generation (OpenAI TTS) - PARALLEL            │
│   - Generates audio fragment per scene                     │
│   - Concatenates all fragments into single MP3             │
│   - Returns durations for timing                           │
├─────────────────────────────────────────────────────────────┤
│ Agent 3: Manim Code Generation (Claude/GPT) - SEQUENTIAL   │
│   - Generates Python code for each scene                   │
│   - Uses previous scene as context for continuity          │
│   - Self-healing REPL loop:                                │
│     1. Write code to file                                  │
│     2. Compile with Manim                                  │
│     3. If error: Send error to LLM                         │
│     4. LLM fixes code                                      │
│     5. Retry (max 3 iterations)                            │
├─────────────────────────────────────────────────────────────┤
│ Agent 4: Video Assembly (FFmpeg)                           │
│   - Concatenates all scene MP4s                            │
│   - Merges with audio track                                │
│   - Outputs final MP4                                      │
└─────────────────────────────────────────────────────────────┘

Strengths:
✅ Working end-to-end pipeline
✅ Self-healing code generation
✅ Progress tracking with job system
✅ Parallel audio generation
✅ LLM fallback (Claude → OpenAI)
✅ Error recovery at each step

Weaknesses for OYKH:
❌ Only 60-second videos (OYKH needs 5 minutes)
❌ 5-10 scenes (OYKH needs 150-180 shots)
❌ No retention optimization
❌ Math animations (not character-based)
❌ Uses OpenAI TTS (not Google TTS)
❌ No image generation (generates code that renders)
```

### OYKH Current Architecture

```
Pipeline (5-minute videos, 150-180 shots):
┌─────────────────────────────────────────────────────────────┐
│ Script Generation (Gemini 1.5 Flash)                       │
│   - Viral retention optimization                           │
│   - Cold open hook (0-3s)                                  │
│   - Retention bombs every 30s                              │
│   - Open loop system                                       │
│   - 150-180 shot breakdown                                 │
│   - Text overlays for 90% of shots                         │
├─────────────────────────────────────────────────────────────┤
│ Image Generation (Imagen 3) - NOT IMPLEMENTED              │
│   - Character-based 3D stick figures                       │
│   - Consistent style across shots                          │
│   - Shot reuse optimization (40% cost savings)             │
├─────────────────────────────────────────────────────────────┤
│ Audio Generation (Google TTS) - NOT IMPLEMENTED            │
│   - 7 Google Journey/Studio/Neural2 voices                 │
│   - Chapter-based narration                                │
├─────────────────────────────────────────────────────────────┤
│ Video Assembly - NOT IMPLEMENTED                           │
│   - Shotstack API or FFmpeg                                │
│   - Shot concatenation                                     │
│   - Audio sync                                             │
│   - Transition effects                                     │
└─────────────────────────────────────────────────────────────┘

Strengths:
✅ Google AI stack integration
✅ Viral retention optimization (unique!)
✅ Character-based storytelling
✅ 5-minute long-form content
✅ Shot-level granularity
✅ Type-safe TypeScript architecture

Weaknesses:
❌ No backend deployment yet (mock services)
❌ No video assembly
❌ No job/progress system
❌ No error recovery
❌ No parallel processing
❌ Sequential pipeline (not async)
```

---

## Integration Recommendation

### Adopt from Topic2Manim

#### 1. **Background Job System**

```typescript
// NEW: services/job-manager.ts
interface VideoJob {
  jobId: string;
  topic: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  progress: number; // 0-100
  currentStep: 'script' | 'images' | 'audio' | 'assembly';
  message: string;
  error?: string;
  videoUrl?: string;
  createdAt: Date;
  updatedAt: Date;
}

class JobManager {
  private jobs: Map<string, VideoJob> = new Map();

  createJob(topic: string): string {
    const jobId = crypto.randomUUID();
    this.jobs.set(jobId, {
      jobId,
      topic,
      status: 'queued',
      progress: 0,
      currentStep: 'script',
      message: 'Job queued',
      createdAt: new Date(),
      updatedAt: new Date(),
    });
    return jobId;
  }

  updateJob(jobId: string, updates: Partial<VideoJob>) {
    const job = this.jobs.get(jobId);
    if (job) {
      Object.assign(job, updates, { updatedAt: new Date() });
    }
  }

  getJob(jobId: string): VideoJob | undefined {
    return this.jobs.get(jobId);
  }
}
```

**Why adopt**: Non-blocking video generation, user sees real-time progress

#### 2. **Self-Healing REPL Loop** (for Imagen/TTS errors)

```typescript
// NEW: services/self-healing.ts
async function generateImageWithRetry(
  shot: Shot,
  vibe: ProductionVibe,
  maxRetries: number = 3
): Promise<string> {
  let lastError: string = '';

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const imageData = await generateImage(shot, vibe);
      return imageData; // Success!
    } catch (error) {
      lastError = error instanceof Error ? error.message : 'Unknown error';

      if (attempt < maxRetries) {
        // Ask Gemini to fix the prompt
        const fixedPrompt = await gemini.models.generateContent({
          model: 'gemini-1.5-flash',
          contents: `The following Imagen 3 prompt failed with error: "${lastError}"

Original prompt:
${shot.imagenPrompt}

Please rewrite the prompt to avoid this error while keeping the same visual concept.
Return ONLY the fixed prompt.`,
        });

        shot.imagenPrompt = fixedPrompt.text || shot.imagenPrompt;
        console.log(`[REPL] Retry ${attempt + 1}/${maxRetries} with fixed prompt`);
      }
    }
  }

  throw new Error(`Failed after ${maxRetries} attempts: ${lastError}`);
}
```

**Why adopt**: Automatic error recovery prevents manual intervention

#### 3. **Parallel Processing Architecture**

```typescript
// UPDATED: services/imagen.ts
export async function generateAllImages(
  shots: Shot[],
  vibe: ProductionVibe,
  onProgress?: (current: number, total: number) => void
): Promise<Map<number, string>> {
  const results = new Map<number, string>();

  // Process in batches of 5 (API rate limiting)
  const batchSize = 5;

  for (let i = 0; i < shots.length; i += batchSize) {
    const batch = shots.slice(i, i + batchSize);

    // Generate all images in batch concurrently
    const batchPromises = batch.map(async (shot) => {
      try {
        const imageData = await generateImageWithRetry(shot, vibe);
        results.set(shot.shotNumber, imageData);
        onProgress?.(results.size, shots.length);
      } catch (error) {
        console.error(`Shot ${shot.shotNumber} failed:`, error);
      }
    });

    await Promise.all(batchPromises);

    // Rate limiting delay between batches
    if (i + batchSize < shots.length) {
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }

  return results;
}
```

**Why adopt**: 5x faster generation for 150 shots (30 min → 6 min)

#### 4. **LLM Fallback System**

```typescript
// NEW: services/llm-provider.ts
type LLMProvider = 'gemini' | 'claude' | 'openai';

interface LLMConfig {
  provider: LLMProvider;
  client: any;
  model: string;
}

async function setupLLMClient(preference: LLMProvider = 'gemini'): Promise<LLMConfig> {
  const geminiKey = process.env.GEMINI_API_KEY;
  const claudeKey = process.env.CLAUDE_API_KEY;
  const openaiKey = process.env.OPENAI_API_KEY;

  // Priority 1: Gemini (Google native)
  if (geminiKey && (preference === 'gemini' || preference === 'auto')) {
    return {
      provider: 'gemini',
      client: new GoogleGenAI({ apiKey: geminiKey }),
      model: 'gemini-1.5-flash',
    };
  }

  // Priority 2: Claude (fallback for script quality)
  if (claudeKey) {
    return {
      provider: 'claude',
      client: new Anthropic({ apiKey: claudeKey }),
      model: 'claude-sonnet-4-5-20250929',
    };
  }

  // Priority 3: OpenAI (final fallback)
  if (openaiKey) {
    return {
      provider: 'openai',
      client: new OpenAI({ apiKey: openaiKey }),
      model: 'gpt-4o',
    };
  }

  throw new Error('No API key configured!');
}
```

**Why adopt**: Resilience - if Gemini rate limits, fallback to Claude/OpenAI

#### 5. **Video Assembly Pipeline**

```typescript
// NEW: services/video-assembly.ts
import { exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';
import fs from 'fs/promises';

const execAsync = promisify(exec);

interface AssemblyOptions {
  shots: Array<{ imageData: string; duration: number }>;
  audioPath: string;
  outputPath: string;
  onProgress?: (percent: number) => void;
}

export async function assembleVideo(options: AssemblyOptions): Promise<string> {
  const { shots, audioPath, outputPath, onProgress } = options;

  // Step 1: Save all shot images as files
  onProgress?.(10);
  const shotFiles: string[] = [];
  for (let i = 0; i < shots.length; i++) {
    const filePath = path.join('/tmp', `shot-${i}.png`);
    const imageBuffer = Buffer.from(shots[i].imageData, 'base64');
    await fs.writeFile(filePath, imageBuffer);
    shotFiles.push(filePath);
  }

  // Step 2: Create FFmpeg filter_complex for shot concatenation
  onProgress?.(30);
  const filterParts: string[] = [];
  let currentTime = 0;

  for (let i = 0; i < shots.length; i++) {
    const duration = shots[i].duration;
    // Ken Burns effect: slow zoom + pan
    filterParts.push(
      `[${i}:v]scale=1920:1080:force_original_aspect_ratio=increase,` +
        `crop=1920:1080,` +
        `zoompan=z='min(zoom+0.0015,1.5)':d=${duration * 30}:s=1920x1080[v${i}]`
    );
    currentTime += duration;
  }

  // Step 3: Concatenate all video segments
  onProgress?.(50);
  const concatFilter =
    shotFiles.map((_, i) => `[v${i}]`).join('') + `concat=n=${shots.length}:v=1:a=0[outv]`;

  const filterComplex = [...filterParts, concatFilter].join(';');

  // Step 4: Build FFmpeg command
  const inputFlags = shotFiles.map((f) => `-loop 1 -t ${shots[0].duration} -i "${f}"`).join(' ');

  const ffmpegCommand = `ffmpeg ${inputFlags} \
    -filter_complex "${filterComplex}" \
    -map "[outv]" \
    -c:v libx264 \
    -preset medium \
    -crf 23 \
    -pix_fmt yuv420p \
    -y \
    /tmp/video_silent.mp4`;

  await execAsync(ffmpegCommand);
  onProgress?.(70);

  // Step 5: Merge audio
  const mergeCommand = `ffmpeg -i /tmp/video_silent.mp4 -i "${audioPath}" \
    -c:v copy \
    -c:a aac \
    -strict experimental \
    -shortest \
    -y "${outputPath}"`;

  await execAsync(mergeCommand);
  onProgress?.(100);

  // Cleanup
  await Promise.all(shotFiles.map((f) => fs.unlink(f)));
  await fs.unlink('/tmp/video_silent.mp4');

  return outputPath;
}
```

**Why adopt**: Final MP4 export (OYKH currently can't produce videos)

---

## Hybrid Architecture: OYKH + Topic2Manim

### New OYKH Architecture (Recommended)

```
Pipeline (5-minute videos, 150-180 shots):
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Script Generation (Gemini 1.5 Flash)               │
│   - Viral retention optimization (OYKH unique)             │
│   - Cold open + retention bombs + open loops               │
│   - 150-180 shot breakdown                                 │
│   - Progress: 0% → 20%                                     │
├─────────────────────────────────────────────────────────────┤
│ Step 2: Image Generation (Imagen 3) - PARALLEL             │
│   - Batch processing (5 images at a time)                  │
│   - Self-healing REPL loop (Topic2Manim pattern)           │
│   - Character consistency via style transfer               │
│   - Progress: 20% → 60%                                    │
├─────────────────────────────────────────────────────────────┤
│ Step 3: Audio Generation (Google TTS) - PARALLEL           │
│   - Chapter-based fragments (Topic2Manim pattern)          │
│   - Concatenate with FFmpeg                                │
│   - Progress: 60% → 75%                                    │
├─────────────────────────────────────────────────────────────┤
│ Step 4: Video Assembly (FFmpeg) - SEQUENTIAL               │
│   - Shot concatenation (Topic2Manim pattern)               │
│   - Ken Burns effects                                      │
│   - Audio merge                                            │
│   - Progress: 75% → 100%                                   │
└─────────────────────────────────────────────────────────────┘

Job System (Topic2Manim pattern):
- Background threading
- Real-time progress updates
- Error recovery
- LLM fallback (Gemini → Claude → OpenAI)
```

---

## Implementation Priority

### Week 1: Backend Infrastructure

**Priority 1: Cloud Functions** (from OYKH current plan)

- ✅ Already started: `functions/src/index.ts` created
- Deploy `generateImage` function
- Deploy `generateAudio` function
- Test with real API calls

**Priority 2: Job Management System** (from Topic2Manim)

- Create `services/job-manager.ts`
- Add background worker threads
- Add progress tracking
- Add WebSocket for real-time updates

### Week 2: Parallel Processing

**Priority 3: Batch Image Generation** (from Topic2Manim)

- Implement `generateAllImages` with batching
- Add self-healing REPL loop
- Add rate limiting (5 concurrent, 1s delays)

**Priority 4: Fragment-based TTS** (from Topic2Manim)

- Generate audio per chapter (not entire script)
- Concatenate fragments with FFmpeg
- Sync with shot timings

### Week 3: Video Assembly

**Priority 5: FFmpeg Pipeline** (from Topic2Manim)

- Implement `assembleVideo` function
- Ken Burns effects for static images
- Audio merging
- MP4 export

**Priority 6: Polish**

- Error recovery for all steps
- LLM fallback system
- Cost monitoring
- Quality validation

---

## Cost Analysis: OYKH + Topic2Manim Hybrid

### Per 5-Minute Video

**Script Generation** (Gemini 1.5 Flash):

- 1 request for full script: $0.002

**Image Generation** (Imagen 3 with optimization):

- 150 shots, 40% reuse → 90 unique images
- 90 × $0.02 = $1.80

**Audio Generation** (Google TTS):

- ~4000 characters @ $16/1M chars = $0.06

**Video Assembly** (FFmpeg - local):

- Free (runs on server)

**Total: ~$1.87 per video**

### Comparison to Topic2Manim

**Topic2Manim** (60-second video):

- Script (Claude): $0.01
- TTS (OpenAI): $0.02
- Manim rendering: Free (local)
- **Total: ~$0.03 per video**

**But:**

- OYKH is 5 minutes (5x longer)
- OYKH has 150 shots vs 5 scenes (30x more images)
- OYKH has character animation (Imagen 3) vs code generation (Manim)

**OYKH cost per minute**: $1.87 / 5 = **$0.37/min**
**Topic2Manim cost per minute**: $0.03 / 1 = **$0.03/min**

OYKH is 12x more expensive per minute due to Imagen 3, but produces higher visual quality (character-based storytelling vs math animations).

**Optimization opportunity**: Use Manim for certain shots (charts, graphs) to reduce Imagen costs.

---

## Next Steps

### Immediate Actions

1. **Test Topic2Manim locally**

   ```bash
   cd /c/projects/topic2manim-test
   # Set up API keys in .env
   docker compose up
   # Navigate to http://localhost:5000
   # Generate test video: "How does photosynthesis work?"
   ```

2. **Extract patterns to OYKH**
   - Copy `services/job-manager.ts` (from video_generator.py)
   - Copy `services/self-healing.ts` (from REPL loop)
   - Copy `services/video-assembly.ts` (from concat_video.py)

3. **Update OYKH architecture**
   - Add background job system
   - Add parallel processing
   - Add FFmpeg video assembly
   - Deploy Cloud Functions backend

### Long-term Strategy

1. **Keep OYKH's unique advantages**
   - Google AI stack
   - Viral retention optimization
   - Character-based animation
   - 5-minute long-form

2. **Adopt Topic2Manim's proven patterns**
   - Job/progress system
   - Self-healing REPL loops
   - Parallel processing
   - FFmpeg video assembly

3. **Future enhancements**
   - Hybrid rendering (Imagen for characters, Manim for charts)
   - Multi-platform export (YouTube, Shorts, TikTok)
   - Analytics feedback loop
   - A/B testing for hooks

---

## Conclusion

**Decision**: **Keep OYKH, integrate Topic2Manim's orchestration patterns**

OYKH has a superior product vision (viral educational explainers with character-based storytelling), but Topic2Manim has proven production infrastructure (job system, error recovery, video assembly).

By combining OYKH's unique retention science with Topic2Manim's robust orchestration, we get the best of both worlds:

✅ **OYKH's vision** (what to build)
✅ **Topic2Manim's execution** (how to build it reliably)

This hybrid approach delivers:

- 5-minute viral educational videos (OYKH)
- Production-ready pipeline with error recovery (Topic2Manim)
- Google AI stack integration (OYKH)
- Proven video assembly (Topic2Manim)
- Character-based storytelling (OYKH)
- Parallel processing efficiency (Topic2Manim)

**Estimated timeline to production**:

- Week 1: Backend + job system
- Week 2: Parallel processing
- Week 3: Video assembly
- **Total: 3 weeks to working MVP**

---

**Built with**: OYKH's vision + Topic2Manim's infrastructure
