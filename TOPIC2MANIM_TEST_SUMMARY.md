# Topic2Manim Test Summary

**Date**: February 19, 2026
**Status**: ✅ Successfully installed and running
**Server**: http://localhost:5000

---

## Installation Results

### Dependencies Installed:

✅ **Manim 0.19.2** - Mathematical animation engine
✅ **Google Generative AI 0.8.6** - For script generation
✅ **OpenAI** - For TTS generation
✅ **Anthropic** - For Claude Sonnet 4.5
✅ **Flask 3.1.2** - Web UI
✅ **SciPy, NumPy, Pillow** - Scientific computing & imaging
✅ **FFmpeg dependencies** (av, pydub) - Video/audio processing

**Total installation time**: ~10 minutes (large packages: scipy 36MB, av 32MB)
**Installation method**: `pip install -e .` (editable mode)

---

## Architecture Analysis

### Topic2Manim's 4-Agent Pipeline:

```
┌──────────────────────────────────────────────────────────┐
│ 1. Script Generation (Claude/GPT)                        │
│    - Input: Topic string                                 │
│    - Output: JSON with 5-10 scenes                       │
│    - Each scene: {text, animation}                       │
│    - LLM fallback: Claude → OpenAI                       │
├──────────────────────────────────────────────────────────┤
│ 2. TTS Generation (OpenAI TTS) - PARALLEL               │
│    - Generates audio fragment per scene                  │
│    - Returns duration for each fragment                  │
│    - Concatenates all fragments into single MP3          │
│    - Uses pydub for audio concatenation                  │
├──────────────────────────────────────────────────────────┤
│ 3. Manim Code Generation (Claude/GPT) - SEQUENTIAL      │
│    - Generates Python code for each scene                │
│    - Uses previous scene as context                      │
│    - SELF-HEALING REPL LOOP:                             │
│      1. Write code to file                               │
│      2. Compile with Manim (subprocess call)             │
│      3. If error → Send error to LLM                     │
│      4. LLM fixes code                                   │
│      5. Retry (max 3 iterations)                         │
│    - Outputs MP4 per scene                               │
├──────────────────────────────────────────────────────────┤
│ 4. Video Assembly (FFmpeg)                               │
│    - Concatenates all scene MP4s                         │
│    - Merges with audio track                             │
│    - Outputs final MP4                                   │
└──────────────────────────────────────────────────────────┘
```

### Background Job System:

```python
# From src/video_generator.py:lines 67-86
def update_job_status(job_id, status=None, progress=None,
                      current_step=None, message=None, error=None, video_url=None):
    """Update job status in storage"""
    if job_id not in jobs:
        jobs[job_id] = {}

    if status:
        jobs[job_id]['status'] = status
    if progress is not None:
        jobs[job_id]['progress'] = progress
    if current_step:
        jobs[job_id]['current_step'] = current_step
    if message:
        jobs[job_id]['message'] = message

    jobs[job_id]['updated_at'] = datetime.now().isoformat()
```

**Key features**:

- In-memory job storage (would use Redis/database in production)
- Real-time progress tracking (0-100%)
- Current step indicators (`script`, `tts`, `code`, `video`)
- Error capture and reporting
- Video URL on completion

### Self-Healing REPL Loop:

```python
# From src/video_generator.py:lines 185-226
for repl_iteration in range(max_repl_iterations):
    # Write current code to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(current_code)

    # Try to compile
    video_path, compile_error = compile_video(filepath, current_class_name, topic_slug, index)

    if video_path and os.path.exists(video_path):
        # Success! Exit the REPL loop
        print(f"[REPL] Scene {index} compiled successfully on iteration {repl_iteration + 1}")
        break

    if compile_error and repl_iteration < max_repl_iterations - 1:
        # Error occurred, try to fix with LLM
        fixed_code = fix_manim_code(
            client=client,
            original_code=current_code,
            error_message=compile_error,
            class_name=current_class_name,
            provider=provider,
            model=model
        )

        if fixed_code:
            current_code = fixed_code.get('content', '')
            current_class_name = fixed_code.get('class_name', current_class_name)
```

**Why this is powerful**:

- Automatically recovers from code generation errors
- LLM reads compilation error and fixes its own code
- Max 3 attempts per scene (prevents infinite loops)
- Preserves context between attempts

---

## What OYKH Should Adopt

### 1. ✅ **Background Job System** (Priority: HIGH)

**Current OYKH limitation**: Synchronous script generation, no progress tracking

**Topic2Manim solution**:

- Background threading for video generation
- Real-time progress updates (0-100%)
- Step-by-step status (`script`, `images`, `audio`, `assembly`)
- User can see progress without blocking

**Implementation for OYKH**:

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

  async startVideoGeneration(topic: string, hook: string, vibe: ProductionVibe): Promise<string> {
    const jobId = crypto.randomUUID();

    this.jobs.set(jobId, {
      jobId,
      topic,
      status: 'queued',
      progress: 0,
      currentStep: 'script',
      message: 'Queued for processing',
      createdAt: new Date(),
      updatedAt: new Date(),
    });

    // Start background worker
    this.runVideoWorkflow(jobId, topic, hook, vibe);

    return jobId;
  }

  private async runVideoWorkflow(jobId: string, topic: string, hook: string, vibe: ProductionVibe) {
    try {
      // Step 1: Script generation (0% → 20%)
      this.updateJob(jobId, { progress: 0, message: 'Generating script...' });
      const script = await generateViralVideoScript(topic, hook, vibe);
      this.updateJob(jobId, {
        progress: 20,
        message: `Script generated (${script.totalShots} shots)`,
      });

      // Step 2: Image generation (20% → 60%)
      this.updateJob(jobId, {
        currentStep: 'images',
        progress: 20,
        message: 'Generating images...',
      });
      const images = await generateAllImages(
        script.chapters.flatMap((c) => c.shots),
        vibe,
        (current, total) => {
          const imageProgress = 20 + (current / total) * 40;
          this.updateJob(jobId, {
            progress: imageProgress,
            message: `Images: ${current}/${total}`,
          });
        }
      );
      this.updateJob(jobId, { progress: 60, message: 'All images generated' });

      // Step 3: Audio generation (60% → 75%)
      this.updateJob(jobId, { currentStep: 'audio', progress: 60, message: 'Generating audio...' });
      const audio = await generateNarration(script.chapters);
      this.updateJob(jobId, { progress: 75, message: 'Audio generated' });

      // Step 4: Video assembly (75% → 100%)
      this.updateJob(jobId, {
        currentStep: 'assembly',
        progress: 75,
        message: 'Assembling video...',
      });
      const videoUrl = await assembleVideo(script, images, audio);

      this.updateJob(jobId, {
        status: 'completed',
        progress: 100,
        message: 'Video generation complete!',
        videoUrl,
      });
    } catch (error) {
      this.updateJob(jobId, {
        status: 'failed',
        error: error.message,
        message: `Error: ${error.message}`,
      });
    }
  }
}
```

**Benefits**:

- User doesn't wait for 15 minutes staring at loading spinner
- Can generate multiple videos concurrently
- Progress bar shows exactly where we are
- Errors don't crash the UI

### 2. ✅ **Self-Healing Error Recovery** (Priority: MEDIUM)

**Current OYKH limitation**: If Imagen 3 API fails, video generation crashes

**Topic2Manim solution**:

- REPL loop detects errors
- Sends error to LLM
- LLM fixes the problem (e.g., adjusts prompt)
- Retries automatically (max 3 attempts)

**Implementation for OYKH**:

```typescript
// NEW: services/self-healing.ts
async function generateImageWithRetry(
  shot: Shot,
  vibe: ProductionVibe,
  maxRetries: number = 3
): Promise<string> {
  let lastError: string = '';
  let currentPrompt = shot.imagenPrompt;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      // Try to generate image
      const imageData = await callImagenAPI(currentPrompt, vibe);
      console.log(`✅ Shot #${shot.shotNumber} succeeded on attempt ${attempt}`);
      return imageData;
    } catch (error) {
      lastError = error.message;
      console.error(
        `❌ Shot #${shot.shotNumber} failed (attempt ${attempt}/${maxRetries}):`,
        lastError
      );

      if (attempt < maxRetries) {
        // Ask Gemini to fix the prompt
        const fixedPrompt = await gemini.models.generateContent({
          model: 'gemini-1.5-flash',
          contents: `The following Imagen 3 prompt failed with this error:
Error: "${lastError}"

Original prompt:
${currentPrompt}

Please rewrite the prompt to avoid this error while keeping the same visual concept.
Make the prompt more specific and descriptive.
Return ONLY the fixed prompt, no explanations.`,
        });

        currentPrompt = fixedPrompt.text || currentPrompt;
        console.log(`🔧 Retrying shot #${shot.shotNumber} with fixed prompt...`);

        // Brief delay before retry
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
    }
  }

  throw new Error(`Shot #${shot.shotNumber} failed after ${maxRetries} attempts: ${lastError}`);
}
```

**Benefits**:

- Automatic error recovery (no manual intervention)
- LLM learns from its mistakes
- Reduces failed video generations
- User never sees transient API errors

### 3. ✅ **Parallel Processing** (Priority: HIGH)

**Current OYKH limitation**: Sequential image generation (150 shots × 30s = 75 minutes!)

**Topic2Manim solution**:

- Generates TTS fragments in parallel
- Batches API calls to respect rate limits
- Progress tracking for each batch

**Implementation for OYKH**:

```typescript
// UPDATED: services/imagen.ts
export async function generateAllImages(
  shots: Shot[],
  vibe: ProductionVibe,
  onProgress?: (current: number, total: number) => void
): Promise<Map<number, string>> {
  const results = new Map<number, string>();
  const batchSize = 5; // Generate 5 images concurrently

  for (let i = 0; i < shots.length; i += batchSize) {
    const batch = shots.slice(i, i + batchSize);

    // Generate all images in batch concurrently
    const batchPromises = batch.map(async (shot) => {
      try {
        const imageData = await generateImageWithRetry(shot, vibe);
        results.set(shot.shotNumber, imageData);
        onProgress?.(results.size, shots.length);
        return { shotNumber: shot.shotNumber, success: true };
      } catch (error) {
        console.error(`Shot ${shot.shotNumber} failed permanently:`, error);
        return { shotNumber: shot.shotNumber, success: false };
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

**Performance improvement**:

- **Before** (sequential): 150 shots × 30s = 75 minutes
- **After** (parallel, batch of 5): 150 shots / 5 × 30s = 15 minutes
- **5x faster** image generation!

### 4. ✅ **LLM Fallback System** (Priority: LOW)

**Current OYKH limitation**: Single LLM (Gemini), no fallback

**Topic2Manim solution**:

- Priority 1: Claude (high quality)
- Priority 2: OpenAI (fallback if Claude unavailable)
- Automatic detection based on API key availability

**Implementation for OYKH**:

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

  // Priority 1: Gemini (Google native, best for OYKH)
  if (geminiKey && preference === 'gemini') {
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

  throw new Error(
    'No API key configured! Please set GEMINI_API_KEY, CLAUDE_API_KEY, or OPENAI_API_KEY'
  );
}
```

**Benefits**:

- Service resilience (if Gemini rate limits, use Claude)
- Quality fallback (Claude often better at creative writing)
- User doesn't see "API quota exceeded" errors

### 5. ✅ **FFmpeg Video Assembly** (Priority: HIGH)

**Current OYKH limitation**: No video assembly (only generates shots)

**Topic2Manim solution**:

- Uses FFmpeg to concatenate scene MP4s
- Merges audio track with video
- Ken Burns effects for static images (zoom/pan)

**Implementation for OYKH**:

```typescript
// NEW: services/video-assembly.ts
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export async function assembleVideo(
  script: ViralVideoScript,
  images: Map<number, string>, // shotNumber → base64 image
  audioPath: string,
  onProgress?: (percent: number) => void
): Promise<string> {
  const tempDir = '/tmp/oykh';
  await fs.mkdir(tempDir, { recursive: true });

  // Step 1: Save all shot images as files
  onProgress?.(10);
  const shotFiles: string[] = [];
  for (const chapter of script.chapters) {
    for (const shot of chapter.shots) {
      const imageData = images.get(shot.shotNumber);
      if (!imageData) continue;

      const filePath = path.join(tempDir, `shot-${shot.shotNumber}.png`);
      const imageBuffer = Buffer.from(imageData, 'base64');
      await fs.writeFile(filePath, imageBuffer);
      shotFiles.push(filePath);
    }
  }

  // Step 2: Create FFmpeg filter_complex for shot concatenation with Ken Burns effects
  onProgress?.(30);
  const filterParts: string[] = [];
  let currentTime = 0;

  for (let i = 0; i < shotFiles.length; i++) {
    const shot = script.chapters.flatMap((c) => c.shots)[i];
    const duration = shot.duration;

    // Ken Burns effect: slow zoom + pan for dynamic feel
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
    shotFiles.map((_, i) => `[v${i}]`).join('') + `concat=n=${shotFiles.length}:v=1:a=0[outv]`;

  const filterComplex = [...filterParts, concatFilter].join(';');

  // Step 4: Build FFmpeg command
  const inputFlags = shotFiles
    .map((f, i) => {
      const shot = script.chapters.flatMap((c) => c.shots)[i];
      return `-loop 1 -t ${shot.duration} -i "${f}"`;
    })
    .join(' ');

  const ffmpegCommand = `ffmpeg ${inputFlags} \
    -filter_complex "${filterComplex}" \
    -map "[outv]" \
    -c:v libx264 \
    -preset medium \
    -crf 23 \
    -pix_fmt yuv420p \
    -y \
    ${tempDir}/video_silent.mp4`;

  await execAsync(ffmpegCommand);
  onProgress?.(70);

  // Step 5: Merge audio
  const outputPath = path.join(tempDir, `output-${Date.now()}.mp4`);
  const mergeCommand = `ffmpeg -i ${tempDir}/video_silent.mp4 -i "${audioPath}" \
    -c:v copy \
    -c:a aac \
    -strict experimental \
    -shortest \
    -y "${outputPath}"`;

  await execAsync(mergeCommand);
  onProgress?.(100);

  // Cleanup temp files
  await Promise.all(shotFiles.map((f) => fs.unlink(f)));
  await fs.unlink(path.join(tempDir, 'video_silent.mp4'));

  return outputPath;
}
```

**Benefits**:

- Final MP4 output (not just individual shots)
- Ken Burns effects make static images feel dynamic
- Audio sync with video
- Standard MP4 format for YouTube upload

---

## Key Differences: Topic2Manim vs OYKH

| Feature                    | Topic2Manim            | OYKH                        |
| -------------------------- | ---------------------- | --------------------------- |
| **Video length**           | 60 seconds             | 5 minutes                   |
| **Shot count**             | 5-10 scenes            | 150-180 shots               |
| **Animation**              | Manim math animations  | Imagen 3 character art      |
| **AI Stack**               | Claude + OpenAI        | Full Google stack           |
| **Retention optimization** | None                   | Cold open + retention bombs |
| **TTS**                    | OpenAI TTS             | Google TTS (Journey voices) |
| **Image generation**       | Manim code → renders   | Imagen 3 API                |
| **Cost per video**         | ~$0.03 (60s)           | ~$1.87 (5min optimized)     |
| **Target audience**        | Math/tech education    | Viral educational content   |
| **Job system**             | ✅ Yes (in-memory)     | ❌ Not yet                  |
| **Self-healing**           | ✅ Yes (REPL loop)     | ❌ Not yet                  |
| **Parallel processing**    | ✅ Yes (TTS fragments) | ❌ Not yet                  |
| **Video assembly**         | ✅ Yes (FFmpeg)        | ❌ Not yet                  |

---

## Implementation Roadmap for OYKH

### Week 1: Core Infrastructure

**Priority 1: Job Management System**

- [ ] Create `services/job-manager.ts`
- [ ] Background worker threads
- [ ] Real-time progress tracking
- [ ] WebSocket for UI updates
- **Time**: 2-3 days

**Priority 2: Parallel Image Generation**

- [ ] Update `services/imagen.ts` with batching
- [ ] Concurrent Promise.all() for 5 images at a time
- [ ] Progress callbacks
- **Time**: 1-2 days

**Priority 3: Self-Healing System**

- [ ] Create `services/self-healing.ts`
- [ ] Retry logic with LLM prompt fixes
- [ ] Error logging and tracking
- **Time**: 1-2 days

### Week 2: Video Assembly

**Priority 4: FFmpeg Integration**

- [ ] Install FFmpeg on server
- [ ] Create `services/video-assembly.ts`
- [ ] Ken Burns effects
- [ ] Audio merging
- [ ] MP4 export
- **Time**: 3-4 days

**Priority 5: LLM Fallback**

- [ ] Create `services/llm-provider.ts`
- [ ] Support Gemini + Claude + OpenAI
- [ ] Automatic detection based on API keys
- **Time**: 1 day

### Week 3: Testing & Polish

**Priority 6: End-to-End Testing**

- [ ] Generate test video (full pipeline)
- [ ] Test error recovery
- [ ] Test parallel processing
- [ ] Measure performance improvements
- **Time**: 2-3 days

**Priority 7: UI Improvements**

- [ ] Progress bar component
- [ ] Real-time job status
- [ ] Error display
- [ ] Video preview
- **Time**: 2-3 days

---

## Performance Benchmarks (Estimated)

### Current OYKH (Sequential):

- Script generation: 30s
- Image generation: 150 shots × 30s = **75 minutes**
- Audio generation: 60s
- Video assembly: N/A (not implemented)
- **Total: ~77 minutes**

### OYKH + Topic2Manim Patterns (Parallel):

- Script generation: 30s
- Image generation: 150 shots / 5 × 30s = **15 minutes**
- Audio generation: 60s (parallel)
- Video assembly: 3 minutes
- **Total: ~19 minutes**

**4x faster** with parallel processing!

### With Self-Healing (Est. 5% failures):

- Current: 5% failure rate = 7.5 failed shots → manual retry → +10 minutes
- With self-healing: Auto-retry → 0 manual intervention
- **Saved time: ~10 minutes per failed generation**

---

## Cost Comparison

### Topic2Manim (60-second video):

- Script (Claude): $0.01
- TTS (OpenAI): $0.02
- Manim rendering: Free (local)
- **Total: ~$0.03 per video**

### OYKH (5-minute video, optimized):

- Script (Gemini): $0.002
- Images (Imagen 3, 90 unique): $1.80
- Audio (Google TTS): $0.06
- Video assembly (FFmpeg): Free (local)
- **Total: ~$1.87 per video**

**OYKH is 62x more expensive** due to Imagen 3, but produces character-based storytelling (not math animations).

**Hybrid optimization idea**: Use Manim for charts/graphs, Imagen 3 for characters → reduce costs by 30%.

---

## Conclusion

### ✅ What We Learned from Topic2Manim:

1. **Background job system** enables non-blocking video generation
2. **Self-healing REPL loops** dramatically reduce manual intervention
3. **Parallel processing** cuts generation time by 75%
4. **FFmpeg video assembly** creates production-ready MP4s
5. **LLM fallback** provides service resilience

### ✅ What OYKH Should Keep:

1. **Google AI stack** (Gemini + Imagen + TTS) - better integration
2. **Viral retention optimization** - unique competitive advantage
3. **Character-based animation** - brand differentiation
4. **5-minute long-form** - deeper educational content
5. **Shot-level granularity** - precise storytelling control

### ✅ Recommended Integration:

**Keep OYKH's vision, adopt Topic2Manim's execution patterns**:

- OYKH's retention science + Topic2Manim's job system = **Best of both worlds**
- OYKH's Google stack + Topic2Manim's parallel processing = **Faster, better videos**
- OYKH's character storytelling + Topic2Manim's FFmpeg assembly = **Production-ready output**

---

**Next Steps**:

1. Implement job manager (Week 1, Priority 1)
2. Add parallel image generation (Week 1, Priority 2)
3. Integrate FFmpeg video assembly (Week 2, Priority 4)
4. Test end-to-end pipeline (Week 3)
5. Deploy to production (Week 4)

**Timeline**: 3-4 weeks to production-ready OYKH with Topic2Manim patterns integrated

---

**Topic2Manim Server**: Running at http://localhost:5000
**Installation**: Complete ✅
**Analysis**: Complete ✅
**Decision**: Integrate patterns into OYKH ✅
