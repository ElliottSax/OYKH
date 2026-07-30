# FFmpeg Setup Guide for OYKH

**Required for**: Video assembly (combining shots into final MP4)

---

## Why FFmpeg?

FFmpeg is an industry-standard tool for video processing. OYKH uses it to:

- Concatenate 150+ shot images into a video
- Apply Ken Burns effects (zoom/pan) for dynamic feel
- Merge audio narration with video
- Export professional-quality MP4 files

---

## Installation

### Windows

**Option 1: Chocolatey (Recommended)**

```powershell
# Install Chocolatey first (if not installed):
# https://chocolatey.org/install

# Install FFmpeg:
choco install ffmpeg

# Verify installation:
ffmpeg -version
```

**Option 2: Manual Installation**

1. Download FFmpeg from https://www.gyan.dev/ffmpeg/builds/
2. Extract the ZIP file (e.g., to `C:\ffmpeg`)
3. Add to PATH:
   - Open "Edit the system environment variables"
   - Click "Environment Variables"
   - Under "System variables", select "Path" and click "Edit"
   - Click "New" and add `C:\ffmpeg\bin`
   - Click "OK" to save
4. Restart terminal and verify:
   ```cmd
   ffmpeg -version
   ```

### macOS

**Using Homebrew (Recommended)**:

```bash
# Install Homebrew first (if not installed):
# https://brew.sh

# Install FFmpeg:
brew install ffmpeg

# Verify installation:
ffmpeg -version
```

### Linux (Ubuntu/Debian)

```bash
# Update package list:
sudo apt update

# Install FFmpeg:
sudo apt install ffmpeg

# Verify installation:
ffmpeg -version
```

### Linux (Fedora/RHEL)

```bash
# Install FFmpeg:
sudo dnf install ffmpeg

# Verify installation:
ffmpeg -version
```

---

## Verification

After installation, verify FFmpeg is working:

```bash
# Check version:
ffmpeg -version

# You should see output like:
# ffmpeg version 6.1.1 Copyright (c) 2000-2024 the FFmpeg developers
# built with gcc 13.2.0 (GCC)
# ...

# Check if it can process video:
ffmpeg -h
```

---

## Testing with OYKH

Once FFmpeg is installed, test the video assembly:

```typescript
import { checkFFmpegInstalled, assembleVideo } from './services/video-assembly';

// Check if FFmpeg is available
const isInstalled = await checkFFmpegInstalled();
console.log('FFmpeg installed:', isInstalled);

// If installed, try generating a test video
if (isInstalled) {
  const videoPath = await assembleVideo(script, imageUrls, audioPath);
  console.log('Video created:', videoPath);
}
```

---

## Troubleshooting

### "ffmpeg: command not found"

**Cause**: FFmpeg is not in your system PATH

**Solution**:

1. **Windows**: Reinstall using Chocolatey, or manually add to PATH (see installation steps above)
2. **macOS**: Run `brew install ffmpeg`
3. **Linux**: Run `sudo apt install ffmpeg` or `sudo dnf install ffmpeg`

After installation, **restart your terminal** and try again.

### "Permission denied" error

**Cause**: Insufficient permissions to execute FFmpeg

**Solution**:

```bash
# macOS/Linux: Make FFmpeg executable
sudo chmod +x /usr/local/bin/ffmpeg

# Windows: Run terminal as Administrator
```

### FFmpeg crashes during video generation

**Cause**: Insufficient memory or disk space

**Solutions**:

1. **Free up disk space** (video generation requires ~2GB temp space)
2. **Close other applications** to free RAM
3. **Reduce video resolution** (edit `video-assembly.ts` to use 1280x720 instead of 1920x1080)

### Video generation is very slow

**Cause**: FFmpeg is using slow encoding settings

**Solutions**:

1. **Use faster preset**: Edit `video-assembly.ts` and change:

   ```typescript
   '-preset medium' → '-preset fast'
   ```

   (Options: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)

2. **Lower quality**: Increase CRF value:

   ```typescript
   '-crf 23' → '-crf 28'
   ```

   (Range: 18-28, higher = lower quality but faster)

3. **Hardware acceleration** (if available):
   ```typescript
   // Add to FFmpeg command:
   '-hwaccel auto';
   ```

### Output video has no audio

**Cause**: Audio file path is incorrect or audio format is unsupported

**Solutions**:

1. **Check audio file exists**:

   ```bash
   ls -la path/to/audio.mp3
   ```

2. **Convert audio to MP3** (if using different format):

   ```bash
   ffmpeg -i input_audio.wav -codec:a libmp3lame -qscale:a 2 output.mp3
   ```

3. **Check FFmpeg audio codecs**:
   ```bash
   ffmpeg -codecs | grep -i aac
   ```

### Output video quality is poor

**Cause**: CRF value is too high (lower quality)

**Solution**: Edit `video-assembly.ts`:

```typescript
// Change from:
'-crf 23';

// To (higher quality):
'-crf 18';

// Note: Lower CRF = better quality but larger file size
```

---

## Advanced Configuration

### Custom Video Settings

Edit `services/video-assembly.ts` to customize:

**Resolution**:

```typescript
// Change from 1920x1080 to 1280x720:
's=1920x1080' → 's=1280x720'
```

**Frame Rate**:

```typescript
// Change from 30fps to 60fps:
const fps = 30; → const fps = 60;
```

**Bitrate**:

```typescript
// Add after '-c:v libx264':
'-b:v 5M', // 5 Mbps video bitrate
```

**Audio Quality**:

```typescript
// Change audio bitrate:
'-b:a 192k' → '-b:a 320k' // Higher quality
```

### Ken Burns Effect Intensity

Edit the zoom parameters in `buildFilterComplex()`:

```typescript
// Slower zoom (more subtle):
'zoompan=z='min(zoom+0.0005,1.1)'' // Default: 0.0015

// Faster zoom (more dramatic):
'zoompan=z='min(zoom+0.003,1.8)''
```

---

## Performance Benchmarks

Typical video assembly times (150 shots, 5-minute video):

| Hardware           | Preset    | Time      |
| ------------------ | --------- | --------- |
| Intel i5 (4 cores) | ultrafast | 2-3 min   |
| Intel i5 (4 cores) | medium    | 5-7 min   |
| Intel i5 (4 cores) | slow      | 10-15 min |
| Intel i7 (8 cores) | medium    | 3-4 min   |
| M1 Mac             | medium    | 2-3 min   |
| M2 Mac             | medium    | 1-2 min   |

**Recommendation**: Use `medium` preset for best balance of speed and quality.

---

## Alternative: Cloud-Based Video Assembly

If FFmpeg installation is problematic, consider using cloud services:

### Option 1: Shotstack API

```typescript
import Shotstack from 'shotstack-sdk';

const client = new Shotstack.Client(process.env.SHOTSTACK_API_KEY);

// Submit render job
const render = await client.render.postRender({
  timeline: {
    /* shots, audio, transitions */
  },
  output: { format: 'mp4', resolution: '1080' },
});

// Poll for completion
const status = await client.render.getRender(render.data.response.id);
```

**Pros**: No local installation, fast cloud rendering
**Cons**: Cost ($0.05-0.10 per video), requires API key

### Option 2: Creatomate API

Similar to Shotstack, cloud-based video rendering.

**Pros**: Good documentation, template system
**Cons**: Cost ($0.08 per video)

### Option 3: FFmpeg.wasm (Browser-based)

```bash
npm install @ffmpeg/ffmpeg @ffmpeg/util
```

**Pros**: No server installation, runs in browser
**Cons**: Slower performance, limited to browser resources

---

## Support

### FFmpeg Documentation

- Official docs: https://ffmpeg.org/documentation.html
- Examples: https://trac.ffmpeg.org/wiki
- Community: https://superuser.com/questions/tagged/ffmpeg

### OYKH-Specific Issues

If you encounter issues with OYKH's video assembly:

1. Check FFmpeg is installed: `ffmpeg -version`
2. Check temp directory permissions: `/temp` folder
3. Check disk space: Requires ~2GB free
4. Review error logs in console

---

**Ready?** Once FFmpeg is installed, run:

```bash
npm install
npm run dev
```

Then test video generation:

```typescript
import { jobManager } from './services/job-manager';

const jobId = jobManager.createJob('Test Video', 'Hook text', 'cosmic');
await jobManager.startJob(jobId);

// Wait for completion, then check output folder:
// ./output/Test_Video_[timestamp].mp4
```

🎬 Happy video creating!
