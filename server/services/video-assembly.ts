/**
 * Video Assembly Service - FFmpeg Pipeline
 *
 * Adapted from Topic2Manim's video concatenation pattern
 * Combines all shots into final MP4 with:
 * - Ken Burns effects (zoom/pan for static images)
 * - Audio narration sync
 * - Smooth transitions
 * - Professional quality output
 */

import { ViralVideoScript, Shot } from '../types';
import { exec } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs/promises';
import * as path from 'path';

const execAsync = promisify(exec);

/**
 * Check if FFmpeg is installed
 */
export async function checkFFmpegInstalled(): Promise<boolean> {
  try {
    await execAsync('ffmpeg -version');
    return true;
  } catch (error) {
    return false;
  }
}

/**
 * Assemble final video from shots and audio
 *
 * @param script - Full video script with shot metadata
 * @param imageUrls - Map of shotNumber → image URL/base64
 * @param audioPath - Path to audio file (or URL)
 * @param onProgress - Progress callback (0-100)
 * @returns Path to final MP4 video
 */
export async function assembleVideo(
  script: ViralVideoScript,
  imageUrls: Map<number, string>,
  audioPath: string,
  onProgress?: (percent: number) => void
): Promise<string> {
  console.log('🎬 Starting video assembly...');

  // Check FFmpeg installation
  const ffmpegInstalled = await checkFFmpegInstalled();
  if (!ffmpegInstalled) {
    throw new Error(
      'FFmpeg not installed! Install it first:\n' +
        'Windows: choco install ffmpeg\n' +
        'Mac: brew install ffmpeg\n' +
        'Linux: sudo apt install ffmpeg'
    );
  }

  // Create temp directory
  const tempDir = path.join(process.cwd(), 'temp', `video-${Date.now()}`);
  await fs.mkdir(tempDir, { recursive: true });

  try {
    // Step 1: Download/save all images to temp files (10%)
    onProgress?.(10);
    console.log('📥 Saving shot images...');
    const shotFiles = await saveImagesToFiles(script, imageUrls, tempDir);
    console.log(`✅ Saved ${shotFiles.length} shot images`);

    // Step 2: Create FFmpeg filter complex (30%)
    onProgress?.(30);
    console.log('🎨 Building FFmpeg filters (Ken Burns effects)...');
    const filterComplex = buildFilterComplex(script, shotFiles);

    // Step 3: Create input list for FFmpeg (40%)
    onProgress?.(40);
    const inputList = await createInputList(script, shotFiles, tempDir);

    // Step 4: Concatenate video without audio (50% → 70%)
    onProgress?.(50);
    console.log('🎞️  Concatenating shots into video...');
    const silentVideoPath = path.join(tempDir, 'video_silent.mp4');
    await concatenateShots(inputList, filterComplex, silentVideoPath, (p) => {
      onProgress?.(50 + p * 0.2); // 50% → 70%
    });
    console.log('✅ Silent video created');

    // Step 5: Download/prepare audio (75%)
    onProgress?.(75);
    console.log('🎵 Preparing audio track...');
    const audioFilePath = await prepareAudio(audioPath, tempDir);

    // Step 6: Merge audio with video (80% → 95%)
    onProgress?.(80);
    console.log('🎶 Merging audio with video...');
    const outputDir = path.join(process.cwd(), 'output');
    await fs.mkdir(outputDir, { recursive: true });
    const finalVideoPath = path.join(
      outputDir,
      `${script.metadata.title.replace(/[^a-z0-9]/gi, '_')}_${Date.now()}.mp4`
    );
    await mergeAudioVideo(silentVideoPath, audioFilePath, finalVideoPath, (p) => {
      onProgress?.(80 + p * 0.15); // 80% → 95%
    });
    console.log('✅ Audio merged');

    // Step 7: Cleanup temp files (100%)
    onProgress?.(100);
    console.log('🧹 Cleaning up temp files...');
    await fs.rm(tempDir, { recursive: true, force: true });

    console.log(`🎉 Video assembly complete: ${finalVideoPath}`);
    return finalVideoPath;
  } catch (error) {
    // Cleanup on error
    try {
      await fs.rm(tempDir, { recursive: true, force: true });
    } catch (cleanupError) {
      console.error('Failed to cleanup temp files:', cleanupError);
    }
    throw error;
  }
}

/**
 * Save all shot images to temporary files
 */
async function saveImagesToFiles(
  script: ViralVideoScript,
  imageUrls: Map<number, string>,
  tempDir: string
): Promise<Array<{ shotNumber: number; filePath: string; duration: number }>> {
  const shotFiles: Array<{ shotNumber: number; filePath: string; duration: number }> = [];

  // Get all shots in order
  const allShots = script.chapters.flatMap((chapter) => chapter.shots);

  for (const shot of allShots) {
    const imageUrl = imageUrls.get(shot.shotNumber);
    if (!imageUrl) {
      console.warn(`⚠️  Missing image for shot #${shot.shotNumber}, skipping`);
      continue;
    }

    const filePath = path.join(tempDir, `shot-${String(shot.shotNumber).padStart(4, '0')}.png`);

    // Handle base64 data URLs
    if (imageUrl.startsWith('data:image')) {
      const base64Data = imageUrl.split(',')[1];
      const buffer = Buffer.from(base64Data, 'base64');
      await fs.writeFile(filePath, buffer);
    }
    // Handle HTTP(S) URLs
    else if (imageUrl.startsWith('http')) {
      const response = await fetch(imageUrl);
      const arrayBuffer = await response.arrayBuffer();
      await fs.writeFile(filePath, Buffer.from(arrayBuffer));
    }
    // Handle file paths
    else {
      await fs.copyFile(imageUrl, filePath);
    }

    shotFiles.push({
      shotNumber: shot.shotNumber,
      filePath,
      duration: shot.duration,
    });
  }

  return shotFiles;
}

/**
 * Build FFmpeg filter_complex with Ken Burns effects
 *
 * Ken Burns effect: Slow zoom and pan for dynamic feel on static images
 */
function buildFilterComplex(
  script: ViralVideoScript,
  shotFiles: Array<{ shotNumber: number; filePath: string; duration: number }>
): string {
  const filters: string[] = [];
  const allShots = script.chapters.flatMap((chapter) => chapter.shots);

  shotFiles.forEach((shotFile, index) => {
    const shot = allShots.find((s) => s.shotNumber === shotFile.shotNumber);
    if (!shot) return;

    const fps = 30; // 30 frames per second
    const frames = Math.round(shot.duration * fps);

    // Ken Burns effect parameters based on shot animation type
    let kenBurnsEffect = '';

    switch (shot.animation) {
      case 'ken-burns-in':
        // Zoom in slowly
        kenBurnsEffect = `zoompan=z='min(zoom+0.0015,1.5)':d=${frames}:s=1920x1080:fps=${fps}`;
        break;

      case 'ken-burns-out':
        // Zoom out slowly
        kenBurnsEffect = `zoompan=z='if(lte(zoom,1.0),1.5,max(1.0,zoom-0.0015))':d=${frames}:s=1920x1080:fps=${fps}`;
        break;

      case 'pan-right':
        // Pan right
        kenBurnsEffect = `zoompan=z='1.2':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=${frames}:s=1920x1080:fps=${fps}`;
        break;

      case 'pan-left':
        // Pan left
        kenBurnsEffect = `zoompan=z='1.2':x='-(iw/zoom-1)*iw/2':y='ih/2-(ih/zoom/2)':d=${frames}:s=1920x1080:fps=${fps}`;
        break;

      case 'static':
      default:
        // Minimal zoom for subtle movement
        kenBurnsEffect = `zoompan=z='min(zoom+0.0005,1.1)':d=${frames}:s=1920x1080:fps=${fps}`;
        break;
    }

    // Full filter chain for this shot
    filters.push(
      `[${index}:v]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,${kenBurnsEffect}[v${index}]`
    );
  });

  // Concatenate all processed shots
  const concatInput = shotFiles.map((_, i) => `[v${i}]`).join('');
  const concatFilter = `${concatInput}concat=n=${shotFiles.length}:v=1:a=0[outv]`;

  filters.push(concatFilter);

  return filters.join(';');
}

/**
 * Create input list file for FFmpeg
 */
async function createInputList(
  script: ViralVideoScript,
  shotFiles: Array<{ shotNumber: number; filePath: string; duration: number }>,
  tempDir: string
): Promise<string> {
  const inputListPath = path.join(tempDir, 'input_list.txt');

  // FFmpeg concat demuxer format
  const lines = shotFiles.map((shot) => `file '${shot.filePath}'\nduration ${shot.duration}`);

  // Add last file without duration (FFmpeg requirement)
  if (shotFiles.length > 0) {
    lines.push(`file '${shotFiles[shotFiles.length - 1].filePath}'`);
  }

  await fs.writeFile(inputListPath, lines.join('\n'));

  return inputListPath;
}

/**
 * Concatenate all shots into single video
 */
async function concatenateShots(
  inputList: string,
  filterComplex: string,
  outputPath: string,
  onProgress?: (percent: number) => void
): Promise<void> {
  // Build FFmpeg command
  // Using filter_complex for Ken Burns effects
  const ffmpegCommand = [
    'ffmpeg',
    '-y', // Overwrite output
    `-f concat`,
    `-safe 0`,
    `-i "${inputList}"`,
    `-filter_complex "${filterComplex}"`,
    `-map "[outv]"`,
    '-c:v libx264', // H.264 codec
    '-preset medium', // Encoding speed vs compression
    '-crf 23', // Quality (lower = better, 18-28 range)
    '-pix_fmt yuv420p', // Pixel format for compatibility
    `"${outputPath}"`,
  ].join(' ');

  console.log('🎬 Running FFmpeg (this may take a few minutes)...');

  try {
    const { stdout, stderr } = await execAsync(ffmpegCommand, {
      maxBuffer: 10 * 1024 * 1024, // 10MB buffer
    });

    // Parse FFmpeg output for progress (optional)
    if (stderr && onProgress) {
      // FFmpeg outputs to stderr by default
      const progressMatch = stderr.match(/time=(\d{2}):(\d{2}):(\d{2})/);
      if (progressMatch) {
        const [_, hours, minutes, seconds] = progressMatch;
        const totalSeconds = parseInt(hours) * 3600 + parseInt(minutes) * 60 + parseInt(seconds);
        // Rough progress estimate
        onProgress(Math.min((totalSeconds / 300) * 100, 100)); // Assume ~5 min video
      }
    }

    onProgress?.(100);
  } catch (error) {
    console.error('FFmpeg error:', error);
    throw new Error(
      `Video concatenation failed: ${error instanceof Error ? error.message : 'Unknown error'}`
    );
  }
}

/**
 * Prepare audio file (download if URL, or use local file)
 */
async function prepareAudio(audioPath: string, tempDir: string): Promise<string> {
  // Handle base64 data URLs
  if (audioPath.startsWith('data:audio')) {
    const base64Data = audioPath.split(',')[1];
    const buffer = Buffer.from(base64Data, 'base64');
    const audioFilePath = path.join(tempDir, 'audio.mp3');
    await fs.writeFile(audioFilePath, buffer);
    return audioFilePath;
  }

  // Handle HTTP(S) URLs
  if (audioPath.startsWith('http')) {
    const response = await fetch(audioPath);
    const arrayBuffer = await response.arrayBuffer();
    const audioFilePath = path.join(tempDir, 'audio.mp3');
    await fs.writeFile(audioFilePath, Buffer.from(arrayBuffer));
    return audioFilePath;
  }

  // Handle local file paths
  return audioPath;
}

/**
 * Merge audio track with video
 */
async function mergeAudioVideo(
  videoPath: string,
  audioPath: string,
  outputPath: string,
  onProgress?: (percent: number) => void
): Promise<void> {
  const ffmpegCommand = [
    'ffmpeg',
    '-y', // Overwrite output
    `-i "${videoPath}"`, // Video input
    `-i "${audioPath}"`, // Audio input
    '-c:v copy', // Copy video stream (no re-encoding)
    '-c:a aac', // AAC audio codec
    '-b:a 192k', // Audio bitrate
    '-shortest', // End when shortest stream ends
    `"${outputPath}"`,
  ].join(' ');

  console.log('🎶 Merging audio with video...');

  try {
    const { stdout, stderr } = await execAsync(ffmpegCommand, {
      maxBuffer: 10 * 1024 * 1024,
    });

    onProgress?.(100);
  } catch (error) {
    console.error('FFmpeg merge error:', error);
    throw new Error(
      `Audio merge failed: ${error instanceof Error ? error.message : 'Unknown error'}`
    );
  }
}

/**
 * Get video metadata (duration, resolution, etc.)
 */
export async function getVideoMetadata(videoPath: string): Promise<{
  duration: number;
  width: number;
  height: number;
  codec: string;
  bitrate: number;
}> {
  const ffprobeCommand = `ffprobe -v quiet -print_format json -show_format -show_streams "${videoPath}"`;

  try {
    const { stdout } = await execAsync(ffprobeCommand);
    const metadata = JSON.parse(stdout);

    const videoStream = metadata.streams.find((s: any) => s.codec_type === 'video');

    return {
      duration: parseFloat(metadata.format.duration || '0'),
      width: videoStream?.width || 0,
      height: videoStream?.height || 0,
      codec: videoStream?.codec_name || 'unknown',
      bitrate: parseInt(metadata.format.bit_rate || '0'),
    };
  } catch (error) {
    throw new Error(
      `Failed to get video metadata: ${error instanceof Error ? error.message : 'Unknown error'}`
    );
  }
}

/**
 * Create thumbnail from video
 */
export async function createThumbnail(
  videoPath: string,
  timestamp: number = 3.0,
  outputPath?: string
): Promise<string> {
  const thumbnailPath = outputPath || videoPath.replace('.mp4', '_thumbnail.jpg');

  const ffmpegCommand = [
    'ffmpeg',
    '-y',
    `-ss ${timestamp}`, // Timestamp to capture
    `-i "${videoPath}"`,
    '-vframes 1', // Extract 1 frame
    '-q:v 2', // High quality
    `"${thumbnailPath}"`,
  ].join(' ');

  await execAsync(ffmpegCommand);

  return thumbnailPath;
}

/**
 * Example usage:
 *
 * import { assembleVideo } from './services/video-assembly';
 *
 * const finalVideo = await assembleVideo(
 *   script,
 *   imageUrls,
 *   audioPath,
 *   (progress) => {
 *     console.log(`Assembly progress: ${progress}%`);
 *   }
 * );
 *
 * console.log(`Video saved to: ${finalVideo}`);
 */
