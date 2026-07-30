import { exec } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs/promises';
import * as path from 'path';
import { getStorage } from 'firebase-admin/storage';
import { Shot } from '../../types'; // Assuming you have a types file

export async function checkFFmpegInstalled(): Promise<boolean> {
  try {
    await execAsync('ffmpeg -version');
    return true;
  } catch (error) {
    return false;
  }
}

const execAsync = promisify(exec);

interface AssemblyOptions {
  shots: Shot[];
  imageResults: Map<number, string>;
  audioResults: Map<number, string>;
  outputDir: string;
  jobId: string;
}

export async function assembleVideo(options: AssemblyOptions): Promise<string | undefined> {
  const { shots, imageResults, audioResults, outputDir, jobId } = options;
  const tempDir = path.join(outputDir, jobId);
  await fs.mkdir(tempDir, { recursive: true });

  // 1. Save images and create a list file for FFmpeg
  const imageFiles: string[] = [];
  let imageListContent = '';
  for (const shot of shots) {
    const imageUrl = imageResults.get(shot.shotNumber);
    if (imageUrl) {
      const imagePath = path.join(tempDir, `shot-${shot.shotNumber}.png`);
      // Assuming imageUrl is a publicly accessible URL
      // You might need to download it first if it's not a local path
      // For this example, let's assume it's a local path for simplicity
      await fs.copyFile(imageUrl, imagePath);
      imageFiles.push(imagePath);
      imageListContent += `file '${imagePath}'\nduration ${shot.duration}\n`;
    }
  }
  const imageListPath = path.join(tempDir, 'images.txt');
  await fs.writeFile(imageListPath, imageListContent);

  // 2. Save audio and create a list file for FFmpeg
  const audioFiles: string[] = [];
  for (const [chapterNumber, audioUrl] of audioResults.entries()) {
    const audioPath = path.join(tempDir, `audio-${chapterNumber}.mp3`);
    // Assuming audioUrl is a publicly accessible URL
    // You might need to download it first if it's not a local path
    await fs.copyFile(audioUrl, audioPath);
    audioFiles.push(audioPath);
  }
  const audioListPath = path.join(tempDir, 'audio.txt');
  await fs.writeFile(audioListPath, audioFiles.map((f) => `file '${f}'`).join('\n'));

  // 3. Concatenate audio
  const concatenatedAudioPath = path.join(tempDir, 'full-audio.mp3');
  const concatAudioCommand = `ffmpeg -f concat -safe 0 -i ${audioListPath} -c copy ${concatenatedAudioPath}`;
  await execAsync(concatAudioCommand);

  // 4. Create video from images
  const silentVideoPath = path.join(tempDir, 'silent-video.mp4');
  const videoCommand = `ffmpeg -f concat -safe 0 -i ${imageListPath} -c:v libx264 -r 30 -pix_fmt yuv420p ${silentVideoPath}`;
  await execAsync(videoCommand);

  // 5. Merge video and audio
  const finalVideoPath = path.join(outputDir, `${jobId}.mp4`);
  const mergeCommand = `ffmpeg -i ${silentVideoPath} -i ${concatenatedAudioPath} -c:v copy -c:a aac -shortest ${finalVideoPath}`;
  await execAsync(mergeCommand);

  // 6. Upload to Cloud Storage
  let videoUrl: string | undefined;
  if (process.env.STORAGE_BUCKET) {
    const bucket = getStorage().bucket(process.env.STORAGE_BUCKET);
    const destination = `videos/${jobId}.mp4`;
    await bucket.upload(finalVideoPath, {
      destination,
      metadata: {
        contentType: 'video/mp4',
        metadata: {
          jobId,
        },
      },
    });
    await bucket.file(destination).makePublic();
    videoUrl = `https://storage.googleapis.com/${process.env.STORAGE_BUCKET}/${destination}`;
    console.log(`✅ Final video uploaded to ${videoUrl}`);
  } else {
    console.warn('STORAGE_BUCKET environment variable not set. Skipping upload.');
  }

  // 7. Cleanup
  await fs.rm(tempDir, { recursive: true, force: true });
  await fs.rm(finalVideoPath, { force: true }); // Also remove the local final video

  return videoUrl;
}
