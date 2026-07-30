import { jobManager } from './job-manager';
import type { Shot } from '../../types';
import { generateViralVideoScript } from '../../services/gemini';
import { generateAllImages } from '../../services/imagen';
import { generateChapterAudio } from '../../services/tts';
import { assembleVideo } from './video-assembly';
import * as fs from 'fs/promises';
import * as path from 'path';

export async function startVideoGeneration(jobId: string, topic: string, hook: string) {
  const tempDir = path.join('/tmp', jobId);

  try {
    await fs.mkdir(tempDir, { recursive: true });

    // 1. Generate Script
    await jobManager.updateJob(jobId, {
      topic,
      status: 'running',
      currentStep: 'script',
      progress: 10,
      message: 'Generating script...',
    });
    const job = await jobManager.getJob(jobId);
    if (!job) {
      throw new Error(`Job with id ${jobId} not found`);
    }
    const script = await generateViralVideoScript(job.topic, hook, job.vibe);
    await jobManager.updateJob(jobId, {
      script,
      progress: 25,
      message: 'Script generated, planning shots...',
    });

    // 2. Generate Images
    await jobManager.updateJob(jobId, {
      currentStep: 'images',
      progress: 30,
      message: `Generating ${script.shots.length} images...`,
    });
    const imageGenStartTime = Date.now();

    const imageShots: Shot[] = script.shots.map((shot) => ({
      ...shot,
      vibe: script.vibe,
    }));

    const imageResults = await generateAllImages(imageShots, script.vibe);

    const imageGenTime = (Date.now() - imageGenStartTime) / 1000;
    console.log(`Image generation took ${imageGenTime}s`);

    await jobManager.updateJob(jobId, {
      imageResults,
      progress: 60,
      message: 'Image generation complete',
    });

    // 3. Generate Audio
    await jobManager.updateJob(jobId, {
      currentStep: 'audio',
      progress: 65,
      message: `Generating audio for ${script.chapters.length} chapters...`,
    });
    const audioGenStartTime = Date.now();

    const { audioMap, failedCount } = await generateChapterAudio(
      script.chapters,
      script.voice as any
    );

    const audioGenTime = (Date.now() - audioGenStartTime) / 1000;
    console.log(`Audio generation took ${audioGenTime}s`);

    if (failedCount > 0) {
      console.warn(`${failedCount} audio chapters failed to generate.`);
    }

    const audioMessage =
      failedCount > 0
        ? `Audio generation complete with ${failedCount} failed chapters.`
        : 'Audio generation complete';
    await jobManager.updateJob(jobId, { progress: 80, message: audioMessage });

    // 4. Assemble Video
    await jobManager.updateJob(jobId, {
      currentStep: 'assembly',
      progress: 85,
      message: `Assembling video with ${script.shots.length} shots...`,
    });
    const assemblyStartTime = Date.now();

    const videoUrl = await assembleVideo({
      shots: script.shots,
      imageResults,
      audioResults: audioMap,
      outputDir: '/tmp',
      jobId,
    });

    if (!videoUrl) {
      throw new Error('Video assembly failed: no video URL returned.');
    }

    const assemblyTime = (Date.now() - assemblyStartTime) / 1000;
    console.log(`Video assembly took ${assemblyTime}s`);

    // 5. Job Complete
    await jobManager.updateJob(jobId, {
      status: 'complete',
      currentStep: 'finished',
      videoUrl,
      progress: 100,
      message: 'Video assembly complete',
    });
  } catch (error) {
    console.error(`Job ${jobId} failed:`, error);
    const job = await jobManager.getJob(jobId);
    await jobManager.updateJob(jobId, {
      status: 'failed',
      error: error instanceof Error ? error.message : 'Unknown error',
      message: `Job failed at step: ${job?.currentStep}`,
    });
  } finally {
    // Cleanup
    await fs.rm(tempDir, { recursive: true, force: true });
  }
}
