import { startVideoGeneration } from './video-pipeline';
import { jobManager } from './job-manager';
import { generateViralVideoScript } from '../../services/gemini';
import { generateAllImages } from '../../services/imagen';
import { generateChapterAudio } from '../../services/tts';
import { assembleVideo } from './video-assembly';

jest.mock('./job-manager');
jest.mock('../../services/gemini');
jest.mock('../../services/imagen');
jest.mock('../../services/tts');
jest.mock('./video-assembly');

describe('startVideoGeneration', () => {
  it('should run the video generation pipeline successfully', async () => {
    const jobId = 'test-job-id';
    const topic = 'test-topic';
    const hook = 'test-hook';

    const mockJob = { id: jobId, topic, vibe: 'cosmic', shots: [], chapters: [] };
    const mockScript = { shots: [{ id: 1 }], chapters: [{ id: 1 }], vibe: 'cosmic', voice: 'echo' };

    (jobManager.getJob as jest.Mock).mockResolvedValue(mockJob);
    (jobManager.updateJob as jest.Mock).mockResolvedValue(undefined);
    (generateViralVideoScript as jest.Mock).mockResolvedValue(mockScript);
    (generateAllImages as jest.Mock).mockResolvedValue([]);
    (generateChapterAudio as jest.Mock).mockResolvedValue({ audioMap: new Map(), failedCount: 0 });
    (assembleVideo as jest.Mock).mockResolvedValue('http://fake-video-url.com/video.mp4');

    await startVideoGeneration(jobId, topic, hook);

    expect(jobManager.getJob).toHaveBeenCalledWith(jobId);
    expect(generateViralVideoScript).toHaveBeenCalledWith(topic, hook, 'cosmic');
    expect(jobManager.updateJob).toHaveBeenCalledWith(
      jobId,
      expect.objectContaining({ status: 'running' })
    );
    expect(generateAllImages).toHaveBeenCalled();
    expect(generateChapterAudio).toHaveBeenCalled();
    expect(assembleVideo).toHaveBeenCalled();
    expect(jobManager.updateJob).toHaveBeenCalledWith(
      jobId,
      expect.objectContaining({
        status: 'complete',
        videoUrl: 'http://fake-video-url.com/video.mp4',
      })
    );
  });
});
