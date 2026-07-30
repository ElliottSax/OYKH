import { VideoJob } from './types';

const API_BASE_URL = 'http://localhost:5001/oykh-1a3b9/us-central1/videoJobs'; // Replace with your actual function URL

export async function createVideoJob(
  topic: string,
  hook?: string,
  vibe?: string
): Promise<{ jobId: string }> {
  const response = await fetch(API_BASE_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ topic }), // Only topic is required
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to create video job');
  }

  return response.json();
}

export async function getVideoJob(jobId: string): Promise<VideoJob> {
  const response = await fetch(`${API_BASE_URL}/${jobId}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to get video job');
  }

  return response.json();
}

export async function getAllVideoJobs(): Promise<VideoJob[]> {
  const response = await fetch(API_BASE_URL);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to get all video jobs');
  }

  return response.json();
}
