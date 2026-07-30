import { onRequest } from 'firebase-functions/v2/https';
import { startVideoGeneration } from './video-pipeline';
import { jobManager } from './job-manager';

// =============================================================================
// VIDEO GENERATION JOBS
// =============================================================================

export const videoJobs = onRequest(async (req, res) => {
  // Enable CORS
  res.set('Access-Control-Allow-Origin', '*');
  res.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.set('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.status(200).send();
    return;
  }

  try {
    // POST /jobs - Create a new job
    if (req.method === 'POST') {
      const { topic } = req.body;
      if (!topic) {
        res.status(400).json({ error: 'Missing required field: topic' });
        return;
      }

      const jobId = await jobManager.createJob(topic);
      // Start the pipeline asynchronously
      startVideoGeneration(jobId, topic);

      res.status(202).json({ jobId });
      return;
    }

    // GET /jobs - Get all jobs
    if (req.method === 'GET' && !req.path.split('/')[1]) {
      const jobs = await jobManager.getAllJobs();
      res.status(200).json(jobs);
      return;
    }

    // GET /jobs/:jobId - Get a specific job
    if (req.method === 'GET') {
      const jobId = req.path.split('/')[1];
      if (!jobId) {
        res.status(400).json({ error: 'Invalid job ID' });
        return;
      }

      const job = await jobManager.getJob(jobId);
      if (job) {
        res.status(200).json(job);
      } else {
        res.status(404).json({ error: `Job with ID ${jobId} not found` });
      }
      return;
    }

    res.status(405).json({ error: 'Method not allowed' });
  } catch (error) {
    console.error('Job endpoint error:', error);
    res.status(500).json({ error: 'Job processing failed' });
  }
});
