import { v4 as uuidv4 } from 'uuid';
import * as admin from 'firebase-admin';

import { VideoJob } from '../../types';

class JobManager {
  private db: admin.firestore.Firestore;
  private jobsCollection: admin.firestore.CollectionReference<VideoJob>;

  constructor() {
    if (!admin.apps.length) {
      admin.initializeApp();
    }
    this.db = admin.firestore();
    this.jobsCollection = this.db.collection(
      'videoJobs'
    ) as admin.firestore.CollectionReference<VideoJob>;
  }

  async createJob(topic: string): Promise<string> {
    const jobId = uuidv4();
    const job: VideoJob = {
      id: jobId,
      jobId,
      topic,
      vibe: 'cosmic', // default vibe
      status: 'queued',
      progress: 0,
      currentStep: 'script',
      message: 'Job queued',
      createdAt: admin.firestore.Timestamp.now(),
      updatedAt: admin.firestore.Timestamp.now(),
    };
    await this.jobsCollection.doc(jobId).set(job);
    return jobId;
  }

  async updateJob(
    jobId: string,
    updates: Partial<VideoJob> & { updatedAt?: admin.firestore.Timestamp }
  ) {
    const jobRef = this.jobsCollection.doc(jobId);
    updates.updatedAt = admin.firestore.Timestamp.now();
    await jobRef.update(updates);
  }

  async getJob(jobId: string): Promise<VideoJob | undefined> {
    const jobDoc = await this.jobsCollection.doc(jobId).get();
    return jobDoc.exists ? jobDoc.data() : undefined;
  }

  async getAllJobs(): Promise<VideoJob[]> {
    const snapshot = await this.jobsCollection.orderBy('createdAt', 'desc').get();
    return snapshot.docs.map((doc) => doc.data());
  }
}

// Export a singleton instance
export const jobManager = new JobManager();
export type { VideoJob };
