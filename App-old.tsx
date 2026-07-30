import React, { useState, useEffect } from 'react';
import { createVideoJob, getVideoJob, getAllVideoJobs } from './src/api';
import { VideoJob, ProductionVibe } from './src/types';

const App: React.FC = () => {
  const [topic, setTopic] = useState('');
  const [vibe, setVibe] = useState<ProductionVibe>('minimal');
  const [jobs, setJobs] = useState<VideoJob[]>([]);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const activeJob = jobs.find((j) => j.jobId === activeJobId);

  // Poll for job updates
  useEffect(() => {
    if (isProcessing) {
      const interval = setInterval(async () => {
        try {
          const updatedJobs = await getAllVideoJobs();
          setJobs(updatedJobs);
        } catch (error) {
          console.error('Failed to fetch job updates:', error);
        }
      }, 2000); // Poll every 2 seconds

      return () => clearInterval(interval);
    }
  }, [isProcessing]);

  const handleStartJob = async () => {
    if (!topic.trim()) return;

    setIsProcessing(true);
    try {
      const { jobId } = await createVideoJob(topic, 'The Secret of ' + topic, 'minimal');
      setActiveJobId(jobId);
    } catch (error) {
      console.error('Failed to start render job:', error);
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#FDFDFD] text-black font-sans selection:bg-blue-600 selection:text-white antialiased">
      <header className="fixed top-0 w-full p-6 md:p-10 flex flex-col md:flex-row justify-between items-center bg-white/95 backdrop-blur-3xl z-50 border-b-[6px] border-black gap-6">
        <div
          className="flex items-center gap-6 cursor-pointer"
          onClick={() => window.location.reload()}
        >
          <div className="w-14 h-14 bg-black rounded-2xl flex items-center justify-center text-white font-black italic text-2xl rotate-3 shadow-[6px_6px_0_0_rgba(37,99,235,1)]">
            OY
          </div>
          <h1 className="text-3xl md:text-4xl font-black uppercase tracking-tighter leading-none">
            Studio <span className="text-blue-600 italic">Core</span>
          </h1>
        </div>
      </header>

      <main className="max-w-7xl mx-auto pt-56 px-8 pb-40">
        {!activeJob && (
          <div className="max-w-5xl mx-auto text-center py-10 animate-in fade-in slide-in-from-bottom-10">
            <h2 className="text-6xl md:text-[10rem] font-black leading-[0.75] tracking-tighter mb-20 uppercase italic">
              ANIMATE.
              <br />
              THE FUTURE.
            </h2>
            <div className="relative max-w-4xl mx-auto mb-16">
              <input
                type="text"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleStartJob()}
                placeholder="Topic for your video..."
                className="w-full px-12 py-10 text-3xl md:text-5xl rounded-[60px] border-[8px] border-black shadow-[16px_16px_0_0_rgba(0,0,0,1)] focus:outline-none focus:ring-[20px] focus:ring-blue-100 font-black transition-all placeholder:text-gray-100 italic"
              />
              <button
                onClick={handleStartJob}
                className="absolute right-6 top-6 bottom-6 px-10 bg-black text-white rounded-[50px] font-black text-xl hover:bg-blue-600 transition-all shadow-xl"
              >
                GENERATE
              </button>
            </div>
          </div>
        )}

        {activeJob && (
          <div className="animate-in fade-in">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-end mb-20 gap-8">
              <div>
                <h3 className="text-7xl md:text-8xl font-black uppercase tracking-tighter italic leading-none">
                  PIPELINE.
                </h3>
                <p className="mt-6 text-xl font-black text-blue-600 uppercase tracking-widest italic">
                  {topic}
                </p>
              </div>
              {activeJob.status !== 'running' && activeJob.status !== 'queued' && (
                <button
                  onClick={() => window.location.reload()}
                  className="px-16 py-8 bg-black text-white rounded-[40px] font-black text-3xl shadow-2xl hover:-translate-y-2 transition-all active:scale-95"
                >
                  START OVER
                </button>
              )}
            </div>

            {/* Progress Bar */}
            <div className="mb-20">
              <div className="h-8 bg-gray-200 rounded-full overflow-hidden border-4 border-black">
                <div
                  className="h-full bg-blue-600 transition-all duration-500"
                  style={{ width: `${activeJob.progress}%` }}
                ></div>
              </div>
              <p className="mt-4 text-center text-xl font-black uppercase tracking-widest italic">
                {activeJob.message}
              </p>
            </div>

            {activeJob.status === 'complete' && activeJob.videoUrl && (
              <div className="animate-in zoom-in-95 duration-700">
                <div className="text-center mb-20">
                  <h3 className="text-8xl md:text-[10rem] font-black uppercase tracking-tighter leading-[0.7] italic mb-6">
                    PREMIERE.
                  </h3>
                  <a
                    href={activeJob.videoUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-block px-10 py-3 bg-black text-white rounded-full font-black text-xs tracking-[1em] uppercase border-[6px] border-blue-600 italic hover:bg-blue-800 transition-all"
                  >
                    DOWNLOAD VIDEO
                  </a>
                </div>
                <video
                  src={activeJob.videoUrl}
                  controls
                  className="w-full max-w-6xl mx-auto rounded-[60px] border-[16px] border-black shadow-2xl"
                />
              </div>
            )}

            {activeJob.status === 'script-failed' && (
              <div className="text-center text-red-500 font-black text-3xl">
                <p>Job Failed: {activeJob.error}</p>
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="p-16 text-center border-t-2 border-black/5 opacity-40">
        <p className="text-[10px] font-black uppercase tracking-[2em] italic">
          Studio Core Hyper-Scale Engine v6.1
        </p>
      </footer>
    </div>
  );
};

export default App;
