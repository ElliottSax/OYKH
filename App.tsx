import React, { useState } from 'react';
import { generateViralVideoScript } from './src/gemini';
import { ViralVideoScript, ProductionVibe } from './src/types';

type AppState = 'input' | 'generating' | 'complete' | 'error' | 'generating-video';

const App: React.FC = () => {
  const [topic, setTopic] = useState('');
  const [vibe, setVibe] = useState<ProductionVibe>('minimal');
  const [state, setState] = useState<AppState>('input');
  const [script, setScript] = useState<ViralVideoScript | null>(null);
  const [error, setError] = useState<string>('');
  const [progress, setProgress] = useState(0);
  const [step, setStep] = useState<string>('');
  const [videoUrl, setVideoUrl] = useState<string>('');

  const vibeOptions: { value: ProductionVibe; label: string; desc: string }[] = [
    { value: 'minimal', label: 'Minimal', desc: 'Clean, simple, educational' },
    { value: 'cosmic', label: 'Cosmic', desc: 'Space, wonder, deep questions' },
    { value: 'hype', label: 'Hype', desc: 'Energetic, exciting, bold' },
    { value: 'suspense', label: 'Suspense', desc: 'Mysterious, dramatic, intense' },
    { value: 'success', label: 'Success', desc: 'Inspiring, motivational, golden' },
  ];

  const handleGenerate = async () => {
    if (!topic.trim()) return;

    setState('generating');
    setProgress(0);
    setError('');

    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setProgress((p) => Math.min(p + 10, 90));
      }, 500);

      // Call backend API instead of Gemini directly
      const response = await fetch('http://localhost:3100/api/generate-script', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic: topic.trim(),
          vibe: vibe,
        }),
      });

      if (!response.ok) {
        throw new Error('Script generation failed');
      }

      const result = await response.json();

      clearInterval(progressInterval);
      setProgress(100);
      setScript(result);
      setState('complete');
    } catch (err) {
      setState('error');
      setError(err instanceof Error ? err.message : 'Failed to generate script');
      console.error('Generation error:', err);
    }
  };

  const handleReset = () => {
    setState('input');
    setTopic('');
    setScript(null);
    setError('');
    setProgress(0);
    setVideoUrl('');
  };

  const handleGenerateVideo = async () => {
    if (!script) return;

    setState('generating-video');
    setProgress(0);
    setStep('Starting...');
    setError('');

    try {
      const response = await fetch('http://localhost:3100/api/generate-video-simple', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ script }),
      });

      if (!response.ok) {
        throw new Error('Video generation failed');
      }

      // Handle streaming response
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      if (!reader) throw new Error('Failed to start stream reader');

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Split by newlines and process each JSON object
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep partial line in buffer

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line);

            if (data.progress !== undefined) setProgress(data.progress);
            if (data.step) setStep(data.step);
            if (data.videoUrl) setVideoUrl(data.videoUrl);
            if (data.error) throw new Error(data.error);

            console.log('📡 Progress Update:', data);
          } catch (e) {
            console.warn('Failed to parse progress chunk:', line, e);
          }
        }
      }

      setProgress(100);
      console.log('✅ Video generation complete');
    } catch (err) {
      setState('error');
      setError(err instanceof Error ? err.message : 'Failed to generate video');
      console.error('Video generation error:', err);
    }
  };

  return (
    <div className="min-h-screen bg-[#FDFDFD] text-black font-sans selection:bg-blue-600 selection:text-white antialiased">
      {/* Header */}
      <header className="fixed top-0 w-full p-6 md:p-10 flex flex-col md:flex-row justify-between items-center bg-white/95 backdrop-blur-3xl z-50 border-b-[6px] border-black gap-6">
        <div className="flex items-center gap-6 cursor-pointer" onClick={handleReset}>
          <div className="w-14 h-14 bg-black rounded-2xl flex items-center justify-center text-white font-black italic text-2xl rotate-3 shadow-[6px_6px_0_0_rgba(37,99,235,1)]">
            OY
          </div>
          <h1 className="text-3xl md:text-4xl font-black uppercase tracking-tighter leading-none">
            Viral Video <span className="text-blue-600 italic">Generator</span>
          </h1>
        </div>
      </header>

      <main className="max-w-7xl mx-auto pt-56 px-8 pb-40">
        {/* Input State */}
        {state === 'input' && (
          <div className="max-w-5xl mx-auto text-center py-10 animate-in fade-in slide-in-from-bottom-10">
            <h2 className="text-6xl md:text-[10rem] font-black leading-[0.75] tracking-tighter mb-20 uppercase italic">
              CREATE.
              <br />
              VIRAL.
            </h2>

            {/* Topic Input */}
            <div className="relative max-w-4xl mx-auto mb-16">
              <input
                type="text"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleGenerate()}
                placeholder="Topic for your video..."
                className="w-full px-12 py-10 text-3xl md:text-5xl rounded-[60px] border-[8px] border-black shadow-[16px_16px_0_0_rgba(0,0,0,1)] focus:outline-none focus:ring-[20px] focus:ring-blue-100 font-black transition-all placeholder:text-gray-300 italic"
              />
              <button
                onClick={handleGenerate}
                disabled={!topic.trim()}
                className="absolute right-6 top-6 bottom-6 px-10 bg-black text-white rounded-[50px] font-black text-xl hover:bg-blue-600 transition-all shadow-xl disabled:opacity-50 disabled:cursor-not-allowed"
              >
                GENERATE
              </button>
            </div>

            {/* Vibe Selector */}
            <div className="max-w-4xl mx-auto">
              <p className="text-xl font-black uppercase tracking-widest mb-6">Choose a Vibe</p>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                {vibeOptions.map((option) => (
                  <button
                    key={option.value}
                    onClick={() => setVibe(option.value)}
                    className={`p-6 rounded-3xl border-4 transition-all ${
                      vibe === option.value
                        ? 'border-blue-600 bg-blue-50 scale-105'
                        : 'border-black hover:border-blue-600'
                    }`}
                  >
                    <div className="font-black text-xl mb-2">{option.label}</div>
                    <div className="text-sm opacity-60">{option.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Example Topics */}
            <div className="mt-16 text-gray-400">
              <p className="text-sm font-black uppercase mb-4">Try these:</p>
              <div className="flex flex-wrap gap-3 justify-center">
                {[
                  'Why Do We Dream?',
                  'The Butterfly Effect',
                  'Dark Matter',
                  'AI Consciousness',
                  'Time Dilation',
                ].map((example) => (
                  <button
                    key={example}
                    onClick={() => setTopic(example)}
                    className="px-6 py-2 bg-gray-100 rounded-full text-sm font-bold hover:bg-blue-100 transition-all"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Generating State */}
        {state === 'generating' && (
          <div className="animate-in fade-in">
            <div className="text-center mb-20">
              <h3 className="text-7xl md:text-8xl font-black uppercase tracking-tighter italic leading-none mb-6">
                CREATING.
              </h3>
              <p className="mt-6 text-xl font-black text-blue-600 uppercase tracking-widest italic">
                {topic}
              </p>
            </div>

            {/* Progress Bar */}
            <div className="max-w-4xl mx-auto mb-20">
              <div className="h-8 bg-gray-200 rounded-full overflow-hidden border-4 border-black">
                <div
                  className="h-full bg-blue-600 transition-all duration-500"
                  style={{ width: `${progress}%` }}
                ></div>
              </div>
              <p className="mt-4 text-center text-xl font-black uppercase tracking-widest italic">
                Analyzing viral patterns...
              </p>
            </div>

            {/* Loading Animation */}
            <div className="text-center text-6xl animate-pulse">🎬</div>
          </div>
        )}

        {/* Complete State */}
        {state === 'complete' && script && (
          <div className="animate-in fade-in">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-end mb-20 gap-8">
              <div>
                <h3 className="text-7xl md:text-8xl font-black uppercase tracking-tighter italic leading-none">
                  SCRIPT.
                </h3>
                <p className="mt-6 text-xl font-black text-blue-600 uppercase tracking-widest italic">
                  {script.metadata.topic}
                </p>
              </div>
              <div className="flex gap-4">
                <button
                  onClick={handleGenerateVideo}
                  className="px-16 py-8 bg-gradient-to-r from-indigo-600 to-blue-600 text-white rounded-[40px] font-black text-3xl shadow-2xl hover:-translate-y-2 transition-all active:scale-95"
                >
                  🎬 GENERATE VIDEO
                </button>
                <button
                  onClick={handleReset}
                  className="px-16 py-8 bg-black text-white rounded-[40px] font-black text-3xl shadow-2xl hover:-translate-y-2 transition-all active:scale-95"
                >
                  NEW VIDEO
                </button>
              </div>
            </div>

            {/* Script Details */}
            <div className="space-y-8">
              {/* Metadata Card */}
              <div className="bg-white rounded-[40px] border-[8px] border-black p-12 shadow-[16px_16px_0_0_rgba(0,0,0,1)]">
                <h4 className="text-4xl font-black mb-8 uppercase italic">📊 Overview</h4>
                <div className="grid md:grid-cols-2 gap-6 text-lg">
                  <div>
                    <span className="font-black">Title:</span> {script.metadata.title}
                  </div>
                  <div>
                    <span className="font-black">Hook:</span> {script.metadata.hook}
                  </div>
                  <div>
                    <span className="font-black">Total Shots:</span> {script.totalShots}
                  </div>
                  <div>
                    <span className="font-black">Estimated Cost:</span> $
                    {script.estimatedCost.toFixed(2)}
                  </div>
                  <div>
                    <span className="font-black">Duration:</span> {script.metadata.targetDuration}s
                    ({Math.floor(script.metadata.targetDuration / 60)}min)
                  </div>
                  <div>
                    <span className="font-black">Vibe:</span> {script.metadata.vibe}
                  </div>
                </div>
              </div>

              {/* Thumbnail Card */}
              <div className="bg-white rounded-[40px] border-[8px] border-black p-12 shadow-[16px_16px_0_0_rgba(0,0,0,1)]">
                <h4 className="text-4xl font-black mb-8 uppercase italic">🎨 Thumbnail Concept</h4>
                <div className="grid md:grid-cols-2 gap-6 text-lg">
                  <div>
                    <span className="font-black">Main Element:</span>{' '}
                    {script.metadata.thumbnailConcept.mainElement}
                  </div>
                  <div>
                    <span className="font-black">Emotion:</span>{' '}
                    {script.metadata.thumbnailConcept.emotion}
                  </div>
                  <div>
                    <span className="font-black">Text:</span> "
                    {script.metadata.thumbnailConcept.text}"
                  </div>
                  <div>
                    <span className="font-black">Colors:</span>{' '}
                    {script.metadata.thumbnailConcept.colorScheme}
                  </div>
                </div>
              </div>

              {/* Chapters */}
              <div className="space-y-6">
                <h4 className="text-4xl font-black uppercase italic">
                  📝 Chapters ({script.chapters.length})
                </h4>
                {script.chapters.map((chapter, idx) => (
                  <div
                    key={idx}
                    className="bg-white rounded-[40px] border-[8px] border-black p-12 shadow-[16px_16px_0_0_rgba(0,0,0,1)]"
                  >
                    <div className="flex justify-between items-start mb-6">
                      <div>
                        <span className="text-blue-600 font-black text-xl">
                          #{chapter.chapterNumber}
                        </span>
                        <h5 className="text-3xl font-black mt-2">{chapter.title}</h5>
                        <p className="text-gray-600 font-bold mt-2">
                          {chapter.timestamp} • {chapter.duration}s • {chapter.purpose}
                        </p>
                      </div>
                      <span className="bg-blue-100 text-blue-800 px-4 py-2 rounded-full font-black text-sm">
                        {chapter.shots.length} shots
                      </span>
                    </div>
                    <p className="text-lg leading-relaxed mb-6">{chapter.narration}</p>
                    <div className="bg-blue-50 rounded-2xl p-6">
                      <span className="font-black text-sm uppercase tracking-wider text-blue-600">
                        Key Message:
                      </span>
                      <p className="mt-2 font-bold">{chapter.keyMessage}</p>
                    </div>
                  </div>
                ))}
              </div>

              {/* Open Loops */}
              {script.openLoops.length > 0 && (
                <div className="bg-white rounded-[40px] border-[8px] border-black p-12 shadow-[16px_16px_0_0_rgba(0,0,0,1)]">
                  <h4 className="text-4xl font-black mb-8 uppercase italic">🪝 Open Loops</h4>
                  <div className="space-y-4">
                    {script.openLoops.map((loop, idx) => (
                      <div key={idx} className="flex items-start gap-4">
                        <span className="bg-blue-600 text-white w-8 h-8 rounded-full flex items-center justify-center font-black flex-shrink-0">
                          {idx + 1}
                        </span>
                        <div className="flex-1">
                          <p className="font-bold text-lg">"{loop.question}"</p>
                          <p className="text-gray-600 text-sm mt-1">
                            Posed at {loop.posedAt}s → Resolved at {loop.resolvedAt}s •{' '}
                            {loop.intensity} intensity
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Retention Bombs */}
              {script.retentionBombs.length > 0 && (
                <div className="bg-white rounded-[40px] border-[8px] border-black p-12 shadow-[16px_16px_0_0_rgba(0,0,0,1)]">
                  <h4 className="text-4xl font-black mb-8 uppercase italic">💣 Retention Bombs</h4>
                  <div className="space-y-4">
                    {script.retentionBombs.map((bomb, idx) => (
                      <div key={idx} className="bg-gray-50 rounded-2xl p-6">
                        <div className="flex justify-between items-start mb-2">
                          <span className="bg-red-600 text-white px-3 py-1 rounded-full font-black text-xs uppercase">
                            {bomb.type}
                          </span>
                          <span className="text-gray-600 font-bold text-sm">{bomb.timestamp}s</span>
                        </div>
                        <p className="font-bold mt-2">"{bomb.content}"</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Video Generation State */}
        {state === 'generating-video' && (
          <div className="animate-in fade-in">
            <div className="text-center mb-20">
              <h3 className="text-7xl md:text-8xl font-black uppercase tracking-tighter italic leading-none mb-6">
                RENDERING.
              </h3>
              <p className="mt-6 text-xl font-black text-blue-600 uppercase tracking-widest italic">
                Creating your viral video...
              </p>
            </div>

            {/* Progress Bar */}
            <div className="max-w-4xl mx-auto mb-20">
              <div className="h-8 bg-gray-200 rounded-full overflow-hidden border-4 border-black">
                <div
                  className="h-full bg-gradient-to-r from-indigo-600 to-blue-600 transition-all duration-500"
                  style={{ width: `${progress}%` }}
                ></div>
              </div>
              <p className="mt-4 text-center text-xl font-black uppercase tracking-widest italic">
                {step || `Assembling ${script?.totalShots || 0} shots with FFmpeg...`}
              </p>
            </div>

            {/* Video Result */}
            {videoUrl && (
              <div className="max-w-5xl mx-auto">
                <div className="bg-white rounded-[40px] border-[8px] border-black p-12 shadow-[16px_16px_0_0_rgba(0,0,0,1)]">
                  <h4 className="text-4xl font-black mb-8 uppercase italic">🎉 VIDEO READY!</h4>
                  <video
                    src={videoUrl}
                    controls
                    className="w-full rounded-2xl border-4 border-black mb-8"
                  >
                    Your browser does not support the video tag.
                  </video>
                  <div className="flex gap-4">
                    <a
                      href={videoUrl}
                      download
                      className="flex-1 px-8 py-6 bg-gradient-to-r from-indigo-600 to-blue-600 text-white rounded-[30px] font-black text-2xl text-center shadow-xl hover:-translate-y-1 transition-all"
                    >
                      DOWNLOAD VIDEO
                    </a>
                    <button
                      onClick={handleReset}
                      className="px-8 py-6 bg-black text-white rounded-[30px] font-black text-2xl shadow-xl hover:-translate-y-1 transition-all"
                    >
                      NEW VIDEO
                    </button>
                  </div>
                </div>
              </div>
            )}

            <div className="text-center text-6xl animate-pulse mt-12">🎬</div>
          </div>
        )}

        {/* Error State */}
        {state === 'error' && (
          <div className="text-center py-20">
            <h3 className="text-6xl font-black text-red-600 mb-8">ERROR</h3>
            <p className="text-2xl mb-12">{error}</p>
            <button
              onClick={handleReset}
              className="px-16 py-8 bg-black text-white rounded-[40px] font-black text-3xl shadow-2xl hover:-translate-y-2 transition-all"
            >
              TRY AGAIN
            </button>
          </div>
        )}
      </main>

      <footer className="p-16 text-center border-t-2 border-black/5 opacity-40">
        <p className="text-[10px] font-black uppercase tracking-[2em] italic">
          OYKH Viral Video Generator • Powered by Gemini 1.5 Flash
        </p>
      </footer>
    </div>
  );
};

export default App;
