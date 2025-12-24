
import React, { useState, useEffect, useMemo, useRef } from 'react';
import { AppStep, Scene } from './types';
import { generateScript, generateSceneImage, generateSceneVideo, generateSceneAudio } from './services/gemini';

const SUGGESTED_TOPICS = [
  "Quantum physics is child's play... once you know how",
  "Trading stocks is a snap... once you know how",
  "Speed reading is effortless... once you know how",
  "Lucid dreaming is second nature... once you know how",
  "Building a cabin is straightforward... once you know how",
  "Mastering AI is a breeze... once you know how",
  "Reading body language is a cinch... once you know how",
  "Winning any debate is a snap... once you know how",
  "Surviving the wild is basic... once you know how",
  "Writing a bestseller is intuitive... once you know how",
  "Starting a startup is a snap... once you know how",
  "Baking sourdough is a breeze... once you know how",
  "Public speaking is child's play... once you know how",
  "Parallel parking is a cinch... once you know how",
  "Mastering chess is second nature... once you know how",
  "Tying a bow tie is a snap... once you know how"
];

const TheaterPlayer: React.FC<{ scenes: Scene[] }> = ({ scenes }) => {
  const [currentIdx, setCurrentIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  const currentScene = scenes[currentIdx];

  const handlePlay = () => {
    setIsPlaying(true);
    videoRef.current?.play();
    audioRef.current?.play();
  };

  const handleEnd = () => {
    if (currentIdx < scenes.length - 1) {
      setCurrentIdx(currentIdx + 1);
    } else {
      setIsPlaying(false);
      setCurrentIdx(0);
    }
  };

  useEffect(() => {
    if (isPlaying) {
      // Sync refs when scene changes
      videoRef.current?.load();
      audioRef.current?.load();
      videoRef.current?.play();
      audioRef.current?.play();
    }
  }, [currentIdx, isPlaying]);

  return (
    <div className="w-full max-w-6xl mx-auto">
      <div className="relative aspect-video bg-black rounded-[48px] overflow-hidden border-[12px] border-black shadow-[0_64px_128px_-24px_rgba(0,0,0,0.5)] group">
        <video 
          ref={videoRef}
          src={currentScene.videoUrl}
          className="w-full h-full object-cover"
          onEnded={handleEnd}
          muted={true}
        />
        <audio 
          ref={audioRef}
          src={currentScene.audioUrl}
        />
        
        {!isPlaying && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/40 backdrop-blur-sm transition-all">
            <button 
              onClick={handlePlay}
              className="w-32 h-32 bg-blue-600 rounded-full flex items-center justify-center shadow-2xl hover:scale-110 active:scale-95 transition-all group"
            >
              <svg className="w-12 h-12 text-white ml-2" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
            </button>
          </div>
        )}

        <div className="absolute top-8 left-8 flex items-center gap-4">
          <div className="px-6 py-2 bg-white/10 backdrop-blur-xl border border-white/20 rounded-full text-white font-black text-xs uppercase tracking-[0.2em]">
            Scene {String(currentIdx + 1).padStart(2, '0')} / {scenes.length}
          </div>
          <div className="px-6 py-2 bg-blue-600 rounded-full text-white font-black text-xs uppercase tracking-[0.2em]">
            Premiere Mode
          </div>
        </div>

        <div className="absolute bottom-12 left-12 right-12">
           <div className="mb-6 opacity-0 group-hover:opacity-100 transition-opacity">
              <h4 className="text-white text-3xl font-black uppercase tracking-tighter drop-shadow-lg">{currentScene.title}</h4>
              <p className="text-white/80 font-bold text-xl drop-shadow-lg italic">"{currentScene.script}"</p>
           </div>
           <div className="w-full h-2 bg-white/20 rounded-full overflow-hidden">
              <div 
                className="h-full bg-blue-500 transition-all duration-300" 
                style={{ width: `${((currentIdx + 1) / scenes.length) * 100}%` }}
              />
           </div>
        </div>
      </div>
    </div>
  );
};

const App: React.FC = () => {
  const [step, setStep] = useState<AppStep>(AppStep.START);
  const [topic, setTopic] = useState('');
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isKeyReady, setIsKeyReady] = useState(false);

  useEffect(() => {
    const checkKey = async () => {
      // @ts-ignore
      const hasKey = await window.aistudio.hasSelectedApiKey();
      setIsKeyReady(hasKey);
    };
    checkKey();
  }, []);

  const handleOpenKeySelector = async () => {
    // @ts-ignore
    await window.aistudio.openSelectKey();
    setIsKeyReady(true);
  };

  const startGeneration = async () => {
    if (!topic.trim()) return;
    setError(null);
    setStep(AppStep.GENERATING_SCRIPT);

    try {
      const generatedScenes = await generateScript(topic);
      setScenes(generatedScenes);
      setStep(AppStep.REFINING_SCENES);
    } catch (err: any) {
      setError(err.message || "Something went wrong.");
      setStep(AppStep.START);
    }
  };

  const processAllScenes = async () => {
    if (!isKeyReady) {
      await handleOpenKeySelector();
    }
    
    setError(null);
    const updatedScenes = [...scenes];

    for (let i = 0; i < updatedScenes.length; i++) {
      try {
        updatedScenes[i].status = 'generating-image';
        setScenes([...updatedScenes]);
        const imageData = await generateSceneImage(updatedScenes[i]);
        updatedScenes[i].imageData = imageData;

        updatedScenes[i].status = 'generating-audio';
        setScenes([...updatedScenes]);
        const audioUrl = await generateSceneAudio(updatedScenes[i]);
        updatedScenes[i].audioUrl = audioUrl;

        updatedScenes[i].status = 'generating-video';
        setScenes([...updatedScenes]);
        const videoUrl = await generateSceneVideo(updatedScenes[i], imageData);
        updatedScenes[i].videoUrl = videoUrl;
        
        updatedScenes[i].status = 'completed';
        setScenes([...updatedScenes]);
      } catch (err: any) {
        console.error(err);
        updatedScenes[i].status = 'error';
        setScenes([...updatedScenes]);
        setError(`Production failed on scene ${i + 1}. Check your API key and try again.`);
        return;
      }
    }
    setStep(AppStep.FINAL_VIDEO);
  };

  const completionProgress = useMemo(() => {
    if (scenes.length === 0) return 0;
    const completed = scenes.filter(s => s.status === 'completed').length;
    return Math.round((completed / scenes.length) * 100);
  }, [scenes]);

  return (
    <div className="min-h-screen bg-[#FAFAFA] text-black selection:bg-blue-600 selection:text-white">
      <header className="fixed top-0 w-full p-6 flex justify-between items-center bg-white/90 backdrop-blur-2xl z-50 border-b-4 border-black">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-black rounded-full flex items-center justify-center shadow-xl">
            <div className="w-4 h-4 bg-white rounded-full"></div>
          </div>
          <div>
            <h1 className="text-2xl font-black tracking-tighter uppercase leading-none">
              Once You <span className="text-blue-600 italic">Know How</span>
            </h1>
            <p className="text-[10px] font-black uppercase tracking-[0.2em] text-gray-400">Professional Studio Engine</p>
          </div>
        </div>
        <div className="flex items-center gap-6">
          {step === AppStep.REFINING_SCENES && (
            <div className="flex flex-col items-end">
              <span className="text-xs font-black uppercase tracking-widest text-gray-400">Batch Processing</span>
              <div className="w-48 h-3 bg-gray-100 rounded-full overflow-hidden border-2 border-black mt-1">
                <div className="h-full bg-blue-600 transition-all duration-500" style={{ width: `${completionProgress}%` }}></div>
              </div>
            </div>
          )}
          {!isKeyReady && (
            <button onClick={handleOpenKeySelector} className="px-5 py-2 bg-yellow-400 border-2 border-black rounded-xl font-black text-xs uppercase shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:translate-y-1 hover:shadow-none transition-all">
              Setup Key
            </button>
          )}
        </div>
      </header>

      <main className="max-w-7xl mx-auto pt-32 pb-24 px-8">
        {step === AppStep.START && (
          <div className="max-w-4xl mx-auto text-center py-20 animate-in fade-in slide-in-from-bottom-10 duration-1000">
            <div className="inline-block px-4 py-2 bg-blue-100 border-2 border-blue-600 rounded-full mb-8">
              <span className="text-xs font-black uppercase tracking-widest text-blue-600">The Ultimate How-To Suite</span>
            </div>
            <h2 className="text-[8rem] font-black leading-[0.8] tracking-tighter mb-12 lg:text-[10rem]">
              ONCE <br/>YOU <br/><span className="text-blue-600 italic underline decoration-[16px] decoration-blue-100 text-nowrap">KNOW HOW.</span>
            </h2>
            <p className="text-3xl font-bold text-gray-400 mb-16 tracking-tight leading-snug">
              60 Scripted Segments. Full Narration Sync. <br/>A 10-minute masterclass, produced in minutes.
            </p>
            <div className="relative group max-w-3xl mx-auto mb-16">
              <input 
                type="text" 
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="Baking sourdough is a breeze... once you know how"
                className="w-full px-12 py-10 text-3xl rounded-[40px] border-8 border-black shadow-[12px_12px_0px_0px_rgba(0,0,0,1)] focus:outline-none focus:ring-8 focus:ring-blue-100 transition-all font-black placeholder:text-gray-200"
                onKeyDown={(e) => e.key === 'Enter' && startGeneration()}
              />
              <button 
                onClick={startGeneration}
                className="absolute right-6 top-6 bottom-6 px-12 bg-black text-white rounded-[32px] font-black text-2xl hover:bg-blue-600 transition-all active:scale-95 disabled:opacity-50"
                disabled={!topic.trim()}
              >
                PRODUCE FILM
              </button>
            </div>

            <div className="max-w-5xl mx-auto">
              <p className="text-xs font-black text-gray-400 uppercase tracking-[0.3em] mb-6">Trending Tutorials</p>
              <div className="flex flex-wrap justify-center gap-4">
                {SUGGESTED_TOPICS.map((t) => (
                  <button
                    key={t}
                    onClick={() => setTopic(t)}
                    className="px-6 py-3 bg-white border-4 border-black rounded-2xl font-black text-sm uppercase tracking-tight hover:bg-blue-50 hover:-translate-y-1 hover:shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] transition-all active:translate-y-0 active:shadow-none"
                  >
                    {t}
                  </button>
                ))}
              </div>
            </div>

            {error && <p className="mt-12 text-red-600 font-black text-xl bg-red-50 border-4 border-red-100 p-6 rounded-3xl inline-block">{error}</p>}
          </div>
        )}

        {step === AppStep.GENERATING_SCRIPT && (
          <div className="flex flex-col items-center justify-center py-40">
            <div className="w-32 h-32 border-[16px] border-gray-100 border-t-black rounded-full animate-spin mb-12"></div>
            <h3 className="text-5xl font-black tracking-tighter uppercase">Scripting...</h3>
            <p className="text-2xl text-gray-400 font-bold mt-4 uppercase tracking-widest">Architecting 60 scenes for: {topic}</p>
          </div>
        )}

        {step === AppStep.REFINING_SCENES && (
          <div className="animate-in fade-in duration-500">
            <div className="flex justify-between items-center mb-16">
              <div>
                <h3 className="text-6xl font-black tracking-tighter uppercase">Production Queue</h3>
                <p className="text-2xl font-bold text-gray-400 uppercase tracking-tight">Synchronizing audio, video, and imagery.</p>
              </div>
              <button 
                onClick={processAllScenes}
                className="px-16 py-8 bg-black text-white rounded-[32px] font-black text-3xl shadow-[0_32px_64px_rgba(0,0,0,0.2)] hover:-translate-y-2 hover:bg-blue-600 transition-all"
              >
                BEGIN ANIMATION
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {scenes.map((scene, idx) => (
                <div key={idx} className="bg-white p-6 rounded-[32px] border-2 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] flex flex-col justify-between group hover:border-blue-600 transition-colors">
                  <div>
                    <div className="flex justify-between items-center mb-4">
                      <span className="text-xl font-black text-gray-200 group-hover:text-blue-600">{String(idx + 1).padStart(2, '0')}</span>
                      <div className="status-indicator">
                        {scene.status === 'idle' && <span className="w-3 h-3 bg-gray-100 rounded-full inline-block"></span>}
                        {scene.status === 'completed' && <span className="w-3 h-3 bg-green-500 rounded-full inline-block"></span>}
                        {['generating-image', 'generating-video', 'generating-audio'].includes(scene.status) && <span className="w-3 h-3 bg-blue-600 rounded-full animate-pulse inline-block"></span>}
                        {scene.status === 'error' && <span className="w-3 h-3 bg-red-600 rounded-full inline-block"></span>}
                      </div>
                    </div>
                    <h4 className="text-lg font-black mb-1 uppercase leading-none truncate">{scene.title}</h4>
                  </div>
                  <div className="mt-4 text-[9px] font-black text-gray-300 uppercase tracking-widest">
                    {scene.status.split('-').join(' ')}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {step === AppStep.FINAL_VIDEO && (
          <div className="max-w-6xl mx-auto py-20 animate-in fade-in duration-1000">
            <div className="text-center mb-24">
              <h3 className="text-[12rem] font-black leading-none tracking-tighter uppercase mb-4">THE PREMIERE.</h3>
              <p className="text-4xl font-bold text-gray-400 uppercase tracking-widest leading-none mt-4 italic">{topic}</p>
            </div>

            <TheaterPlayer scenes={scenes} />

            <div className="mt-40 grid grid-cols-1 gap-20">
              <div className="flex flex-col items-center">
                 <h4 className="text-2xl font-black uppercase mb-12 italic tracking-tighter">Production Overview (60 Segments)</h4>
                 <div className="flex flex-wrap justify-center gap-2 max-w-4xl">
                   {scenes.map((_, i) => (
                     <div key={i} className="w-8 h-8 bg-black rounded-sm flex items-center justify-center text-[10px] text-white font-black">
                       {i+1}
                     </div>
                   ))}
                 </div>
              </div>

              <div className="text-center pt-20">
                <button 
                  onClick={() => window.location.reload()}
                  className="px-24 py-12 bg-black text-white rounded-full font-black text-4xl hover:bg-blue-600 hover:scale-110 transition-all shadow-[0_40px_80px_rgba(0,0,0,0.3)] uppercase tracking-tighter"
                >
                  New Production
                </button>
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="w-full py-24 border-t-8 border-black bg-white flex flex-col items-center gap-6">
        <div className="flex gap-4">
          {[1,2,3,4,5].map(i => <div key={i} className="w-4 h-4 bg-gray-100 rounded-full"></div>)}
        </div>
        <p className="font-black text-xs uppercase tracking-[0.5em] text-gray-300">Once You Know How Studio — v2.0</p>
      </footer>
    </div>
  );
};

export default App;
