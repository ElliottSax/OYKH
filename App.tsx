
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { AppStep, Scene, ProductionVibe } from './types.ts';
import { generateScript, generateSceneImage, generateSceneAudio, fetchSuggestions, fetchHooks, MUSIC_FOR_VIBE } from './services/gemini.ts';

const PRODUCTION_LOGS = [
  "Baking global illumination...",
  "Simulating character physics...",
  "Optimizing cel-shaded density...",
  "Calibrating neural vocals...",
  "Generating kinetic paths...",
  "Mastering audio buffers..."
];

const SeamlessPlayer: React.FC<{ scenes: Scene[], vibe: ProductionVibe }> = ({ scenes, vibe }) => {
  const [currentIdx, setCurrentIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement>(null);
  const bgMusicRef = useRef<HTMLAudioElement>(null);
  
  const currentScene = scenes[currentIdx];

  const handleEnd = useCallback(() => {
    if (currentIdx < scenes.length - 1) {
      setCurrentIdx(prev => prev + 1);
    } else {
      setIsPlaying(false);
      setCurrentIdx(0);
      if (bgMusicRef.current) {
        bgMusicRef.current.pause();
        bgMusicRef.current.currentTime = 0;
      }
    }
  }, [currentIdx, scenes.length]);

  const togglePlay = useCallback(() => {
    if (isPlaying) {
      audioRef.current?.pause();
      bgMusicRef.current?.pause();
      setIsPlaying(false);
    } else {
      setIsPlaying(true);
      // Attempt playback
      const playAudio = async () => {
        try {
          if (audioRef.current) await audioRef.current.play();
          if (bgMusicRef.current) {
            bgMusicRef.current.volume = 0.15;
            await bgMusicRef.current.play();
          }
        } catch (e) {
          console.warn("Autoplay blocked or failed:", e);
          setIsPlaying(false); // Revert to paused state if blocked
        }
      };
      playAudio();
    }
  }, [isPlaying]);

  useEffect(() => {
    const handleKeydown = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        e.preventDefault();
        togglePlay();
      }
      if (e.code === 'Escape') {
        setIsPlaying(false);
        setCurrentIdx(0);
        audioRef.current?.pause();
        bgMusicRef.current?.pause();
      }
    };
    window.addEventListener('keydown', handleKeydown);
    return () => window.removeEventListener('keydown', handleKeydown);
  }, [togglePlay]);

  // Sync audio track when slide changes automatically
  useEffect(() => {
    if (isPlaying && audioRef.current) {
      audioRef.current.play().catch(() => {});
    }
  }, [currentIdx, isPlaying]);

  if (!currentScene) return <div className="p-10 text-center font-bold text-red-500">Error: Scene data missing.</div>;

  return (
    <div className="w-full max-w-6xl mx-auto flex flex-col gap-12">
      <div 
        onClick={togglePlay}
        className="relative aspect-video bg-black rounded-[60px] overflow-hidden border-[16px] border-black shadow-[0_100px_200px_-50px_rgba(0,0,0,0.9)] ring-1 ring-white/10 group cursor-pointer select-none"
      >
        {/* Grain overlay */}
        <div className="absolute inset-0 z-40 pointer-events-none opacity-[0.05] mix-blend-overlay">
          <div className="absolute inset-0 animate-[noise_0.2s_infinite_steps(1)] bg-[url('https://grainy-gradients.vercel.app/noise.svg')]"></div>
        </div>

        {/* Visuals */}
        <div className="absolute inset-0 overflow-hidden bg-[#111]">
          {scenes.map((scene, i) => (
            <div key={i} className={`absolute inset-0 transition-opacity duration-300 ${i === currentIdx ? 'opacity-100 z-10' : 'opacity-0 z-0'}`}>
              {scene.imageData ? (
                <img 
                  src={scene.imageData}
                  className={`w-full h-full object-cover transition-transform duration-[2000ms] ${isPlaying && i === currentIdx ? 'scale-110' : 'scale-100'}`}
                  alt={`Scene ${i+1}`}
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center bg-gray-900 text-gray-700 font-black text-6xl">
                  RENDERING...
                </div>
              )}
            </div>
          ))}
        </div>
        
        {/* Audio Tracks */}
        <audio ref={audioRef} src={currentScene.audioUrl} onEnded={handleEnd} />
        <audio ref={bgMusicRef} src={MUSIC_FOR_VIBE(vibe)} loop />

        {/* Typography Overlay */}
        <div className="absolute inset-0 flex flex-col items-center justify-center z-30 pointer-events-none p-10">
          <h2 
            key={`text-${currentIdx}`} 
            className="text-[6rem] md:text-[10rem] font-black text-white uppercase tracking-tighter leading-none italic animate-[impact_0.4s_cubic-bezier(0.175,0.885,0.32,1.275)_forwards] drop-shadow-[0_20px_60px_rgba(0,0,0,1)] text-center"
            style={{ WebkitTextStroke: '4px black' }}
          >
            {currentScene.screenText}
          </h2>
          
          <div className="absolute bottom-32 px-20 text-center w-full">
             <p key={`sub-${currentIdx}`} className="inline-block text-white bg-black/80 px-8 py-4 rounded-3xl border-2 border-white/20 backdrop-blur-md text-2xl font-bold tracking-tight animate-in fade-in slide-in-from-bottom-4 duration-500 shadow-2xl">
               {currentScene.script}
             </p>
          </div>
        </div>

        {/* Play Button Overlay */}
        {!isPlaying && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/70 backdrop-blur-2xl z-40 animate-in fade-in duration-500">
            <div className="w-56 h-56 bg-white rounded-full flex items-center justify-center shadow-2xl group-hover:scale-110 transition-transform active:scale-95">
              <svg className="w-28 h-28 text-black ml-4" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
            </div>
            <div className="mt-12 text-white font-black uppercase tracking-[1em] text-xs animate-pulse">Click or Space to Play</div>
          </div>
        )}

        {/* Progress Bar */}
        <div className="absolute bottom-12 left-12 right-12 flex items-center gap-10 z-50">
          <div className="flex-1 h-3 bg-white/5 rounded-full overflow-hidden backdrop-blur-md border border-white/10">
            <div className="h-full bg-blue-500 transition-all duration-300" style={{ width: `${((currentIdx + 1) / scenes.length) * 100}%` }}></div>
          </div>
          <div className="text-white font-black text-[10px] tracking-widest uppercase bg-black/60 px-6 py-2 rounded-xl border border-white/10 backdrop-blur-md tabular-nums">
            {currentIdx + 1} / {scenes.length}
          </div>
        </div>
      </div>

      <style>{`
        @keyframes impact { 0% { transform: scale(0.6) translateY(20px); opacity: 0; } 100% { transform: scale(1) translateY(0); opacity: 1; } }
        @keyframes noise { 0%, 100% { transform: translate(0,0) } 10% { transform: translate(-1%,-1%) } 20% { transform: translate(-2%,1%) } 50% { transform: translate(1%,-2%) } }
      `}</style>
    </div>
  );
};

const App: React.FC = () => {
  const [step, setStep] = useState<AppStep>(AppStep.START);
  const [topic, setTopic] = useState('');
  const [hooks, setHooks] = useState<string[]>([]);
  const [selectedHook, setSelectedHook] = useState('');
  const [vibe, setVibe] = useState<ProductionVibe>('minimal');
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [isTestMode, setIsTestMode] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [logIdx, setLogIdx] = useState(0);

  // Initialize data on mount
  useEffect(() => {
    fetchSuggestions()
      .then(setSuggestions)
      .catch((e) => {
        console.warn("Falling back to internal suggestions", e);
        setSuggestions(["Space Paradoxes", "The 1% Rule", "Brain Chemistry"]);
      });
  }, []);

  // Cycle production logs
  useEffect(() => {
    if (isProcessing) {
      const interval = setInterval(() => setLogIdx(p => (p + 1) % PRODUCTION_LOGS.length), 2500);
      return () => clearInterval(interval);
    }
  }, [isProcessing]);

  // Clean up Blob URLs to prevent memory leaks
  useEffect(() => {
    return () => {
      scenes.forEach(s => { if(s.audioUrl?.startsWith('blob:')) URL.revokeObjectURL(s.audioUrl); });
    };
  }, [scenes]);

  const initiateHookSearch = async (val?: string) => {
    const t = val || topic;
    if (!t) return;
    setTopic(t);
    setStep(AppStep.HOOK_SELECTION);
    try {
      const result = await fetchHooks(t);
      setHooks(result.hooks || []);
      setVibe((result.vibe as ProductionVibe) || 'minimal');
    } catch (e) {
      console.error("Hook failure:", e);
      setHooks([`Secret of ${t}`, `Why ${t} matters`, `How ${t} works`]);
    }
  };

  const startProduction = async (h: string) => {
    setSelectedHook(h);
    setStep(AppStep.GENERATING_SCRIPT);
    try {
      const s = await generateScript(topic, h, vibe, isTestMode);
      setScenes(s);
      setStep(AppStep.REFINING_SCENES);
    } catch (e) {
      console.error("Script failure:", e);
      setStep(AppStep.START);
    }
  };

  const runQuickDemo = async () => {
    console.log("Quick Demo Triggered");
    setIsTestMode(true);
    setTopic("Rapid AI Growth");
    setStep(AppStep.HOOK_SELECTION);
    setHooks(["AI's Secret Speed", "Future in Minutes", "The Silicon Edge"]);
    setVibe('cosmic');
  };

  const runRender = async () => {
    setIsProcessing(true);
    
    // Process scenes sequentially to manage API rate limits and state updates cleanly
    for (let i = 0; i < scenes.length; i++) {
      try {
        // 1. Generate Image
        setScenes(prev => {
          const next = [...prev];
          next[i] = { ...next[i], status: 'generating-image' };
          return next;
        });
        
        const imgData = await generateSceneImage(scenes[i], vibe, isTestMode);
        
        setScenes(prev => {
          const next = [...prev];
          next[i] = { ...next[i], imageData: imgData, status: 'generating-audio' };
          return next;
        });

        // 2. Generate Audio
        const audioUrl = await generateSceneAudio(scenes[i], isTestMode);

        setScenes(prev => {
          const next = [...prev];
          next[i] = { ...next[i], audioUrl: audioUrl, status: 'completed' };
          return next;
        });

      } catch (e) {
        console.error(`Render scene ${i} error:`, e);
        setScenes(prev => {
          const next = [...prev];
          next[i] = { ...next[i], status: 'error' };
          return next;
        });
        // Stop processing on fatal error, or continue? We'll stop to avoid cascading failures.
        setIsProcessing(false);
        return; 
      }
    }
    
    setIsProcessing(false);
    setStep(AppStep.FINAL_VIDEO);
  };

  return (
    <div className="min-h-screen bg-[#FDFDFD] text-black font-sans selection:bg-blue-600 selection:text-white antialiased">
      <header className="fixed top-0 w-full p-6 md:p-10 flex flex-col md:flex-row justify-between items-center bg-white/95 backdrop-blur-3xl z-50 border-b-[6px] border-black gap-6">
        <div className="flex items-center gap-6 cursor-pointer" onClick={() => window.location.reload()}>
          <div className="w-14 h-14 bg-black rounded-2xl flex items-center justify-center text-white font-black italic text-2xl rotate-3 shadow-[6px_6px_0_0_rgba(37,99,235,1)]">OY</div>
          <h1 className="text-3xl md:text-4xl font-black uppercase tracking-tighter leading-none">Studio <span className="text-blue-600 italic">Core</span></h1>
        </div>
        <div className="flex gap-4">
          <button onClick={runQuickDemo} className="px-6 py-2 rounded-full bg-cyan-50 border-2 border-cyan-400 text-cyan-700 font-black uppercase text-[10px] tracking-widest shadow-[4px_4px_0_0_rgba(34,211,238,1)] hover:-translate-y-1 transition-all active:shadow-none">TEST RUN</button>
          <button onClick={() => setIsTestMode(!isTestMode)} className={`px-6 py-2 rounded-full border-2 border-black font-black uppercase text-[10px] tracking-widest transition-all ${isTestMode ? 'bg-black text-white' : 'bg-white'}`}>{isTestMode ? 'MOCK MODE' : 'LIVE STUDIO'}</button>
        </div>
      </header>

      <main className="max-w-7xl mx-auto pt-56 px-8 pb-40">
        {step === AppStep.START && (
          <div className="max-w-5xl mx-auto text-center py-10 animate-in fade-in slide-in-from-bottom-10">
            <h2 className="text-6xl md:text-[10rem] font-black leading-[0.75] tracking-tighter mb-20 uppercase italic">ANIMATE.<br/>THE FUTURE.</h2>
            <div className="relative max-w-4xl mx-auto mb-16">
              <input 
                type="text" 
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && initiateHookSearch()}
                placeholder="Topic for your video..."
                className="w-full px-12 py-10 text-3xl md:text-5xl rounded-[60px] border-[8px] border-black shadow-[16px_16px_0_0_rgba(0,0,0,1)] focus:outline-none focus:ring-[20px] focus:ring-blue-100 font-black transition-all placeholder:text-gray-100 italic"
              />
              <button onClick={() => initiateHookSearch()} className="absolute right-6 top-6 bottom-6 px-10 bg-black text-white rounded-[50px] font-black text-xl hover:bg-blue-600 transition-all shadow-xl">HOOK IT</button>
            </div>
            <div className="flex flex-wrap justify-center gap-4">
              {suggestions.map((s, i) => (
                <button key={i} onClick={() => initiateHookSearch(s)} className="px-6 py-3 bg-white border-2 border-black rounded-[20px] font-black uppercase text-[10px] tracking-widest hover:bg-black hover:text-white transition-all shadow-[6px_6px_0_0_rgba(0,0,0,1)] active:shadow-none italic">{s}</button>
              ))}
            </div>
          </div>
        )}

        {step === AppStep.HOOK_SELECTION && (
          <div className="max-w-6xl mx-auto animate-in fade-in zoom-in-95">
            <h3 className="text-7xl md:text-[8rem] font-black uppercase tracking-tighter leading-[0.8] italic mb-20">CHOOSE<br/>A HOOK.</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-10">
              {hooks.length > 0 ? hooks.map((h, i) => (
                <button 
                  key={i} 
                  onClick={() => startProduction(h)}
                  className="p-10 text-left bg-white border-[6px] border-black rounded-[45px] shadow-[12px_12px_0_0_rgba(0,0,0,1)] hover:-translate-y-2 hover:shadow-[20px_20px_0_0_rgba(0,0,0,1)] transition-all flex flex-col justify-between aspect-square group"
                >
                  <span className="text-5xl font-black text-gray-100 italic">0{i+1}</span>
                  <p className="text-3xl font-black uppercase tracking-tighter italic leading-tight group-hover:text-blue-600 transition-colors">{h}</p>
                </button>
              )) : <div className="col-span-3 text-center text-5xl font-black animate-pulse opacity-10 py-20 italic">ORCHESTRATING...</div>}
            </div>
          </div>
        )}

        {(step === AppStep.GENERATING_SCRIPT || step === AppStep.REFINING_SCENES) && (
          <div className="animate-in fade-in">
             <div className="flex flex-col md:flex-row justify-between items-start md:items-end mb-20 gap-8">
                <div>
                   <h3 className="text-7xl md:text-8xl font-black uppercase tracking-tighter italic leading-none">PIPELINE.</h3>
                   <p className="mt-6 text-xl font-black text-blue-600 uppercase tracking-widest italic">{selectedHook}</p>
                </div>
                {!isProcessing && step === AppStep.REFINING_SCENES && (
                  <button onClick={runRender} className="px-16 py-8 bg-black text-white rounded-[40px] font-black text-3xl shadow-2xl hover:-translate-y-2 transition-all active:scale-95">START RENDER</button>
                )}
             </div>

             <div className="grid grid-cols-2 md:grid-cols-6 gap-6">
                {scenes.map((s, i) => (
                  <div key={i} className={`p-8 rounded-[40px] border-[5px] border-black shadow-[10px_10px_0_0_rgba(0,0,0,1)] aspect-[4/6] flex flex-col justify-between transition-all ${s.status === 'completed' ? 'bg-blue-50 border-blue-600 shadow-[12px_12px_0_0_rgba(37,99,235,1)]' : 'bg-white'}`}>
                     <div>
                        <span className="text-4xl font-black italic opacity-10">0{i+1}</span>
                        <p className="mt-6 text-sm font-black uppercase tracking-tighter italic line-clamp-3">"{s.screenText}"</p>
                     </div>
                     <div className={`py-3 rounded-2xl border-2 border-black text-center font-black uppercase text-[8px] tracking-widest ${s.status === 'completed' ? 'bg-black text-white' : 'bg-white text-black'}`}>
                        {s.status.replace('-', ' ')}
                     </div>
                  </div>
                ))}
             </div>

             {isProcessing && (
                <div className="mt-20 text-center animate-pulse">
                   <p className="text-3xl font-black uppercase italic tracking-tighter text-blue-600">{PRODUCTION_LOGS[logIdx]}</p>
                </div>
             )}
          </div>
        )}

        {step === AppStep.FINAL_VIDEO && (
          <div className="animate-in zoom-in-95 duration-700">
             <div className="text-center mb-20">
                <h3 className="text-8xl md:text-[10rem] font-black uppercase tracking-tighter leading-[0.7] italic mb-6">PREMIERE.</h3>
                <div className="inline-block px-10 py-3 bg-black text-white rounded-full font-black text-xs tracking-[1em] uppercase border-[6px] border-blue-600 italic">Mastered for 4K Playback</div>
             </div>
             <SeamlessPlayer scenes={scenes} vibe={vibe} />
             <div className="mt-24 text-center">
                <button onClick={() => window.location.reload()} className="text-gray-200 font-black uppercase tracking-[1em] hover:text-blue-600 transition-all text-xl underline decoration-[8px] underline-offset-[20px]">Return to Start</button>
             </div>
          </div>
        )}
      </main>

      <footer className="p-16 text-center border-t-2 border-black/5 opacity-40">
        <p className="text-[10px] font-black uppercase tracking-[2em] italic">Studio Core Hyper-Scale Engine v6.1</p>
      </footer>
    </div>
  );
};

export default App;
