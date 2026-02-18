
export type ProductionVibe = 'cosmic' | 'hype' | 'minimal' | 'suspense' | 'success';

export interface Scene {
  title: string;
  script: string;
  visualDescription: string;
  screenText: string; 
  imageData?: string;
  audioUrl?: string;
  status: 'idle' | 'generating-image' | 'generating-audio' | 'completed' | 'error';
}

export interface ProjectConfig {
  topic: string;
  vibe: ProductionVibe;
  hook: string;
}

export enum AppStep {
  START = 'START',
  HOOK_SELECTION = 'HOOK_SELECTION',
  GENERATING_SCRIPT = 'GENERATING_SCRIPT',
  REFINING_SCENES = 'REFINING_SCENES',
  FINAL_VIDEO = 'FINAL_VIDEO'
}
