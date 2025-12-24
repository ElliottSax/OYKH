
export interface Scene {
  title: string;
  script: string;
  visualDescription: string;
  imageData?: string;
  videoUrl?: string;
  audioUrl?: string;
  status: 'idle' | 'generating-image' | 'generating-video' | 'generating-audio' | 'completed' | 'error';
}

export interface ExplainerProject {
  topic: string;
  scenes: Scene[];
}

export enum AppStep {
  START = 'START',
  GENERATING_SCRIPT = 'GENERATING_SCRIPT',
  REFINING_SCENES = 'REFINING_SCENES',
  FINAL_VIDEO = 'FINAL_VIDEO'
}
