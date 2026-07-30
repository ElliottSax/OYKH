export type ProductionVibe = 'cosmic' | 'hype' | 'minimal' | 'suspense' | 'success';

export type VideoJobStatus =
  | 'new'
  | 'queued'
  | 'running'
  | 'script-generating'
  | 'script-failed'
  | 'script-generated'
  | 'video-generating'
  | 'video-failed'
  | 'video-generated'
  | 'complete'
  | 'failed';

export interface VideoJob {
  id: string;
  status: VideoJobStatus;
  topic: string;
  vibe: ProductionVibe;
  script?: ViralVideoScript;
  assetUrls?: string[];
  finalVideoUrl?: string;
  createdAt: any;
  updatedAt: any;
  progress?: number;
  currentStep?: string;
  error?: string;
  jobId?: string;
  message?: string;
  videoUrl?: string;
  imageResults?: Map<number, string>;
}

// Core Video Script Structure
export interface ViralVideoScript {
  metadata: VideoMetadata;
  chapters: ViralChapter[];
  openLoops: OpenLoop[];
  retentionBombs: RetentionBomb[];
  totalShots: number;
  estimatedCost: number;
  shots: Shot[];
  vibe: ProductionVibe;
  voice: GoogleVoice;
}

export interface VideoMetadata {
  topic: string;
  hook: string;
  title: string; // 60 chars max, curiosity-driven
  vibe: ProductionVibe;
  targetDuration: number; // 300 seconds
  thumbnailConcept: ThumbnailConcept;
}

export interface ViralChapter {
  chapterNumber: number;
  title: string;
  timestamp: string; // "0:00"
  duration: number;
  purpose: ChapterPurpose;

  // Content
  narration: string;
  keyMessage: string;

  // Structure
  shots: Shot[];

  // Engagement
  emotionalTone: EmotionalTone;
}

export type ChapterPurpose =
  | 'cold-open' // First 3 seconds
  | 'hook' // Grab attention
  | 'setup' // Establish context
  | 'build-tension' // Increase curiosity
  | 'payoff' // Answer question
  | 'surprise' // Unexpected twist
  | 'deepdive' // Detailed explanation
  | 'relate' // Make it personal
  | 'resolve' // Close loops
  | 'cta'; // Call to action

export interface Shot {
  shotNumber: number;
  duration: number; // 1.5-3 seconds

  // Visual Description
  characterAction: CharacterAction;
  characterEmotion: CharacterEmotion;
  cameraAngle: CameraAngle;
  cameraMovement: CameraMovement;

  // Environment
  backgroundStyle: BackgroundStyle;

  // AI Generation Prompt (for Imagen 3)
  prompt: string;

  // Text Overlay
  textOverlay?: TextOverlay;

  // Animation
  animation: AnimationType;
  transition: TransitionType;

  // Generated Content
  imageData?: string;
  status: ShotStatus;
}

export type CharacterAction =
  | 'standing-neutral'
  | 'thinking-chin'
  | 'excited-jumping'
  | 'confused-questionmark'
  | 'explaining-pointing'
  | 'running-forward'
  | 'holding-object'
  | 'sitting-desk'
  | 'looking-magnifying-glass'
  | 'lightbulb-idea'
  | 'two-characters-talking'
  | 'climbing-stairs'
  | 'presenting-chart'
  | 'transforming'
  | 'celebrating';

export type CharacterEmotion =
  | 'neutral'
  | 'happy'
  | 'excited'
  | 'shocked'
  | 'confused'
  | 'concerned'
  | 'thoughtful'
  | 'determined'
  | 'surprised'
  | 'satisfied';

export type CameraAngle =
  | 'wide-full-body'
  | 'medium-waist-up'
  | 'closeup-shoulders'
  | 'extreme-closeup-face'
  | 'over-shoulder'
  | 'top-down'
  | 'three-quarter'
  | 'side-profile'
  | 'dutch-angle';

export type CameraMovement =
  | 'static'
  | 'slow-push-in'
  | 'slow-pull-out'
  | 'whip-pan'
  | 'float'
  | 'shake'
  | 'orbit'
  | 'dolly-zoom';

export type BackgroundStyle =
  | 'gradient-simple'
  | 'gradient-radial'
  | 'gradient-dynamic'
  | 'geometric-minimal'
  | 'void-black'
  | 'void-white'
  | 'spotlight';

export type AnimationType =
  | 'ken-burns-in'
  | 'ken-burns-out'
  | 'pan-right'
  | 'pan-left'
  | 'dolly-forward'
  | 'dolly-back'
  | 'rotate-slow'
  | 'static'
  | 'bounce'
  | 'pulse';

export type TransitionType =
  | 'cut'
  | 'fade'
  | 'wipe-left'
  | 'wipe-right'
  | 'iris-in'
  | 'iris-out'
  | 'morph';

export type ShotStatus = 'pending' | 'generating' | 'completed' | 'error';

export interface TextOverlay {
  text: string;
  position: TextPosition;
  style: TextStyle;
  animation: TextAnimation;
  timing: {
    delay: number;
    duration: number;
    fadeOut: number;
  };
  highlightWords?: string[];
  fontSize?: 'small' | 'medium' | 'large' | 'huge';
}

export type TextPosition =
  | 'top-left'
  | 'top-center'
  | 'top-right'
  | 'middle-center'
  | 'bottom-left'
  | 'bottom-center'
  | 'bottom-right'
  | 'lower-third';

export type TextStyle =
  | 'caption'
  | 'title'
  | 'keyword'
  | 'stat'
  | 'quote'
  | 'label'
  | 'emphasis'
  | 'question'
  | 'answer';

export type TextAnimation =
  | 'fade-in'
  | 'slide-up'
  | 'slide-down'
  | 'typewriter'
  | 'word-by-word'
  | 'scale-in'
  | 'bounce-in'
  | 'glitch-in';

export type EmotionalTone =
  | 'curiosity'
  | 'surprise'
  | 'awe'
  | 'tension'
  | 'satisfaction'
  | 'concern'
  | 'relief'
  | 'excitement'
  | 'contemplation';

export interface OpenLoop {
  question: string;
  posedAt: number; // Timestamp in seconds
  resolvedAt: number;
  intensity: 'low' | 'medium' | 'high';
}

export interface RetentionBomb {
  timestamp: number;
  type: RetentionType;
  content: string;
  shotNumbers: number[]; // Which shots execute this
}

export type RetentionType =
  | 'pattern-interrupt'
  | 'surprise-fact'
  | 'plot-twist'
  | 'callback'
  | 'escalation'
  | 'cliffhanger'
  | 'visualization'
  | 'contrast';

export interface ThumbnailConcept {
  mainElement: string;
  emotion: 'shocked' | 'curious' | 'excited' | 'concerned';
  text: string; // 3-5 words max
  colorScheme: 'high-contrast' | 'vibrant' | 'mysterious';
}

// App State
export enum AppStep {
  START = 'START',
  GENERATING_SCRIPT = 'GENERATING_SCRIPT',
  SCRIPT_REVIEW = 'SCRIPT_REVIEW',
  REFINING_SCRIPT = 'REFINING_SCRIPT',
  GENERATING_VISUALS = 'GENERATING_VISUALS',
  GENERATING_AUDIO = 'GENERATING_AUDIO',
  ASSEMBLING_VIDEO = 'ASSEMBLING_VIDEO',
  FINAL_VIDEO = 'FINAL_VIDEO',
  ERROR = 'ERROR',
}

// Voice Configuration
export type GoogleVoice =
  | 'en-US-Journey-D' // Energetic male
  | 'en-US-Journey-F' // Warm female
  | 'en-US-Journey-O' // Authoritative
  | 'en-US-Studio-M' // Documentary male
  | 'en-US-Studio-O' // Professional female
  | 'en-US-Neural2-D' // Friendly male
  | 'en-US-Neural2-F'; // Clear female

export interface GenerationProgress {
  step: AppStep;
  progress: number; // 0-100
  currentShot?: number;
  totalShots?: number;
  message: string;
}
