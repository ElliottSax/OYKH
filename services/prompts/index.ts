/**
 * Centralized AI Prompt Library for OYKH Video Generation
 *
 * This module exports all AI prompts used across the application.
 * All prompts are organized by domain (script, image, audio, etc.)
 * and follow consistent patterns for maintainability.
 */

// Script Generation Prompts
export {
  SCRIPT_GENERATION_PROMPT,
  SCRIPT_SCHEMA,
  REFINEMENT_PROMPT,
  SUGGESTIONS_PROMPT,
  HOOKS_PROMPT,
  getStyleGuide,
} from './script-generation';

// Image Generation Prompts
export {
  IMAGE_GENERATION_PROMPT,
  PROMPT_REFINEMENT_PROMPT,
  CAMERA_DESCRIPTIONS,
  ACTION_DESCRIPTIONS,
  EMOTION_DESCRIPTIONS,
} from './image-generation';

// Audio Generation Prompts (Future)
// export * from './audio-generation';

// Video Assembly Prompts (Future)
// export * from './video-assembly';

/**
 * Prompt Engineering Best Practices
 *
 * 1. Clarity: Prompts should be clear and unambiguous
 * 2. Context: Provide sufficient context for the AI model
 * 3. Examples: Use few-shot examples where applicable
 * 4. Constraints: Set clear boundaries and requirements
 * 5. Iteration: Test and refine prompts based on results
 */

/**
 * Prompt Categories
 *
 * 1. Generation Prompts: Create new content (scripts, images, audio)
 * 2. Refinement Prompts: Improve existing content
 * 3. Error Handling Prompts: Fix failed generations
 * 4. Optimization Prompts: Enhance content quality
 * 5. Validation Prompts: Check content compliance
 */
