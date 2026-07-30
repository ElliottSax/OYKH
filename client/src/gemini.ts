import { GoogleGenerativeAI } from '@google/generative-ai';
import { ViralVideoScript, ProductionVibe } from './types';
import { SCRIPT_GENERATION_PROMPT, SCRIPT_SCHEMA } from './prompts';

const apiKey = (process.env.VITE_GEMINI_API_KEY || process.env.GEMINI_API_KEY) as string;

if (!apiKey) {
  console.error('GEMINI_API_KEY not found in environment variables');
}

const genAI = new GoogleGenerativeAI(apiKey);

/**
 * Generate a complete viral 5-minute video script optimized for retention
 */
export const generateViralVideoScript = async (
  topic: string,
  hook: string = `The Secret of ${topic}`,
  vibe: ProductionVibe = 'minimal'
): Promise<ViralVideoScript> => {
  try {
    console.log('🎬 Generating viral video script...');
    console.log('Topic:', topic);
    console.log('Hook:', hook);
    console.log('Vibe:', vibe);

    const model = genAI.getGenerativeModel({
      model: 'gemini-2.5-flash',
    });

    const prompt =
      SCRIPT_GENERATION_PROMPT(topic, hook, vibe) +
      '\n\nRETURN ONLY VALID JSON. No markdown, no explanations, just the JSON object.';

    console.log('⏳ Calling Gemini API (this may take 15-45 seconds)...');
    const startTime = Date.now();
    const result = (await Promise.race([
      model.generateContent(prompt),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error('Request timed out after 60 seconds')), 60000)
      ),
    ])) as any;
    console.log(`⏱️  API responded in ${((Date.now() - startTime) / 1000).toFixed(1)}s`);

    const response = result.response;
    let text = response.text();

    if (!text) {
      throw new Error('No response from Gemini API');
    }

    // Strip markdown code blocks if present
    console.log('📝 Response length:', text.length, 'characters');
    text = text.replace(/^```json\s*/i, '').replace(/\s*```$/, '');
    console.log('🧹 After markdown strip:', text.substring(0, 100));

    console.log('🔍 Parsing JSON...');
    const parsed = JSON.parse(text);
    console.log('✅ JSON parsed successfully');

    // Add status to all shots
    parsed.chapters.forEach((chapter: any) => {
      chapter.shots.forEach((shot: any) => {
        shot.status = 'pending';
      });
    });

    console.log('✅ Script generated successfully!');
    console.log(`📊 Total shots: ${parsed.totalShots}`);
    console.log(`💰 Estimated cost: $${parsed.estimatedCost.toFixed(2)}`);

    return parsed as ViralVideoScript;
  } catch (error) {
    console.error('❌ Script generation failed:', error);
    throw error;
  }
};

/**
 * Generate multiple hook options for a topic
 */
export const generateHookOptions = async (topic: string): Promise<string[]> => {
  try {
    const prompt = `Generate 5 attention-grabbing hooks for a viral YouTube video about: "${topic}"

Each hook should:
- Create immediate curiosity
- Use pattern interrupts ("Wait, WHAT?!", "You won't believe...")
- Be under 10 words
- Make viewers NEED to keep watching

Return as JSON array of strings.`;

    const model = genAI.getGenerativeModel({
      model: 'gemini-2.5-flash',
      generationConfig: {
        responseMimeType: 'application/json',
      },
    });

    const result = await model.generateContent(prompt);
    const text = result.response.text();

    if (!text) {
      throw new Error('No response from Gemini API');
    }

    return JSON.parse(text) as string[];
  } catch (error) {
    console.error('Hook generation failed:', error);
    // Return fallback hooks
    return [
      `The Secret of ${topic}`,
      `What Scientists Don't Tell You About ${topic}`,
      `The ${topic} Mystery Nobody Can Explain`,
      `Why ${topic} Changes Everything`,
      `The Hidden Truth About ${topic}`,
    ];
  }
};
