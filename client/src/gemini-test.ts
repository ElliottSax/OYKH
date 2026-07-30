// Quick test to find available models
import { GoogleGenerativeAI } from '@google/generative-ai';

const apiKey = process.env.GEMINI_API_KEY as string;

async function testModels() {
  const ai = new GoogleGenerativeAI(apiKey);

  const modelsToTry = [
    'gemini-1.5-flash',
    'gemini-1.5-pro',
    'gemini-pro',
    'gemini-flash',
    'models/gemini-1.5-flash',
    'models/gemini-pro',
  ];

  for (const modelName of modelsToTry) {
    try {
      console.log(`Testing: ${modelName}...`);
      const model = ai.getGenerativeModel({ model: modelName });
      const result = await model.generateContent('Hello, say "OK" if you can hear me.');
      const response = result.response;
      console.log('Response:', response.candidates[0].content.parts[0].text?.substring(0, 50));
      break;
    } catch (error: any) {
      console.log(`❌ ${modelName} failed:`, error.message?.substring(0, 100));
    }
  }
}

testModels();
