import { GoogleGenerativeAI } from '@google/generative-ai';

const apiKey = 'AIzaSyBuq3Ws0EdfB6RGU94_7HdwREFvHXvqUmw';
const genAI = new GoogleGenerativeAI(apiKey);

console.log('🧪 Testing Gemini API...');

const model = genAI.getGenerativeModel({
  model: 'gemini-2.5-flash',
});

const prompt = `Generate a simple JSON object with 3 video chapters for a video about "Why do we dream?". Keep it minimal - just title, duration, and 2 shots per chapter. Return ONLY valid JSON.`;

console.log('📡 Calling API...');

try {
  const result = await Promise.race([
    model.generateContent(prompt),
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error('Timeout after 30 seconds')), 30000)
    ),
  ]);

  const response = result.response;
  let text = response.text();

  console.log('✅ Response received!');
  console.log('Length:', text.length, 'characters');
  console.log('First 200 chars:', text.substring(0, 200));

  // Strip markdown code blocks if present
  text = text.replace(/^```json\s*/i, '').replace(/\s*```$/, '');

  // Try to parse JSON
  const parsed = JSON.parse(text);
  console.log('✅ Valid JSON!');
  console.log('Parsed:', JSON.stringify(parsed, null, 2).substring(0, 500));
} catch (error) {
  console.error('❌ Error:', error.message);
  console.error('Stack:', error.stack);
}
