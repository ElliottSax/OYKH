/**
 * Quick Test of Together.ai API
 * Verify your key works and see the cost savings!
 */

import Together from 'together-ai';

const TOGETHER_API_KEY = 'b5b8f9dc6d35a80a4b85dc46dac3d7ffff94ee3f6c31b165c5c68d50e0baab03';

async function testTogether() {
  console.log('🧪 Testing Together.ai API');
  console.log('===========================\n');

  const together = new Together({ apiKey: TOGETHER_API_KEY });

  // Test 1: Simple completion
  console.log('[Test 1] Simple completion...');
  try {
    const response = await together.chat.completions.create({
      model: "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
      messages: [
        {
          role: "user",
          content: "Say hello and confirm you're working!"
        }
      ],
      max_tokens: 50
    });

    console.log('✅ SUCCESS!');
    console.log('Response:', response.choices[0].message.content);
    console.log('\nTokens used:', response.usage.total_tokens);
    console.log('Cost: $' + (response.usage.total_tokens / 1000000 * 0.88).toFixed(6));
    console.log('Same on OpenAI GPT-4: $' + (response.usage.total_tokens / 1000000 * 30).toFixed(4));
    console.log('Savings per call: 97%!\n');
  } catch (error) {
    console.log('❌ FAILED:', error.message);
    return;
  }

  // Test 2: Content summarization (like RepurposeAI)
  console.log('[Test 2] Content summarization...');
  try {
    const article = `
      Artificial Intelligence is transforming content creation. AI tools can now
      generate high-quality text, images, and videos in seconds. This technology
      enables creators to produce more content faster while maintaining quality.
      The future of content creation is AI-assisted, not AI-replaced.
    `;

    const response = await together.chat.completions.create({
      model: "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
      messages: [
        {
          role: "system",
          content: "You are an expert content summarizer. Create concise, engaging summaries."
        },
        {
          role: "user",
          content: `Summarize this in 2 sentences:\n\n${article}`
        }
      ],
      max_tokens: 100,
      temperature: 0.7
    });

    console.log('✅ SUCCESS!');
    console.log('Summary:', response.choices[0].message.content);
    console.log('\nTokens used:', response.usage.total_tokens);
    console.log('Together.ai cost: $' + (response.usage.total_tokens / 1000000 * 0.88).toFixed(6));
    console.log('OpenAI cost: $' + (response.usage.total_tokens / 1000000 * 30).toFixed(4));
    console.log('');
  } catch (error) {
    console.log('❌ FAILED:', error.message);
    return;
  }

  // Test 3: Chat conversation (like Coach app)
  console.log('[Test 3] Chat conversation...');
  try {
    const response = await together.chat.completions.create({
      model: "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
      messages: [
        {
          role: "system",
          content: "You are an empathetic life coach providing actionable advice."
        },
        {
          role: "user",
          content: "I'm struggling with procrastination. What's one thing I can do right now?"
        }
      ],
      max_tokens: 150,
      temperature: 0.8
    });

    console.log('✅ SUCCESS!');
    console.log('Coach response:', response.choices[0].message.content);
    console.log('\nTokens used:', response.usage.total_tokens);
    console.log('Cost: $' + (response.usage.total_tokens / 1000000 * 0.88).toFixed(6));
    console.log('');
  } catch (error) {
    console.log('❌ FAILED:', error.message);
    return;
  }

  // Summary
  console.log('===========================');
  console.log('🎉 ALL TESTS PASSED!');
  console.log('===========================\n');
  console.log('Your Together.ai API is working perfectly!');
  console.log('');
  console.log('Next steps:');
  console.log('1. Add key to .env files in all projects');
  console.log('2. Install SDK: npm install together-ai');
  console.log('3. Replace OpenAI calls');
  console.log('4. Start saving $2,047/month!');
  console.log('');
}

testTogether().catch(console.error);
