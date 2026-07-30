/**
 * Test All Game-Changer Services
 * Run this after getting API keys to verify everything works
 */

import 'dotenv/config';

console.log('🧪 Testing All Game-Changer Services');
console.log('======================================\n');

const results = {
  passed: [],
  failed: [],
  skipped: []
};

// Test 1: Hugging Face
async function testHuggingFace() {
  console.log('[1/4] Testing Hugging Face...');

  if (!process.env.HF_TOKEN) {
    console.log('⚠️  SKIP - No HF_TOKEN in .env\n');
    results.skipped.push('Hugging Face');
    return;
  }

  try {
    const response = await fetch(
      'https://api-inference.huggingface.co/models/gpt2',
      {
        headers: { Authorization: `Bearer ${process.env.HF_TOKEN}` },
        method: 'POST',
        body: JSON.stringify({ inputs: 'Hello world' })
      }
    );

    if (response.ok) {
      console.log('✅ Hugging Face API working!\n');
      results.passed.push('Hugging Face');
    } else {
      throw new Error(`Status ${response.status}`);
    }
  } catch (error) {
    console.log(`❌ Hugging Face failed: ${error.message}\n`);
    results.failed.push('Hugging Face');
  }
}

// Test 2: Together.ai
async function testTogether() {
  console.log('[2/4] Testing Together.ai...');

  if (!process.env.TOGETHER_API_KEY) {
    console.log('⚠️  SKIP - No TOGETHER_API_KEY in .env\n');
    results.skipped.push('Together.ai');
    return;
  }

  try {
    const { default: Together } = await import('together-ai');
    const together = new Together({ apiKey: process.env.TOGETHER_API_KEY });

    const response = await together.chat.completions.create({
      model: "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
      messages: [{ role: "user", content: "Say hello" }],
      max_tokens: 10
    });

    console.log('✅ Together.ai API working!');
    console.log(`   Response: ${response.choices[0].message.content}`);
    console.log(`   Cost: $${(response.usage.total_tokens / 1000000 * 0.18).toFixed(6)}\n`);
    results.passed.push('Together.ai');
  } catch (error) {
    if (error.code === 'MODULE_NOT_FOUND') {
      console.log('⚠️  SKIP - together-ai not installed (run: npm install together-ai)\n');
      results.skipped.push('Together.ai');
    } else {
      console.log(`❌ Together.ai failed: ${error.message}\n`);
      results.failed.push('Together.ai');
    }
  }
}

// Test 3: Modal
async function testModal() {
  console.log('[3/4] Testing Modal...');

  if (!process.env.MODAL_ENDPOINT) {
    console.log('⚠️  SKIP - No MODAL_ENDPOINT in .env (deploy modal-flux-app.py first)\n');
    results.skipped.push('Modal');
    return;
  }

  try {
    const response = await fetch(process.env.MODAL_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: 'test image',
        use_lora: false,
        num_steps: 4
      }),
      timeout: 60000
    });

    if (response.ok) {
      console.log('✅ Modal API working!\n');
      results.passed.push('Modal');
    } else {
      throw new Error(`Status ${response.status}`);
    }
  } catch (error) {
    console.log(`❌ Modal failed: ${error.message}\n`);
    results.failed.push('Modal');
  }
}

// Test 4: Kaggle
async function testKaggle() {
  console.log('[4/4] Testing Kaggle...');

  try {
    const { exec } = await import('child_process');
    const { promisify } = await import('util');
    const execAsync = promisify(exec);

    const { stdout } = await execAsync('kaggle competitions list --page-size 1');

    if (stdout.includes('ref')) {
      console.log('✅ Kaggle CLI working!\n');
      results.passed.push('Kaggle');
    } else {
      throw new Error('Unexpected output');
    }
  } catch (error) {
    console.log(`❌ Kaggle failed: ${error.message}\n`);
    results.failed.push('Kaggle');
  }
}

// Run all tests
async function runTests() {
  await testHuggingFace();
  await testTogether();
  await testModal();
  await testKaggle();

  // Summary
  console.log('======================================');
  console.log('Test Summary');
  console.log('======================================\n');

  if (results.passed.length > 0) {
    console.log(`✅ Passed (${results.passed.length}):`);
    results.passed.forEach(service => console.log(`   - ${service}`));
    console.log('');
  }

  if (results.failed.length > 0) {
    console.log(`❌ Failed (${results.failed.length}):`);
    results.failed.forEach(service => console.log(`   - ${service}`));
    console.log('');
  }

  if (results.skipped.length > 0) {
    console.log(`⚠️  Skipped (${results.skipped.length}):`);
    results.skipped.forEach(service => console.log(`   - ${service}`));
    console.log('');
  }

  const total = results.passed.length + results.failed.length + results.skipped.length;
  console.log(`Total: ${results.passed.length}/${total} services ready\n`);

  if (results.passed.length === total) {
    console.log('🎉 All services operational! Ready to save $3,000-5,000/year!\n');
  } else if (results.passed.length > 0) {
    console.log('⚡ Some services ready. Complete setup to unlock full savings.\n');
    console.log('Next steps:');
    if (results.skipped.includes('Hugging Face')) {
      console.log('  - Get HF token: https://huggingface.co/settings/tokens');
    }
    if (results.skipped.includes('Together.ai')) {
      console.log('  - Get Together.ai key: https://api.together.xyz/settings/api-keys');
    }
    if (results.skipped.includes('Modal')) {
      console.log('  - Deploy Modal: modal deploy modal-flux-app.py');
    }
    console.log('');
  } else {
    console.log('📝 Complete service setup to start saving!\n');
    console.log('Run: bash setup-all-services.sh\n');
  }
}

runTests().catch(console.error);
