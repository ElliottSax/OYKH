/**
 * Test Kaggle CLI Setup
 * Verifies everything is working correctly
 */

import { exec } from 'child_process';
import { promisify } from 'util';
import fs from 'fs';
import os from 'os';
import path from 'path';

const execAsync = promisify(exec);

console.log('🧪 Testing Kaggle CLI Setup');
console.log('============================');
console.log('');

async function testKaggleSetup() {
  const tests = [];
  let allPassed = true;

  // Test 1: Check if Kaggle CLI is installed
  console.log('Test 1: Checking Kaggle CLI installation...');
  try {
    const { stdout } = await execAsync('kaggle --version');
    console.log('✅ PASS - Kaggle CLI installed:', stdout.trim());
    tests.push({ name: 'CLI Installation', status: 'PASS' });
  } catch (error) {
    console.log('❌ FAIL - Kaggle CLI not found');
    console.log('   Install with: pip install kaggle');
    tests.push({ name: 'CLI Installation', status: 'FAIL' });
    allPassed = false;
  }
  console.log('');

  // Test 2: Check for credentials file
  console.log('Test 2: Checking API credentials...');
  const kaggleDir = path.join(os.homedir(), '.kaggle');
  const credFile = path.join(kaggleDir, 'kaggle.json');

  if (fs.existsSync(credFile)) {
    console.log('✅ PASS - Credentials file found:', credFile);

    // Check file contents
    try {
      const creds = JSON.parse(fs.readFileSync(credFile, 'utf8'));
      if (creds.username && creds.key) {
        console.log('   Username:', creds.username);
        console.log('   API Key: ****' + creds.key.slice(-8));
        tests.push({ name: 'API Credentials', status: 'PASS' });
      } else {
        console.log('❌ FAIL - Invalid credentials format');
        tests.push({ name: 'API Credentials', status: 'FAIL' });
        allPassed = false;
      }
    } catch (error) {
      console.log('❌ FAIL - Cannot read credentials file');
      tests.push({ name: 'API Credentials', status: 'FAIL' });
      allPassed = false;
    }
  } else {
    console.log('❌ FAIL - Credentials file not found');
    console.log('   Expected location:', credFile);
    console.log('   Download from: https://www.kaggle.com/settings/account');
    tests.push({ name: 'API Credentials', status: 'FAIL' });
    allPassed = false;
  }
  console.log('');

  // Test 3: Test API connection
  console.log('Test 3: Testing API connection...');
  try {
    const { stdout } = await execAsync('kaggle competitions list');
    console.log('✅ PASS - API connection working');
    console.log('   Connected to Kaggle successfully!');
    tests.push({ name: 'API Connection', status: 'PASS' });
  } catch (error) {
    console.log('❌ FAIL - Cannot connect to Kaggle API');
    console.log('   Error:', error.message);
    tests.push({ name: 'API Connection', status: 'FAIL' });
    allPassed = false;
  }
  console.log('');

  // Test 4: Check GPU quota
  console.log('Test 4: Checking GPU quota...');
  try {
    // Note: This is a placeholder - actual quota checking requires web scraping or API
    console.log('ℹ️  INFO - GPU quota check requires web login');
    console.log('   Check at: https://www.kaggle.com/settings');
    console.log('   Free tier: 30 hours/week GPU time');
    tests.push({ name: 'GPU Quota', status: 'INFO' });
  } catch (error) {
    tests.push({ name: 'GPU Quota', status: 'INFO' });
  }
  console.log('');

  // Summary
  console.log('============================');
  console.log('Test Summary:');
  console.log('============================');
  tests.forEach(test => {
    const icon = test.status === 'PASS' ? '✅' : test.status === 'FAIL' ? '❌' : 'ℹ️';
    console.log(`${icon} ${test.name}: ${test.status}`);
  });
  console.log('');

  if (allPassed) {
    console.log('🎉 ALL TESTS PASSED!');
    console.log('');
    console.log('You can now:');
    console.log('  • Upload datasets: kaggle datasets create');
    console.log('  • Run notebooks: kaggle kernels push');
    console.log('  • Train LoRAs for FREE!');
    console.log('');
    console.log('Next steps:');
    console.log('  1. Read: KAGGLE_PROGRAMMATIC_GUIDE.md');
    console.log('  2. Upload training data');
    console.log('  3. Start free LoRA training!');
  } else {
    console.log('⚠️  SOME TESTS FAILED');
    console.log('');
    console.log('Fix issues above, then run this test again.');
    console.log('');
    console.log('Quick fixes:');
    console.log('  • Install CLI: pip install kaggle');
    console.log('  • Get credentials: https://www.kaggle.com/settings/account');
    console.log(`  • Move to: ${credFile}`);
  }
  console.log('');
}

testKaggleSetup().catch(console.error);
