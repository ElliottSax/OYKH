/**
 * Monitor Kaggle LoRA Training Progress
 * Checks every 2 minutes and notifies when complete
 */

import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);
const KERNEL_SLUG = 'elliottsax/oykhchar-lora-training';
const CHECK_INTERVAL = 120000; // 2 minutes

console.log('🔍 Monitoring Kaggle LoRA Training');
console.log('=====================================');
console.log(`Kernel: ${KERNEL_SLUG}`);
console.log(`Check interval: ${CHECK_INTERVAL / 1000} seconds`);
console.log('');
console.log('View progress: https://www.kaggle.com/code/elliottsax/oykhchar-lora-training');
console.log('');

let checkCount = 0;

async function checkStatus() {
  try {
    const { stdout } = await execAsync(`kaggle kernels status ${KERNEL_SLUG}`);
    checkCount++;

    const timestamp = new Date().toLocaleTimeString();

    if (stdout.includes('RUNNING')) {
      console.log(`[${timestamp}] Check #${checkCount}: Still training... 🎓`);
    } else if (stdout.includes('COMPLETE')) {
      console.log('');
      console.log('=====================================');
      console.log('🎉 TRAINING COMPLETE!');
      console.log('=====================================');
      console.log('');
      console.log('Download LoRA weights:');
      console.log(`  kaggle kernels output ${KERNEL_SLUG} -p ./lora-output`);
      console.log('');

      // Auto-download
      console.log('📥 Downloading trained LoRA...');
      const { stdout: downloadOut } = await execAsync(
        `kaggle kernels output ${KERNEL_SLUG} -p C:/projects/oykh-temp/lora-output`
      );
      console.log(downloadOut);

      console.log('');
      console.log('✅ LoRA downloaded to: C:/projects/oykh-temp/lora-output/');
      console.log('');
      console.log('Next steps:');
      console.log('  1. Test the LoRA with test images');
      console.log('  2. Integrate into server-simple.js');
      console.log('  3. Generate production videos!');

      process.exit(0);
    } else if (stdout.includes('ERROR') || stdout.includes('FAILED')) {
      console.log('');
      console.log('❌ Training failed!');
      console.log(stdout);
      console.log('');
      console.log('Check logs at: https://www.kaggle.com/code/elliottsax/oykhchar-lora-training');
      process.exit(1);
    }
  } catch (error) {
    console.error('Error checking status:', error.message);
  }
}

// Initial check
checkStatus();

// Check every 2 minutes
const interval = setInterval(checkStatus, CHECK_INTERVAL);

// Keep process alive
process.on('SIGINT', () => {
  console.log('\n\nMonitoring stopped.');
  clearInterval(interval);
  process.exit(0);
});
