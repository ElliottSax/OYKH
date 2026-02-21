/**
 * Monitor Kaggle LoRA Training V2
 * Checks every 2 minutes
 */

import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

const KERNEL_ID = 'elliottsax/oykhchar-lora-training-v2';
const CHECK_INTERVAL = 2 * 60 * 1000; // 2 minutes

console.log('🔍 Monitoring Kaggle LoRA Training V2');
console.log('====================================');
console.log('');
console.log(`Kernel: ${KERNEL_ID}`);
console.log(`Checking every 2 minutes...`);
console.log('Press Ctrl+C to stop');
console.log('');

let lastStatus = null;

async function checkStatus() {
  try {
    const { stdout } = await execAsync(`kaggle kernels status ${KERNEL_ID}`);
    const match = stdout.match(/KernelWorkerStatus\.(\w+)/);
    const status = match ? match[1] : 'UNKNOWN';

    const timestamp = new Date().toLocaleTimeString();

    if (status !== lastStatus) {
      console.log(`[${timestamp}] Status: ${status}`);
      lastStatus = status;

      if (status === 'COMPLETE') {
        console.log('');
        console.log('✅ Training complete!');
        console.log('');
        console.log('Download output:');
        console.log(`  kaggle kernels output ${KERNEL_ID} -p C:/projects/oykh-temp/lora-output`);
        console.log('');
        console.log('View notebook:');
        console.log(`  https://www.kaggle.com/code/${KERNEL_ID.replace('/', '/')}`);
        console.log('');
        process.exit(0);
      }

      if (status === 'ERROR') {
        console.log('');
        console.log('❌ Training failed!');
        console.log('');
        console.log('View logs:');
        console.log(`  kaggle kernels output ${KERNEL_ID} -p C:/projects/oykh-temp/lora-error`);
        console.log('');
        process.exit(1);
      }
    } else {
      // Same status, just show a dot
      process.stdout.write('.');
    }
  } catch (error) {
    console.error(`✗ Error checking status: ${error.message}`);
  }
}

// Check immediately
checkStatus();

// Then check every 2 minutes
setInterval(checkStatus, CHECK_INTERVAL);
