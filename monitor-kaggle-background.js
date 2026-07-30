/**
 * Background Kaggle Monitor
 * Checks status every 5 minutes and logs updates
 */

import { exec } from 'child_process';
import { promisify } from 'util';
import { appendFileSync } from 'fs';

const execAsync = promisify(exec);
const KERNEL_SLUG = 'elliottsax/oykhchar-lora-training';
const CHECK_INTERVAL = 5 * 60 * 1000; // 5 minutes
const LOG_FILE = 'C:/projects/oykh-temp/kaggle-monitor.log';

function log(message) {
  const timestamp = new Date().toLocaleString();
  const logMessage = `[${timestamp}] ${message}\n`;
  console.log(logMessage.trim());
  appendFileSync(LOG_FILE, logMessage);
}

async function checkStatus() {
  try {
    const { stdout } = await execAsync(`kaggle kernels status ${KERNEL_SLUG}`);

    if (stdout.includes('RUNNING')) {
      log('✅ Training is RUNNING');
    } else if (stdout.includes('COMPLETE')) {
      log('🎉 TRAINING COMPLETE! Ready to download.');
      log('Run: kaggle kernels output elliottsax/oykhchar-lora-training -p C:/projects/oykh-temp/lora-output');
      process.exit(0);
    } else if (stdout.includes('ERROR')) {
      log('❌ Training has ERROR status (needs to be restarted)');
    } else {
      log(`Status: ${stdout.trim()}`);
    }
  } catch (error) {
    log(`Error checking status: ${error.message}`);
  }
}

log('🔍 Started Kaggle monitoring');
log(`Checking every ${CHECK_INTERVAL / 1000 / 60} minutes`);

// Check immediately
checkStatus();

// Then check every 5 minutes
setInterval(checkStatus, CHECK_INTERVAL);

// Keep alive
process.on('SIGINT', () => {
  log('Monitoring stopped');
  process.exit(0);
});
