import { jobManager } from './functions/src/job-manager';
import { checkFFmpegInstalled } from './functions/src/video-assembly';

async function testFullPipeline() {
  console.log('🎬 OYKH Full Pipeline Test\n');
  console.log('='.repeat(60));

  // Step 1: Verify FFmpeg installation
  console.log('\n📋 Step 1: Checking FFmpeg installation...');
  const ffmpegInstalled = await checkFFmpegInstalled();

  if (!ffmpegInstalled) {
    console.error('❌ FFmpeg not found! Please install it first.');
    console.log('\nInstallation instructions:');
    console.log('  Windows: choco install ffmpeg  OR  winget install Gyan.FFmpeg');
    console.log('  macOS:   brew install ffmpeg');
    console.log('  Linux:   sudo apt install ffmpeg');
    return;
  }

  console.log('✅ FFmpeg installed and ready!');

  // Step 2: Check for API keys
  console.log('\n📋 Step 2: Checking API configuration...');
  const hasGeminiKey =
    !!process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY !== 'your_gemini_api_key_here';

  if (!hasGeminiKey) {
    console.log('⚠️  GEMINI_API_KEY not configured');
    console.log('\nTo run the full pipeline:');
    console.log('  1. Copy .env.example to .env.local');
    console.log('  2. Add your Gemini API key from https://aistudio.google.com/app/apikey');
    console.log('  3. Run this test again');
    console.log('\n📚 Showing what WOULD happen with API keys...\n');
  } else {
    console.log('✅ API keys configured!');
  }

  // Step 3: Demonstrate job creation (works without API keys)
  console.log('\n📋 Step 3: Creating video generation job...');

  const testTopic = 'Why Do We Dream?';
  console.log(`\n  Topic: "${testTopic}"`);

  const jobId = await jobManager.createJob(testTopic);
  console.log(`\n✅ Job created: ${jobId}`);

  // Step 4: Show job details
  console.log('\n📋 Step 4: Job details...');
  const job = await jobManager.getJob(jobId);
  if (job) {
    console.log(`  Status: ${job.status}`);
    console.log(`  Progress: ${job.progress}%`);
    console.log(`  Current step: ${job.currentStep}`);
    console.log(`  Message: ${job.message}`);
    console.log(`  Created: ${new Date(job.createdAt.seconds * 1000).toISOString()}`);
  }

  // Step 5: Explain what happens when job starts (if we had API keys)
  if (!hasGeminiKey) {
    console.log('\n📋 Step 5: What happens when you run startJob()...\n');
    console.log('  With API keys, the pipeline would:');
    console.log('  ┌─ Step 1: Script Generation (0% → 20%)');
    console.log('  │  • Gemini 1.5 Flash generates viral script');
    console.log('  │  • ~150-180 shots with retention optimization');
    console.log('  │  • Cold open + retention bombs + open loops');
    console.log('  │  • Cost: $0.002');
    console.log('  │');
    console.log('  ├─ Step 2: Image Generation (20% → 60%)');
    console.log('  │  • Parallel processing: 5 images at a time');
    console.log('  │  • Self-healing retry with AI prompt fixing');
    console.log('  │  • ~15 minutes for 150 shots (5x faster!)');
    console.log('  │  • Cost: $1.80 for ~90 unique images');
    console.log('  │');
    console.log('  ├─ Step 3: Audio Generation (60% → 75%)');
    console.log('  │  • Google Cloud TTS for narration');
    console.log('  │  • Cost: $0.01 for 5 minutes');
    console.log('  │');
    console.log('  ├─ Step 4: Video Assembly (75% → 95%)');
    console.log('  │  • FFmpeg combines images and audio');
    console.log('  │  • Uploads to Google Cloud Storage');
    console.log('  │');
    console.log('  └─ Step 5: Job Complete (95% → 100%)');
    console.log('     • Final video URL is available');
  }
}

testFullPipeline();
