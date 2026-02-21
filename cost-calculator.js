/**
 * Cost Calculator - Compare Old vs New Infrastructure
 * Shows potential savings across all services
 */

console.log('💰 Infrastructure Cost Calculator');
console.log('===================================\n');

// Monthly usage estimates (adjust based on your actual usage)
const usage = {
  // OYKH Video Generation
  images_generated: 3000,        // Images per month
  video_frames_needed: 5000,     // Frames for videos

  // LLM Usage (RepurposeAI, Coach, Dream, Membership)
  llm_tokens_input: 50_000_000,  // 50M tokens/month
  llm_tokens_output: 10_000_000, // 10M tokens/month

  // LoRA Training
  lora_trainings: 4,             // Times per month

  // Storage
  storage_gb: 100,               // GB stored
  bandwidth_gb: 500,             // GB transferred

  // Database queries
  db_queries: 5_000_000          // Queries per month
};

// OLD COSTS (Current Setup)
const oldCosts = {
  // Replicate
  images: usage.images_generated * 0.05,
  lora_training: usage.lora_trainings * 9, // Average $6-12

  // OpenAI
  llm_input: (usage.llm_tokens_input / 1_000_000) * 30, // GPT-4
  llm_output: (usage.llm_tokens_output / 1_000_000) * 60,

  // AWS S3
  storage: usage.storage_gb * 0.023,
  bandwidth: usage.bandwidth_gb * 0.09,

  // Database (estimated)
  database: 50,

  // Redis (estimated)
  redis: 25
};

const oldTotal = Object.values(oldCosts).reduce((a, b) => a + b, 0);

// NEW COSTS (Optimized Setup)
const newCosts = {
  // Modal (60% cheaper than Replicate)
  images: usage.images_generated * 0.02,

  // Kaggle (FREE!)
  lora_training: 0,
  batch_frames: 0, // Included in Kaggle free tier

  // Together.ai (90% cheaper)
  llm_input: (usage.llm_tokens_input / 1_000_000) * 0.88,
  llm_output: (usage.llm_tokens_output / 1_000_000) * 0.88,

  // Cloudflare R2 (cheaper + no egress fees)
  storage: usage.storage_gb * 0.015,
  bandwidth: 0, // No egress fees!

  // Supabase (FREE tier)
  database: 0,

  // Vercel KV (FREE tier)
  redis: 0
};

const newTotal = Object.values(newCosts).reduce((a, b) => a + b, 0);

const savings = oldTotal - newTotal;
const savingsPercent = ((savings / oldTotal) * 100).toFixed(1);

// Display Results
console.log('OLD INFRASTRUCTURE (Current)');
console.log('─────────────────────────────');
console.log(`Images (Replicate):          $${oldCosts.images.toFixed(2)}`);
console.log(`LoRA Training (Replicate):   $${oldCosts.lora_training.toFixed(2)}`);
console.log(`LLM Calls (OpenAI):          $${(oldCosts.llm_input + oldCosts.llm_output).toFixed(2)}`);
console.log(`Storage (S3):                $${oldCosts.storage.toFixed(2)}`);
console.log(`Bandwidth (S3):              $${oldCosts.bandwidth.toFixed(2)}`);
console.log(`Database (Prisma/PlanetScale): $${oldCosts.database.toFixed(2)}`);
console.log(`Redis:                       $${oldCosts.redis.toFixed(2)}`);
console.log(`─────────────────────────────`);
console.log(`TOTAL:                       $${oldTotal.toFixed(2)}/month\n`);

console.log('NEW INFRASTRUCTURE (Optimized)');
console.log('─────────────────────────────');
console.log(`Images (Modal):              $${newCosts.images.toFixed(2)}`);
console.log(`LoRA Training (Kaggle):      $${newCosts.lora_training.toFixed(2)} (FREE!)`);
console.log(`Batch Frames (Kaggle):       $${newCosts.batch_frames.toFixed(2)} (FREE!)`);
console.log(`LLM Calls (Together.ai):     $${(newCosts.llm_input + newCosts.llm_output).toFixed(2)}`);
console.log(`Storage (R2):                $${newCosts.storage.toFixed(2)}`);
console.log(`Bandwidth (R2):              $${newCosts.bandwidth.toFixed(2)} (FREE!)`);
console.log(`Database (Supabase):         $${newCosts.database.toFixed(2)} (FREE!)`);
console.log(`Redis (Vercel KV):           $${newCosts.redis.toFixed(2)} (FREE!)`);
console.log(`─────────────────────────────`);
console.log(`TOTAL:                       $${newTotal.toFixed(2)}/month\n`);

console.log('💰 SAVINGS');
console.log('─────────────────────────────');
console.log(`Per Month:                   $${savings.toFixed(2)} (${savingsPercent}% reduction)`);
console.log(`Per Year:                    $${(savings * 12).toFixed(2)}\n`);

// Breakdown by category
console.log('SAVINGS BY CATEGORY');
console.log('─────────────────────────────');

const categories = {
  'Image Generation': oldCosts.images - newCosts.images,
  'LoRA Training': oldCosts.lora_training - newCosts.lora_training,
  'LLM Calls': (oldCosts.llm_input + oldCosts.llm_output) - (newCosts.llm_input + newCosts.llm_output),
  'Storage + Bandwidth': (oldCosts.storage + oldCosts.bandwidth) - (newCosts.storage + newCosts.bandwidth),
  'Database': oldCosts.database - newCosts.database,
  'Redis': oldCosts.redis - newCosts.redis
};

Object.entries(categories)
  .sort((a, b) => b[1] - a[1])
  .forEach(([category, saving]) => {
    console.log(`${category.padEnd(25)} $${saving.toFixed(2)}/month`);
  });

console.log('\n');

// ROI Analysis
console.log('⏱️  IMPLEMENTATION TIME vs ROI');
console.log('─────────────────────────────');
console.log(`Setup time:                  10-15 hours`);
console.log(`Monthly savings:             $${savings.toFixed(2)}`);
console.log(`Hourly value:                $${(savings / 12.5).toFixed(2)}/hour`);
console.log(`Payback period:              Immediate!\n`);

// Projections
console.log('📈 SAVINGS PROJECTIONS');
console.log('─────────────────────────────');
console.log(`3 months:                    $${(savings * 3).toFixed(2)}`);
console.log(`6 months:                    $${(savings * 6).toFixed(2)}`);
console.log(`1 year:                      $${(savings * 12).toFixed(2)}`);
console.log(`2 years:                     $${(savings * 24).toFixed(2)}\n`);

// Recommendations
console.log('🎯 TOP PRIORITIES (Highest Savings)');
console.log('─────────────────────────────');

const priorities = [
  { name: 'Together.ai (LLM)', saving: categories['LLM Calls'], time: '3 hours' },
  { name: 'Modal (Images)', saving: categories['Image Generation'], time: '2 hours' },
  { name: 'Cloudflare R2', saving: categories['Storage + Bandwidth'], time: '2 hours' },
  { name: 'Kaggle (LoRA)', saving: categories['LoRA Training'], time: '1 hour (done!)' },
  { name: 'Supabase', saving: categories['Database'], time: '4 hours' }
];

priorities
  .sort((a, b) => b.saving - a.saving)
  .forEach((p, i) => {
    console.log(`${i + 1}. ${p.name.padEnd(20)} $${p.saving.toFixed(2)}/mo (${p.time})`);
  });

console.log('\n🚀 Start with #1-3 for 80% of savings in 7 hours!\n');
