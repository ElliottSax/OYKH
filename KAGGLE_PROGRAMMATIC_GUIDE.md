# Kaggle Notebooks - Programmatic Training

## Complete Automation Guide

### Why Use Kaggle Programmatically?

**Benefits:**
- ✅ 30 hours/week FREE GPU (better than Colab!)
- ✅ Fully automated LoRA training
- ✅ CLI/API for everything
- ✅ Integrate into your workflow
- ✅ Schedule retraining automatically
- ✅ No manual clicking

---

## Step 1: Install Kaggle CLI

```bash
npm install -g kaggle-api
# or
pip install kaggle
```

---

## Step 2: Get API Credentials

1. Go to: https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New Token"
4. Downloads `kaggle.json`
5. Move to correct location:

**Windows:**
```bash
mkdir %USERPROFILE%\.kaggle
move kaggle.json %USERPROFILE%\.kaggle\
```

**Mac/Linux:**
```bash
mkdir ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## Step 3: Create Kaggle Dataset (Your Training Images)

```bash
# Create dataset metadata
cat > dataset-metadata.json << EOF
{
  "title": "OYKHCHAR LoRA Training Data",
  "id": "yourusername/oykhchar-training",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

# Upload your training data
kaggle datasets create -p C:/projects/oykh-temp/lora-training-perfect
```

---

## Step 4: Create Training Notebook

**File: `lora-training-notebook.ipynb`**

This is a Jupyter notebook with the training code:

```json
{
  "metadata": {
    "kernelspec": {
      "language": "python",
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.10.12"
    },
    "kaggle": {
      "accelerator": "gpu",
      "dataSources": [
        {
          "datasetId": "yourusername/oykhchar-training",
          "sourceType": "datasetVersion"
        }
      ]
    }
  },
  "nbformat_minor": 4,
  "nbformat": 4,
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "# Install dependencies\n",
        "!pip install -q diffusers transformers accelerate peft safetensors"
      ],
      "metadata": {},
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Your training code here\n",
        "# (Same as Colab example)"
      ],
      "metadata": {},
      "execution_count": null,
      "outputs": []
    }
  ]
}
```

---

## Step 5: Push Notebook to Kaggle

```bash
# Push notebook
kaggle kernels push -p /path/to/notebook/folder
```

**Notebook folder structure:**
```
lora-training/
├── lora-training-notebook.ipynb
└── kernel-metadata.json
```

**kernel-metadata.json:**
```json
{
  "id": "yourusername/oykhchar-lora-training",
  "title": "OYKHCHAR LoRA Training",
  "code_file": "lora-training-notebook.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["yourusername/oykhchar-training"],
  "kernel_sources": []
}
```

---

## Step 6: Run Training Programmatically

```bash
# Start training
kaggle kernels output yourusername/oykhchar-lora-training -p ./output

# This will:
# 1. Start the notebook
# 2. Run all cells
# 3. Download outputs when complete
```

---

## Step 7: Automated Workflow Script

**File: `train-lora-automated.js`**

```javascript
import { exec } from 'child_process';
import { promisify } from 'util';
import fs from 'fs/promises';

const execAsync = promisify(exec);

async function trainLoRAOnKaggle() {
  console.log('🚀 Starting automated LoRA training on Kaggle...');

  // Step 1: Upload training data
  console.log('📤 Uploading training dataset...');
  await execAsync(
    'kaggle datasets create -p C:/projects/oykh-temp/lora-training-perfect'
  );

  // Step 2: Trigger training
  console.log('🎓 Starting training notebook...');
  const { stdout } = await execAsync(
    'kaggle kernels push -p ./kaggle-notebooks/lora-training'
  );

  const kernelSlug = extractKernelSlug(stdout);
  console.log(`✅ Training started: ${kernelSlug}`);

  // Step 3: Monitor progress
  console.log('⏳ Monitoring training progress...');
  await monitorTraining(kernelSlug);

  // Step 4: Download results
  console.log('📥 Downloading trained LoRA...');
  await execAsync(
    `kaggle kernels output ${kernelSlug} -p ./lora-output`
  );

  console.log('🎉 LoRA training complete!');
  console.log('📁 Output: ./lora-output/');
}

async function monitorTraining(kernelSlug) {
  let status = 'running';

  while (status === 'running') {
    await new Promise(resolve => setTimeout(resolve, 60000)); // Check every minute

    const { stdout } = await execAsync(
      `kaggle kernels status ${kernelSlug}`
    );

    status = JSON.parse(stdout).status;
    console.log(`Status: ${status}`);

    if (status === 'complete') {
      return;
    } else if (status === 'error') {
      throw new Error('Training failed!');
    }
  }
}

function extractKernelSlug(output) {
  // Extract kernel slug from push output
  const match = output.match(/Successfully pushed to (.+)/);
  return match ? match[1] : null;
}

// Run it!
trainLoRAOnKaggle().catch(console.error);
```

---

## Step 8: Schedule Automatic Retraining

**Option A: Cron Job (Linux/Mac)**

```bash
# Retrain every Sunday at 2 AM
0 2 * * 0 node /path/to/train-lora-automated.js
```

**Option B: Windows Task Scheduler**

```powershell
$action = New-ScheduledTaskAction -Execute "node" -Argument "C:\projects\oykh-temp\train-lora-automated.js"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 2am
Register-ScheduledTask -Action $action -Trigger $trigger -TaskName "LoRA Retraining"
```

**Option C: GitHub Actions**

```yaml
# .github/workflows/retrain-lora.yml
name: Retrain LoRA
on:
  schedule:
    - cron: '0 2 * * 0'  # Every Sunday at 2 AM
  workflow_dispatch:  # Manual trigger

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Setup Kaggle
        env:
          KAGGLE_USERNAME: ${{ secrets.KAGGLE_USERNAME }}
          KAGGLE_KEY: ${{ secrets.KAGGLE_KEY }}
        run: |
          mkdir -p ~/.kaggle
          echo '{"username":"'$KAGGLE_USERNAME'","key":"'$KAGGLE_KEY'"}' > ~/.kaggle/kaggle.json
          chmod 600 ~/.kaggle/kaggle.json

      - name: Train LoRA
        run: node train-lora-automated.js

      - name: Upload LoRA artifact
        uses: actions/upload-artifact@v2
        with:
          name: trained-lora
          path: ./lora-output/
```

---

## Advanced: Full Integration

**Integrate into video generation pipeline:**

```javascript
// server-simple.js

async function ensureLoRAUpToDate() {
  const loraAge = await getLoRAAgeInDays();

  if (loraAge > 30) {
    console.log('🔄 LoRA is over 30 days old - retraining...');

    // Trigger Kaggle training
    await execAsync('node train-lora-automated.js');

    console.log('✅ LoRA retrained and updated!');
  }
}

// Call before generating videos
await ensureLoRAUpToDate();
```

---

## Monitoring & Notifications

**Add Slack/Discord notifications:**

```javascript
async function notifyTrainingComplete(webhookUrl) {
  await fetch(webhookUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text: '🎉 LoRA training complete! Ready for production.',
      attachments: [{
        color: 'good',
        fields: [
          { title: 'Training Time', value: '45 minutes' },
          { title: 'Loss', value: '0.0023' },
          { title: 'Status', value: '✅ Success' }
        ]
      }]
    })
  });
}
```

---

## Cost Analysis

**Fully Automated Setup:**
- Kaggle training: **$0** (free GPU)
- Automation scripts: **$0** (run locally)
- Scheduling: **$0** (cron/GitHub Actions)
- **Total: $0**

**vs Manual Replicate:**
- Each training: $6-12
- **Total: $6-12 per training**

---

## Use Cases

**1. Weekly Retraining**
- Add new training images each week
- Automatically retrain on Sundays
- Always have the latest LoRA

**2. A/B Testing**
- Train multiple LoRAs with different settings
- Compare results automatically
- Pick the best one

**3. Continuous Improvement**
- Automatically curate best video frames
- Add to training set
- Retrain monthly
- Character consistency improves over time

---

## Complete Setup Checklist

- [ ] Install Kaggle CLI
- [ ] Get API credentials
- [ ] Create dataset with training images
- [ ] Push training notebook
- [ ] Test manual run
- [ ] Set up automated script
- [ ] Configure scheduling (optional)
- [ ] Add notifications (optional)
- [ ] Integrate with video pipeline

---

## Next Steps

**Immediate:**
1. Let Replicate training finish (almost done!)
2. Test the results
3. Use in production

**This Week:**
1. Set up Kaggle CLI
2. Upload training data
3. Test one manual Kaggle training

**Next Month:**
4. Automate with scripts
5. Set up scheduled retraining
6. Never pay for training again! 🎉

---

## Kaggle CLI Quick Reference

```bash
# Upload dataset
kaggle datasets create -p /path/to/data

# Update dataset
kaggle datasets version -p /path/to/data -m "Updated training data"

# Push notebook
kaggle kernels push -p /path/to/notebook

# Check status
kaggle kernels status username/notebook-name

# Download output
kaggle kernels output username/notebook-name -p ./output

# List your kernels
kaggle kernels list --mine

# Pull notebook
kaggle kernels pull username/notebook-name
```

---

**Bottom Line:**
- ✅ Kaggle is fully scriptable
- ✅ Free GPU forever (30 hrs/week)
- ✅ Can fully automate training
- ✅ Integrate into your workflow
- ✅ Zero ongoing costs

Perfect for production! 🚀
