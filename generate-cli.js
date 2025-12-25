#!/usr/bin/env node
/**
 * OYKH CLI Video Generator
 * Generates a full video and saves to desktop
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Load environment variables from .env
const envPath = path.join(__dirname, '.env');
if (fs.existsSync(envPath)) {
  const envContent = fs.readFileSync(envPath, 'utf-8');
  envContent.split('\n').forEach(line => {
    const [key, ...valueParts] = line.split('=');
    if (key && !key.startsWith('#')) {
      process.env[key.trim()] = valueParts.join('=').trim();
    }
  });
}

const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY;
const REPLICATE_API_TOKEN = process.env.REPLICATE_API_TOKEN;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

// Desktop path for Windows via WSL
const DESKTOP_PATH = '/mnt/c/Users/ellio/Desktop';

const VISUAL_STYLE_PROMPT = `
Style Guide: Ultra-clean professional vector-style educational illustration.
- Character: A minimalist white stick figure with round head and dot eyes.
- Outlines: Thick, heavy, uniform black vector outlines.
- Shading: Heavy stylized black shadows for depth.
- Background: Modern, vibrant gradient background.
- Composition: 16:9 aspect ratio, clean and professional.
`;

async function generateScript(topic) {
  console.log('📝 Generating script with DeepSeek...');

  const prompt = `Write a 5-scene instructional "How-To" script for: "${topic}".
    Return ONLY a valid JSON array with this structure:
    [{"title": "Scene Title", "script": "15-20 word narration", "visualDescription": "Visual description"}]

    Keep it short - exactly 5 scenes for a demo.`;

  const response = await fetch("https://api.deepseek.com/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${DEEPSEEK_API_KEY}`
    },
    body: JSON.stringify({
      model: "deepseek-chat",
      messages: [
        { role: "system", content: "You are a script writer. Respond with valid JSON only." },
        { role: "user", content: prompt }
      ],
      temperature: 0.7,
      max_tokens: 4000
    })
  });

  const data = await response.json();
  let content = data.choices?.[0]?.message?.content || "";

  // Extract JSON from markdown if present
  const jsonMatch = content.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (jsonMatch) content = jsonMatch[1];

  return JSON.parse(content.trim());
}

async function generateImage(scene, index, outputDir) {
  console.log(`🎨 Generating image for scene ${index + 1}...`);

  const prompt = `${VISUAL_STYLE_PROMPT}
  Scene: ${scene.visualDescription}
  Style: Clean vector illustration, stick figure character, thick outlines, 16:9 ratio`;

  const response = await fetch("https://api.replicate.com/v1/predictions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${REPLICATE_API_TOKEN}`
    },
    body: JSON.stringify({
      version: "black-forest-labs/flux-schnell",
      input: {
        prompt: prompt,
        aspect_ratio: "16:9",
        num_outputs: 1,
        output_format: "png"
      }
    })
  });

  let prediction = await response.json();

  // Poll for completion
  while (prediction.status !== "succeeded" && prediction.status !== "failed") {
    await new Promise(r => setTimeout(r, 2000));
    const poll = await fetch(prediction.urls.get, {
      headers: { "Authorization": `Bearer ${REPLICATE_API_TOKEN}` }
    });
    prediction = await poll.json();
    process.stdout.write('.');
  }
  console.log(' done');

  if (prediction.status === "failed") throw new Error("Image generation failed");

  const imageUrl = prediction.output?.[0];
  const imageResponse = await fetch(imageUrl);
  const imageBuffer = Buffer.from(await imageResponse.arrayBuffer());

  const imagePath = path.join(outputDir, `scene_${index + 1}.png`);
  fs.writeFileSync(imagePath, imageBuffer);

  return imagePath;
}

async function generateVideo(imagePath, scene, index, outputDir) {
  console.log(`🎬 Generating video for scene ${index + 1}...`);

  const imageData = fs.readFileSync(imagePath);
  const base64Image = imageData.toString('base64');
  const dataUrl = `data:image/png;base64,${base64Image}`;

  const response = await fetch("https://api.replicate.com/v1/predictions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${REPLICATE_API_TOKEN}`
    },
    body: JSON.stringify({
      version: "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438",
      input: {
        input_image: dataUrl,
        video_length: "14_frames_with_svd",
        sizing_strategy: "maintain_aspect_ratio",
        frames_per_second: 6,
        motion_bucket_id: 127
      }
    })
  });

  let prediction = await response.json();

  // Poll for completion (video takes longer)
  while (prediction.status !== "succeeded" && prediction.status !== "failed") {
    await new Promise(r => setTimeout(r, 5000));
    const poll = await fetch(prediction.urls.get, {
      headers: { "Authorization": `Bearer ${REPLICATE_API_TOKEN}` }
    });
    prediction = await poll.json();
    process.stdout.write('.');
  }
  console.log(' done');

  if (prediction.status === "failed") throw new Error("Video generation failed");

  const videoUrl = prediction.output;
  const videoResponse = await fetch(videoUrl);
  const videoBuffer = Buffer.from(await videoResponse.arrayBuffer());

  const videoPath = path.join(outputDir, `scene_${index + 1}.mp4`);
  fs.writeFileSync(videoPath, videoBuffer);

  return videoPath;
}

async function generateAudio(scene, index, outputDir) {
  console.log(`🔊 Generating audio for scene ${index + 1}...`);

  // Use Google Cloud TTS via Gemini's text-to-speech endpoint
  // Try the TTS-specific model
  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key=${GEMINI_API_KEY}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{
          parts: [{
            text: scene.script
          }]
        }],
        generationConfig: {
          responseModalities: ["AUDIO"],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName: "Kore" }
            }
          }
        }
      })
    }
  );

  if (!response.ok) {
    const errText = await response.text();
    console.log(`  (TTS error: ${response.status}, using fallback)`);
    // Fallback: save script as text file for manual TTS
    const txtPath = path.join(outputDir, `scene_${index + 1}_narration.txt`);
    fs.writeFileSync(txtPath, scene.script);
    return null;
  }

  const data = await response.json();
  const base64Audio = data.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;

  if (!base64Audio) {
    console.log('  (no audio data returned, saving script text)');
    const txtPath = path.join(outputDir, `scene_${index + 1}_narration.txt`);
    fs.writeFileSync(txtPath, scene.script);
    return null;
  }

  // The audio is raw PCM, convert to WAV
  const audioBuffer = Buffer.from(base64Audio, 'base64');
  const wavBuffer = pcmToWav(audioBuffer, 24000, 1, 16);

  const audioPath = path.join(outputDir, `scene_${index + 1}.wav`);
  fs.writeFileSync(audioPath, wavBuffer);
  console.log('  done');

  return audioPath;
}

// Convert raw PCM to WAV format
function pcmToWav(pcmBuffer, sampleRate, numChannels, bitsPerSample) {
  const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
  const blockAlign = numChannels * (bitsPerSample / 8);
  const dataSize = pcmBuffer.length;
  const headerSize = 44;
  const fileSize = headerSize + dataSize;

  const wav = Buffer.alloc(fileSize);

  // RIFF header
  wav.write('RIFF', 0);
  wav.writeUInt32LE(fileSize - 8, 4);
  wav.write('WAVE', 8);

  // fmt chunk
  wav.write('fmt ', 12);
  wav.writeUInt32LE(16, 16); // chunk size
  wav.writeUInt16LE(1, 20); // PCM format
  wav.writeUInt16LE(numChannels, 22);
  wav.writeUInt32LE(sampleRate, 24);
  wav.writeUInt32LE(byteRate, 28);
  wav.writeUInt16LE(blockAlign, 32);
  wav.writeUInt16LE(bitsPerSample, 34);

  // data chunk
  wav.write('data', 36);
  wav.writeUInt32LE(dataSize, 40);
  pcmBuffer.copy(wav, 44);

  return wav;
}

async function combineScenes(outputDir, sceneCount, topicName) {
  console.log('\n🎬 Combining scenes into final video...');

  const ffmpeg = '/home/elliott/.local/bin/ffmpeg';

  // Merge each video with its audio
  for (let i = 1; i <= sceneCount; i++) {
    const videoPath = path.join(outputDir, `scene_${i}.mp4`);
    const audioPath = path.join(outputDir, `scene_${i}.wav`);
    const mergedPath = path.join(outputDir, `merged_${i}.mp4`);

    if (fs.existsSync(audioPath)) {
      process.stdout.write(`  Merging scene ${i}...`);
      execSync(`${ffmpeg} -y -stream_loop -1 -i "${videoPath}" -i "${audioPath}" -map 0:v -map 1:a -c:v libx264 -c:a aac -shortest -pix_fmt yuv420p "${mergedPath}" 2>/dev/null`);
      console.log(' done');
    } else {
      // No audio, just copy video
      fs.copyFileSync(videoPath, mergedPath);
    }
  }

  // Create concat list
  const concatList = path.join(outputDir, 'concat_list.txt');
  const concatContent = Array.from({ length: sceneCount }, (_, i) =>
    `file 'merged_${i + 1}.mp4'`
  ).join('\n');
  fs.writeFileSync(concatList, concatContent);

  // Generate safe filename from topic
  const safeName = topicName
    .replace(/[^a-zA-Z0-9\s]/g, '')
    .replace(/\s+/g, '_')
    .slice(0, 30);

  const finalPath = path.join(outputDir, `FINAL_${safeName}.mp4`);

  process.stdout.write('  Concatenating all scenes...');
  execSync(`${ffmpeg} -y -f concat -safe 0 -i "${concatList}" -c:v libx264 -c:a aac -pix_fmt yuv420p "${finalPath}" 2>/dev/null`);
  console.log(' done');

  // Cleanup intermediate files
  for (let i = 1; i <= sceneCount; i++) {
    fs.unlinkSync(path.join(outputDir, `merged_${i}.mp4`));
  }
  fs.unlinkSync(concatList);

  return finalPath;
}

async function main() {
  const topic = process.argv[2] || "Making the perfect coffee is easy... once you know how";

  console.log('\n🎬 OYKH Video Generator');
  console.log('========================');
  console.log(`Topic: ${topic}\n`);

  // Create output directory on desktop
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const outputDir = path.join(DESKTOP_PATH, `OYKH_${timestamp}`);
  fs.mkdirSync(outputDir, { recursive: true });
  console.log(`📁 Output: ${outputDir}\n`);

  try {
    // Generate script
    const scenes = await generateScript(topic);
    console.log(`✅ Generated ${scenes.length} scenes\n`);

    // Save script
    fs.writeFileSync(
      path.join(outputDir, 'script.json'),
      JSON.stringify(scenes, null, 2)
    );

    // Process each scene
    for (let i = 0; i < scenes.length; i++) {
      console.log(`\n--- Scene ${i + 1}: ${scenes[i].title} ---`);

      const imagePath = await generateImage(scenes[i], i, outputDir);
      const videoPath = await generateVideo(imagePath, scenes[i], i, outputDir);
      await generateAudio(scenes[i], i, outputDir);
    }

    // Combine all scenes into final video
    const finalVideo = await combineScenes(outputDir, scenes.length, topic);

    console.log('\n✅ Complete!');
    console.log(`📁 Files saved to: ${outputDir}`);
    console.log(`\n🎥 Final video: ${path.basename(finalVideo)}`);

  } catch (error) {
    console.error('\n❌ Error:', error.message);
    process.exit(1);
  }
}

main();
