#!/usr/bin/env python3
"""
OYKH Video Generator - Simple, Cost-Effective Educational Videos

Generates "Once You Know How" style videos with:
- Blob character style (NOT stick figures)
- Ken Burns zoom/pan effects (NOT expensive video gen)
- Professional Gemini TTS narration (Rasalgethi voice)

Cost: ~$0.25 per 60-scene video

Usage:
    python generate.py "Your topic once you know how"
    python generate.py "Topic" --scenes 10
"""

import os
import sys
import json
import time
import base64
import struct
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional
import requests


# ============================================================================
# BLOB CHARACTER STYLE - Critical for visual consistency
# ============================================================================

BLOB_CHARACTER_STYLE = """Ultra-clean vector-style illustration. White blob character with round head,
two black dot eyes, thick black outlines, soft gray edge shading, curved
white highlight on head and body. Organic bean-shaped body, NOT a stick
figure. Black ellipse shadow beneath character. Vibrant smooth gradient
background (orange, purple, blue, cyan tones). Modern, friendly, educational
style. 16:9 aspect ratio, clean uncluttered composition."""


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Scene:
    """A single scene in the storyboard."""
    index: int
    title: str
    script: str
    visual_description: str
    pose: str
    duration: float = 10.0
    image_path: Optional[str] = None
    audio_path: Optional[str] = None
    video_path: Optional[str] = None


@dataclass
class Storyboard:
    """Complete storyboard for a video."""
    topic: str
    scenes: List[Scene]


# ============================================================================
# API FUNCTIONS
# ============================================================================

def create_storyboard(topic: str, scene_count: int, api_key: str) -> Storyboard:
    """Generate storyboard using DeepSeek."""
    print(f"  Creating storyboard with {scene_count} scenes...")

    prompt = f"""Create a {scene_count}-scene storyboard for an educational video.

Topic: "{topic}"

For each scene, provide:
- title: Short scene title
- script: Narration text (2-3 sentences, educational tone)
- visualDescription: What to show (blob character doing something related to topic)
- pose: Character's pose/action

Style: Friendly educational content like "Once You Know How" videos.
The character is a simple white blob with dot eyes (NOT a stick figure).

Return ONLY valid JSON array:
[
  {{"title": "...", "script": "...", "visualDescription": "...", "pose": "..."}},
  ...
]"""

    response = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }
    )
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]

    # Extract JSON from response
    import re
    json_match = re.search(r'\[[\s\S]*\]', content)
    if json_match:
        scenes_data = json.loads(json_match.group())
    else:
        scenes_data = json.loads(content)

    scenes = []
    for i, s in enumerate(scenes_data[:scene_count]):
        scenes.append(Scene(
            index=i,
            title=s.get("title", f"Scene {i+1}"),
            script=s.get("script", ""),
            visual_description=s.get("visualDescription", s.get("visual_description", "")),
            pose=s.get("pose", "standing")
        ))

    print(f"  Created {len(scenes)} scenes")
    return Storyboard(topic=topic, scenes=scenes)


def generate_image(scene: Scene, output_dir: str, api_key: str) -> bool:
    """Generate image using Replicate Flux with blob style."""
    output_path = os.path.join(output_dir, "images", f"scene_{scene.index + 1}.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Combine blob style with scene description
    full_prompt = f"{BLOB_CHARACTER_STYLE}\n\nScene: {scene.visual_description}\nCharacter pose: {scene.pose}"

    try:
        # Start prediction
        response = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers={
                "Authorization": f"Token {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "version": "black-forest-labs/flux-schnell",
                "input": {
                    "prompt": full_prompt,
                    "aspect_ratio": "16:9",
                    "num_outputs": 1
                }
            }
        )
        prediction = response.json()

        # Poll for completion
        prediction_url = prediction.get("urls", {}).get("get") or f"https://api.replicate.com/v1/predictions/{prediction['id']}"

        for _ in range(60):  # Max 60 attempts (2 minutes)
            time.sleep(2)
            poll = requests.get(
                prediction_url,
                headers={"Authorization": f"Token {api_key}"}
            )
            result = poll.json()

            if result["status"] == "succeeded":
                # Download image
                image_url = result["output"][0] if isinstance(result["output"], list) else result["output"]
                img_response = requests.get(image_url)
                with open(output_path, "wb") as f:
                    f.write(img_response.content)
                scene.image_path = output_path
                return True
            elif result["status"] == "failed":
                return False

        return False

    except Exception as e:
        print(f"    Error generating image: {e}")
        return False


def generate_images_parallel(storyboard: Storyboard, output_dir: str, api_key: str, workers: int = 5):
    """Generate all images in parallel."""
    print(f"  Generating {len(storyboard.scenes)} images ({workers} workers)...")

    success = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(generate_image, scene, output_dir, api_key): scene
            for scene in storyboard.scenes
        }

        for future in as_completed(futures):
            scene = futures[future]
            try:
                if future.result():
                    success += 1
                    print(f"    Scene {scene.index + 1}: OK")
                else:
                    print(f"    Scene {scene.index + 1}: FAILED")
            except Exception as e:
                print(f"    Scene {scene.index + 1}: ERROR - {e}")

    print(f"  Generated {success}/{len(storyboard.scenes)} images")
    return success


def generate_audio(scene: Scene, output_dir: str, api_key: str) -> bool:
    """Generate audio using Gemini TTS with Rasalgethi voice."""
    output_path = os.path.join(output_dir, "audio", f"scene_{scene.index + 1}.wav")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": scene.script}]}],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {"voiceName": "Rasalgethi"}
                        }
                    }
                }
            }
        )

        data = response.json()
        audio_data = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("inlineData", {}).get("data")

        if audio_data:
            # Convert PCM to WAV
            pcm_bytes = base64.b64decode(audio_data)
            wav_bytes = pcm_to_wav(pcm_bytes)
            with open(output_path, "wb") as f:
                f.write(wav_bytes)
            scene.audio_path = output_path
            return True

        return False

    except Exception as e:
        print(f"    Error generating audio: {e}")
        return False


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000) -> bytes:
    """Convert raw PCM to WAV format."""
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm_data)

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,
        b'WAVE',
        b'fmt ',
        16,
        1,  # PCM
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )

    return header + pcm_data


def generate_all_audio(storyboard: Storyboard, output_dir: str, api_key: str):
    """Generate audio for all scenes."""
    print(f"  Generating {len(storyboard.scenes)} audio clips...")

    success = 0
    for scene in storyboard.scenes:
        if generate_audio(scene, output_dir, api_key):
            success += 1
            print(f"    Scene {scene.index + 1}: OK")
        else:
            print(f"    Scene {scene.index + 1}: FAILED")

    print(f"  Generated {success}/{len(storyboard.scenes)} audio clips")
    return success


# ============================================================================
# KEN BURNS VIDEO EFFECTS
# ============================================================================

KEN_BURNS_EFFECTS = [
    # Zoom in to center
    "scale=2496:1404,zoompan=z='zoom+0.001':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1920x1080:fps=30",
    # Zoom out from center
    "scale=2496:1404,zoompan=z='1.3-0.001*on':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1920x1080:fps=30",
    # Pan left to right
    "scale=2496:1404,zoompan=z='1.1':x='0+on*2':y='ih/2-(ih/zoom/2)':d={frames}:s=1920x1080:fps=30",
    # Pan right to left
    "scale=2496:1404,zoompan=z='1.1':x='iw-iw/zoom-on*2':y='ih/2-(ih/zoom/2)':d={frames}:s=1920x1080:fps=30",
    # Zoom in + pan down
    "scale=2496:1404,zoompan=z='zoom+0.0008':x='iw/2-(iw/zoom/2)':y='0+on':d={frames}:s=1920x1080:fps=30",
]


def create_video_clip(scene: Scene, output_dir: str, ffmpeg_path: str = "ffmpeg") -> bool:
    """Create video clip with Ken Burns effect."""
    if not scene.image_path or not os.path.exists(scene.image_path):
        return False

    output_path = os.path.join(output_dir, "clips", f"scene_{scene.index + 1}.mp4")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Get audio duration if available
    duration = 5.0  # Default
    if scene.audio_path and os.path.exists(scene.audio_path):
        try:
            result = subprocess.run(
                [ffmpeg_path, "-i", scene.audio_path],
                capture_output=True, text=True
            )
            import re
            match = re.search(r"Duration: (\d+):(\d+):(\d+\.?\d*)", result.stderr)
            if match:
                h, m, s = match.groups()
                duration = int(h) * 3600 + int(m) * 60 + float(s)
        except:
            pass

    frames = int(duration * 30)  # 30 fps

    # Select effect based on scene index (cycle through effects)
    effect = KEN_BURNS_EFFECTS[scene.index % len(KEN_BURNS_EFFECTS)].format(frames=frames)

    # Add fade in/out
    fade_duration = min(0.5, duration / 4)
    fade_out_start = max(0, duration - fade_duration)
    effect += f",fade=t=in:st=0:d={fade_duration},fade=t=out:st={fade_out_start}:d={fade_duration}"

    cmd = [ffmpeg_path, "-y"]

    # Input image
    cmd.extend(["-loop", "1", "-i", scene.image_path])

    # Input audio if available
    if scene.audio_path and os.path.exists(scene.audio_path):
        cmd.extend(["-i", scene.audio_path])

    # Video filter
    cmd.extend(["-vf", effect])

    # Output settings
    cmd.extend([
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-t", str(duration)
    ])

    # Audio codec if audio present
    if scene.audio_path and os.path.exists(scene.audio_path):
        cmd.extend(["-c:a", "aac", "-shortest"])

    cmd.append(output_path)

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode == 0:
            scene.video_path = output_path
            return True
        return False
    except Exception as e:
        print(f"    Error creating video: {e}")
        return False


def create_all_clips(storyboard: Storyboard, output_dir: str, ffmpeg_path: str = "ffmpeg"):
    """Create video clips for all scenes."""
    print(f"  Creating {len(storyboard.scenes)} video clips with Ken Burns...")

    success = 0
    for scene in storyboard.scenes:
        if create_video_clip(scene, output_dir, ffmpeg_path):
            success += 1
            print(f"    Scene {scene.index + 1}: OK")
        else:
            print(f"    Scene {scene.index + 1}: FAILED")

    print(f"  Created {success}/{len(storyboard.scenes)} clips")
    return success


def concatenate_videos(storyboard: Storyboard, output_dir: str, ffmpeg_path: str = "ffmpeg") -> str:
    """Concatenate all clips into final video."""
    print("  Concatenating clips into final video...")

    # Create concat list
    concat_path = os.path.join(output_dir, "concat_list.txt")
    with open(concat_path, "w") as f:
        for scene in storyboard.scenes:
            if scene.video_path and os.path.exists(scene.video_path):
                f.write(f"file '{scene.video_path}'\n")

    # Generate safe filename
    safe_name = storyboard.topic.replace(" ", "_")[:30]
    safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")
    final_path = os.path.join(output_dir, f"FINAL_{safe_name}.mp4")

    cmd = [
        ffmpeg_path, "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_path,
        "-c:v", "libx264", "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        final_path
    ]

    try:
        subprocess.run(cmd, capture_output=True, timeout=120)
        os.remove(concat_path)
        print(f"  Final video: {final_path}")
        return final_path
    except Exception as e:
        print(f"  Error concatenating: {e}")
        return None


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def load_env():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().split("\n"):
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())


def run_pipeline(topic: str, scene_count: int = 5, output_dir: str = None, workers: int = 5):
    """Run the complete video generation pipeline."""
    start_time = time.time()

    # Load environment
    load_env()

    # Get API keys
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    replicate_key = os.environ.get("REPLICATE_API_TOKEN")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    ffmpeg_path = os.environ.get("FFMPEG_PATH", "/home/elliott/.local/bin/ffmpeg")

    missing = []
    if not deepseek_key: missing.append("DEEPSEEK_API_KEY")
    if not replicate_key: missing.append("REPLICATE_API_TOKEN")
    if not gemini_key: missing.append("GEMINI_API_KEY")

    if missing:
        print(f"Missing API keys: {', '.join(missing)}")
        print("Set them in .env file or environment")
        sys.exit(1)

    # Set output directory
    if not output_dir:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"/mnt/c/Users/ellio/Desktop/OYKH_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    print()
    print("=" * 60)
    print("  OYKH VIDEO GENERATOR")
    print("=" * 60)
    print(f"  Topic: {topic}")
    print(f"  Scenes: {scene_count}")
    print(f"  Output: {output_dir}")
    print("=" * 60)
    print()

    # Step 1: Create storyboard
    print("[1/5] Creating storyboard...")
    storyboard = create_storyboard(topic, scene_count, deepseek_key)

    # Save storyboard
    storyboard_path = os.path.join(output_dir, "storyboard.json")
    with open(storyboard_path, "w") as f:
        json.dump({
            "topic": storyboard.topic,
            "scenes": [
                {
                    "index": s.index,
                    "title": s.title,
                    "script": s.script,
                    "visualDescription": s.visual_description,
                    "pose": s.pose
                }
                for s in storyboard.scenes
            ]
        }, f, indent=2)

    # Step 2: Generate images
    print("\n[2/5] Generating images (Flux)...")
    generate_images_parallel(storyboard, output_dir, replicate_key, workers)

    # Step 3: Generate audio
    print("\n[3/5] Generating audio (Gemini TTS)...")
    generate_all_audio(storyboard, output_dir, gemini_key)

    # Step 4: Create video clips
    print("\n[4/5] Creating video clips (Ken Burns)...")
    create_all_clips(storyboard, output_dir, ffmpeg_path)

    # Step 5: Concatenate
    print("\n[5/5] Creating final video...")
    final_path = concatenate_videos(storyboard, output_dir, ffmpeg_path)

    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("  COMPLETE!")
    print(f"  Time: {elapsed:.1f} seconds")
    print(f"  Output: {final_path}")
    print("=" * 60)

    return final_path


def main():
    parser = argparse.ArgumentParser(
        description="OYKH Video Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate.py "Making coffee is easy once you know how"
    python generate.py "Chess tactics" --scenes 10
    python generate.py "Speed reading" --scenes 5 --workers 3
        """
    )

    parser.add_argument("topic", help="Video topic")
    parser.add_argument("--scenes", type=int, default=5, help="Number of scenes (default: 5)")
    parser.add_argument("--workers", type=int, default=5, help="Parallel image workers (default: 5)")
    parser.add_argument("--output", help="Output directory")

    args = parser.parse_args()

    try:
        run_pipeline(args.topic, args.scenes, args.output, args.workers)
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
