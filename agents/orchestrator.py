"""
Video Orchestrator - Coordinates the tri-agent feedback loop.

Implements the Code2Video-style closed-loop system:
1. Planner creates storyboard
2. ImageCoder generates images in parallel
3. Critic reviews and flags issues
4. Bad scenes get regenerated with feedback
5. Repeat until quality threshold or max iterations
6. Generate audio and video
7. Combine into final video
"""

import os
import json
import time
import subprocess
from dataclasses import dataclass
from typing import Optional, Callable
from pathlib import Path

from .planner import PlannerAgent, Storyboard
from .coder import ImageCoderAgent, GenerationResult
from .critic import CriticAgent, CriticReport, CriticProvider


@dataclass
class PipelineConfig:
    """Configuration for the video generation pipeline."""
    # API Keys (required)
    deepseek_api_key: str
    replicate_api_key: str
    gemini_api_key: str
    output_dir: str

    # Optional API keys
    groq_api_key: str = ""
    qwen_api_key: str = ""

    # Output settings
    scene_count: int = 60

    # Quality control
    max_feedback_rounds: int = 2
    max_regeneration_per_scene: int = 3
    quality_threshold: float = 0.7

    # Critic provider: "gemini", "groq", or "qwen"
    critic_provider: str = "gemini"

    # Performance
    image_workers: int = 5
    enable_critic: bool = True

    # Paths
    ffmpeg_path: str = "/home/elliott/.local/bin/ffmpeg"


@dataclass
class PipelineStats:
    """Statistics from a pipeline run."""
    topic: str
    total_scenes: int
    images_generated: int
    images_regenerated: int
    critic_rounds: int
    final_pass_rate: float
    total_cost: float
    total_time: float


class VideoOrchestrator:
    """
    Orchestrates the complete video generation pipeline.

    Flow:
    1. Planner → Storyboard
    2. ImageCoder → Images (parallel)
    3. Critic → Review → Feedback
    4. Regenerate bad scenes (loop back to 2-3)
    5. Generate audio (Gemini TTS)
    6. Generate video clips (Replicate SVD)
    7. Combine with ffmpeg
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

        # Initialize agents
        self.planner = PlannerAgent(config.deepseek_api_key)
        self.coder = ImageCoderAgent(
            config.replicate_api_key,
            config.output_dir,
            max_workers=config.image_workers
        )

        # Select critic provider and API key
        critic_provider = CriticProvider(config.critic_provider)
        if critic_provider == CriticProvider.GROQ:
            critic_key = config.groq_api_key
        elif critic_provider == CriticProvider.QWEN:
            critic_key = config.qwen_api_key
        else:
            critic_key = config.gemini_api_key

        self.critic = CriticAgent(
            critic_key,
            provider=critic_provider,
            quality_threshold=config.quality_threshold
        )

        # Tracking
        self.storyboard: Optional[Storyboard] = None
        self.stats = None
        self.start_time = None

    def _log(self, message: str):
        """Print with timestamp."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"[{elapsed:6.1f}s] {message}")

    def generate_storyboard(self, topic: str) -> Storyboard:
        """Step 1: Generate structured storyboard."""
        self._log("📝 Generating storyboard with Planner...")

        self.storyboard = self.planner.create_storyboard_sync(topic, self.config.scene_count)

        self._log(f"✅ Created {len(self.storyboard.scenes)} scenes")

        # Save storyboard
        storyboard_path = os.path.join(self.config.output_dir, "storyboard.json")
        with open(storyboard_path, "w") as f:
            json.dump(self.storyboard.to_dict(), f, indent=2)

        return self.storyboard

    def generate_images(self, feedback_map: dict = None) -> list:
        """Step 2: Generate images in parallel."""
        self._log(f"🎨 Generating images ({self.config.image_workers} workers)...")

        results = self.coder.generate_all(self.storyboard, feedback_map)

        success = sum(1 for r in results if r.success)
        failed = len(results) - success

        self._log(f"✅ Generated {success} images ({failed} failed)")

        return results

    def run_critic_review(self) -> CriticReport:
        """Step 3: Critic reviews all images."""
        self._log("🔍 Critic reviewing images...")

        report = self.critic.review_storyboard(self.storyboard)

        self._log(f"📊 Pass rate: {report.pass_rate:.1%} ({report.scenes_with_issues} issues)")

        return report

    def regenerate_with_feedback(self, report: CriticReport) -> list:
        """Step 4: Regenerate bad scenes with critic feedback."""
        if not report.regeneration_needed:
            return []

        # Filter scenes that haven't exceeded max regenerations
        to_regenerate = [
            idx for idx in report.regeneration_needed
            if self.storyboard.scenes[idx].regeneration_count < self.config.max_regeneration_per_scene
        ]

        if not to_regenerate:
            self._log("⚠️ Max regenerations reached for remaining issues")
            return []

        self._log(f"🔄 Regenerating {len(to_regenerate)} scenes...")

        feedback_map = self.critic.get_feedback_map(report)
        results = self.coder.regenerate_scenes(
            self.storyboard,
            to_regenerate,
            feedback_map
        )

        success = sum(1 for r in results if r.success)
        self._log(f"✅ Regenerated {success}/{len(to_regenerate)} scenes")

        return results

    def generate_audio(self) -> int:
        """Step 5: Generate audio for all scenes using Gemini TTS."""
        self._log("🔊 Generating audio with Gemini TTS...")

        success_count = 0

        for scene in self.storyboard.scenes:
            audio_path = os.path.join(
                self.config.output_dir,
                f"scene_{scene.index + 1}.wav"
            )

            try:
                response = __import__('requests').post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key={self.config.gemini_api_key}",
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{
                            "parts": [{"text": scene.script}]
                        }],
                        "generationConfig": {
                            "responseModalities": ["AUDIO"],
                            "speechConfig": {
                                "voiceConfig": {
                                    "prebuiltVoiceConfig": {"voiceName": "Kore"}
                                }
                            }
                        }
                    }
                )

                data = response.json()
                audio_data = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("inlineData", {}).get("data")

                if audio_data:
                    import base64
                    audio_bytes = base64.b64decode(audio_data)

                    # Convert PCM to WAV
                    wav_data = self._pcm_to_wav(audio_bytes)
                    with open(audio_path, "wb") as f:
                        f.write(wav_data)

                    scene.audio_path = audio_path
                    success_count += 1

            except Exception as e:
                # Save script as fallback
                txt_path = audio_path.replace(".wav", "_narration.txt")
                with open(txt_path, "w") as f:
                    f.write(scene.script)

        self._log(f"✅ Generated {success_count}/{len(self.storyboard.scenes)} audio files")
        return success_count

    def _pcm_to_wav(self, pcm_data: bytes, sample_rate: int = 24000) -> bytes:
        """Convert raw PCM to WAV format."""
        import struct

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

    def generate_videos(self) -> int:
        """Step 6: Generate video clips from images."""
        self._log("🎬 Generating video clips...")

        success_count = 0
        import requests
        import base64
        import time as time_module

        for scene in self.storyboard.scenes:
            if not scene.image_path or not os.path.exists(scene.image_path):
                continue

            video_path = os.path.join(
                self.config.output_dir,
                f"scene_{scene.index + 1}.mp4"
            )

            try:
                # Read image
                with open(scene.image_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode()

                data_url = f"data:image/png;base64,{image_data}"

                # Start prediction
                response = requests.post(
                    "https://api.replicate.com/v1/predictions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.config.replicate_api_key}"
                    },
                    json={
                        "version": "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438",
                        "input": {
                            "input_image": data_url,
                            "video_length": "14_frames_with_svd",
                            "sizing_strategy": "maintain_aspect_ratio",
                            "frames_per_second": 6,
                            "motion_bucket_id": 127
                        }
                    }
                )
                prediction = response.json()

                # Poll for completion
                while prediction.get("status") not in ("succeeded", "failed", "canceled"):
                    time_module.sleep(3)
                    poll = requests.get(
                        prediction["urls"]["get"],
                        headers={"Authorization": f"Bearer {self.config.replicate_api_key}"}
                    )
                    prediction = poll.json()

                if prediction["status"] == "succeeded":
                    video_url = prediction["output"]
                    video_response = requests.get(video_url)
                    with open(video_path, "wb") as f:
                        f.write(video_response.content)
                    scene.video_path = video_path
                    success_count += 1

                print(f"  Scene {scene.index + 1}: {'✓' if prediction['status'] == 'succeeded' else '✗'}")

            except Exception as e:
                print(f"  Scene {scene.index + 1}: Error - {str(e)[:50]}")

        self._log(f"✅ Generated {success_count}/{len(self.storyboard.scenes)} video clips")
        return success_count

    def combine_final_video(self) -> str:
        """Step 7: Combine all scenes into final video."""
        self._log("🎬 Combining scenes into final video...")

        output_dir = self.config.output_dir
        ffmpeg = self.config.ffmpeg_path

        # Merge each scene's video with audio
        for scene in self.storyboard.scenes:
            video_path = scene.video_path or os.path.join(output_dir, f"scene_{scene.index + 1}.mp4")
            audio_path = scene.audio_path or os.path.join(output_dir, f"scene_{scene.index + 1}.wav")
            merged_path = os.path.join(output_dir, f"merged_{scene.index + 1}.mp4")

            if os.path.exists(video_path) and os.path.exists(audio_path):
                subprocess.run([
                    ffmpeg, "-y",
                    "-stream_loop", "-1",
                    "-i", video_path,
                    "-i", audio_path,
                    "-map", "0:v", "-map", "1:a",
                    "-c:v", "libx264", "-c:a", "aac",
                    "-shortest", "-pix_fmt", "yuv420p",
                    merged_path
                ], capture_output=True)
            elif os.path.exists(video_path):
                # No audio, just copy
                subprocess.run(["cp", video_path, merged_path], capture_output=True)

        # Create concat list
        concat_path = os.path.join(output_dir, "concat_list.txt")
        with open(concat_path, "w") as f:
            for scene in self.storyboard.scenes:
                merged = os.path.join(output_dir, f"merged_{scene.index + 1}.mp4")
                if os.path.exists(merged):
                    f.write(f"file '{merged}'\n")

        # Generate safe filename
        safe_name = self.storyboard.topic.replace(" ", "_")[:30]
        safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")
        final_path = os.path.join(output_dir, f"FINAL_{safe_name}.mp4")

        # Concatenate
        subprocess.run([
            ffmpeg, "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_path,
            "-c:v", "libx264", "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            final_path
        ], capture_output=True)

        # Cleanup
        for scene in self.storyboard.scenes:
            merged = os.path.join(output_dir, f"merged_{scene.index + 1}.mp4")
            if os.path.exists(merged):
                os.remove(merged)
        if os.path.exists(concat_path):
            os.remove(concat_path)

        self._log(f"✅ Final video: {os.path.basename(final_path)}")
        return final_path

    def run(self, topic: str) -> PipelineStats:
        """
        Run the complete pipeline with feedback loop.

        Returns:
            PipelineStats with metrics from the run
        """
        self.start_time = time.time()
        os.makedirs(self.config.output_dir, exist_ok=True)

        self._log(f"🚀 Starting OYKH Pipeline")
        self._log(f"   Topic: {topic}")
        self._log(f"   Output: {self.config.output_dir}")
        self._log("")

        # Step 1: Generate storyboard
        self.generate_storyboard(topic)

        # Step 2: Generate images
        self.generate_images()

        images_regenerated = 0

        # Step 3-4: Critic feedback loop
        if self.config.enable_critic:
            for round_num in range(self.config.max_feedback_rounds):
                self._log(f"\n--- Critic Round {round_num + 1}/{self.config.max_feedback_rounds} ---")

                report = self.run_critic_review()

                if report.pass_rate >= 0.95:
                    self._log("✨ Quality threshold met!")
                    break

                if report.regeneration_needed:
                    results = self.regenerate_with_feedback(report)
                    images_regenerated += len(results)
                else:
                    break

        # Final review
        final_report = self.run_critic_review() if self.config.enable_critic else None
        final_pass_rate = final_report.pass_rate if final_report else 1.0

        # Step 5: Generate audio
        self._log("")
        self.generate_audio()

        # Step 6: Generate video clips
        self.generate_videos()

        # Step 7: Combine final video
        final_video = self.combine_final_video()

        # Calculate stats
        total_time = time.time() - self.start_time
        total_cost = self.coder.total_cost + 0.01  # Add ~$0.01 for DeepSeek

        self.stats = PipelineStats(
            topic=topic,
            total_scenes=len(self.storyboard.scenes),
            images_generated=len(self.storyboard.scenes),
            images_regenerated=images_regenerated,
            critic_rounds=self.config.max_feedback_rounds,
            final_pass_rate=final_pass_rate,
            total_cost=total_cost,
            total_time=total_time
        )

        self._log("")
        self._log("=" * 50)
        self._log("📊 PIPELINE COMPLETE")
        self._log(f"   Scenes: {self.stats.total_scenes}")
        self._log(f"   Regenerated: {self.stats.images_regenerated}")
        self._log(f"   Pass rate: {self.stats.final_pass_rate:.1%}")
        self._log(f"   Cost: ${self.stats.total_cost:.2f}")
        self._log(f"   Time: {self.stats.total_time/60:.1f} minutes")
        self._log(f"   Output: {final_video}")
        self._log("=" * 50)

        return self.stats


def create_video(
    topic: str,
    output_dir: str,
    deepseek_key: str,
    replicate_key: str,
    gemini_key: str,
    scene_count: int = 5,  # Default to 5 for quick testing
    enable_critic: bool = True
) -> str:
    """
    Convenience function to create a video.

    Returns path to final video.
    """
    config = PipelineConfig(
        deepseek_api_key=deepseek_key,
        replicate_api_key=replicate_key,
        gemini_api_key=gemini_key,
        output_dir=output_dir,
        scene_count=scene_count,
        enable_critic=enable_critic
    )

    orchestrator = VideoOrchestrator(config)
    orchestrator.run(topic)

    # Find final video
    for f in os.listdir(output_dir):
        if f.startswith("FINAL_") and f.endswith(".mp4"):
            return os.path.join(output_dir, f)

    return None


if __name__ == "__main__":
    import sys

    # Load env
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().split("\n"):
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

    topic = sys.argv[1] if len(sys.argv) > 1 else "Making perfect coffee is easy... once you know how"

    config = PipelineConfig(
        deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        replicate_api_key=os.environ.get("REPLICATE_API_TOKEN", ""),
        gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
        output_dir=f"/mnt/c/Users/ellio/Desktop/OYKH_triagent_{int(time.time())}",
        scene_count=5,  # Quick test
        enable_critic=True
    )

    orchestrator = VideoOrchestrator(config)
    orchestrator.run(topic)
