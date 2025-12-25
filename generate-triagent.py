#!/usr/bin/env python3
"""
OYKH Tri-Agent Video Generator

Uses Code2Video-style architecture:
- Planner: Creates structured 60-scene storyboards
- ImageCoder: Generates images in parallel
- Critic: Reviews with Gemini Vision, suggests fixes
- Feedback loop: Auto-regenerates bad scenes

Usage:
    python generate-triagent.py "Your topic... once you know how"
    python generate-triagent.py "Topic" --scenes 5 --no-critic
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add agents to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.orchestrator import VideoOrchestrator, PipelineConfig


def load_env():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().split("\n"):
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()


def main():
    parser = argparse.ArgumentParser(
        description="OYKH Tri-Agent Video Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate-triagent.py "Making coffee is easy... once you know how"
    python generate-triagent.py "Chess tactics" --scenes 10
    python generate-triagent.py "Speed reading" --no-critic --workers 3
        """
    )

    parser.add_argument("topic", help="Video topic")
    parser.add_argument("--scenes", type=int, default=5, help="Number of scenes (default: 5)")
    parser.add_argument("--workers", type=int, default=5, help="Parallel image workers (default: 5)")
    parser.add_argument("--no-critic", action="store_true", help="Disable critic review")
    parser.add_argument("--critic", choices=["gemini", "groq", "qwen"], default="groq",
                        help="Critic vision model: groq (free, default), gemini, or qwen")
    parser.add_argument("--feedback-rounds", type=int, default=2, help="Max critic feedback rounds")
    parser.add_argument("--quality", type=float, default=0.7, help="Quality threshold 0-1 (default: 0.7)")
    parser.add_argument("--output", help="Output directory (default: Desktop)")

    args = parser.parse_args()

    # Load environment
    load_env()

    # Validate API keys
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    replicate_key = os.environ.get("REPLICATE_API_TOKEN")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")
    qwen_key = os.environ.get("QWEN_API_KEY")

    missing = []
    if not deepseek_key:
        missing.append("DEEPSEEK_API_KEY")
    if not replicate_key:
        missing.append("REPLICATE_API_TOKEN")
    if not gemini_key:
        missing.append("GEMINI_API_KEY")

    # Check critic-specific keys
    if args.critic == "groq" and not groq_key:
        missing.append("GROQ_API_KEY")
    if args.critic == "qwen" and not qwen_key:
        missing.append("QWEN_API_KEY")

    if missing:
        print(f"❌ Missing API keys: {', '.join(missing)}")
        print("   Set them in .env file or environment")
        sys.exit(1)

    # Set output directory
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = args.output or f"/mnt/c/Users/ellio/Desktop/OYKH_{timestamp}"

    # Create config
    config = PipelineConfig(
        deepseek_api_key=deepseek_key,
        replicate_api_key=replicate_key,
        gemini_api_key=gemini_key,
        groq_api_key=groq_key or "",
        qwen_api_key=qwen_key or "",
        output_dir=output_dir,
        scene_count=args.scenes,
        max_feedback_rounds=args.feedback_rounds,
        quality_threshold=args.quality,
        critic_provider=args.critic,
        image_workers=args.workers,
        enable_critic=not args.no_critic,
    )

    # Print header
    print()
    print("=" * 60)
    print("  🎬 OYKH TRI-AGENT VIDEO GENERATOR")
    print("=" * 60)
    print(f"  Topic: {args.topic}")
    print(f"  Scenes: {args.scenes}")
    print(f"  Workers: {args.workers}")
    critic_status = f"{args.critic.upper()}" if not args.no_critic else "Disabled"
    print(f"  Critic: {critic_status}")
    print(f"  Output: {output_dir}")
    print("=" * 60)
    print()

    # Run pipeline
    orchestrator = VideoOrchestrator(config)

    try:
        stats = orchestrator.run(args.topic)

        print()
        print("🎉 Video generation complete!")
        print(f"   Find your video at: {output_dir}")

    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
