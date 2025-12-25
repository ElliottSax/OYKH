"""
Critic Agent - Reviews generated images and suggests improvements.

Supports multiple vision model providers:
- Gemini Vision (default)
- Groq + Llama 3.2 Vision (free, fast)
- Qwen-VL-Max (Alibaba Cloud)

Responsibilities:
- Analyze images using vision models
- Check for common issues (cut-off figures, text overlap, wrong poses)
- Provide specific actionable feedback for regeneration
- Flag scenes that need regeneration
"""

import os
import base64
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import requests

from .planner import Scene, Storyboard


class CriticProvider(Enum):
    GEMINI = "gemini"
    GROQ = "groq"
    QWEN = "qwen"


@dataclass
class CriticFeedback:
    """Feedback for a single scene."""
    scene_index: int
    has_issues: bool
    issues: List[str]
    suggested_improvements: str
    quality_score: float  # 0-1


@dataclass
class CriticReport:
    """Complete report from critic review."""
    total_reviewed: int
    scenes_with_issues: int
    feedback: List[CriticFeedback]
    regeneration_needed: List[int]  # Scene indices to regenerate

    @property
    def pass_rate(self) -> float:
        if self.total_reviewed == 0:
            return 1.0
        return 1 - (self.scenes_with_issues / self.total_reviewed)


class CriticAgent:
    """
    Reviews generated images using vision models and provides feedback.

    Supported providers:
    - gemini: Gemini 2.0 Flash (default)
    - groq: Llama 3.2 90B Vision (free, fast)
    - qwen: Qwen-VL-Max (Alibaba Cloud)

    The critic checks for:
    - Character visibility (not cut off at edges)
    - Text/figure overlap issues
    - Correct pose matching description
    - Visual clarity and composition
    - Style consistency
    """

    REVIEW_PROMPT = """You are a quality control critic for educational video frames.

Review this image for VIDEO PRODUCTION quality:
- Title: {title}
- Visual Description: {visual_description}
- Required Pose/Action: {pose}

CHECK FOR MAJOR ISSUES ONLY:
1. Is the main subject fully visible (not cut off at edges)?
2. Is the composition balanced and usable for video?
3. Does it roughly match the visual description concept?
4. Is it clear and professional looking?

IGNORE minor style differences. Focus on: visibility, composition, clarity.

RESPOND IN JSON FORMAT ONLY:
{{"has_issues": true/false, "issues": ["major issue"], "suggested_improvements": "fix", "quality_score": 0.0-1.0}}

Scoring guide:
- 0.9+: Excellent, ready for production
- 0.8+: Good, minor imperfections OK
- 0.7+: Acceptable for video use
- <0.7: Has major issues (cut off, wrong subject, blurry)

Be LENIENT. Only fail images with MAJOR production issues."""

    def __init__(
        self,
        api_key: str,
        provider: CriticProvider = CriticProvider.GEMINI,
        quality_threshold: float = 0.7
    ):
        self.api_key = api_key
        self.provider = provider
        self.quality_threshold = quality_threshold

        # Provider-specific settings
        self.provider_config = {
            CriticProvider.GEMINI: {
                "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                "model": "gemini-2.0-flash"
            },
            CriticProvider.GROQ: {
                "url": "https://api.groq.com/openai/v1/chat/completions",
                "model": "meta-llama/llama-4-scout-17b-16e-instruct"
            },
            CriticProvider.QWEN: {
                "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                "model": "qwen-vl-max"
            }
        }

    def _encode_image(self, image_path: str, max_size: int = 1024) -> str:
        """Encode image to base64, resizing if needed for API limits."""
        try:
            from PIL import Image
            import io

            img = Image.open(image_path)

            # Resize if larger than max_size
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)

            # Convert to JPEG for smaller size
            buffer = io.BytesIO()
            img.convert('RGB').save(buffer, format='JPEG', quality=85)
            return base64.b64encode(buffer.getvalue()).decode()

        except ImportError:
            # Fallback without PIL
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode()

    def _review_single_scene(self, scene: Scene) -> CriticFeedback:
        """Review a single scene's image using configured provider."""
        if not scene.image_path or not os.path.exists(scene.image_path):
            return CriticFeedback(
                scene_index=scene.index,
                has_issues=True,
                issues=["No image file found"],
                suggested_improvements="Generate the image first",
                quality_score=0.0
            )

        image_data = self._encode_image(scene.image_path)

        prompt = self.REVIEW_PROMPT.format(
            title=scene.title,
            visual_description=scene.visual_description,
            pose=scene.pose,
            text_overlay=scene.text_overlay or "None"
        )

        try:
            if self.provider == CriticProvider.GEMINI:
                content = self._call_gemini(prompt, image_data)
            elif self.provider == CriticProvider.GROQ:
                content = self._call_groq(prompt, image_data)
            elif self.provider == CriticProvider.QWEN:
                content = self._call_qwen(prompt, image_data)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            # Parse JSON from response
            json_str = content
            if "```" in content:
                import re
                match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
                if match:
                    json_str = match.group(1)

            result = json.loads(json_str.strip())

            return CriticFeedback(
                scene_index=scene.index,
                has_issues=result.get("has_issues", False),
                issues=result.get("issues", []),
                suggested_improvements=result.get("suggested_improvements", ""),
                quality_score=result.get("quality_score", 0.8)
            )

        except Exception as e:
            # On error, assume it's okay to avoid blocking
            return CriticFeedback(
                scene_index=scene.index,
                has_issues=False,
                issues=[f"Review error: {str(e)}"],
                suggested_improvements="",
                quality_score=0.75
            )

    def _call_gemini(self, prompt: str, image_data: str) -> str:
        """Call Gemini Vision API."""
        response = requests.post(
            f"{self.provider_config[CriticProvider.GEMINI]['url']}?key={self.api_key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/png", "data": image_data}}
                    ]
                }],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 500}
            }
        )
        response.raise_for_status()
        data = response.json()
        return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")

    def _call_groq(self, prompt: str, image_data: str) -> str:
        """Call Groq Llama 3.2 Vision API (free!)."""
        # Note: Groq vision doesn't support system messages
        # Use smaller model (11b) for reliability, 90b can timeout
        response = requests.post(
            self.provider_config[CriticProvider.GROQ]["url"],
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            json={
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",  # Current Groq vision model
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }],
                "temperature": 0.1,
                "max_tokens": 500
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "{}")

    def _call_qwen(self, prompt: str, image_data: str) -> str:
        """Call Qwen-VL-Max API (Alibaba Cloud)."""
        response = requests.post(
            self.provider_config[CriticProvider.QWEN]["url"],
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            json={
                "model": self.provider_config[CriticProvider.QWEN]["model"],
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            }
                        }
                    ]
                }],
                "temperature": 0.1,
                "max_tokens": 500
            }
        )
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "{}")

    def review_storyboard(
        self,
        storyboard: Storyboard,
        scene_indices: Optional[List[int]] = None
    ) -> CriticReport:
        """
        Review all or specific scenes in the storyboard.

        Args:
            storyboard: The storyboard with generated images
            scene_indices: Optional list of specific scenes to review

        Returns:
            CriticReport with feedback for each scene
        """
        indices = scene_indices or list(range(len(storyboard.scenes)))
        feedback_list = []
        regeneration_needed = []

        for idx in indices:
            scene = storyboard.scenes[idx]
            print(f"  Reviewing scene {idx + 1}...", end=" ")

            feedback = self._review_single_scene(scene)
            feedback_list.append(feedback)

            # Store feedback in scene
            scene.critic_feedback = feedback.suggested_improvements

            # Check if regeneration is needed
            if feedback.has_issues or feedback.quality_score < self.quality_threshold:
                regeneration_needed.append(idx)
                print(f"ISSUES ({feedback.quality_score:.2f})")
            else:
                print(f"OK ({feedback.quality_score:.2f})")

        scenes_with_issues = len(regeneration_needed)

        return CriticReport(
            total_reviewed=len(indices),
            scenes_with_issues=scenes_with_issues,
            feedback=feedback_list,
            regeneration_needed=regeneration_needed
        )

    def get_feedback_map(self, report: CriticReport) -> Dict[int, str]:
        """
        Extract feedback map for regeneration.

        Returns:
            Dict mapping scene_index to improvement suggestions
        """
        feedback_map = {}
        for fb in report.feedback:
            if fb.has_issues or fb.quality_score < self.quality_threshold:
                feedback_map[fb.scene_index] = fb.suggested_improvements
        return feedback_map

    def quick_check(self, scene: Scene) -> bool:
        """
        Quick pass/fail check for a single scene.

        Returns:
            True if scene passes quality check
        """
        feedback = self._review_single_scene(scene)
        return not feedback.has_issues and feedback.quality_score >= self.quality_threshold


class BatchCriticAgent(CriticAgent):
    """
    Optimized critic that batches multiple images per request.
    More efficient for large storyboards.
    """

    BATCH_REVIEW_PROMPT = """
    You are a quality control critic for educational video frames.
    Review these {count} images from a storyboard.

    For each image, check:
    1. Stick figure fully visible (not cut off)
    2. Correct pose
    3. No awkward text overlap
    4. Good composition
    5. Consistent style

    RESPOND IN JSON FORMAT:
    {{
        "reviews": [
            {{"scene_index": 0, "has_issues": false, "issues": [], "quality_score": 0.9}},
            {{"scene_index": 1, "has_issues": true, "issues": ["cut off"], "quality_score": 0.4}}
        ]
    }}
    """

    def review_batch(
        self,
        storyboard: Storyboard,
        batch_size: int = 4
    ) -> CriticReport:
        """Review scenes in batches for efficiency."""
        all_feedback = []
        regeneration_needed = []

        scenes = storyboard.scenes
        for i in range(0, len(scenes), batch_size):
            batch = scenes[i:i + batch_size]

            # Prepare batch request
            parts = [{"text": self.BATCH_REVIEW_PROMPT.format(count=len(batch))}]

            for scene in batch:
                if scene.image_path and os.path.exists(scene.image_path):
                    image_data = self._encode_image(scene.image_path)
                    parts.append({
                        "text": f"\nScene {scene.index}: {scene.title} (pose: {scene.pose})"
                    })
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_data
                        }
                    })

            try:
                response = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}",
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": parts}],
                        "generationConfig": {"temperature": 0.1}
                    }
                )
                data = response.json()
                content = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")

                # Parse response
                result = json.loads(content.strip())
                for review in result.get("reviews", []):
                    fb = CriticFeedback(
                        scene_index=review["scene_index"],
                        has_issues=review.get("has_issues", False),
                        issues=review.get("issues", []),
                        suggested_improvements=", ".join(review.get("issues", [])),
                        quality_score=review.get("quality_score", 0.8)
                    )
                    all_feedback.append(fb)

                    if fb.has_issues or fb.quality_score < self.quality_threshold:
                        regeneration_needed.append(fb.scene_index)

            except Exception as e:
                # Fallback to individual review
                for scene in batch:
                    fb = self._review_single_scene(scene)
                    all_feedback.append(fb)
                    if fb.has_issues:
                        regeneration_needed.append(fb.scene_index)

        return CriticReport(
            total_reviewed=len(scenes),
            scenes_with_issues=len(regeneration_needed),
            feedback=all_feedback,
            regeneration_needed=regeneration_needed
        )


if __name__ == "__main__":
    # Test the critic
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY environment variable")
        exit(1)

    from planner import Scene, SceneType, Storyboard

    # Create test scene with existing image
    storyboard = Storyboard(
        topic="Test",
        scenes=[
            Scene(
                index=0,
                title="Test Scene",
                script="Testing the critic agent",
                visual_description="A stick figure waving hello",
                scene_type=SceneType.HOOK,
                pose="waving",
                image_path="/mnt/c/Users/ellio/Desktop/OYKH_2025-12-24T04-20-41/scene_1.png"
            )
        ]
    )

    critic = CriticAgent(api_key)
    report = critic.review_storyboard(storyboard)

    print(f"\nReviewed: {report.total_reviewed}")
    print(f"Issues: {report.scenes_with_issues}")
    print(f"Pass rate: {report.pass_rate:.1%}")

    for fb in report.feedback:
        print(f"\nScene {fb.scene_index}:")
        print(f"  Score: {fb.quality_score:.2f}")
        print(f"  Issues: {fb.issues}")
