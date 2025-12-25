"""
Planner Agent - Creates structured storyboards with proper pacing and visual variety.

Responsibilities:
- Generate 60-scene storyboards with YouTube retention best practices
- Include pattern interrupts every 30-60 seconds
- Vary scene types (concept, comparison, data, quote, action)
- Embed hooks and catchphrases at strategic points
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class SceneType(Enum):
    HOOK = "hook"           # Attention grabber (scenes 1-5)
    CONCEPT = "concept"     # Explain an idea with visual
    ACTION = "action"       # Step-by-step instruction
    COMPARISON = "comparison"  # Before/after or vs
    DATA = "data"           # Statistics or facts
    QUOTE = "quote"         # Memorable phrase
    TRANSITION = "transition"  # Pattern interrupt
    REVEAL = "reveal"       # Aha moment
    RECAP = "recap"         # Summary


@dataclass
class Scene:
    """A single scene in the storyboard."""
    index: int
    title: str
    script: str  # Narration text (15-20 words)
    visual_description: str  # What the image should show
    scene_type: SceneType
    duration: float = 10.0  # seconds
    pose: str = "explaining"  # Stick figure pose
    text_overlay: Optional[str] = None  # On-screen text
    status: str = "pending"
    image_path: Optional[str] = None
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    critic_feedback: Optional[str] = None
    regeneration_count: int = 0

    def to_dict(self):
        return {
            "index": self.index,
            "title": self.title,
            "script": self.script,
            "visualDescription": self.visual_description,
            "sceneType": self.scene_type.value,
            "duration": self.duration,
            "pose": self.pose,
            "textOverlay": self.text_overlay,
            "status": self.status,
        }


@dataclass
class Storyboard:
    """Complete storyboard for a video."""
    topic: str
    scenes: List[Scene] = field(default_factory=list)
    total_duration: float = 0.0

    def to_dict(self):
        return {
            "topic": self.topic,
            "scenes": [s.to_dict() for s in self.scenes],
            "totalDuration": self.total_duration,
        }


class PlannerAgent:
    """
    Creates structured storyboards optimized for YouTube retention.

    Uses DeepSeek for cost-effective generation with structured output.
    """

    VISUAL_STYLE = """
    Style: Ultra-clean professional vector-style educational illustration.
    Character: Minimalist white stick figure with round head and two dot eyes.
    Body: Slightly thicker limbs for visibility.
    Outlines: Thick, uniform black vector outlines on everything.
    Shadows: Heavy stylized black shadows beneath character and objects.
    Reflections: White curved highlights in upper-right of major shapes.
    Background: Modern vibrant gradient (varies by scene mood).
    Composition: 16:9 aspect ratio, clean and balanced.
    """

    POSES = [
        "explaining with hands out",
        "pointing upward",
        "thinking with hand on chin",
        "celebrating with arms up",
        "walking confidently",
        "demonstrating with tool",
        "presenting to audience",
        "counting on fingers",
        "giving thumbs up",
        "holding object",
    ]

    STRUCTURE_TEMPLATE = """
    Create a 60-scene instructional video storyboard for: "{topic}"

    STRUCTURE (follow exactly):
    - Scenes 1-5: THE HOOK - Show the impressive end result that seems impossible
    - Scenes 6-15: THE SETUP - Introduce tools, mindset, and prerequisites
    - Scenes 16-45: STEP-BY-STEP - Core instruction, rapid-fire teaching
    - Scenes 46-55: PRO TIPS - Advanced nuances that separate amateurs from pros
    - Scenes 56-60: THE REVEAL - "Aha!" moment and final mastery demonstration

    RULES:
    1. Each scene narration: exactly 15-20 words (fits 10-second window)
    2. Use "once you know how" phrase at least 8 times throughout
    3. Scene 60 MUST end with: "...once you know how."
    4. Include pattern interrupts every 8-10 scenes (surprising visual, question, stat)
    5. Vary poses - don't repeat same pose in consecutive scenes
    6. Visual descriptions must be specific and actionable

    SCENE TYPES (use all of these):
    - hook: Attention grabber
    - concept: Explain idea with visual metaphor
    - action: Step-by-step instruction
    - comparison: Before/after or A vs B
    - data: Statistics or facts
    - quote: Memorable phrase emphasized
    - transition: Pattern interrupt
    - reveal: Aha moment
    - recap: Quick summary

    OUTPUT FORMAT (JSON array):
    [
      {{
        "title": "Scene Title",
        "script": "Narration text here",
        "visualDescription": "Detailed visual description",
        "sceneType": "hook|concept|action|comparison|data|quote|transition|reveal|recap",
        "pose": "stick figure pose description",
        "textOverlay": "Optional on-screen text or null"
      }}
    ]

    Generate exactly 60 scenes.
    """

    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.deepseek.com/chat/completions"

    async def create_storyboard(self, topic: str) -> Storyboard:
        """Generate a complete 60-scene storyboard for the topic."""
        import aiohttp

        prompt = self.STRUCTURE_TEMPLATE.format(topic=topic)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": f"You are a professional video storyboard creator. Visual style: {self.VISUAL_STYLE}. Always respond with valid JSON only."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 20000
                }
            ) as response:
                data = await response.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Parse JSON (handle markdown code blocks)
        json_str = content
        if "```" in content:
            import re
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
            if match:
                json_str = match.group(1)

        scenes_data = json.loads(json_str.strip())

        # Convert to Scene objects
        scenes = []
        for i, s in enumerate(scenes_data):
            scene_type = SceneType(s.get("sceneType", "concept"))
            scenes.append(Scene(
                index=i,
                title=s.get("title", f"Scene {i+1}"),
                script=s.get("script", ""),
                visual_description=s.get("visualDescription", ""),
                scene_type=scene_type,
                pose=s.get("pose", "explaining"),
                text_overlay=s.get("textOverlay"),
            ))

        storyboard = Storyboard(
            topic=topic,
            scenes=scenes,
            total_duration=len(scenes) * 10.0
        )

        return storyboard

    def create_storyboard_sync(self, topic: str, scene_count: int = 60) -> Storyboard:
        """Synchronous version for non-async contexts."""
        import requests

        # Adjust template for different scene counts
        if scene_count <= 10:
            prompt = f"""Create a {scene_count}-scene instructional video storyboard for: "{topic}"

Each scene needs:
- title: Short scene title
- script: 15-20 word narration
- visualDescription: What the image should show
- sceneType: one of hook/concept/action/comparison/data/quote/transition/reveal/recap
- pose: stick figure pose

Use "once you know how" at least once. Last scene MUST end with "...once you know how."

Return ONLY a JSON array, no markdown:
[{{"title":"...", "script":"...", "visualDescription":"...", "sceneType":"...", "pose":"..."}}]
"""
        else:
            prompt = self.STRUCTURE_TEMPLATE.format(topic=topic)

        response = requests.post(
            self.api_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a video storyboard creator. Return ONLY valid JSON arrays, no markdown."
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 8000
            }
        )

        if not response.ok:
            raise Exception(f"DeepSeek API error: {response.status_code} - {response.text[:200]}")

        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        if not content:
            raise Exception(f"Empty response from DeepSeek: {data}")

        # Parse JSON
        json_str = content
        if "```" in content:
            import re
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
            if match:
                json_str = match.group(1)

        scenes_data = json.loads(json_str.strip())

        scenes = []
        for i, s in enumerate(scenes_data):
            try:
                scene_type = SceneType(s.get("sceneType", "concept"))
            except ValueError:
                scene_type = SceneType.CONCEPT

            scenes.append(Scene(
                index=i,
                title=s.get("title", f"Scene {i+1}"),
                script=s.get("script", ""),
                visual_description=s.get("visualDescription", ""),
                scene_type=scene_type,
                pose=s.get("pose", "explaining"),
                text_overlay=s.get("textOverlay"),
            ))

        return Storyboard(
            topic=topic,
            scenes=scenes,
            total_duration=len(scenes) * 10.0
        )


if __name__ == "__main__":
    # Test the planner
    import os

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Set DEEPSEEK_API_KEY environment variable")
        exit(1)

    planner = PlannerAgent(api_key)
    storyboard = planner.create_storyboard_sync("Making perfect coffee is easy... once you know how")

    print(f"Generated {len(storyboard.scenes)} scenes")
    for scene in storyboard.scenes[:5]:
        print(f"\n{scene.index+1}. {scene.title} [{scene.scene_type.value}]")
        print(f"   Script: {scene.script}")
        print(f"   Visual: {scene.visual_description[:80]}...")
