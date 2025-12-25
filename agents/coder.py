"""
ImageCoder Agent - Generates images in parallel with retry logic.

Responsibilities:
- Generate images for all scenes using Replicate Flux
- Run multiple workers in parallel
- Handle failures and retries
- Track generation costs
"""

import os
import base64
import asyncio
import aiohttp
from dataclasses import dataclass
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import requests
import time

from .planner import Scene, Storyboard


@dataclass
class GenerationResult:
    """Result of an image generation attempt."""
    scene_index: int
    success: bool
    image_path: Optional[str] = None
    image_data: Optional[str] = None  # base64
    error: Optional[str] = None
    attempts: int = 1
    cost: float = 0.003  # ~$0.003 per Flux image


class ImageCoderAgent:
    """
    Generates images for storyboard scenes using Replicate Flux.

    Features:
    - Parallel generation with configurable workers
    - Automatic retries on failure
    - Cost tracking
    - Accepts critic feedback for regeneration
    """

    VISUAL_STYLE = """
    Ultra-clean professional vector-style educational illustration.
    - Minimalist white stick figure character
    - Perfectly round head with two small black dot eyes
    - Thick, heavy, uniform black outlines on everything
    - Heavy stylized black shadows beneath character and objects
    - White curved highlights in upper-right of major shapes
    - Modern vibrant gradient background
    - 16:9 aspect ratio, clean composition
    """

    def __init__(
        self,
        api_key: str,
        output_dir: str,
        max_workers: int = 5,
        max_retries: int = 3
    ):
        self.api_key = api_key
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.total_cost = 0.0

        os.makedirs(output_dir, exist_ok=True)

    def _build_prompt(self, scene: Scene, feedback: Optional[str] = None) -> str:
        """Build the image generation prompt for a scene."""
        prompt_parts = [
            self.VISUAL_STYLE,
            f"\nScene: {scene.visual_description}",
            f"\nPose: Stick figure {scene.pose}",
        ]

        if scene.text_overlay:
            prompt_parts.append(f"\nText overlay: \"{scene.text_overlay}\"")

        if feedback:
            prompt_parts.append(f"\n\nIMPORTANT CORRECTIONS: {feedback}")

        return "".join(prompt_parts)

    def _generate_single_image(
        self,
        scene: Scene,
        feedback: Optional[str] = None
    ) -> GenerationResult:
        """Generate a single image synchronously."""
        prompt = self._build_prompt(scene, feedback)

        for attempt in range(1, self.max_retries + 1):
            try:
                # Start prediction
                response = requests.post(
                    "https://api.replicate.com/v1/predictions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    },
                    json={
                        "version": "black-forest-labs/flux-schnell",
                        "input": {
                            "prompt": prompt,
                            "aspect_ratio": "16:9",
                            "num_outputs": 1,
                            "output_format": "png"
                        }
                    }
                )
                response.raise_for_status()
                prediction = response.json()

                # Poll for completion
                while prediction["status"] not in ("succeeded", "failed", "canceled"):
                    time.sleep(1.5)
                    poll = requests.get(
                        prediction["urls"]["get"],
                        headers={"Authorization": f"Bearer {self.api_key}"}
                    )
                    prediction = poll.json()

                if prediction["status"] != "succeeded":
                    raise Exception(f"Prediction failed: {prediction.get('error')}")

                # Download image
                image_url = prediction["output"][0]
                img_response = requests.get(image_url)
                img_response.raise_for_status()

                # Save to file
                image_path = os.path.join(
                    self.output_dir,
                    f"scene_{scene.index + 1}.png"
                )
                with open(image_path, "wb") as f:
                    f.write(img_response.content)

                # Also keep base64 for video generation
                image_data = base64.b64encode(img_response.content).decode()

                self.total_cost += 0.003  # Track cost

                return GenerationResult(
                    scene_index=scene.index,
                    success=True,
                    image_path=image_path,
                    image_data=image_data,
                    attempts=attempt
                )

            except Exception as e:
                if attempt == self.max_retries:
                    return GenerationResult(
                        scene_index=scene.index,
                        success=False,
                        error=str(e),
                        attempts=attempt
                    )
                time.sleep(2 ** attempt)  # Exponential backoff

    def generate_all(
        self,
        storyboard: Storyboard,
        feedback_map: Optional[dict] = None
    ) -> List[GenerationResult]:
        """
        Generate images for all scenes in parallel.

        Args:
            storyboard: The storyboard with scenes to generate
            feedback_map: Optional dict of scene_index -> feedback string

        Returns:
            List of GenerationResult objects
        """
        feedback_map = feedback_map or {}
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for scene in storyboard.scenes:
                feedback = feedback_map.get(scene.index)
                future = executor.submit(
                    self._generate_single_image,
                    scene,
                    feedback
                )
                futures.append((scene.index, future))

            for scene_index, future in futures:
                result = future.result()
                results.append(result)

                # Update scene status
                scene = storyboard.scenes[scene_index]
                if result.success:
                    scene.image_path = result.image_path
                    scene.status = "image_generated"
                else:
                    scene.status = "image_failed"

        return sorted(results, key=lambda r: r.scene_index)

    def regenerate_scenes(
        self,
        storyboard: Storyboard,
        scene_indices: List[int],
        feedback_map: dict
    ) -> List[GenerationResult]:
        """
        Regenerate specific scenes with critic feedback.

        Args:
            storyboard: The storyboard
            scene_indices: Indices of scenes to regenerate
            feedback_map: Dict of scene_index -> improvement suggestion

        Returns:
            List of GenerationResult for regenerated scenes
        """
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for idx in scene_indices:
                scene = storyboard.scenes[idx]
                scene.regeneration_count += 1
                feedback = feedback_map.get(idx, "")
                future = executor.submit(
                    self._generate_single_image,
                    scene,
                    feedback
                )
                futures.append((idx, future))

            for scene_index, future in futures:
                result = future.result()
                results.append(result)

                scene = storyboard.scenes[scene_index]
                if result.success:
                    scene.image_path = result.image_path
                    scene.status = "image_regenerated"
                else:
                    scene.status = "image_failed"

        return results

    async def generate_all_async(
        self,
        storyboard: Storyboard,
        feedback_map: Optional[dict] = None,
        progress_callback=None
    ) -> List[GenerationResult]:
        """Async version for better concurrency."""
        feedback_map = feedback_map or {}
        results = []
        semaphore = asyncio.Semaphore(self.max_workers)

        async def generate_with_semaphore(scene: Scene):
            async with semaphore:
                # Run sync generation in thread pool
                loop = asyncio.get_event_loop()
                feedback = feedback_map.get(scene.index)
                result = await loop.run_in_executor(
                    None,
                    self._generate_single_image,
                    scene,
                    feedback
                )

                if progress_callback:
                    await progress_callback(scene.index, result)

                return result

        tasks = [generate_with_semaphore(s) for s in storyboard.scenes]
        results = await asyncio.gather(*tasks)

        # Update scene statuses
        for result in results:
            scene = storyboard.scenes[result.scene_index]
            if result.success:
                scene.image_path = result.image_path
                scene.status = "image_generated"
            else:
                scene.status = "image_failed"

        return sorted(results, key=lambda r: r.scene_index)

    def get_stats(self) -> dict:
        """Get generation statistics."""
        return {
            "total_cost": self.total_cost,
            "output_dir": self.output_dir,
            "max_workers": self.max_workers,
        }


if __name__ == "__main__":
    # Test the image coder
    from planner import PlannerAgent, Storyboard, Scene, SceneType

    api_key = os.environ.get("REPLICATE_API_TOKEN")
    if not api_key:
        print("Set REPLICATE_API_TOKEN environment variable")
        exit(1)

    # Create a mini storyboard for testing
    storyboard = Storyboard(
        topic="Test",
        scenes=[
            Scene(
                index=0,
                title="Test Scene",
                script="This is a test scene",
                visual_description="A stick figure waving hello",
                scene_type=SceneType.HOOK,
                pose="waving"
            )
        ]
    )

    coder = ImageCoderAgent(api_key, output_dir="/tmp/oykh_test")
    results = coder.generate_all(storyboard)

    for r in results:
        print(f"Scene {r.scene_index}: {'Success' if r.success else 'Failed'}")
        if r.success:
            print(f"  Path: {r.image_path}")
        else:
            print(f"  Error: {r.error}")
