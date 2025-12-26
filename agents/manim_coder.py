"""
Manim Coder Agent - Generates Manim code for deterministic animation rendering.

Based on Code2Video architecture:
- LLM generates Manim Python code (not image prompts)
- ScopeRefine debugging: line -> block -> global scope
- Visual Anchor system: 6x6 grid for precise positioning
- Deterministic output (no AI hallucination in rendering)

Benefits over AI image generation:
- 100% reproducible output
- Precise text/element positioning
- No cut-off figures or overlap issues
- Easy to iterate and fix
"""

import os
import re
import json
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import requests

from .planner import Scene, Storyboard


@dataclass
class ManimCode:
    """Generated Manim code for a scene."""
    scene_index: int
    code: str
    class_name: str
    output_path: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0


@dataclass
class ScopeRefineConfig:
    """Configuration for hierarchical debugging."""
    line_attempts: int = 3      # K1: attempts at line scope
    block_attempts: int = 2     # K2: attempts at block scope
    global_attempts: int = 1    # K3: full regeneration attempts


class VisualAnchor:
    """
    6x6 grid anchor system for precise element positioning.

    Converts fuzzy descriptions like "center" or "top-left" into
    discrete grid coordinates that Manim can render precisely.

    Grid layout (6x6):
        (0,5) (1,5) (2,5) (3,5) (4,5) (5,5)  <- TOP
        (0,4) (1,4) (2,4) (3,4) (4,4) (5,4)
        (0,3) (1,3) (2,3) (3,3) (4,3) (5,3)
        (0,2) (1,2) (2,2) (3,2) (4,2) (5,2)
        (0,1) (1,1) (2,1) (3,1) (4,1) (5,1)
        (0,0) (1,0) (2,0) (3,0) (4,0) (5,0)  <- BOTTOM
          ^                           ^
         LEFT                       RIGHT
    """

    # Map natural language positions to grid coordinates
    POSITION_MAP = {
        "center": (2.5, 2.5),
        "top": (2.5, 4.5),
        "bottom": (2.5, 0.5),
        "left": (0.5, 2.5),
        "right": (4.5, 2.5),
        "top-left": (0.5, 4.5),
        "top-right": (4.5, 4.5),
        "bottom-left": (0.5, 0.5),
        "bottom-right": (4.5, 0.5),
        "upper-center": (2.5, 3.5),
        "lower-center": (2.5, 1.5),
    }

    @classmethod
    def to_manim_coords(cls, anchor: str) -> Tuple[float, float]:
        """Convert anchor name to Manim coordinates."""
        grid = cls.POSITION_MAP.get(anchor.lower(), (2.5, 2.5))
        # Convert 6x6 grid to Manim's coordinate system (-7 to 7 x, -4 to 4 y)
        x = (grid[0] - 2.5) * 2.8  # Scale to Manim x range
        y = (grid[1] - 2.5) * 1.6  # Scale to Manim y range
        return (round(x, 2), round(y, 2))

    @classmethod
    def grid_to_manim(cls, gx: int, gy: int) -> Tuple[float, float]:
        """Convert raw grid coordinates to Manim coordinates."""
        x = (gx - 2.5) * 2.8
        y = (gy - 2.5) * 1.6
        return (round(x, 2), round(y, 2))


class ManimCoder:
    """
    Generates Manim animation code from scene descriptions.

    Uses DeepSeek to write Python Manim code, then renders
    with the Manim CLI for deterministic output.
    """

    CODE_PROMPT = '''You are a Manim animation expert. Generate Python code for the Manim Community library.

SCENE REQUIREMENTS:
- Title: {title}
- Description: {description}
- Pose/Action: {pose}
- Duration: {duration} seconds (at 30fps = {frames} frames)

VISUAL ANCHOR COORDINATES (use these for positioning):
{anchor_guide}

STYLE REQUIREMENTS:
- Clean, minimal educational style
- Use simple shapes and stick figures
- Clear text with good contrast
- Smooth animations

OUTPUT FORMAT:
```python
from manim import *

class Scene{index}(Scene):
    def construct(self):
        # Your animation code here
        pass
```

IMPORTANT:
- Use self.play() for animations
- Use self.wait() for pauses
- Position elements using the anchor coordinates provided
- Keep animations simple and clear
- The scene should be self-contained

Generate ONLY the Python code, no explanations.'''

    ANCHOR_GUIDE = """
Manim coordinate system:
- Center: (0, 0)
- Top: (0, 3)
- Bottom: (0, -3)
- Left: (-5, 0)
- Right: (5, 0)
- Top-left: (-5, 3)
- Top-right: (5, 3)
- Bottom-left: (-5, -3)
- Bottom-right: (5, -3)

Use UP, DOWN, LEFT, RIGHT, ORIGIN for common positions.
Use .shift() or .move_to() for precise positioning.
"""

    FIX_PROMPT = '''The following Manim code has an error. Fix it.

ORIGINAL CODE:
```python
{code}
```

ERROR MESSAGE:
{error}

SCOPE: {scope}
{scope_context}

Fix the error and return the complete corrected code.
Output ONLY the fixed Python code, no explanations.'''

    def __init__(
        self,
        api_key: str,
        python_path: str = "python3",
        output_dir: str = "./output",
        scope_config: Optional[ScopeRefineConfig] = None
    ):
        self.api_key = api_key
        self.python_path = python_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scope_config = scope_config or ScopeRefineConfig()

    def _call_deepseek(self, prompt: str, temperature: float = 0.3) -> str:
        """Call DeepSeek API for code generation."""
        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": 2000
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Try to find code block
        match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try without language specifier
        match = re.search(r"```\s*(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Return as-is if no code block found
        return response.strip()

    def _render_manim(self, code: str, class_name: str, output_name: str) -> Tuple[bool, str]:
        """
        Render Manim code to image/video.

        Returns:
            (success, error_or_path)
        """
        # Write code to temp file
        temp_dir = tempfile.mkdtemp()
        code_path = os.path.join(temp_dir, "scene.py")

        with open(code_path, "w") as f:
            f.write(code)

        # Run manim
        output_path = self.output_dir / f"{output_name}.png"

        try:
            result = subprocess.run(
                [
                    self.python_path, "-m", "manim", "render",
                    "-ql",  # Low quality for speed
                    "--format", "png",
                    "-o", str(output_path),
                    code_path,
                    class_name
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir
            )

            if result.returncode != 0:
                error = result.stderr or result.stdout
                return False, error

            # Find the actual output file (manim creates subdirs)
            for root, dirs, files in os.walk(temp_dir):
                for f in files:
                    if f.endswith(".png"):
                        import shutil
                        actual_path = os.path.join(root, f)
                        shutil.copy(actual_path, output_path)
                        return True, str(output_path)

            return False, "No output file generated"

        except subprocess.TimeoutExpired:
            return False, "Render timeout (60s)"
        except Exception as e:
            return False, str(e)

    def _scope_refine(
        self,
        code: str,
        error: str,
        class_name: str,
        output_name: str
    ) -> Tuple[bool, str, str]:
        """
        Hierarchical debugging: line -> block -> global.

        Returns:
            (success, code_or_error, output_path)
        """
        current_code = code

        # Line scope (K1 attempts)
        for i in range(self.scope_config.line_attempts):
            print(f"    ScopeRefine: line attempt {i+1}/{self.scope_config.line_attempts}")

            # Extract error line if possible
            line_match = re.search(r"line (\d+)", error)
            line_num = int(line_match.group(1)) if line_match else None

            scope_context = ""
            if line_num:
                lines = current_code.split("\n")
                start = max(0, line_num - 2)
                end = min(len(lines), line_num + 2)
                scope_context = f"Focus on lines {start+1}-{end}:\n" + "\n".join(lines[start:end])

            prompt = self.FIX_PROMPT.format(
                code=current_code,
                error=error,
                scope="LINE",
                scope_context=scope_context
            )

            fixed = self._extract_code(self._call_deepseek(prompt, temperature=0.2))
            success, result = self._render_manim(fixed, class_name, output_name)

            if success:
                return True, fixed, result

            current_code = fixed
            error = result

        # Block scope (K2 attempts)
        for i in range(self.scope_config.block_attempts):
            print(f"    ScopeRefine: block attempt {i+1}/{self.scope_config.block_attempts}")

            prompt = self.FIX_PROMPT.format(
                code=current_code,
                error=error,
                scope="BLOCK",
                scope_context="Focus on the construct() method and its structure."
            )

            fixed = self._extract_code(self._call_deepseek(prompt, temperature=0.4))
            success, result = self._render_manim(fixed, class_name, output_name)

            if success:
                return True, fixed, result

            current_code = fixed
            error = result

        # Global scope (K3 attempts) - full regeneration
        for i in range(self.scope_config.global_attempts):
            print(f"    ScopeRefine: global attempt {i+1}/{self.scope_config.global_attempts}")

            prompt = self.FIX_PROMPT.format(
                code=current_code,
                error=error,
                scope="GLOBAL",
                scope_context="Regenerate the entire scene from scratch with a simpler approach."
            )

            fixed = self._extract_code(self._call_deepseek(prompt, temperature=0.6))
            success, result = self._render_manim(fixed, class_name, output_name)

            if success:
                return True, fixed, result

            error = result

        return False, error, ""

    def generate_scene(self, scene: Scene) -> ManimCode:
        """
        Generate Manim code for a single scene.

        Uses ScopeRefine for automatic error correction.
        """
        class_name = f"Scene{scene.index}"
        output_name = f"scene_{scene.index + 1}"

        print(f"  Generating Manim code for scene {scene.index + 1}...")

        # Generate initial code
        prompt = self.CODE_PROMPT.format(
            title=scene.title,
            description=scene.visual_description,
            pose=scene.pose,
            duration=scene.duration,
            frames=int(scene.duration * 30),
            index=scene.index,
            anchor_guide=self.ANCHOR_GUIDE
        )

        response = self._call_deepseek(prompt)
        code = self._extract_code(response)

        # Try to render
        success, result = self._render_manim(code, class_name, output_name)

        if success:
            print(f"    Rendered successfully: {result}")
            return ManimCode(
                scene_index=scene.index,
                code=code,
                class_name=class_name,
                output_path=result
            )

        # Apply ScopeRefine debugging
        print(f"    Initial render failed, applying ScopeRefine...")
        success, final_code, output_path = self._scope_refine(
            code, result, class_name, output_name
        )

        if success:
            print(f"    Fixed and rendered: {output_path}")
            return ManimCode(
                scene_index=scene.index,
                code=final_code,
                class_name=class_name,
                output_path=output_path
            )

        # All attempts failed
        print(f"    All ScopeRefine attempts failed")
        return ManimCode(
            scene_index=scene.index,
            code=final_code if 'final_code' in dir() else code,
            class_name=class_name,
            error=result
        )

    def generate_storyboard(self, storyboard: Storyboard) -> List[ManimCode]:
        """Generate Manim code for all scenes in storyboard."""
        results = []

        for scene in storyboard.scenes:
            manim_code = self.generate_scene(scene)
            results.append(manim_code)

            # Update scene with output path
            if manim_code.output_path:
                scene.image_path = manim_code.output_path

        return results


class HybridCoder:
    """
    Hybrid approach: Manim for text/diagrams + Flux for backgrounds.

    Uses the best of both worlds:
    - Manim: Precise text positioning, animations, diagrams
    - Flux: Complex backgrounds, characters, artistic elements
    """

    def __init__(
        self,
        deepseek_key: str,
        replicate_key: str,
        python_path: str = "python3",
        output_dir: str = "./output"
    ):
        self.manim_coder = ManimCoder(deepseek_key, python_path, output_dir)
        self.replicate_key = replicate_key
        self.output_dir = Path(output_dir)

    def _generate_flux_background(self, description: str, output_name: str) -> str:
        """Generate background image with Flux."""
        import replicate

        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={
                "prompt": f"Simple clean background for educational video: {description}. Minimal, professional, no text.",
                "aspect_ratio": "16:9",
                "num_outputs": 1
            }
        )

        # Download the image
        import urllib.request
        output_path = self.output_dir / f"{output_name}_bg.png"
        urllib.request.urlretrieve(output[0], output_path)

        return str(output_path)

    def generate_hybrid_scene(self, scene: Scene) -> Dict:
        """
        Generate scene with hybrid approach.

        Returns dict with background_path and overlay_path.
        """
        # Generate Flux background (async-friendly)
        bg_path = self._generate_flux_background(
            scene.visual_description,
            f"scene_{scene.index + 1}"
        )

        # Generate Manim overlay (text, diagrams)
        manim_code = self.manim_coder.generate_scene(scene)

        return {
            "background": bg_path,
            "overlay": manim_code.output_path,
            "code": manim_code.code,
            "error": manim_code.error
        }


if __name__ == "__main__":
    # Test the Manim coder
    import os

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Set DEEPSEEK_API_KEY environment variable")
        exit(1)

    from planner import Scene, SceneType, Storyboard

    # Create test scene
    scene = Scene(
        index=0,
        title="Test Animation",
        script="This is a test of Manim code generation",
        visual_description="A title card with the text 'Hello World' that fades in",
        scene_type=SceneType.HOOK,
        pose="None",
        duration=3.0
    )

    coder = ManimCoder(
        api_key=api_key,
        python_path="/mnt/e/projects/pod/venv/bin/python3",
        output_dir="/tmp/manim_test"
    )

    result = coder.generate_scene(scene)

    print(f"\nGenerated code:\n{result.code}")
    print(f"\nOutput: {result.output_path}")
    print(f"Error: {result.error}")
