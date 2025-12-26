"""
HTML/SVG Coder Agent - Generates deterministic animations using web tech.

Alternative to Manim that works without system dependencies:
- LLM generates HTML/CSS/SVG animation code
- Puppeteer renders to images/video
- No cairo/pycairo issues on WSL

Benefits:
- Uses familiar web technologies
- Deterministic output
- Precise positioning (CSS grid/flexbox)
- Easy to debug in browser
- Works everywhere Node.js works
"""

import os
import re
import json
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path
import requests


@dataclass
class HTMLScene:
    """Generated HTML animation for a scene."""
    scene_index: int
    html: str
    output_path: Optional[str] = None
    error: Optional[str] = None


class HTMLCoder:
    """
    Generates HTML/SVG animations from scene descriptions.

    Uses DeepSeek to write HTML/CSS/SVG code, then renders
    with Puppeteer for deterministic output.
    """

    CODE_PROMPT = '''You are a web animation expert. Generate HTML/CSS/SVG for an educational video frame.

SCENE REQUIREMENTS:
- Title: {title}
- Description: {description}
- Pose/Action: {pose}

OUTPUT: Create a single HTML file with embedded CSS and SVG that renders a 1920x1080 frame.

STYLE REQUIREMENTS:
- Clean, minimal educational style
- Use SVG for shapes and stick figures
- Clear text with good contrast (white on dark or dark on light)
- Center the main content
- Use modern CSS (flexbox, grid)

POSITIONING GRID (use for layout):
- Use CSS Grid with 6 columns x 6 rows
- Top-left: grid-column: 1; grid-row: 1
- Center: grid-column: 3/5; grid-row: 3/5
- Bottom-right: grid-column: 6; grid-row: 6

STICK FIGURE SVG TEMPLATE:
```svg
<svg viewBox="0 0 100 200">
  <circle cx="50" cy="30" r="20" fill="black"/> <!-- head -->
  <line x1="50" y1="50" x2="50" y2="120" stroke="black" stroke-width="3"/> <!-- body -->
  <line x1="50" y1="70" x2="20" y2="100" stroke="black" stroke-width="3"/> <!-- left arm -->
  <line x1="50" y1="70" x2="80" y2="100" stroke="black" stroke-width="3"/> <!-- right arm -->
  <line x1="50" y1="120" x2="30" y2="180" stroke="black" stroke-width="3"/> <!-- left leg -->
  <line x1="50" y1="120" x2="70" y2="180" stroke="black" stroke-width="3"/> <!-- right leg -->
</svg>
```

OUTPUT FORMAT:
```html
<!DOCTYPE html>
<html>
<head>
  <style>
    body {{ margin: 0; width: 1920px; height: 1080px; }}
    /* Your CSS */
  </style>
</head>
<body>
  <!-- Your content -->
</body>
</html>
```

Generate ONLY the HTML code, no explanations.'''

    FIX_PROMPT = '''The following HTML has an issue. Fix it.

ORIGINAL HTML:
```html
{html}
```

ERROR/ISSUE:
{error}

Fix the error and return the complete corrected HTML.
Output ONLY the fixed HTML code, no explanations.'''

    def __init__(
        self,
        api_key: str,
        output_dir: str = "./output",
        node_path: str = "node"
    ):
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.node_path = node_path

        # Create the Puppeteer render script
        self._setup_renderer()

    def _setup_renderer(self):
        """Create Node.js Puppeteer script for rendering."""
        renderer_script = '''
const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

async function render(htmlPath, outputPath, width = 1920, height = 1080) {
    const browser = await puppeteer.launch({
        headless: 'new',
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });

    try {
        const page = await browser.newPage();
        await page.setViewport({ width, height });

        const html = fs.readFileSync(htmlPath, 'utf8');
        await page.setContent(html, { waitUntil: 'networkidle0' });

        // Wait for any CSS animations to settle
        await page.waitForTimeout(100);

        await page.screenshot({ path: outputPath, type: 'png' });
        console.log(JSON.stringify({ success: true, path: outputPath }));
    } catch (error) {
        console.log(JSON.stringify({ success: false, error: error.message }));
    } finally {
        await browser.close();
    }
}

const [htmlPath, outputPath] = process.argv.slice(2);
if (htmlPath && outputPath) {
    render(htmlPath, outputPath);
}
'''
        renderer_path = self.output_dir / "render.js"
        renderer_path.write_text(renderer_script)
        self.renderer_path = str(renderer_path)

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
                "max_tokens": 4000
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _extract_html(self, response: str) -> str:
        """Extract HTML from LLM response."""
        # Try to find HTML block
        match = re.search(r"```html\s*(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try without language specifier
        match = re.search(r"```\s*(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Return as-is if starts with <!DOCTYPE or <html
        if response.strip().startswith("<!DOCTYPE") or response.strip().startswith("<html"):
            return response.strip()

        return response.strip()

    def _render_html(self, html: str, output_name: str) -> Tuple[bool, str]:
        """
        Render HTML to PNG using Puppeteer.

        Returns:
            (success, error_or_path)
        """
        # Write HTML to temp file
        html_path = self.output_dir / f"{output_name}.html"
        output_path = self.output_dir / f"{output_name}.png"

        html_path.write_text(html)

        try:
            # Check if puppeteer renderer exists
            motion_dir = Path("/mnt/e/projects/OYKH/motion-renderer")
            if (motion_dir / "node_modules" / "puppeteer").exists():
                render_script = motion_dir / "render.js"

                # Create render script if not exists
                if not render_script.exists():
                    render_script.write_text('''
const puppeteer = require('puppeteer');
const fs = require('fs');

async function render(htmlPath, outputPath) {
    const browser = await puppeteer.launch({
        headless: 'new',
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    const page = await browser.newPage();
    await page.setViewport({ width: 1920, height: 1080 });
    const html = fs.readFileSync(htmlPath, 'utf8');
    await page.setContent(html, { waitUntil: 'networkidle0' });
    await page.screenshot({ path: outputPath, type: 'png' });
    await browser.close();
    console.log(JSON.stringify({ success: true, path: outputPath }));
}
render(process.argv[2], process.argv[3]);
''')

                result = subprocess.run(
                    [self.node_path, str(render_script), str(html_path), str(output_path)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(motion_dir)
                )

                if result.returncode == 0 and output_path.exists():
                    return True, str(output_path)
                else:
                    return False, result.stderr or result.stdout or "Render failed"

            else:
                # Fallback: just save the HTML
                return True, str(html_path)

        except subprocess.TimeoutExpired:
            return False, "Render timeout (30s)"
        except Exception as e:
            return False, str(e)

    def generate_scene(self, scene) -> HTMLScene:
        """
        Generate HTML for a single scene.
        """
        output_name = f"scene_{scene.index + 1}"

        print(f"  Generating HTML for scene {scene.index + 1}...")

        # Generate HTML
        prompt = self.CODE_PROMPT.format(
            title=scene.title,
            description=scene.visual_description,
            pose=scene.pose
        )

        response = self._call_deepseek(prompt)
        html = self._extract_html(response)

        # Try to render
        success, result = self._render_html(html, output_name)

        if success:
            print(f"    Generated: {result}")
            # Update scene
            if result.endswith('.png'):
                scene.image_path = result
            return HTMLScene(
                scene_index=scene.index,
                html=html,
                output_path=result
            )

        # Try to fix
        print(f"    Render failed, attempting fix...")
        fix_prompt = self.FIX_PROMPT.format(html=html, error=result)
        fixed = self._extract_html(self._call_deepseek(fix_prompt, temperature=0.2))

        success, result = self._render_html(fixed, output_name)

        if success:
            print(f"    Fixed and rendered: {result}")
            if result.endswith('.png'):
                scene.image_path = result
            return HTMLScene(
                scene_index=scene.index,
                html=fixed,
                output_path=result
            )

        return HTMLScene(
            scene_index=scene.index,
            html=fixed if 'fixed' in dir() else html,
            error=result
        )

    def generate_storyboard(self, storyboard) -> List[HTMLScene]:
        """Generate HTML for all scenes in storyboard."""
        results = []

        for scene in storyboard.scenes:
            html_scene = self.generate_scene(scene)
            results.append(html_scene)

        return results


if __name__ == "__main__":
    import os

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Set DEEPSEEK_API_KEY environment variable")
        exit(1)

    # Create simple test
    class MockScene:
        def __init__(self):
            self.index = 0
            self.title = "Hello World"
            self.visual_description = "A stick figure waving hello with a speech bubble"
            self.pose = "waving"
            self.image_path = None

    scene = MockScene()

    coder = HTMLCoder(
        api_key=api_key,
        output_dir="/tmp/html_test"
    )

    result = coder.generate_scene(scene)

    print(f"\nGenerated HTML:\n{result.html[:500]}...")
    print(f"\nOutput: {result.output_path}")
    print(f"Error: {result.error}")
