"""manim_evalator.py: Evaluate Manim code snippets.

This module provides a class to evaluate Manim code snippets by executing them in a subprocess.
It captures the output and error messages, allowing for easy debugging and validation of the code.
"""

__author__      = "Ravidu Silva"

from pathlib import Path
from src.utils.manim_renderer import ManimRenderer, RenderQuality, RenderResult


class ManimEvaluator:
    """Class to evaluate Manim code snippets."""

    def __init__(
            self, 
            evaluator_temp_output: Path = Path("tmp/temp_eval_vid_output"), 
            render_quality:RenderQuality = RenderQuality.P_480,
            timeout: int = None
        ):
        """
        Initialize the ManimEvaluator with the path to the Manim executable and render quality.

        Args:
            render_quality (RenderQuality): Render quality setting. Default is RenderQuality.P_480.
        """
        self.evaluator_temp_output = evaluator_temp_output
        self.evaluator_temp_output.mkdir(parents=True, exist_ok=True)
        self.manim_renderer = ManimRenderer(
            output_dir=evaluator_temp_output,
            render_quality=render_quality,
            timeout=timeout
        )

    def __del__(self):
        """Clean up temporary files and directories."""
        if self.evaluator_temp_output.exists():
            for item in self.evaluator_temp_output.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    item.rmdir()
            self.evaluator_temp_output.rmdir()

    def evaluate_code(self, manim_code: str, clear_output: bool = True) -> RenderResult:
        """Run Manim code, returning success status, info logs, and error logs."""
        render_result = self.manim_renderer.render_code(
            manim_code=manim_code,
            video_name="test_video",
            preview=False
        )

        if render_result.video_path is not None:
            # Delete the temporary video file after evaluation
            if clear_output and render_result.video_path.exists():
                render_result.video_path.unlink()

        return render_result

