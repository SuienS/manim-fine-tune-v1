"""manim_renderer.py: Manim renderer for generating videos from code snippets.
This module provides a class to render videos using Manim Community Edition (ManimCE).
"""

__author__      = "Ravidu Silva"

import subprocess
import tempfile
import os
import time
import random
from pathlib import Path
from enum import Enum

class RenderQuality(Enum):
    """Enum for Manim render quality settings."""
    P_480 = "-ql"
    P_720 = "-qm"
    P_1080 = "-qh"
    P_1440 = "-qp"
    P_2160 = "-qk"

    def __str__(self):
        return self.value

class RenderResult:
    """Class to represent the result of a Manim render operation."""
    def __init__(self, video_path: Path, success: bool, info: str, errors: str):
        self.video_path = video_path
        self.success = success
        self.info = info
        self.errors = errors

    def __repr__(self):
        return f"RenderResult(video_path={self.video_path}, success={self.success}, info={self.info}, errors={self.errors})"

class ManimRenderer:
    """Class to render videos using Manim Community Edition (ManimCE)."""
    MANIM_COMMAND = "manimce"

    def __init__(self, output_dir: Path, manim_exec: str = MANIM_COMMAND, render_quality: RenderQuality = RenderQuality.P_480, timeout: int = None):
        """
        Initialize the ManimRenderer with the path to the Manim executable and render quality.

        Args:
            output_dir (str): Directory where the rendered video(s) will be saved.
            manim_exec (str): Path to the Manim executable. Default is "manim".
            render_quality (RenderQuality): Render quality setting. Default is RenderQuality.P_480.
            timeout (int): Timeout for the Manim command in seconds. Default is None (no timeout).
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manim_exec = manim_exec
        self.render_quality = render_quality
        self.timeout = timeout

    def render_code(self, manim_code: str, video_name: str, scene_name: str = None, format:str = "mp4", preview: bool = False) -> RenderResult:
        """Run Manim code, returning success status, info logs, and error logs.

        Args:
            manim_code (str): The Manim code to be rendered.
            video_name (str): The name of the video to be generated.
            scene_name (str): The name of the scene to be rendered.
            preview (bool): Whether to preview the video after rendering. Default is False.

        Returns:
            RenderResult: An object containing the video path, success status, info logs, and error logs.
        """

        try:
            # Create a temporary file to store the Manim code
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
                temp_file.write(manim_code)
                temp_file.flush()
                temp_file_name = temp_file.name

            # Video name: <scene_name>_<epoch_time>.mp4
            video_name = f"{video_name}_{random.randrange(100000,999999)}_{int(time.time())}"
            video_path = self.output_dir / video_name

            # Manim command
            manim_cmd = [
                self.manim_exec,
                temp_file_name
            ]

            if scene_name:
                manim_cmd.append(scene_name)

            if preview: manim_cmd.append("-p")

            manim_cmd += [
                str(self.render_quality),
                f"--format={format}",
                "-o",
                str(video_path.absolute()),
            ]

            result = subprocess.run(
                manim_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # Check if the command was successful
            if result.returncode != 0:
                raise RuntimeError(f"Manim command failed with error: {result.stderr}")

            # Check file existence at video_path and get its extension
            ext = "."+format
            if (video_path.with_suffix(ext)).exists():
                video_path = video_path.with_suffix(ext)
            elif (video_path.with_suffix(".png")).exists():
                video_path = video_path.with_suffix(".png")
            else:
                raise FileNotFoundError(f"Video file not found at {str(video_path)}.<extension>")


            return RenderResult(
                video_path=video_path,
                success=(result.returncode == 0),
                info=result.stdout,
                errors=result.stderr
            )

        except subprocess.TimeoutExpired:
            return RenderResult(
                video_path=None,
                success=False,
                info="",
                errors=f"ERROR: Rendering timed out after {self.timeout} seconds."
            )
        except Exception as e:
            return RenderResult(
                video_path=None,
                success=False,
                info="",
                errors=str(e)
            )
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_name):
                os.remove(temp_file_name)
            