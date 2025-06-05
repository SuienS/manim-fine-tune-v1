"""prompt_template.py: Contains the defined prompt templates for the LLM."""

class PromptTemplate:
    MANIM_VID_GEN_PROMPT = """You are an expert Manim Community Edition (ManimCE) educator and Python developer.

Please follow these steps precisely:
1. Read the `<TEXT_SCRIPT>` block which contains a description of a Manim animation that I want to create.
2. DO NOT think out loud or provide any explanations.
3. Generate only executable Python code for Manim, wrapped between `<CODE>` and `</CODE>`.
4. Do not include any explanations, comments, or instructions on how to run the code. Only include the code.

Example format:

<TEXT_SCRIPT>
Display a red square centered on screen, then transform it into a circle.
</TEXT_SCRIPT>

<CODE>
```python
from manim import *
class RedSquareToCircle(Scene):
    def construct(self):
        square = Square(color=RED)
        self.play(Create(square))
        circle = Circle(color=RED)
        self.play(Transform(square, circle))
        self.wait()
``` 
</CODE>

Now, generate code for the following text script:

<TEXT_SCRIPT>
{text_script}
</TEXT_SCRIPT>

{response}"""