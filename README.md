# Manim Fine-Tune

[![License: CC-BY-NC-SA-4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-Model-orange.svg)](https://huggingface.co/models)
[![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace-Dataset-blue.svg)](https://huggingface.co/datasets/SuienR/ManimBench-v1)



Fine-tune smaller LLM models to generate high-quality [Manim](https://www.manim.community/) animation code from natural language descriptions.

**📝 Research Paper: [Upcoming](https://arxiv.org/abs/....)**

**🚧 Note:** This project is still in development. Some features may not be fully implemented or tested yet.

## 📋 Overview

This project allows you to generate Manim code from textual descriptions using fine-tuned language models. It is designed to help educators, content creators, and developers create educational animations easily by leveraging the power of relatively smaller language models.
The project includes:
- Fine-tuning scripts for smaller LLMs (e.g., Qwen3 8B)
- A dataset of Manim code and corresponding textual descriptions
- Jupyter notebooks for testing and development
- Example prompts and generated Manim code
##

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/manim-fine-tune.git
cd manim-fine-tune

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Install Manim if not already installed
pip install manim
```

## 🚀 Usage

### Fine-tuning a model
To fine-tune a model, you can run the train_unsloth.py script. This script will load the dataset and fine-tune the specified model using the provided configuration.

```bash
python train_unsloth.py \
    --train_model "unsloth/model-name" \
    --load_in_4bit \
    --epochs 1 \
    --max_seq_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --train_data_path "/path/to/manim_sft_dataset.parquet" \
    --learning_rate 2e-5 \
    --token "your_hf_token"
```
### Evaluating a model
To evaluate the fine-tuned model, you can use the `evaluate.py` script. This script will load the model and dataset, and then generate Manim code based on the textual descriptions.

```bash
python evaluate.py \
    --hf-model-name "hf-model-name" \
    --use-unsloth \
    --load-in-4bit \
    --peft-model-path "/path/to/peft_weights" \
    --cache-path "/your/cache/path" \
    --run-test-sample \
    --run-eval
```
### Generating Manim code - Upcoming
To generate Manim code from a textual description, you can use the `main.py` script. This script will prompt the user for a description and then generate the corresponding Manim code.

```bash
python main.py #To be implemented
```

<!-- ### Using the notebooks - Upcoming -->


## 📚 Dataset: ManimBench v1
The dataset used for fine-tuning consists of Manim code snippets paired with natural language descriptions. It is available in both CSV and Parquet formats in the `data/` directory.

The dataset can be also accessed directly from the [HuggingFace Hub](https://huggingface.co/datasets/SuienR/ManimBench-v1) and [Kaggle](https://www.kaggle.com/datasets/ravidussilva/manim-sft/).

The dataset is structured as follows:

| Column Name | Description |
|-------------|-------------|
| `Generated Description` | LLM-generated Natural language description of the Manim animation |
| `Reviewed Description` | Human-reviewed Natural language description of the Manim animation |
| `Code` | Corresponding Manim code snippet |
| `Type` | Complexity type of the animation: `Basic`, `Intermediate`, `Advanced` |
| `Split` | Split the sample belongs to: `train` or `test` |

## 📂 Project Structure

```
├── config.py             # Configuration settings
├── evaluate.py           # Evaluation scripts
├── train_unsloth.py      # Fine-tuning script for Unsloth models
├── main.py               # Main script
├── requirements.txt      # Python dependencies
├── data/                 # Training and evaluation data and models
│   ├── manim_sft_dataset.csv
│   ├── manim_sft_dataset.parquet
│   └── ...
├── output/               # Output directory for evaluation results
└── src/                  # Source code
```

## 📝 Example

Here's an example of generating a Manim animation from a text description:

```python
# Input description
manim_instructions = """Axes and Labels: The video starts with the creation of a set of axes with the x-range from 0 to 10 and the y-range from 0 to 100. The axes are labeled with "x" for the x-axis and "f(x)" for the y-axis.

Graph of the Function: The function f(x) = 2(x - 5)^2 is plotted on the axes. This will appear as a parabola opening upwards, with its vertex at x = 5.

Dot Initialization: A dot is placed at the initial point (0, f(0)).

Dot Movement: The dot will move along the curve of the function as the value of t changes.

Finding the Minimum: The code calculates the minimum value of the function over the specified x-range.

Final Display: The video will end with the dot positioned at the minimum point of the function, (5, 0).
"""

# Generated Manim code
# from manim import *
# 
# class FunctionAnimation(Scene):
#     def construct(self):
#         # Create axes
#         axes = Axes(
#             x_range=[0, 10, 1],
#             y_range=[0, 100, 10],
#             axis_config={"color": BLUE},
#         )
#         axes_labels = axes.get_axis_labels(x_label="x", y_label="f(x)")
#         self.play(Create(axes), Write(axes_labels))
#
#         # Define the function
#         func = lambda x: 2 * (x - 5) ** 2
#         graph = axes.plot(func, color=GREEN)
#
#         # ... more generated code ...
```

## 🔬 Technical Details

_Await the research paper for detailed technical insights into the fine-tuning process, model architecture, and evaluation metrics._

## 🧪 Test Results
_Await the research paper for detailed test results and performance metrics of the fine-tuned models._
<!--
| Model Name | Val | Val |
|------------|---------------|-----------|
| Qwen3 8B | 0.11 | 0.11 |
| Qwen3 14B | 0.11 | 0.11 |
| Qwen3 32B | 0.11 | 0.11 |
| Qwen3 70B | 0.11 | 0.11 | -->

## 📄 License

This project is licensed under the [CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/). You are free to share and adapt the material for non-commercial purposes, as long as you give appropriate credit, provide a link to the license, and indicate if changes were made. You may not use the material for commercial purposes.

## 🙏 Acknowledgments

- [Manim Community](https://www.manim.community/) for their amazing animation engine
- Grant Sanderson ([3Blue1Brown](https://www.3blue1brown.com/)) for creating the original Manim
- All contributors to the Manim documentation used in our dataset

## 📚 References

- [Manim Community Documentation](https://docs.manim.community/en/stable/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [HuggingFace PEFT](https://huggingface.co/docs/peft/index)

## 👥 Contributors
- [Ravidu Silva](https://github.com/SuienS)
- [Jordan J. Bird](https://github.com/jordan-bird)
