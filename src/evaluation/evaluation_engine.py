"""evaluation_engine.py
Evaluation engine for the Manim LLM project.
"""
from tqdm import tqdm

from src.inference.inference_engine import InferenceEngine
from src.evaluation.code_evaluator import CodeEvaluator
from src.evaluation.manim_evaluator import ManimEvaluator
from src.utils import utils

class EvaluationEngine:
    """Evaluation engine for the Manim LLM project."""


    def __init__(self, inference_engine: InferenceEngine, code_evaluator: CodeEvaluator, manim_evaluator: ManimEvaluator):
        """
        Initialize the evaluation engine with the specified inference engine.

        Args:
            inference_engine (InferenceEngine): The inference engine to use for evaluation.
            code_evaluator (CodeEvaluator): The code evaluator to use for evaluation.
            manim_evaluator (ManimEvaluator): The Manim evaluator to use for evaluation.
        """

        self.inference_engine = inference_engine
        self.code_evaluator = code_evaluator
        self.manim_evaluator = manim_evaluator

        # Set the model to evaluation mode
        self.inference_engine.set_eval_mode()

        print(f"Evaluation engine initialized with model: {self.inference_engine.hf_model_name}")


    def evaluate_sample(self, textual_script: str, expected_code: str) -> dict:
        """
        Evaluate a single sample of textual script against expected code.

        Args:
            textual_script (str): The textual script to evaluate.
            expected_code (str): The expected ManimCE code.

        Returns:
            dict: A dictionary containing the evaluation results.
        """

        # Generate ManimCE code from the textual script
        generated_code = self.inference_engine.generate_manim_code(textual_script).generated_code

        # Extract the generated ManimCE code from the response
        generated_code = utils.extract_manim_code_from_llm_response(generated_code)

        # Evaluate the generated code against the expected code
        evaluation_results = self.code_evaluator.evaluate_code(generated_code, expected_code)

        # Evaluate the generated code using Manim
        manim_result = self.manim_evaluator.evaluate_code(generated_code)

        # Add Manim evaluation result to the evaluation results
        evaluation_results["manim_render_success"] = manim_result.success
        evaluation_results["manim_render_info"] = manim_result.info
        evaluation_results["manim_render_error"] = manim_result.errors

        evaluation_results["text_script"] = textual_script
        evaluation_results["expected_code"] = expected_code
        evaluation_results["generated_code"] = generated_code

        return evaluation_results

    
    def evaluate_batch(self, textual_scripts: list, expected_codes: list) -> list:
        """
        Evaluate a batch of textual scripts against expected codes.

        Args:
            textual_scripts (list): A list of textual scripts to evaluate.
            expected_codes (list): A list of expected ManimCE codes.

        Returns:
            list: A list of dictionaries containing the evaluation results for each sample.
        """

        # ToDo: Parallelize this function - Use num_poc=4 in map
        results = list(map(self.evaluate_sample, textual_scripts, expected_codes))

        return results
    
    def run_evaluation(self, batch_size: int, textual_scripts: list, expected_codes: list) -> list:
        """
        Run the evaluation process for a batch of textual scripts and expected codes.

        Args:
            batch_size (int): The size of the batch to evaluate.
            textual_scripts (list): A list of textual scripts to evaluate.
            expected_codes (list): A list of expected ManimCE codes.

        Returns:
            list: A list of dictionaries containing the evaluation results for each sample.
        """

        # Split the textual scripts and expected codes into batches
        textual_scripts_batches = [textual_scripts[i:i + batch_size] for i in range(0, len(textual_scripts), batch_size)]
        expected_code_batches = [expected_codes[i:i + batch_size] for i in range(0, len(expected_codes), batch_size)]

        results = []
        # Evaluate each batch with tqdm progress bar
        for batch_textual_scripts, batch_expected_codes in tqdm(zip(textual_scripts_batches, expected_code_batches), total=len(textual_scripts_batches), desc="Evaluating batches"):
            batch_results = self.evaluate_batch(batch_textual_scripts, batch_expected_codes)
            results.extend(batch_results)

        return results
