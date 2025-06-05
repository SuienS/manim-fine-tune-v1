#!/usr/bin/env python

"""evaluate.py: Script to evaluate code generation using a Manim-based pipeline.

Example usage:
- python evaluate.py --hf-model-name "hf_model_name" --load-in-4bit --run-test-sample --run-eval
- python evaluate.py --hf-model-name "hf_model_name" --use-unsloth --load-in-4bit --peft-model-path "path/to/peft/model" --run-eval
- python evaluate.py --hf-model-name "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit" --use-unsloth --load-in-4bit --peft-model-path "./data/manim_sft_Qwen2.5_Coder_3B_Instruct_bnb_4bit_finetuned_lora_20250528_205436" --cache-path "/your/cache/path" --run-eval


"""

__author__ = "Ravidu Silva"

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv
sys.path.append(str(Path(__file__).resolve().parent))

TEST_TEXT_SCRIPT = """Set the background color to a light beige. Display a large black double-struck "M" shifted to the left and up. Show a green circle shifted to the left, a blue square shifted up, and a red triangle shifted to the right. Group these shapes and the "M" together, centering the group on the screen."""
TEST_EXPECTED_CODE = """from manim import *

class ManimCELogo(Scene):
    def construct(self):
        self.camera.background_color = "#ece6e2"
        logo_green = "#87c2a5"
        logo_blue = "#525893"
        logo_red = "#e07a5f"
        logo_black = "#343434"
        ds_m = MathTex(r"\\mathbb\\{M\\}", fill_color=logo_black).scale(7)
        ds_m.shift(2.25 * LEFT + 1.5 * UP)
        circle = Circle(color=logo_green, fill_opacity=1).shift(LEFT)
        square = Square(color=logo_blue, fill_opacity=1).shift(UP)
        triangle = Triangle(color=logo_red, fill_opacity=1).shift(RIGHT)
        logo = VGroup(triangle, square, circle, ds_m)
        logo.move_to(ORIGIN)
        self.add(logo)
"""

def main():
    parser = argparse.ArgumentParser(description="Evaluate code generation with a Manim-based pipeline.")
    parser.add_argument("--hf-model-name", help="Name of the Hugging Face model to use for inference.")
    parser.add_argument("--peft-model-path", default=None, help="Path to the Unsloth PEFT model for inference. Only supports Unsloth models at the moment.")
    parser.add_argument("--use-unsloth", action="store_true", help="Use Unsloth PEFT model for inference. Use ONLY with Unsloth models.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use 4-bit quantization.")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Max new tokens for generation.")
    parser.add_argument("--device-map", default="auto", help="Device map for model loading.")
    parser.add_argument("--cache-path", default=None, help="Cache path (sets environment variables).")
    parser.add_argument("--test-file", default="./data/manim_sft_dataset.parquet", help="Path to test dataset.")
    parser.add_argument("--output-dir", default="./output/eval_results", help="Output directory for results.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluation. Use 1.")
    parser.add_argument("--run-test-sample", action="store_true", help="Run a sample test case.")
    parser.add_argument("--run-eval", action="store_true", help="Run evaluation on the test set.")
    args = parser.parse_args()


    load_dotenv()
    if args.cache_path is not None:
        os.environ["CACHE_PATH"] = args.cache_path
    if os.getenv('CACHE_PATH') is not None:
        print(f"Setting cache path to: {os.getenv('CACHE_PATH')}")
        os.environ["HF_HOME"] = os.getenv('CACHE_PATH') + '/transformers'
        os.environ["HF_DATASETS_CACHE"] = os.getenv('CACHE_PATH') + '/datasets'
        os.environ["TORCH_HOME"] = os.getenv('CACHE_PATH') + '/torch'
        os.environ["TFHUB_CACHE_DIR"] = os.getenv('CACHE_PATH') + '/tfhub'

    # Model configs
    selected_model_name = args.hf_model_name
    selected_model_name_str = selected_model_name.replace('/', '_').replace('-', '_')

    if args.use_unsloth:
        from unsloth import FastLanguageModel
    else:
        from peft import PeftModel

    import config
    from src.inference.inference_engine import InferenceEngine
    from src.evaluation.code_evaluator import CodeEvaluator
    from src.evaluation.manim_evaluator import ManimEvaluator
    from src.evaluation.evaluation_engine import EvaluationEngine


    inference_engine = InferenceEngine(
        selected_model_name,
        load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_new_tokens,
        device_map=args.device_map
    )

    if args.peft_model_path is not None:
        if args.use_unsloth:
            print(f"Loading Unsloth PEFT model from: {args.peft_model_path}")
            saved_peft_model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = args.peft_model_path,
                max_seq_length = args.max_new_tokens,
                dtype = None,
                load_in_4bit = args.load_in_4bit,
            )
            FastLanguageModel.for_inference(saved_peft_model)

            inference_engine.tokenizer = tokenizer
        else:
            print(f"Loading PEFT model from: {args.peft_model_path}")
            saved_peft_model = PeftModel.from_pretrained(
                inference_engine.base_model,
                args.peft_model_path,
            )
        inference_engine.set_peft_model(saved_peft_model)
    
    code_evaluator = CodeEvaluator(config.SupportedModels.CODE_EVALUATOR_MODEL)
    manim_evaluator = ManimEvaluator(Path(config.Config.EVAL_TEMP_DIR), timeout=240)
    evaluation_engine = EvaluationEngine(
        inference_engine=inference_engine,
        code_evaluator=code_evaluator,
        manim_evaluator=manim_evaluator,
    )

    if args.run_test_sample:
        print("Running test sample...")
        response = inference_engine.generate_manim_code(TEST_TEXT_SCRIPT)
        print("Generated code:\n", response.generated_code)
        result_sample = evaluation_engine.evaluate_sample(
            textual_script=TEST_TEXT_SCRIPT,
            expected_code=TEST_EXPECTED_CODE
        )
        print("Sample result:\n", result_sample)

    if args.run_eval:
        print("Running evaluation on the test set...")
        if not Path(args.test_file).exists():
            print(f"Test file not found: {args.test_file}")
            return
        print(f"Loading test dataset from: {args.test_file}")
        # Load the test dataset
        test_set = pd.read_parquet(args.test_file)
        test_set = test_set[test_set['Split'] == 'test']
        test_set_x_y = test_set[['Reviewed Description', 'Code']]
        result_batch = evaluation_engine.run_evaluation(
            batch_size=args.batch_size,
            textual_scripts=list(test_set_x_y['Reviewed Description']),
            expected_codes=list(test_set_x_y['Code'])
        )
        df_result_sample = pd.DataFrame(result_batch)
        numeric_columns = [
            'codebert_similarity',
            'codebleu',
            'ngram_match_score',
            'weighted_ngram_match_score',
            'syntax_match_score',
            'dataflow_match_score',
            'ast_distance_norm',
            'ast_distance_raw',
            'ast_distance_max',
            'syntax_error',
            'manim_render_success'
        ]
        print("Mean scores:\n", df_result_sample[numeric_columns].mean())
        print("Std dev:\n", df_result_sample[numeric_columns].std())

        from datetime import datetime
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"eval_result_{selected_model_name_str}_{timestamp}.csv"
        output_path = output_dir / file_name
        df_result_sample.to_csv(output_path)
        print(f"Evaluation results saved to {output_path}")

if __name__ == "__main__":
    main()
