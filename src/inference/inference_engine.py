"""inference_engine.py
Inference engine for the Manim LLM project.
"""
import typing

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.inference.model_config import ModelConfig
from src.utils.prompt_template import PromptTemplate
from src.utils.constants import Constants


class ModelResponse:
    """Model response class to handle the output from the model."""

    def __init__(self, generated_code: str, thinking_content: str):
        """
        Initialize the model response with generated code and thinking content.

        Args:
            generated_code (str): The generated ManimCE code.
            thinking_content (str): The internal thought process of the model.
        """
        self.generated_code = generated_code
        self.thinking_content = thinking_content


class InferenceEngine:
    """Inference engine for the Manim LLM project."""

    def __init__(
            self, 
            hf_model_name: str, 
            load_in_4bit: bool = False, 
            device_map: str = "auto", 
            max_new_tokens: int = ModelConfig.MAX_NEW_TOKENS,
            remvove_token_type_ids: bool = False,
            #ToDo: backend: typing.Literal['huggingface', 'unsloth'] = "huggingface"
        ):
        """
        Initialize the inference engine with the specified model.

        Args:
            hf_model_name (str): The name of the Hugging Face model to use.
        """
        # load the tokenizer and the model
        print(f"Loading model: '{hf_model_name}'" + (" with 4bit..." if load_in_4bit else "..."))
        self.hf_model_name = hf_model_name
        self.load_in_4bit = load_in_4bit
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        # Setting the pad token to eos token since the model is causal
        # and the model does not have a pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base_model = None
        if load_in_4bit:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                hf_model_name,
                torch_dtype="auto",
                device_map=device_map,
                trust_remote_code=True,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            )
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                hf_model_name,
                torch_dtype="auto",
                trust_remote_code=True,
                device_map=device_map
            )

        # PEFT model
        self.peft_model = None
        self.model_in_use = self.base_model
        print(f"Model in use: {self.get_inference_model_type()}")
        self.max_new_tokens = max_new_tokens

        self.remvove_token_type_ids = remvove_token_type_ids


    def generate_manim_code(self, textual_script: str) -> ModelResponse:
        """
        Generate ManimCE code based on the textual script.
        
        Args:
            textual_script (str): The textual script describing the Manim animation.
        
        Returns:
            ModelResponse: An object containing the generated ManimCE code and the model's internal thought process.
        """

        # Tokenize the prompt
        model_inputs = self.tokenizer(
        [
            PromptTemplate.MANIM_VID_GEN_PROMPT.format(text_script=textual_script, response="")+self.tokenizer.eos_token,
        ], return_tensors = "pt").to(Constants.DEVICE)

        if self.remvove_token_type_ids:
            del model_inputs['token_type_ids']

        generated_ids = self.model_in_use.generate(
            **model_inputs, 
            max_new_tokens = self.max_new_tokens,
            pad_token_id = self.tokenizer.pad_token_id,
            use_cache = True
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content =  self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        generated_code =  self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        model_response = ModelResponse(generated_code=generated_code, thinking_content=thinking_content)

        return model_response
    

    def set_peft_model(self, peft_model: PeftModel):
        """
        Set the PEFT model for the inference engine.

        Args:
            peft_model (PeftModel): The PEFT model to use.
        """

        if not isinstance(peft_model, PeftModel):
            raise ValueError("The provided model is not a PEFT model. Please provide a valid PEFT model.")
        if self.peft_model is not None:
            print(f"Warning: Overwriting existing PEFT model: {self.peft_model}.")

        # Set the PEFT model
        self.peft_model = peft_model
        self.select_model(use_peft_model=True)
        print(f"`peft_model` of the inference engine set to {self.peft_model}.")


    def select_model(self, use_peft_model: bool = False):
        """
        Select the model to use for inference.

        Args:
            use_peft_model (bool): Whether to use the PEFT model or the base model.
        """
        if use_peft_model:
            if self.peft_model is None:
                raise ValueError("PEFT model is not set. Please set the PEFT model before selecting it.")
            self.model_in_use = self.peft_model
        else:
            self.model_in_use = self.base_model

        print(f"Model in use: {self.get_inference_model_type()}")

    def set_eval_mode(self):
        """
        Set the model to evaluation mode.
        """
        self.base_model.eval()
        if self.peft_model is not None:
            self.peft_model.eval()
        else:
            print("Warning: PEFT model is not set. Skipping evaluation mode setting for PEFT model.")


        print(f"Model set to evaluation mode.")


    def set_train_mode(self):
        """
        Set the model to training mode.
        """
        self.base_model.train()
        if self.peft_model is not None:
            self.peft_model.train()
        else:
            print("Warning: PEFT model is not set. Skipping training mode setting for PEFT model.")
        print(f"Model set to training mode.")


    def get_inference_model_type(self) -> typing.Literal["BASE", "PEFT"]:
        """
        Get the type of inference model in use.

        Returns:
            str: The type of inference model in use.
                - "BASE" for the base model
                - "PEFT" for the PEFT model
        """
        return "PEFT" if isinstance(self.model_in_use, PeftModel) else "BASE"
