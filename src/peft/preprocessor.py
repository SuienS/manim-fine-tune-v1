"""preprocessor.py: Preprocessor for PEFT models."""

import pandas as pd
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset, DatasetDict


from src.inference.inference_engine import InferenceEngine
from src.utils.prompt_template import PromptTemplate
from src.utils.constants import Constants

# # Constants for special tokens to be added when more data is available
# TOK_START_INST = "<|INST|>"
# TOK_END_INST = "<|ENDINST|>"
# TOK_START_RESPONSE = "<|START_RESPONSE|>"
# TOK_END_RESPONSE = "<|END_RESPONSE|>"

class Preprocessor:

    """Preprocessor for PEFT models."""

    def __init__(self, inference_engine: InferenceEngine, is_sft: bool = True):
        """
        Initialize the preprocessor with the specified tokenizer.

        Args:
            tokenizer (Tokenizer): The tokenizer to use for preprocessing.
            is_sft (bool): Whether to use SFT (Supervised Fine-Tuning) or not (Pretraining type Fine-Tuning).
                Default is True.
        """
        self.tokenizer = inference_engine.tokenizer
        self.is_sft = is_sft

        self.data_collator = None
        if is_sft:
            self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                return_tensors="pt"
            )
            # ToDO: Add special tokens to the tokenizer when more data is available
            # self.tokenizer.add_special_tokens({
            #     "additional_special_tokens": [
            #         TOK_START_INST,
            #         TOK_END_INST,
            #         TOK_START_RESPONSE,
            #         TOK_END_RESPONSE
            #     ]
            # })

            # # Resize the tokenizer to accommodate the new special tokens
            # inference_engine.base_model.resize_token_embeddings(len(self.tokenizer))
        else:
            raise ValueError("Preprocessor currently only supports SFT (Supervised Fine-Tuning) mode.")
        print(f"Preprocessor initialized with SFT mode: {self.is_sft}")
        print(f"Data collator initialized...")

    def preprocess_sample_for_sft(self, example: dict, x_name: str, y_name: str, max_prompt_length=2048) -> dict:
        """
        Preprocess the example for SFT (Supervised Fine-Tuning).
        Args:
            example (dict): The example to preprocess.
            x_name (str): The name of the input field in the example.
            y_name (str): The name of the output field in the example.
            max_prompt_length (int): The maximum length of the prompt. Default is 2028.
        Returns:
            dict: The preprocessed example.
        """
        sft_prompt = (
            f"<TEXT_SCRIPT>\n"
            + example[x_name].strip()
            + f"\n</TEXT_SCRIPT>\n"
        )
        sft_full_prompt = sft_prompt + f"<CODE>\n" + example[y_name].strip() + f"\n</CODE>"

        toks = self.tokenizer(
            sft_full_prompt, 
            truncation=True, 
            max_length=max_prompt_length,
            padding="max_length",
            return_attention_mask=True
        )
        input_ids = toks["input_ids"]
        attention_mask = toks["attention_mask"]
        # Compute where the response starts and mask the input tokens
        # to -100 so that they are ignored in the loss calculation
        prompt_len = len(self.tokenizer(sft_prompt, add_special_tokens=False)["input_ids"])
        labels = [
            -100 if i < prompt_len else token_id
            for i, token_id in enumerate(input_ids)
        ]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    
    
    def preprocess_dataset(
            self, 
            pandas_dataset_df: pd.DataFrame, 
            x_name: str, y_name: str, 
            max_prompt_length=2048
        ) -> Dataset:
        """
        Preprocess the dataset for SFT (Supervised Fine-Tuning).
        
        Args:
            pandas_dataset_df (pd.DataFrame): The dataset to preprocess.
            max_prompt_length (int): The maximum length of the prompt. Default is 2028.
            test_size (float): The proportion of the dataset to include in the test split. Default is 0.1.
        
        Returns:
            Dataset: The preprocessed dataset.
        """
        # Convert the DataFrame to a Dataset
        hf_dataset = Dataset.from_pandas(pandas_dataset_df)

        print("Preprocessing dataset for SFT...")
        # Map the preprocessing function to the dataset
        hf_dataset_preprocessed = hf_dataset.map(
            lambda x: self.preprocess_sample_for_sft(x, x_name, y_name, max_prompt_length),
            batched=False,
            remove_columns=hf_dataset.column_names
        )
        print("Dataset preprocessing completed.")
        
        return hf_dataset_preprocessed
    
    def get_data_collator(self):
        """
        Get the data collator for the preprocessor.
        
        Returns:
            DataCollatorForLanguageModeling: The data collator.
        """
        return self.data_collator

