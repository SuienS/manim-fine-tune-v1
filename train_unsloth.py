"""train_unsloth.py: A script to fine-tune a Seed-Coder model with Manim SFT data using Unsloth.

This script is designed to fine-tune a language model for generating Manim animations based on textual descriptions.

Example usage:
    - python train_unsloth.py --train_model "unsloth/Seed-Coder-8B-Instruct-unsloth-bnb-4bit" --epochs 1 --max_seq_length 1024 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --train_data_path "/path/to/manim_sft_dataset.parquet" --load_in_4bit --learning_rate 2e-5
    - python train_unsloth.py --train_model "unsloth/Seed-Coder-8B-Instruct-unsloth-bnb-4bit" --epochs 1 --max_seq_length 1024 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --train_data_path "/path/to/manim_sft_dataset.parquet" --load_in_4bit --learning_rate 2e-5 --token "your_hf_token"
"""

import argparse
import pandas as pd
import torch
from datetime import datetime
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from trl import SFTTrainer


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

"""

MANIM_VID_SFT_GEN_RESPONSE = """<CODE>
```python
{response}
```
</CODE>"""


def preprocess_sample_for_sft(example, max_prompt_length, tokenizer, MANIM_VID_GEN_PROMPT, MANIM_VID_SFT_GEN_RESPONSE, eos_token):
    """
    Preprocess the example for SFT (Supervised Fine-Tuning).
    """
    def format_prompt(ex):
        sft_prompt = MANIM_VID_GEN_PROMPT.format(
            text_script=ex['Reviewed Description'].strip()) + eos_token
        sft_full_prompt = sft_prompt + \
            MANIM_VID_SFT_GEN_RESPONSE.format(
                response=ex['Code'].strip()) + eos_token
        return sft_prompt, sft_full_prompt

    sft_prompt, sft_full_prompt = format_prompt(example)
    toks = tokenizer(
        sft_full_prompt,
        truncation=True,
        max_length=max_prompt_length,
        padding="max_length",
        return_attention_mask=True
    )
    input_ids = toks["input_ids"]
    attention_mask = toks["attention_mask"]
    prompt_len = len(
        tokenizer(sft_prompt, add_special_tokens=False)["input_ids"])
    labels = [
        -100 if i < prompt_len else token_id
        for i, token_id in enumerate(input_ids)
    ]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def preprocess_dataset(pandas_dataset_df, max_prompt_length, tokenizer, MANIM_VID_GEN_PROMPT, MANIM_VID_SFT_GEN_RESPONSE, eos_token):
    """
    Preprocess the dataset for SFT (Supervised Fine-Tuning).
    """
    hf_dataset = Dataset.from_pandas(pandas_dataset_df)
    print("Preprocessing dataset for SFT...")

    def _map_func(x):
        return preprocess_sample_for_sft(
            example=x,
            max_prompt_length=max_prompt_length,
            tokenizer=tokenizer,
            MANIM_VID_GEN_PROMPT=MANIM_VID_GEN_PROMPT,
            MANIM_VID_SFT_GEN_RESPONSE=MANIM_VID_SFT_GEN_RESPONSE,
            eos_token=eos_token
        )

    hf_dataset_preprocessed = hf_dataset.map(
        _map_func,
        batched=False,
        remove_columns=hf_dataset.column_names
    )
    print("Dataset preprocessing completed.")
    return hf_dataset_preprocessed


def main():
    parser = argparse.ArgumentParser(
        description="Simple script to fine-tune a Seed-Coder model with Manim SFT data.")
    parser.add_argument(
        "--train_model", default="unsloth/Seed-Coder-8B-Instruct-unsloth-bnb-4bit", help="HF model name.")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs.")
    parser.add_argument("--max_seq_length", type=int,
                        default=1024, help="Max sequence length.")
    parser.add_argument("--per_device_train_batch_size",
                        type=int, default=4, help="Per device train batch size.")
    parser.add_argument("--gradient_accumulation_steps",
                        type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument(
        "--train_data_path", default="/kaggle/input/manim-sft/manim_sft_dataset.parquet", help="Path to dataset parquet.")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Use 4-bit quantization.")
    parser.add_argument("--learning_rate", type=float,
                        default=2e-5, help="Learning rate.")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face token for gated models.")

    args = parser.parse_args()
    train_model = args.train_model
    train_model_name = f"{train_model.split('/')[-1].replace('-', '_')}_finetuned"
    train_epochs = args.epochs
    max_seq_length = args.max_seq_length
    per_device_train_batch_size = args.per_device_train_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    load_in_4bit = args.load_in_4bit
    learning_rate = args.learning_rate
    train_data_path = args.train_data_path

    # Load the base model
    print(f"Loading model: {train_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=train_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        # For gated models, set your HF token if needed
        token=args.token,
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha=16,
        lora_dropout=0,
        use_gradient_checkpointing=True,
        random_state=1230,
        use_rslora=False,
        loftq_config=None,
        bias='none'
    )

    print(f"Reading dataset from: {train_data_path}")
    manim_sft_dataset_df = pd.read_parquet(train_data_path)

    # Split
    manim_sft_train_df = manim_sft_dataset_df[manim_sft_dataset_df['Split'] == 'train'][[
        'Reviewed Description', 'Code']]
    manim_sft_test_df = manim_sft_dataset_df[manim_sft_dataset_df['Split'] == 'test'][[
        'Reviewed Description', 'Code']]

    eos_token = tokenizer.eos_token

    # Preprocess
    manim_sft_train_hf_ds = preprocess_dataset(
        manim_sft_train_df,
        max_seq_length,
        tokenizer,
        MANIM_VID_GEN_PROMPT,
        MANIM_VID_SFT_GEN_RESPONSE,
        eos_token
    )
    manim_sft_test_hf_ds = preprocess_dataset(
        manim_sft_test_df,
        max_seq_length,
        tokenizer,
        MANIM_VID_GEN_PROMPT,
        MANIM_VID_SFT_GEN_RESPONSE,
        eos_token
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt"
    )

    print("Setting up trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=manim_sft_train_hf_ds,
        eval_dataset=manim_sft_test_hf_ds,
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            label_names=["labels"],
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.1,
            num_train_epochs=train_epochs,
            learning_rate=learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=1230,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            output_dir="outputs",
            eval_strategy="steps",
            save_strategy="steps",
            report_to="none",
        ),
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(
        torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    print("Training...")
    trainer_stats = trainer.train()

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    model_save_name = f"manim_sft_{train_model_name}_lora_{timestamp}"
    model.save_pretrained(model_save_name)
    print("Model saved:", model_save_name)

    used_memory = round(torch.cuda.max_memory_reserved() /
                        1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

    print(
        f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(
        f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


if __name__ == "__main__":
    main()
