"""peft_engine.py
PEFT engine for the Manim LLM project.
"""
from peft import LoraConfig, TaskType, get_peft_model
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

from src.inference.inference_engine import InferenceEngine


class PEFTEngine:
    """PEFT engine for the Manim LLM project."""

    def __init__(
            self,
            inference_engine: InferenceEngine,
            peft_config: LoraConfig,
            training_args: TrainingArguments
    ):
        """
        Initialize the PEFT engine with the specified model and PEFT configuration.

        Args:
            inference_engine (InferenceEngine): The inference engine to use.
            peft_config (LoraConfig): The PEFT configuration to use.
            training_args (TrainingArguments): The training arguments to use.
            training_dataset (Dataset, optional): The training dataset to use.
            eval_dataset (Dataset, optional): The evaluation dataset to use.
            compute_metrics (function, optional): The function to compute metrics.
        """

        # Initialize the PEFT engine with the specified model and PEFT configuration
        self.trainer_initialized = False
        self.training_dataset = None
        self.eval_dataset = None
        self.inference_engine = inference_engine
        self.peft_config = peft_config
        self.training_args = training_args

        # Create the PEFT model
        self.peft_model = get_peft_model(
            self.inference_engine.base_model,
            peft_config
        )
        print(f"PEFT model loaded...")
        self.peft_model.print_trainable_parameters()


    
    def initialize_trainer(self, training_dataset, eval_dataset, data_collator, compute_metrics=None):
        """
        Initialize the trainer with the specified datasets.

        Args:
            training_dataset (Dataset): The training dataset to use.
            eval_dataset (Dataset): The evaluation dataset to use.
        """
        self.training_dataset = training_dataset
        self.eval_dataset = eval_dataset

        # Create the trainer
        self.trainer = Trainer(
            model=self.peft_model,
            args=self.training_args,
            data_collator=data_collator,
            train_dataset=self.training_dataset,
            eval_dataset=self.eval_dataset
        )
        if compute_metrics is not None:
            self.trainer.compute_metrics = compute_metrics

        self.trainer_initialized = True
        print(f"Trainer initialized...")


    def train(self, save_peft_adapters=False):
        """
        Train the model using the specified training dataset.
        
        Args:
            save_peft_model (bool): Whether to save the PEFT model.

        Returns:
            model (PreTrainedModel): The trained PEFT model.
        """
        if not self.trainer_initialized:
            raise ValueError("Trainer not initialized. Please initialize the trainer before training.")
        
        print(f"Training started...")
        self.trainer.train()
        print(f"Training completed.")

        if save_peft_adapters:
            self.peft_model.save_pretrained(self.training_args.output_dir+"/peft_adapters")
            print(f"PEFT adapters saved to {self.training_args.output_dir}/peft_adapters")

        self.inference_engine.set_peft_model(self.peft_model)
        
        return self.peft_model