"""config.py: Contains the configuration settings."""

class Config:
    CACHE_PATH = "cache"
    EVAL_TEMP_DIR = "tmp/eval"

class SupportedModels:
    """Supported models for the LLM."""
    QWEN3_8B_BASE = "Qwen/Qwen3-8B-Base"
    QWEN3_8B_UNSLOTH_4bit = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
    QWEN3_4B_BASE = "Qwen/Qwen3-4B-Base"
    LLAMA_3_2_3B = "meta-llama/Llama-3.2-3B-Instruct"
    CODE_EVALUATOR_MODEL = "microsoft/codebert-base"