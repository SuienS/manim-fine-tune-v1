"""constants.py: Contains the defined constants for the LLM."""

import torch
class Constants:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_SEED = 1230