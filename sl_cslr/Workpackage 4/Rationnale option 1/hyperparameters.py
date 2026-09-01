import torch
import pandas as pd

# === Hyperparameters=============

### tbh, we need to decide about that, this is just place holder numbers 

BATCH_SIZE = 4
SEQ_LEN = 120
INPUT_DIM = 768

VOCAB_SIZE =  5131
MAX_TARGET_LEN = 12 #tbd en vrai j'ai mis 12 signes, mais je sais pas à quel point 12 mots/idées ensemble ça fait une phrase.

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
