import torch

#Config and hyper parameters

#####################################

#### PSA : WE SHOULD CHANGE THESE HYPER PARAMETERS HERE !

BATCH_SIZE = 4 ### can be changed later. again, it is only for it to work here
SEQ_LEN = 100

CODEBOOK_SIZE = 1024   # nbrs codes VQVAE --> to remove later
D_MODEL = 768
NHEAD = 12 ## ARBITRARY CHOICE
NUM_LAYERS = 6 ## ARBITRARY CHOICE
MASK_RATIO = 0.4 ### 40% of mask but can be changed after

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"