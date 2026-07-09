import torch

URL = "https://www.gutenberg.org/files/2591/2591-0.txt"
FILE_PATH = "fairy_tales.txt"
TEMP_CORPUS = "temp_corpus.txt"
VOCAB_SIZE = 5000
MODEL_PATH = "fairy_tale_gpt.pth"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BLOCK_SIZE = 256
BATCH_SIZE = 16
N_EMBD = 384
NUM_HEADS = 6
N_LAYER = 4
DROPOUT = 0.2
HEAD_SIZE = N_EMBD // NUM_HEADS

LEARNING_RATE = 3e-4
MAX_ITERS = 3500
EVAL_INTERVAL = 500
EVAL_ITERS = 200
