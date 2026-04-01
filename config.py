import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"