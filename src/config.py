import torch

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Models
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_NAME = "llama-3.3-70b-versatile"
REWRITE_MODEL_NAME = "llama-3.1-8b-instant"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FAITHFULNESS_MODEL_NAME = "llama-3.1-8b-instant"

# Paths
DATA_DIR = "data"
CHROMA_DB_DIR = "db/chroma_db"

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MIN_CHUNK_LENGTH = 100

# Retrieval
SEMANTIC_TOP_K = 15
BM25_TOP_K = 15
RERANK_TOP_K = 6
RERANK_SCORE_THRESHOLD = 0.2

# Generation / Evaluation
MAX_CONTEXT_CHARS = 6000
FAITHFULNESS_MIN_SCORE = 7
MAX_RETRIES = 2
