# 🧠 Self-Evaluating RAG System for Technical Question Answering

A modular **Self-Evaluating Retrieval-Augmented Generation (RAG)** system designed to improve factual grounding and reduce hallucinations in technical question answering.

The system combines **query rewriting, hybrid retrieval, ChromaDB vector search, BM25 keyword retrieval, cross-encoder reranking, grounded answer generation, runtime faithfulness evaluation, and retry-based response refinement**.

This project also includes a **Basic RAG baseline** and a full **DeepEval-based evaluation pipeline** to compare retrieval quality, grounding, answer relevancy, contextual precision, contextual recall, and latency.

---

## 📌 Project Title

**A Self-Evaluating RAG System for Technical Question Answering**

---

## 🎯 Project Objective

Large Language Models can generate fluent answers, but they may also produce unsupported or hallucinated responses.

This project aims to build a RAG system that can:

- retrieve relevant technical context,
- generate grounded answers,
- evaluate its own generated answer,
- detect unsupported claims,
- retry generation with stricter grounding instructions,
- compare performance against a Basic RAG baseline.

---

## 🧩 Key Features

- 🔍 **Hybrid Retrieval** — combines semantic vector search and BM25 keyword search.
- 🧠 **Query Rewriting** — uses `llama-3.1-8b-instant` to rewrite user queries for better retrieval.
- 🗂️ **ChromaDB Vector Store** — stores document embeddings persistently.
- 📄 **PDF + TXT Knowledge Base** — supports official documentation `.txt` files and research paper PDFs.
- ✂️ **Document Chunking** — uses overlapping chunks for better context retrieval.
- 🎯 **Cross-Encoder Reranking** — uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to rerank retrieved chunks.
- 🤖 **Grounded Answer Generation** — uses `llama-3.3-70b-versatile` for final answer generation.
- ✅ **Runtime Faithfulness Evaluation** — uses `llama-3.1-8b-instant` to verify whether generated answers are supported by retrieved context.
- 🔁 **Retry-Based Refinement** — retries generation with stricter grounding prompts when answers are not faithful.
- 📊 **DeepEval Evaluation** — evaluates Answer Relevancy, Faithfulness, Contextual Precision, Contextual Recall, and Latency.
- ⚖️ **Basic RAG Baseline** — includes a Basic RAG system for comparison.

---

## 🏗️ System Architecture

```text
USER QUESTION
      │
      ▼
QUERY REWRITING
llama-3.1-8b-instant
      │
      ▼
HYBRID RETRIEVAL
Semantic Search + BM25
      │
      ▼
CHROMADB KNOWLEDGE BASE
Technical Docs + PDF
      │
      ▼
DEDUPLICATION
Remove duplicate chunks
      │
      ▼
CROSS-ENCODER RERANKING
ms-marco-MiniLM-L-6-v2
      │
      ▼
TOP RELEVANT CONTEXT CHUNKS
      │
      ▼
CONTEXT CONSTRUCTION
Merge top chunks into prompt context
      │
      ▼
ANSWER GENERATION
llama-3.3-70b-versatile
      │
      ▼
FAITHFULNESS EVALUATION
llama-3.1-8b-instant
      │
      ├── Faithful → Final Answer + Sources
      │
      └── Not Faithful
              │
              ▼
       Retry with stricter prompt
              │
              ▼
       Regenerate answer
```

---

## 📁 Project Structure

```text
Self_Evaluating_RAG_System_refactored/
│
├── data/
│   ├── Attention_Is_All_You_Need.pdf
│   ├── docs.langchain.com_*.txt
│   └── docs.trychroma.com_*.txt
│
├── db/
│   └── chroma_db/
│
├── src/
│   ├── config.py
│   │
│   ├── ingestion/
│   │   ├── download_doc.py
│   │   └── ingestion_pipeline.py
│   │
│   ├── generation/
│   │   └── answer_generator.py
│   │
│   ├── retrieval/
│   │   ├── contextual_retrieval.py
│   │   ├── retriever.py
│   │   └── reranker.py
│   │
│   ├── evaluation/
│   │   ├── benchmark_questions.json
│   │   ├── eval_config.py
│   │   ├── metrics.py
│   │   ├── run_evaluation.py
│   │   ├── run_baseline_evaluation.py
│   │   └── results/
│   │       ├── baseline/
│   │       └── self_eval/
│   │
│   └── prompts/
│
├── requirements.txt
├── README.md
└── .env
```

---

## 🧠 Final Model Stack

| Component                      | Model                                  |
| ------------------------------ | -------------------------------------- |
| Embedding Model                | `BAAI/bge-base-en-v1.5`                |
| Query Rewriter                 | `llama-3.1-8b-instant`                 |
| Answer Generator               | `llama-3.3-70b-versatile`              |
| Runtime Faithfulness Evaluator | `llama-3.1-8b-instant`                 |
| Cross-Encoder Reranker         | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector Database                | ChromaDB                               |
| Evaluation Framework           | DeepEval                               |
| Evaluation Judge               | Local Mistral / DeepSeek via Ollama    |

---

## ⚙️ Configuration

```python
DEVICE = "cuda"

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_NAME = "llama-3.3-70b-versatile"
REWRITE_MODEL_NAME = "llama-3.1-8b-instant"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FAITHFULNESS_MODEL_NAME = "llama-3.1-8b-instant"

DATA_DIR = "data"
CHROMA_DB_DIR = "db/chroma_db"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MIN_CHUNK_LENGTH = 100

SEMANTIC_TOP_K = 15
BM25_TOP_K = 15
RERANK_TOP_K = 6
RERANK_SCORE_THRESHOLD = 0.2

MAX_CONTEXT_CHARS = 6000
FAITHFULNESS_MIN_SCORE = 7
MAX_RETRIES = 2
```

---

## 📚 Knowledge Base

The knowledge base contains:

- LangChain documentation
- ChromaDB documentation
- RAG and retrieval documentation
- Embedding model documentation
- Research paper PDF: `Attention_Is_All_You_Need.pdf`

Supported file formats:

| File Type | Supported |
| --------- | --------- |
| `.txt`    | Yes       |
| `.pdf`    | Yes       |

---

## 🔄 Main Pipeline

### 1. Document Ingestion

The ingestion pipeline:

- loads `.txt` files,
- loads `.pdf` files using `PyPDFLoader`,
- splits documents into chunks,
- removes very short chunks,
- creates embeddings,
- stores embeddings in ChromaDB.

Run:

```bash
python -m src.ingestion.ingestion_pipeline
```

---

### 2. Query Rewriting

The query rewriting step improves retrieval by making the user query more specific and keyword-rich.

Example:

```text
Original Query:
Why is retrieval useful for LLMs?

Rewritten Query:
Retrieval usefulness for large language models, finite context, static knowledge, external knowledge access
```

---

### 3. Hybrid Retrieval

The system retrieves documents using:

- semantic search from ChromaDB,
- BM25 keyword retrieval.

This improves both semantic matching and exact keyword matching.

---

### 4. Cross-Encoder Reranking

After hybrid retrieval, duplicate chunks are removed and the remaining chunks are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Only the top relevant chunks are passed to the answer generation model.

---

### 5. Context Construction

The top reranked chunks are combined into a single context block.

The maximum context length is controlled using:

```python
MAX_CONTEXT_CHARS = 6000
```

---

### 6. Answer Generation

The answer generation model uses only the retrieved context.

Model:

```text
llama-3.3-70b-versatile
```

The prompt instructs the model to:

- answer only from context,
- avoid outside knowledge,
- keep answers concise,
- avoid unsupported claims.

---

### 7. Runtime Faithfulness Evaluation

After generation, the answer is evaluated using:

```text
llama-3.1-8b-instant
```

The evaluator returns:

```json
{
  "faithful": true,
  "score": 9,
  "reason": "The answer is supported by the retrieved context."
}
```

The answer is accepted if:

```python
faithfulness_score >= 7
```

---

### 8. Retry-Based Refinement

If the answer is not faithful:

- the system adds a stricter retry note,
- regenerates the answer,
- checks faithfulness again,
- retries up to 2 times.

```python
MAX_RETRIES = 2
```

---

## 🧪 Basic RAG Baseline

This project includes a Basic RAG baseline for comparison.

The Basic RAG system uses:

- ChromaDB semantic retrieval,
- no query rewriting,
- no BM25,
- no reranking,
- no faithfulness evaluation,
- no retry mechanism.

Basic RAG flow:

```text
User Query
   │
   ▼
Semantic Retrieval
   │
   ▼
Context Construction
   │
   ▼
Answer Generation
   │
   ▼
Final Answer
```

Run Basic RAG evaluation:

```bash
python -m src.evaluation.run_baseline_evaluation
```

---

## 📊 Evaluation

The system was evaluated using DeepEval on 50 benchmark questions.

Evaluation categories include:

- RAG concepts
- Retrieval
- Embeddings
- LangChain documentation
- ChromaDB
- Transformer PDF
- Multi-hop questions
- Edge cases

Metrics used:

| Metric               | Purpose                                                     |
| -------------------- | ----------------------------------------------------------- |
| Answer Relevancy     | Checks whether the answer addresses the question            |
| Faithfulness         | Checks whether the answer is supported by retrieved context |
| Contextual Precision | Checks ranking quality of retrieved chunks                  |
| Contextual Recall    | Checks whether required context was retrieved               |
| Latency              | Measures response time                                      |

---

## 📈 Final Evaluation Results

### Basic RAG Results

| Metric               |   Score |
| -------------------- | ------: |
| Answer Relevancy     |  0.8373 |
| Faithfulness         |  0.7880 |
| Contextual Precision |  0.7804 |
| Contextual Recall    |  0.7952 |
| Average Latency      | 0.7828s |

---

### Self-Evaluating RAG Results

| Metric               |   Score |
| -------------------- | ------: |
| Answer Relevancy     |  0.9310 |
| Faithfulness         |  0.7830 |
| Contextual Precision |  0.8260 |
| Contextual Recall    |  0.8791 |
| Average Latency      | 2.7183s |

---

### Basic RAG vs Self-Evaluating RAG

| Metric               | Basic RAG | Self-Evaluating RAG |
| -------------------- | --------: | ------------------: |
| Answer Relevancy     |    0.8373 |              0.9310 |
| Faithfulness         |    0.7880 |              0.7830 |
| Contextual Precision |    0.7804 |              0.8260 |
| Contextual Recall    |    0.7952 |              0.8791 |
| Average Latency      |   0.7828s |             2.7183s |

---

## 📌 Result Summary

The Self-Evaluating RAG system improved:

- Answer Relevancy
- Contextual Precision
- Contextual Recall

The system introduced higher latency because of:

- query rewriting,
- hybrid retrieval,
- reranking,
- runtime faithfulness evaluation,
- retry mechanism.

This creates a realistic tradeoff:

```text
Better answer quality and grounding
at the cost of increased latency.
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Dhruv06000/Self_Evaluating_RAG_System.git
cd Self_Evaluating_RAG_System
```

---

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If needed:

```bash
pip install pypdf
pip install ollama
pip install langchain-ollama
```

---

### 4. Create `.env` File

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

### 5. Add Documents

Place your documents inside the `data/` folder.

Supported:

```text
.txt
.pdf
```

Example:

```text
data/
├── Attention_Is_All_You_Need.pdf
├── docs.langchain.com_oss_python_langchain_rag.txt
├── docs.langchain.com_oss_python_langchain_retrieval.txt
└── docs.trychroma.com_docs_overview_introduction.txt
```

---

### 6. Download Documentation Files

Run:

```bash
python .\src\ingestion\download_doc.py
```

---

### 7. Run Ingestion

```bash
python -m src.ingestion.ingestion_pipeline
```

This creates:

```text
db/chroma_db/
```

---

### 8. Run the Self-Evaluating RAG System

```bash
python -m src.retrieval.contextual_retrieval
```

---

### 9. Run Basic RAG

```bash
python -m src.retrieval.retriever
```

---

### 10. Run Full Self-Evaluating RAG Evaluation

```bash
python -m src.evaluation.run_evaluation
```

---

### 11. Run Basic RAG Evaluation

```bash
python -m src.evaluation.run_baseline_evaluation
```

---

## 🧾 Example Questions

```text
What is Retrieval-Augmented Generation?
Why is retrieval useful for large language models?
What is hybrid retrieval?
How do embeddings help semantic search?
What is the Transformer architecture?
What is self-attention?
What is ChromaDB used for?
```

---

## 🧪 Example Output

```text
Question:
What is Retrieval-Augmented Generation?

Answer:
Retrieval-Augmented Generation is a method that improves
a language model's answer by retrieving relevant external
knowledge at query time and using that retrieved context
during generation.

Sources:
- Retrieval - Docs by LangChain
- Build a RAG agent with LangChain

Faithfulness Evaluation:
faithful: true
score: 9
reason: The answer is supported by the retrieved context.
```

---

## 🛠️ Tech Stack

| Component               | Technology                        |
| ----------------------- | --------------------------------- |
| Language                | Python                            |
| RAG Framework           | LangChain                         |
| Vector Database         | ChromaDB                          |
| Embeddings              | HuggingFace                       |
| Reranking               | SentenceTransformers CrossEncoder |
| Keyword Retrieval       | BM25                              |
| LLM Provider            | Groq                              |
| Local Evaluation Models | Ollama                            |
| Evaluation Framework    | DeepEval                          |
| PDF Loader              | PyPDFLoader                       |

## 🔮 Future Improvements

- Add Streamlit UI
- Add retrieval caching
- Add multi-query retrieval
- Add query decomposition
- Add local answer generation
- Add semantic chunking
- Add table and figure extraction from PDFs
- Add conversation/session management
- Add advanced hallucination severity scoring

---

## 📌 Final Conclusion

This project demonstrates that adding self-evaluation to RAG improves technical question answering quality.

The proposed Self-Evaluating RAG system improves answer relevancy, contextual precision, and contextual recall compared to Basic RAG by using:

- query rewriting,
- hybrid retrieval,
- reranking,
- runtime faithfulness evaluation,
- retry-based refinement.

The main tradeoff is increased latency due to additional evaluation steps.

---

## 🤝 Contributing

Contributions are welcome. You can open issues or submit pull requests for improvements.

---

## 📄 License

This project is open-source and available under the MIT License.
