# 🧠 Self-Evaluating RAG System

A Retrieval-Augmented Generation (RAG) system built with context-awareness and modular pipelines. This project demonstrates how to ingest documents, store embeddings in a vector database, and retrieve relevant information to generate accurate answers—with and without conversational context.

## 📁 Project Structure

```
SELF_EVALUATING_RAG_SYSTEM/
│
├── data/ # Raw text documents (.txt files)
├── db/
│   └── chroma_db/ # ChromaDB vector store (auto-generated)
│
├── ingestion_pipeline.py # Document loading, chunking, embedding, storage
├── retriever_pipeline.py # Basic retrieval + QA (no memory)
├── contextual_pipeline.py # Context-aware retrieval + QA (with memory)
│
└── README.md
```

## ⚙️ Features

- 📄 Document ingestion and preprocessing
- ✂️ Smart text chunking
- 🔢 Embedding generation
- 🗂️ Vector storage using ChromaDB
- 🔍 Semantic search over documents
- 🤖 Answer generation from retrieved context
- 🧠 Context-aware conversation support
- 🧪 Modular pipelines for experimentation

## 🔄 Pipelines Overview

1. Ingestion Pipeline (ingestion_pipeline.py)

Responsible for preparing your data for retrieval:

- Loads .txt documents from the data/ folder
- Splits text into chunks
- Converts chunks into embeddings
- Stores embeddings in ChromaDB (db/ folder)

2. Retriever Pipeline (retriever_pipeline.py)

Basic RAG implementation:

- Takes a user query
- Converts query into embedding
- Retrieves similar chunks from vector DB
- Generates an answer

⚠️ Limitation:

- No memory → Each query is independent (stateless)

3. Contextual Pipeline (contextual_pipeline.py)

Improved RAG with context awareness:

- Maintains conversation history
- Uses previous interactions as context
- Produces more coherent and relevant answers

✅ Suitable for chatbot-like interactions

## 🧪 Example

User: What does Tesla do?  
→ Generates answer from documents

User: Who is its CEO?  
→ (Contextual pipeline remembers previous question)

## 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/self-evaluating-rag.git
cd self-evaluating-rag_system
```

## 2. Install Dependencies

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 3.Create a .env file:

- GEMINI_API_KEY=your_api_key_here

## 4. Run Ingestion Pipeline
```
- python ingestion_pipeline.py
```
This will:

- Process documents
- Generate embeddings
- Store them in db/

## 5. Run Query Pipelines

Without Context:

```
- python retriever_pipeline.py
```

With Context:

```
- python contextual_pipeline.py
```

## 🧠 How It Works

1. Documents are embedded into vector space
2. User query is also embedded
3. Similar vectors are retrieved using semantic search
4. Retrieved context is passed to an LLM
5. The LLM generates a final answer

## 🔍 Future Improvements

- ✅ Add evaluation metrics (faithfulness, relevance, etc.)
- 📊 Integrate logging & monitoring
- 💬 Build a UI (Streamlit / React)
- 🔁 Add feedback loop for self-evaluation
- ⚡ Optimize retrieval (hybrid search, reranking)

## 🛠️ Tech Stack

- Python
- RAG(Retrieval-Augmented Generation)
- ChromaDB (Vector DB)
- Gemini API (LLM)
- Embedding Model

# 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.
