import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import (
    DEVICE, EMBEDDING_MODEL_NAME, DATA_DIR, CHROMA_DB_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LENGTH,
)
from src.utils.text_cleaner import clean_text, extract_title_and_content

load_dotenv()


def load_documents(docs_path: str = DATA_DIR):
    print(f"Loading documents from: {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory not found: {docs_path}")

    path = Path(docs_path)
    documents = []

    pdf_files = list(path.glob("*.pdf"))
    txt_files = list(path.glob("*.txt"))

    print(f"Found: {len(pdf_files)} PDFs, {len(txt_files)} text files\n")

    if not pdf_files and not txt_files:
        raise ValueError("No PDF or TXT files found in directory!")

    for pdf_path in pdf_files:
        print(f"Loading PDF: {pdf_path.name}")
        try:
            loader = PyPDFLoader(str(pdf_path))
            raw_pages = loader.load()

            loaded_pages = 0
            for page in raw_pages:
                content = clean_text(page.page_content)
                if not content.strip():
                    continue
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "title": pdf_path.stem,
                        "source": str(pdf_path),
                        "type": "book",
                        "page": page.metadata.get("page", 0),
                    },
                ))
                loaded_pages += 1
            print(f" Loaded {loaded_pages} pages from {pdf_path.name}")

        except Exception as e:
            print(f" Failed to load {pdf_path.name}: {e}")

    for txt_path in txt_files:
        print(f"Loading TXT: {txt_path.name}")
        try:
            loader = TextLoader(str(txt_path), encoding="utf-8-sig")
            raw_docs = loader.load()

            for doc in raw_docs:
                title, content = extract_title_and_content(doc.page_content)
                content = clean_text(content)
                if not content.strip():
                    continue
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "title": title,
                        "source": str(txt_path),
                        "type": "wikipedia",
                    },
                ))
            print(f" Loaded {txt_path.name}")

        except Exception as e:
            print(f" Failed to load {txt_path.name}: {e}")

    if not documents:
        raise ValueError("No content extracted from any file!")

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f"  Title:   {doc.metadata['title']}")
        print(f"  Type:    {doc.metadata['type']}")
        print(f"  Preview: {doc.page_content[:150]}...")

    print(f"\nLoaded {len(documents)} total documents")
    print(f"  from {len(pdf_files)} PDFs + {len(txt_files)} text files\n")
    return documents


def split_documents(documents, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    print("Splitting documents into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.\n")

    chunks = [c for c in chunks if len(c.page_content.strip()) > MIN_CHUNK_LENGTH]
    print(f"After filtering short chunks: {len(chunks)} chunks remaining\n")

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Title:  {chunk.metadata.get('title')}")
        print(f"Type:   {chunk.metadata.get('type')}")
        print(f"Length: {len(chunk.page_content)}")
        print(f"Content: {chunk.page_content[:200]}...")
        print("-" * 50)

    return chunks


def create_vector_store(chunks, persist_directory: str = CHROMA_DB_DIR):
    print("Creating embeddings...")

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": DEVICE},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 128,
        },
    )

    print("Building Chroma vector store...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print(f"✅ Vector store saved at: {persist_directory}")
    print(f"   Total chunks indexed: {vector_store._collection.count()}")
    return vector_store


def main():
    print("=== RAG Ingestion Pipeline ===\n")

    if os.path.exists(CHROMA_DB_DIR):
        print("✅ Vector store already exists. Loading...")

        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )

        db = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embedding_model,
        )

        print(f"   Total chunks loaded: {db._collection.count()}")
        return db

    print("No existing DB found. Creating new one...\n")

    documents = load_documents(DATA_DIR)
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks, CHROMA_DB_DIR)

    print("\n✅ Ingestion complete!")
    return vector_store


if __name__ == "__main__":
    main()
