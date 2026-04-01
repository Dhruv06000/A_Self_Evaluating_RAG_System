import os 
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import DEVICE, EMBEDDING_MODEL_NAME
from dotenv import load_dotenv


load_dotenv()

def load_documents(docs_path = "data"):
   """Load all text files from the docs directory"""
   print(f"Loading documents from {docs_path}...")

   #Check if docs directory exists
   if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory '{docs_path}' not found. Please create it and add your text files.")
   
    #Load all .txtes from the docs directory
   loader = DirectoryLoader(path = docs_path, glob = "*.txt", loader_cls = TextLoader,loader_kwargs={"encoding": "utf-8"})

   documents = loader.load()

   if len(documents) == 0:
        raise ValueError(f"No text files found in '{docs_path}'. Please add some .txt files to the directory.")

   for i,doc in enumerate(documents[:2]): #Show first 2 files
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

   return documents

def split_documents(documents, chunk_size = 400, chunk_overlap = 100):
  """Split documents into smaller chunks with overlap"""
  print("Splitting documents into chunks...")
  
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", ".", " ", ""]
)

  chunks = text_splitter.split_documents(documents)
  
  if chunks:
    
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

  return chunks

def create_vector_store(chunks, persist_directory = "db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")
    
    embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": DEVICE},  # or "cpu" if you don't have a GPU
    encode_kwargs={"normalize_embeddings": True}
)

    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vector_store = Chroma.from_documents(
        documents = chunks,
        embedding = embedding_model,
        persist_directory = persist_directory,
        collection_metadata = {"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")

    print(f"Vector store created and saved to {persist_directory}")

    return vector_store



def main():
    """Main ingestion pipeline"""
    print("=== RAG Document Ingestion Pipeline ===\n")

    # Define paths
    docs_path = "data"
    persistent_directory = "db/chroma_db"
    
    # Check if vector store already exists
    if os.path.exists(persistent_directory):
        print("✅ Vector store already exists. No need to re-process documents.")
        
        embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device":DEVICE},  # or "cpu" if you don't have a GPU
    encode_kwargs={"normalize_embeddings": True}
)
        vector_store = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model, 
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"Loaded existing vector store with {vector_store._collection.count()} documents")
        return vector_store
    print("Persistent directory does not exist. Initializing vector store...\n")
    #Step 1: Load documents
    documents = load_documents(docs_path)

    # Step 2: Split into chunks
    chunks = split_documents(documents)

    # Step 3: Create vector store
    vector_store = create_vector_store(chunks, persist_directory = persistent_directory)

    print("\n✅ Ingestion complete! Your documents are now ready for RAG queries.")
    return vector_store


if __name__ == "__main__":
   main()
