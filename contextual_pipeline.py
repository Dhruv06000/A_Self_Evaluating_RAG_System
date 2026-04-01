from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage , SystemMessage,AIMessage
from config import DEVICE, EMBEDDING_MODEL_NAME

load_dotenv()

persistent_directory = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": DEVICE},  # or "cpu" if you don't have a GPU
    encode_kwargs={"normalize_embeddings": True}
)

db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model
        )
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
# Store our conversation as messages
chat_history = []

def ask_question(user_question):
  # Step 1: Make the question clear using conversation history
  if chat_history:
    # Ask AI to make the question standalone
    messages = [
      SystemMessage(content="Given the chat history, rephrase the user's question to be clear and standalone, without losing the original intent and return the rewritten question."),
    ] + chat_history + [
      HumanMessage(content=f"New question: {user_question}")
    ]
    result = model.invoke(messages)
    search_query = result.content.strip()
    print(f"Searching for: {search_query}")
  else: 
    search_query = user_question

  # Step 2: Retrieve relevant documents
  retriver = db.as_retriever(
  search_type = "similarity_score_threshold",
  search_kwargs = {
    "score_threshold": 0.3,

    "k": 5
  })
  relevant_docs = retriver.invoke(search_query)

  print(f"Found {len(relevant_docs)} relevant documents:")
  for i, doc in enumerate(relevant_docs, 1):
      # Show first 2 lines of each document
      lines = doc.page_content.split('\n')[:2]
      preview = '\n'.join(lines)
      print(f"  Doc {i}: {preview}...")

  # Step 3: Create final prompt
  combined_input = f"""Based on the following documents, please answer this question: {user_question}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer and  Do not make up answers. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""
  # Step 4: Get the answer
  messages = [
    SystemMessage(content="You are a helpful assistant that answers question based on the provided documents."),
  ] + chat_history + [
    HumanMessage(content=combined_input)
  ]
  result = model.invoke(messages)
  answer = result.content

  # Step 5: Remember this conversation
  chat_history.append(HumanMessage(content = user_question))
  chat_history.append(AIMessage(content = answer))

  print(f"\nAnswer: {answer}")
  return answer


# Simple chat loop
def start_chat():
  print("Ask me a question! (type 'exit' to quit)")

  while True:
    question = input("\nYour question: ")

    if question.lower() == "exit":
      print("Goodbye!")
      break

    ask_question(question)

if __name__ == "__main__":
  start_chat()
