"""
RAG Builder — FAISS (cosine) + OpenAI Embeddings
- Put 5–10 (or more) .txt files into ./docs
- Builds FAISS index at ./faiss_index_txt
- Uses OpenAI embeddings so it's 100% compatible with your chatbot
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# --------------------
# Load environment
# --------------------
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file!")

# --------------------
# Load documents
# --------------------
loader = DirectoryLoader('./docs', glob='**/*.txt', show_progress=True)
documents = loader.load()

# --------------------
# Split into chunks
# --------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=60
)
docs = text_splitter.split_documents(documents)

print(f"✅ Split {len(documents)} documents into {len(docs)} chunks.")

# --------------------
# Embeddings (OpenAI)
# --------------------
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)

# --------------------
# Build FAISS index
# --------------------
vector_store = FAISS.from_documents(docs, embedding_model)

# Save FAISS locally
vector_store.save_local("faiss_index_txt")
print("✅ FAISS index built and saved to ./faiss_index_txt")

# --------------------
# Quick test
# --------------------
retriever = vector_store.as_retriever()

sample_query = "What is the best skincare routine for dry skin?"
results = retriever.get_relevant_documents(sample_query)

print(f"🔎 Query: {sample_query}")
print("Top result snippet:", results[0].page_content[:200])
