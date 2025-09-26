"""
RAG Builder — FAISS (cosine) + Enriched Chunks (EN/AR)
- Put 5–10 (or more) .txt files into ./docs
- Builds FAISS index at ./faiss_skincare_index
- Uses multilingual-e5-large embeddings

pip install -U langchain langchain-community langchain-huggingface faiss-cpu
"""
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint

# Load documents from a directory
loader = DirectoryLoader('./docs', glob='**/*.txt', show_progress=True)
documents = loader.load()

# Split the loaded documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=60)
docs = text_splitter.split_documents(documents)

print(f"Split {len(documents)} documents into {len(docs)} chunks.")

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create a FAISS vector store from the text chunks and embeddings
vector_store = FAISS.from_documents(docs, embedding_model)

# Save the vector store locally for later use
vector_store.save_local("faiss_index_txt")

# Load the saved FAISS index
vector_store = FAISS.load_local("faiss_index_txt", embedding_model, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever()

# Initialize a Hugging Face LLM (Google Flan-T5-large)

from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    model_kwargs={"temperature": 0.5}
)

llm = HuggingFacePipeline(pipeline=generator)

# Create the Retrieval-Augmented Generation chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Get a response for a query
query = "What information is available about [topic from your text files]?"
response = qa_chain.invoke(query)

print("Query:", query)
print("Result:", response['result'])
print("Source Documents:", response['source_documents'])