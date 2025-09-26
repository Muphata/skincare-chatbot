import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# --------------------
# Load environment
# --------------------
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# --------------------
# Load FAISS index
# --------------------
print("🔄 Loading FAISS index...")
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
vector_store = FAISS.load_local(
    "faiss_index_txt",
    embedding_model,
    allow_dangerous_deserialization=True
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
print("✅ FAISS index loaded.\n")

# --------------------
# Init LLM (GPT-4o-mini or gpt-3.5-turbo)
# --------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_KEY
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# --------------------
# Simple REPL
# --------------------
print("💬 Type 'exit' to quit.\n")
while True:
    query = input("❓ Your question: ").strip()
    if query.lower() in ["exit", "quit", "q"]:
        print("👋 Bye!")
        break

    response = qa_chain.invoke({"query": query})

    print("\n--- 📝 Answer ---\n")
    print(response["result"].strip())

    print("\n--- 📚 Sources ---\n")
    for i, doc in enumerate(response["source_documents"], 1):
        source = doc.metadata.get("source", "unknown")
        snippet = doc.page_content[:200].replace("\n", " ")
        print(f"{i}. [Source: {source}] {snippet}{'...' if len(snippet) > 200 else ''}")

    print("\n" + "="*70 + "\n")
