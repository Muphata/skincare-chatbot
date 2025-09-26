# Skincare Chatbot Assistant 💬✨

This project is a **Retrieval-Augmented Generation (RAG) skincare chatbot** that helps users analyze skincare products, ingredients, and routines. It uses **FAISS for semantic search** and integrates with **OpenAI embeddings**.

---

## 🚀 Features
- OCR support for extracting skincare ingredients from images
- FAISS-powered semantic search
- RAG pipeline for answering skincare-related questions
- Gradio UI for easy interaction

---

## 📂 Project Structure
- `skincarefinalproject/` → Core chatbot code
- `build_rag.py` → Build FAISS index
- `chatbot_logic.py` → Main chatbot logic
- `rag_draft.py` → Experimental RAG scripts
- `testrag.py` → Testing script
- `requirements.txt` → Dependencies
- `notebooks/skincare_demo.ipynb` → Colab demo

---

## ⚙️ Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/skincare-chatbot.git
cd skincare-chatbot
