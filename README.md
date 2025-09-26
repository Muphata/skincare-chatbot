# 🌸 Skincare Chatbot Assistant

This project is a **Retrieval-Augmented Generation (RAG) chatbot** for skincare advice.  
It uses **FAISS**, **OpenAI embeddings**, and a **Gradio UI** to provide answers based on curated skincare documents.

---

## 🚀 Try it in Google Colab

Click below to launch the chatbot directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Muphata/skincare-chatbot/blob/main/notebooks/skincare_demo.ipynb)

---

## 🛠️ Quickstart (Colab)

Run these steps inside the Colab notebook:

```python
# 1. Clone the repository
!git clone https://github.com/Muphata/skincare-chatbot.git
%cd skincare-chatbot

# 2. Install dependencies
!pip install -r requirements.txt

# 3. ⚠️ Add your OpenAI API key
import os
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"

# 4. Build FAISS index
!python build_rag.py

# 5. Launch chatbot (Gradio UI)
!python chatbot_logic.py
