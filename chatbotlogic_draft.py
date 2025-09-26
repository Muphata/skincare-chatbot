import os
import re
import traceback
import gradio as gr
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

# OCR
import pytesseract
from PIL import Image

# Function calling
from pydantic import BaseModel, Field

# --------------------
# Load environment
# --------------------
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# --------------------
# FAISS Index
# --------------------
print("✅ Loading FAISS index...")
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
vector_store = FAISS.load_local(
    "faiss_index_txt",
    embedding_model,
    allow_dangerous_deserialization=True
)
retriever = vector_store.as_retriever()
print("✅ FAISS index loaded.")

# --------------------
# LLM (GPT-4o-mini)
# --------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    max_tokens=500,
    openai_api_key=OPENAI_KEY,
)


# --------------------
# Greeting & Off-topic detector
# --------------------
def detect_chat_mode(user_text: str) -> str:
    greetings_en = ["hi", "hello", "hey", "good morning", "good evening", "hiya"]
    greetings_ar = ["مرحبا", "اهلا", "السلام عليكم", "صباح الخير", "مساء الخير"]

    low = user_text.strip().lower()

    if low in greetings_en or any(word in user_text for word in greetings_ar):
        return "greeting"

    # Check if it's clearly off-topic (not mentioning skincare/skin/ingredient/etc.)
    keywords = ["skin", "skincare", "ingredient", "acne", "serum", "cream", "moisturizer", "product", "use", "redness", "niacinamide"
                "sunscreen", "تونر", "بشرة", "جلد", "كريم", "سيروم", "مكونات", "الهيالورونيك", "الريتينول", "بشرتي", "جافة" ,"ندوب ", "حب الشباب"]
    if not any(k in user_text.lower() for k in keywords):
        return "offtopic"

    return "skincare"


# --------------------
# Memory
# --------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# --------------------
# Prompts
# --------------------
system_prompt = """You are a helpful skincare assistant. Follow the user's instructions exactly and answer clearly."""

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
    ("system", "Relevant information: {context}")
])

no_retrieval_prompt = PromptTemplate.from_template(
    "You are a helpful skincare assistant. Answer the question clearly.\n\nQuestion: {question}"
)

ingredient_prompt = PromptTemplate.from_template(
    """You are a skincare expert. The user provided product ingredients.
Analyze carefully and answer the question specifically.
Ingredients: {ingredients}
User Question: {question}
Give ingredient-specific advice: mention helpful ones, possible irritants, and why.
Be clear and practical."""
)

# --------------------
# Chains
# --------------------
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": rag_prompt},
    return_source_documents=False,
    output_key="answer"
)

no_retrieval_chain = LLMChain(
    llm=llm,
    prompt=no_retrieval_prompt,
    memory=memory,
    output_key="answer"
)

ingredient_chain = LLMChain(
    llm=llm,
    prompt=ingredient_prompt,
    output_key="answer"
)

# --------------------
# Pydantic schema for OCR cleanup
# --------------------
class ExtractIngredients(BaseModel):
    ingredients: list[str] = Field(..., description="List of ingredient names")

fn_chain = llm.with_structured_output(ExtractIngredients)

# --------------------
# Detect Arabic helper
# --------------------
def detect_language(text: str) -> str:
    if re.search(r"[\u0600-\u06FF]", text):
        return "ar"
    return "en"

# --------------------
# Format Answer
# --------------------
def format_answer(result):
    answer = result.get("answer") or result.get("result", "")
    if not answer.strip():
        answer = "⚠️ لم أتمكن من العثور على إجابة واضحة. حاول إعادة صياغة سؤالك."
    return f"--- Answer ---\n\n{answer}\n\n"

# --------------------
# OCR Function
# --------------------
def _ocr(image_path, history, ocr_state):
    if image_path:
        img = Image.open(image_path)
        raw_text = pytesseract.image_to_string(img, lang="eng+ara")

        try:
            fc = fn_chain.invoke(raw_text)
            ingredients = fc.ingredients
        except Exception as e:
            print("⚠️ OCR cleaning failed:", e)
            ingredients = [x.strip() for x in re.split(r",|\n", raw_text) if x.strip()]

        ocr_state["ingredients"] = ingredients

        reply = f"--- Extracted Ingredients ---\n\n{ingredients}"
        history.append({"role": "user", "content": "[Uploaded image]"})
        history.append({"role": "assistant", "content": reply})
        return history, history, ocr_state
    return history, history, ocr_state

# --------------------
# Main QA Function
# --------------------
def answer_text(user_text, history, ocr_state, use_retrieval):
    try:
        lang = detect_language(user_text)
        if lang == "ar":
            system_instruction = "أجب بالعربية فقط"
        else:
            system_instruction = "Answer in English only"

        # 🔎 New: detect chat mode
        mode = detect_chat_mode(user_text)

        if mode == "greeting":
            if lang == "ar":
                reply = "👋 مرحباً! كيف يمكنني مساعدتك في العناية بالبشرة اليوم؟"
            else:
                reply = "👋 Hello! How can I help you with skincare today?"
        elif mode == "offtopic":
            if lang == "ar":
                reply = "⚠️ أنا مساعد متخصص في العناية بالبشرة فقط. إذا كان لديك سؤال متعلق بالبشرة فسأكون سعيداً بالمساعدة. ✨"
            else:
                reply = "⚠️ I am a skincare assistant. I can't answer that question. If you have any skincare-related question, I’d be happy to help. ✨"
        else:
            # ✅ Normal skincare flow
            final_query = f"{system_instruction}: {user_text}"

            if ocr_state.get("ingredients"):
                ocr_text = ", ".join(ocr_state["ingredients"])
                result = ingredient_chain.invoke({
                    "ingredients": ocr_text,
                    "question": final_query
                })
                reply = format_answer(result)
                ocr_state["ingredients"] = []  # reset
            else:
                if use_retrieval:
                    result = qa_chain.invoke({"question": final_query})
                else:
                    result = no_retrieval_chain.invoke({"question": final_query})
                reply = format_answer(result)

        # Update chat history
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})
        return history, history, ocr_state

    except Exception as e:
        print("❌ ERROR:", e)
        traceback.print_exc()
        reply = f"⚠️ Error: {str(e)}"
        history.append({"role": "assistant", "content": reply})
        return history, history, ocr_state


# --------------------
# Gradio UI
# --------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## Skincare RAG Chatbot (Retrieval ON/OFF + Memory + OCR Ingredient Analysis)")

    with gr.Row():
        chatbot = gr.Chatbot(type="messages", height=550, scale=2, label="Chat History")
        with gr.Column():
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask me anything about skincare...",
                    label="Your Question",
                    lines=4,
                    scale=8
                )
            send_btn = gr.Button("Send")

            img_input = gr.Image(type="filepath", label="Upload product image (OCR)")
            retrieval_toggle = gr.Checkbox(value=True, label="Retrieval")
            clear = gr.Button("Clear Chat")

    state = gr.State([])
    ocr_state = gr.State({"ingredients": []})

    msg.submit(answer_text, [msg, state, ocr_state, retrieval_toggle], [chatbot, state, ocr_state])
    send_btn.click(answer_text, [msg, state, ocr_state, retrieval_toggle], [chatbot, state, ocr_state])
    img_input.upload(_ocr, [img_input, state, ocr_state], [chatbot, state, ocr_state])
    clear.click(lambda: ([], [], {"ingredients": []}), None, [chatbot, state, ocr_state])

if __name__ == "__main__":
    demo.launch()