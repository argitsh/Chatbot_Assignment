import streamlit as st
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import chromadb
from chromadb.utils import embedding_functions

# ---------------- CONFIG ----------------
CHROMA_DIR = "/content/chroma_db"  # Path to your stored Chroma DB
COLLECTION_NAME = "training_docs"
MODEL_NAME = "stabilityai/stablelm-zephyr-3b"
TOP_K = 3
TEMPERATURE = 0.3
MAX_NEW_TOKENS = 300
# ----------------------------------------

st.set_page_config(page_title="Colab RAG Chatbot", layout="wide")

# ---- Load Chroma DB ----
@st.cache_resource
def load_chroma():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name=COLLECTION_NAME)

collection = load_chroma()

def get_collection_size():
    try:
        return collection.count()
    except:
        return 0

# ---- Load Model & Tokenizer ----
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_model()

# ---- Retrieval ----
def retrieve_top_k(query, k=3):
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents","metadatas","distances"]
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    return [{"text": doc, "meta": meta, "distance": dist} for doc, meta, dist in zip(docs, metas, dists)]

# ---- Prompt Template ----
PROMPT_TEMPLATE = """
You are a helpful assistant. Use ONLY the provided context to answer the question.
If the answer is not in the context, say "I could not find the answer in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

# ---- Streaming Generation ----
def stream_generate(prompt, temperature, max_new_tokens):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=temperature,
        streamer=streamer
    )
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    partial = ""
    for token in streamer:
        partial += token
        yield partial
    thread.join()

# ---- Sidebar ----
with st.sidebar:
    st.header("Chatbot Info")
    st.write("**Model in use:**", MODEL_NAME)
    st.write("**Indexed Chunks:**", get_collection_size())
    if st.button("Clear Chat"):
        st.session_state.clear()

# ---- Chat State ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources" not in st.session_state:
    st.session_state.sources = []

# ---- Chat UI ----
st.title("📄 RAG Chatbot (Phi-3, Colab, Streamlit)")

# Display past messages
for role, msg in st.session_state.messages:
    if role == "user":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Assistant:** {msg}")

# User input
user_query = st.text_input("Ask a question about the document:")
if st.button("Send") and user_query:
    st.session_state.messages.append(("user", user_query))
    sources = retrieve_top_k(user_query, k=TOP_K)
    st.session_state.sources = sources
    context_text = "\n\n".join([f"[{i+1}] {src['text']}" for i, src in enumerate(sources)])
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=user_query)

    st.markdown("**Assistant:**")
    response_placeholder = st.empty()
    partial_text = ""
    for partial_text in stream_generate(prompt, TEMPERATURE, MAX_NEW_TOKENS):
        response_placeholder.markdown(f"{partial_text}")
    st.session_state.messages.append(("assistant", partial_text))

# ---- Sources ----
if st.session_state.sources:
    st.subheader("Sources used")
    for i, src in enumerate(st.session_state.sources):
        st.markdown(f"**[{i+1}]** {src['text'][:400]}...")
        st.caption(f"Distance: {src['distance']:.4f}")
